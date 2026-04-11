use crate::filters::{FilterVertex, Filters};
use crate::layouts::BindLayouts;
use crate::pipelines::VERTEX_BUFFERS_DESCRIPTION_POS;
use crate::shaders::Shaders;
use crate::{
    BitmapSamplers, Pipelines, PosColorVertex, PosVertex, TextureTransforms,
    create_buffer_with_data,
};
use crate::buffer_pool::VertexInstancePool;
use fnv::FnvHashMap;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use swf::{GradientInterpolation, GradientRecord};
use wgpu::Backend;

/// Compute a 64-bit FNV-1a hash of a gradient definition without cloning the
/// records slice.  Used as the gradient texture cache key so that repeated
/// lookups (common for shared gradient shapes) pay only O(n) hash cost with no
/// heap allocation, instead of the O(n) clone that a `Vec`-keyed map would require.
///
/// Collision probability with FNV-64 over the bounded inputs (≤15 gradient
/// stops × 5 bytes each) is negligible; Flash's SWF specification caps gradients
/// at 15 colour stops and this constraint is enforced at the SWF-parser layer
/// before `gradient_cache_key` is ever called.
pub(crate) fn gradient_cache_key(interpolation: GradientInterpolation, records: &[GradientRecord]) -> u64 {
    let mut h = fnv::FnvHasher::default();
    interpolation.hash(&mut h);
    records.hash(&mut h);
    h.finish()
}

pub struct Descriptors {
    pub wgpu_instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub limits: wgpu::Limits,
    pub backend: Backend,
    pub queue: wgpu::Queue,
    pub bitmap_samplers: BitmapSamplers,
    pub bind_layouts: BindLayouts,
    pub quad: Quad,
    copy_pipeline: Mutex<FnvHashMap<(u32, wgpu::TextureFormat), wgpu::RenderPipeline>>,
    copy_srgb_pipeline: Mutex<FnvHashMap<(u32, wgpu::TextureFormat), wgpu::RenderPipeline>>,
    /// Pipeline cache for the final post-process pass (formats match).
    post_process_pipeline: Mutex<FnvHashMap<(u32, wgpu::TextureFormat), wgpu::RenderPipeline>>,
    /// Pipeline cache for the final post-process pass (sRGB surface variant).
    post_process_srgb_pipeline: Mutex<FnvHashMap<(u32, wgpu::TextureFormat), wgpu::RenderPipeline>>,
    pub shaders: Shaders,
    pipelines: Mutex<FnvHashMap<(u32, wgpu::TextureFormat), Arc<Pipelines>>>,
    pub filters: Filters,
    /// Pool of reusable `VERTEX | COPY_DST` buffers for per-frame instance data.
    /// Shared across all rendering paths via `Descriptors`; thread-safe
    /// through the pool's internal `Mutex`.
    pub vertex_instance_pool: VertexInstancePool,
    /// Cache of gradient textures keyed by a pre-computed FNV-1a hash of the
    /// `(GradientInterpolation, Vec<GradientRecord>)` pair.
    ///
    /// Using a pre-hashed `u64` key (computed by [`gradient_cache_key`]) avoids
    /// the `Vec::clone()` heap allocation that a `Vec`-keyed map would require on
    /// every lookup — important because gradient lookups occur at shape-register
    /// time, which is hot for SWFs with many gradient fills.
    ///
    /// `wgpu::Texture` is cheaply cloneable (internally `Arc`-backed).
    ///
    /// The cache is bounded by [`Self::purge_gradient_cache_if_oversized`],
    /// which is called at the end of every frame.
    pub(crate) gradient_texture_cache: Mutex<FnvHashMap<u64, wgpu::Texture>>,
}

impl Debug for Descriptors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Descriptors")
    }
}

impl Descriptors {
    pub fn new(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        let limits = device.limits();
        let bind_layouts = BindLayouts::new(&device);
        let bitmap_samplers = BitmapSamplers::new(&device);
        let shaders = Shaders::new(&device);
        let quad = Quad::new(&device);
        let filters = Filters::new(&device);
        let backend = adapter.get_info().backend;

        Self {
            wgpu_instance: instance,
            adapter,
            device,
            limits,
            backend,
            queue,
            bitmap_samplers,
            bind_layouts,
            quad,
            copy_pipeline: Default::default(),
            copy_srgb_pipeline: Default::default(),
            post_process_pipeline: Default::default(),
            post_process_srgb_pipeline: Default::default(),
            shaders,
            pipelines: Default::default(),
            filters,
            vertex_instance_pool: VertexInstancePool::new(),
            gradient_texture_cache: Mutex::new(FnvHashMap::default()),
        }
    }

    /// Evict the gradient texture cache once it grows beyond `MAX_ENTRIES`.
    ///
    /// Each entry is a 256×1 RGBA8 texture (≈ 1 KiB GPU memory).  The cap
    /// keeps worst-case VRAM usage bounded at roughly `MAX_ENTRIES` KiB while
    /// still achieving good reuse across typical SWF content (which tends to
    /// reuse a small number of gradient definitions).
    ///
    /// When the cache exceeds the limit, half of the entries are discarded
    /// rather than clearing everything, so the most-recently-inserted half
    /// is retained for the next frame.
    pub fn purge_gradient_cache_if_oversized(&self) {
        const MAX_ENTRIES: usize = 256;
        let mut cache = self
            .gradient_texture_cache
            .lock()
            .expect("gradient_texture_cache lock poisoned");
        if cache.len() > MAX_ENTRIES {
            // Retain the most-recently-inserted half instead of clearing all
            // entries at once.  This avoids a cliff-edge where a SWF with
            // MAX_ENTRIES+1 unique gradients causes a full-cache miss every frame.
            let keep = MAX_ENTRIES / 2;
            let to_remove = cache.len().saturating_sub(keep);
            let keys_to_remove: Vec<_> = cache.keys().take(to_remove).cloned().collect();
            for key in keys_to_remove {
                cache.remove(&key);
            }
        }
    }

    pub fn copy_srgb_pipeline(
        &self,
        format: wgpu::TextureFormat,
        msaa_sample_count: u32,
    ) -> wgpu::RenderPipeline {
        let mut pipelines = self
            .copy_srgb_pipeline
            .lock()
            .expect("Pipelines should not be already locked");
        pipelines
            .entry((msaa_sample_count, format))
            .or_insert_with(|| {
                let copy_texture_pipeline_layout =
                    &self
                        .device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: create_debug_label!("Copy sRGB pipeline layout").as_deref(),
                            bind_group_layouts: &[
                                &self.bind_layouts.globals,
                                &self.bind_layouts.transforms,
                                &self.bind_layouts.bitmap,
                            ],
                            push_constant_ranges: &[],
                        });
                self.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: create_debug_label!("Copy sRGB pipeline").as_deref(),
                        layout: Some(copy_texture_pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &self.shaders.copy_srgb_shader,
                            entry_point: Some("main_vertex"),
                            buffers: &VERTEX_BUFFERS_DESCRIPTION_POS,
                            compilation_options: Default::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &self.shaders.copy_srgb_shader,
                            entry_point: Some("main_fragment"),
                            targets: &[Some(wgpu::ColorTargetState {
                                format,
                                // All of our blending has been done by now, so we want
                                // to overwrite the target pixels without any blending
                                blend: Some(wgpu::BlendState::REPLACE),
                                write_mask: Default::default(),
                            })],
                            compilation_options: Default::default(),
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            strip_index_format: None,
                            front_face: wgpu::FrontFace::Ccw,
                            cull_mode: None,
                            polygon_mode: wgpu::PolygonMode::default(),
                            unclipped_depth: false,
                            conservative: false,
                        },
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState {
                            count: msaa_sample_count,
                            mask: !0,
                            alpha_to_coverage_enabled: false,
                        },
                        multiview: None,
                        cache: None,
                    })
            })
            .clone()
    }

    pub fn copy_pipeline(
        &self,
        format: wgpu::TextureFormat,
        msaa_sample_count: u32,
    ) -> wgpu::RenderPipeline {
        let mut pipelines = self
            .copy_pipeline
            .lock()
            .expect("Pipelines should not be already locked");
        pipelines
            .entry((msaa_sample_count, format))
            .or_insert_with(|| {
                let copy_texture_pipeline_layout =
                    &self
                        .device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: create_debug_label!("Copy pipeline layout").as_deref(),
                            bind_group_layouts: &[
                                &self.bind_layouts.globals,
                                &self.bind_layouts.transforms,
                                &self.bind_layouts.bitmap,
                            ],
                            push_constant_ranges: &[],
                        });
                self.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: create_debug_label!("Copy pipeline").as_deref(),
                        layout: Some(copy_texture_pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &self.shaders.copy_shader,
                            entry_point: Some("main_vertex"),
                            buffers: &VERTEX_BUFFERS_DESCRIPTION_POS,
                            compilation_options: Default::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &self.shaders.copy_shader,
                            entry_point: Some("main_fragment"),
                            targets: &[Some(wgpu::ColorTargetState {
                                format,
                                // All of our blending has been done by now, so we want
                                // to overwrite the target pixels without any blending
                                blend: Some(wgpu::BlendState::REPLACE),
                                write_mask: Default::default(),
                            })],
                            compilation_options: Default::default(),
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            strip_index_format: None,
                            front_face: wgpu::FrontFace::Ccw,
                            cull_mode: None,
                            polygon_mode: wgpu::PolygonMode::default(),
                            unclipped_depth: false,
                            conservative: false,
                        },
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState {
                            count: msaa_sample_count,
                            mask: !0,
                            alpha_to_coverage_enabled: false,
                        },
                        multiview: None,
                        cache: None,
                    })
            })
            .clone()
    }

    /// Returns the post-process pipeline for the given format and sample count.
    /// This pipeline replaces the plain copy for the scene → swapchain step,
    /// adding bilinear filtering, FXAA, sharpening, and colour correction.
    /// Used when the internal format equals the surface format.
    pub fn post_process_pipeline(
        &self,
        format: wgpu::TextureFormat,
        msaa_sample_count: u32,
    ) -> wgpu::RenderPipeline {
        let mut pipelines = self
            .post_process_pipeline
            .lock()
            .expect("Pipelines should not be already locked");
        pipelines
            .entry((msaa_sample_count, format))
            .or_insert_with(|| {
                let layout = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: create_debug_label!("Post-process pipeline layout").as_deref(),
                        bind_group_layouts: &[
                            &self.bind_layouts.globals,
                            &self.bind_layouts.transforms,
                            &self.bind_layouts.bitmap,
                        ],
                        push_constant_ranges: &[],
                    });
                self.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: create_debug_label!("Post-process pipeline").as_deref(),
                        layout: Some(&layout),
                        vertex: wgpu::VertexState {
                            module: &self.shaders.post_process_shader,
                            entry_point: Some("main_vertex"),
                            buffers: &VERTEX_BUFFERS_DESCRIPTION_POS,
                            compilation_options: Default::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &self.shaders.post_process_shader,
                            entry_point: Some("main_fragment"),
                            targets: &[Some(wgpu::ColorTargetState {
                                format,
                                blend: Some(wgpu::BlendState::REPLACE),
                                write_mask: Default::default(),
                            })],
                            compilation_options: Default::default(),
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            strip_index_format: None,
                            front_face: wgpu::FrontFace::Ccw,
                            cull_mode: None,
                            polygon_mode: wgpu::PolygonMode::default(),
                            unclipped_depth: false,
                            conservative: false,
                        },
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState {
                            count: msaa_sample_count,
                            mask: !0,
                            alpha_to_coverage_enabled: false,
                        },
                        multiview: None,
                        cache: None,
                    })
            })
            .clone()
    }

    /// Returns the post-process pipeline for the given sRGB surface format.
    /// Identical to `post_process_pipeline` but additionally applies the
    /// sRGB colour-space conversion before writing to the sRGB swapchain.
    pub fn post_process_srgb_pipeline(
        &self,
        format: wgpu::TextureFormat,
        msaa_sample_count: u32,
    ) -> wgpu::RenderPipeline {
        let mut pipelines = self
            .post_process_srgb_pipeline
            .lock()
            .expect("Pipelines should not be already locked");
        pipelines
            .entry((msaa_sample_count, format))
            .or_insert_with(|| {
                let layout = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: create_debug_label!("Post-process sRGB pipeline layout").as_deref(),
                        bind_group_layouts: &[
                            &self.bind_layouts.globals,
                            &self.bind_layouts.transforms,
                            &self.bind_layouts.bitmap,
                        ],
                        push_constant_ranges: &[],
                    });
                self.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: create_debug_label!("Post-process sRGB pipeline").as_deref(),
                        layout: Some(&layout),
                        vertex: wgpu::VertexState {
                            module: &self.shaders.post_process_srgb_shader,
                            entry_point: Some("main_vertex"),
                            buffers: &VERTEX_BUFFERS_DESCRIPTION_POS,
                            compilation_options: Default::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &self.shaders.post_process_srgb_shader,
                            entry_point: Some("main_fragment"),
                            targets: &[Some(wgpu::ColorTargetState {
                                format,
                                blend: Some(wgpu::BlendState::REPLACE),
                                write_mask: Default::default(),
                            })],
                            compilation_options: Default::default(),
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            strip_index_format: None,
                            front_face: wgpu::FrontFace::Ccw,
                            cull_mode: None,
                            polygon_mode: wgpu::PolygonMode::default(),
                            unclipped_depth: false,
                            conservative: false,
                        },
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState {
                            count: msaa_sample_count,
                            mask: !0,
                            alpha_to_coverage_enabled: false,
                        },
                        multiview: None,
                        cache: None,
                    })
            })
            .clone()
    }

    pub fn pipelines(&self, msaa_sample_count: u32, format: wgpu::TextureFormat) -> Arc<Pipelines> {
        let mut pipelines = self
            .pipelines
            .lock()
            .expect("Pipelines should not be already locked");
        pipelines
            .entry((msaa_sample_count, format))
            .or_insert_with(|| {
                Arc::new(Pipelines::new(
                    &self.device,
                    &self.shaders,
                    format,
                    msaa_sample_count,
                    &self.bind_layouts,
                ))
            })
            .clone()
    }
}

pub struct Quad {
    pub vertices_pos: wgpu::Buffer,
    pub vertices_pos_color: wgpu::Buffer,
    pub filter_vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub indices_line: wgpu::Buffer,
    pub indices_line_rect: wgpu::Buffer,
    pub texture_transforms: wgpu::Buffer,
}

impl Quad {
    pub fn new(device: &wgpu::Device) -> Self {
        let vertices_pos = [
            PosVertex {
                position: [0.0, 0.0],
            },
            PosVertex {
                position: [1.0, 0.0],
            },
            PosVertex {
                position: [1.0, 1.0],
            },
            PosVertex {
                position: [0.0, 1.0],
            },
        ];
        let vertices_pos_color = [
            PosColorVertex {
                position: [0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            PosColorVertex {
                position: [1.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            PosColorVertex {
                position: [1.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            PosColorVertex {
                position: [0.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
        ];
        let filter_vertices = [
            FilterVertex {
                position: [0.0, 0.0],
                uv: [0.0, 0.0],
            },
            FilterVertex {
                position: [1.0, 0.0],
                uv: [1.0, 0.0],
            },
            FilterVertex {
                position: [1.0, 1.0],
                uv: [1.0, 1.0],
            },
            FilterVertex {
                position: [0.0, 1.0],
                uv: [0.0, 1.0],
            },
        ];
        let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
        let indices_line: [u32; 2] = [0, 1];
        let indices_line_rect: [u32; 5] = [0, 1, 2, 3, 0];

        let vbo_pos = create_buffer_with_data(
            device,
            bytemuck::cast_slice(&vertices_pos),
            wgpu::BufferUsages::VERTEX,
            create_debug_label!("Quad vbo (pos)"),
        );

        let vbo_pos_color = create_buffer_with_data(
            device,
            bytemuck::cast_slice(&vertices_pos_color),
            wgpu::BufferUsages::VERTEX,
            create_debug_label!("Quad vbo (pos & color)"),
        );

        let vbo_filter = create_buffer_with_data(
            device,
            bytemuck::cast_slice(&filter_vertices),
            wgpu::BufferUsages::VERTEX,
            create_debug_label!("Quad vbo (filter)"),
        );

        let ibo = create_buffer_with_data(
            device,
            bytemuck::cast_slice(&indices),
            wgpu::BufferUsages::INDEX,
            create_debug_label!("Quad ibo"),
        );
        let ibo_line = create_buffer_with_data(
            device,
            bytemuck::cast_slice(&indices_line),
            wgpu::BufferUsages::INDEX,
            create_debug_label!("Line ibo"),
        );
        let ibo_line_rect = create_buffer_with_data(
            device,
            bytemuck::cast_slice(&indices_line_rect),
            wgpu::BufferUsages::INDEX,
            create_debug_label!("Line rect ibo"),
        );

        let tex_transforms = create_buffer_with_data(
            device,
            bytemuck::cast_slice(&[TextureTransforms {
                u_matrix: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }]),
            wgpu::BufferUsages::UNIFORM,
            create_debug_label!("Quad tex transforms"),
        );

        Self {
            vertices_pos: vbo_pos,
            vertices_pos_color: vbo_pos_color,
            filter_vertices: vbo_filter,
            indices: ibo,
            indices_line: ibo_line,
            indices_line_rect: ibo_line_rect,
            texture_transforms: tex_transforms,
        }
    }
}
