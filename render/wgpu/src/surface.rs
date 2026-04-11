mod commands;
pub mod target;

use crate::backend::RenderTargetMode;
use crate::blend::ComplexBlend;
use crate::buffer_pool::TexturePool;
use crate::dynamic_transforms::DynamicTransforms;
use crate::filters::FilterSource;
use crate::mesh::Mesh;
use crate::pixel_bender::{ShaderMode, run_pixelbender_shader_impl};
use crate::surface::commands::{Chunk, CommandRenderer, chunk_blends};
use crate::utils::{remove_srgb, supported_sample_count};
use crate::{Descriptors, MaskState, Pipelines, PostProcessQuality};
use ruffle_render::commands::CommandList;
use ruffle_render::pixel_bender_support::{ImageInputTexture, PixelBenderShaderArgument};
use ruffle_render::quality::StageQuality;
use std::sync::Arc;
use target::CommandTarget;
use tracing::instrument;

use crate::utils::run_post_process_pipeline;

pub use crate::surface::commands::LayerRef;

use self::commands::ChunkBlendMode;

#[derive(Debug)]
pub struct Surface {
    size: wgpu::Extent3d,
    quality: StageQuality,
    post_process_quality: PostProcessQuality,
    sample_count: u32,
    pipelines: Arc<Pipelines>,
    format: wgpu::TextureFormat,
    actual_surface_format: wgpu::TextureFormat,
    /// Total GPU draw calls issued during the most recent `draw_commands` call.
    last_draw_call_count: u32,
}

impl Surface {
    pub fn new(
        descriptors: &Descriptors,
        quality: StageQuality,
        post_process_quality: PostProcessQuality,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let frame_buffer_format = remove_srgb(surface_format);

        let sample_count = supported_sample_count(
            &descriptors.adapter,
            quality.sample_count(),
            frame_buffer_format,
        );
        let pipelines = descriptors.pipelines(sample_count, frame_buffer_format);
        Self {
            size,
            quality,
            post_process_quality,
            sample_count,
            pipelines,
            format: frame_buffer_format,
            actual_surface_format: surface_format,
            last_draw_call_count: 0,
        }
    }

    #[expect(clippy::too_many_arguments)]
    #[instrument(level = "debug", skip_all)]
    pub fn draw_commands_and_copy_to<'frame, 'global: 'frame>(
        &mut self,
        frame_view: &wgpu::TextureView,
        render_target_mode: RenderTargetMode,
        descriptors: &'global Descriptors,
        staging_belt: &'frame mut wgpu::util::StagingBelt,
        dynamic_transforms: &'global DynamicTransforms,
        draw_encoder: &'frame mut wgpu::CommandEncoder,
        meshes: &'global Vec<Mesh>,
        commands: CommandList,
        layer: LayerRef,
        texture_pool: &mut TexturePool,
        rect_batch_limit: usize,
        bitmap_batch_limit: usize,
    ) {
        let target = self.draw_commands(
            render_target_mode,
            descriptors,
            meshes,
            commands,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            layer,
            texture_pool,
            rect_batch_limit,
            bitmap_batch_limit,
        );

        run_post_process_pipeline(
            descriptors,
            self.format,
            self.actual_surface_format,
            frame_view,
            target.color_view(),
            target.whole_frame_bind_group(descriptors),
            target.globals(),
            1,
            draw_encoder,
            self.post_process_quality,
        );
    }

    #[expect(clippy::too_many_arguments)]
    #[instrument(level = "debug", skip_all)]
    pub fn draw_commands<'frame, 'global: 'frame>(
        &mut self,
        render_target_mode: RenderTargetMode,
        descriptors: &'global Descriptors,
        meshes: &'global Vec<Mesh>,
        commands: CommandList,
        staging_belt: &'global mut wgpu::util::StagingBelt,
        dynamic_transforms: &'global DynamicTransforms,
        draw_encoder: &'frame mut wgpu::CommandEncoder,
        nearest_layer: LayerRef<'frame>,
        texture_pool: &mut TexturePool,
        rect_batch_limit: usize,
        bitmap_batch_limit: usize,
    ) -> CommandTarget {
        let target = CommandTarget::new(
            descriptors,
            texture_pool,
            self.size,
            self.format,
            self.sample_count,
            render_target_mode,
            draw_encoder,
        );

        let mut num_masks = 0;
        let mut mask_state = MaskState::NoMask;
        let mut frame_draw_calls: u32 = 0;
        let chunks = chunk_blends(
            commands,
            descriptors,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            meshes,
            self.quality,
            target.width(),
            target.height(),
            match nearest_layer {
                LayerRef::Current => LayerRef::Parent(&target),
                layer => layer,
            },
            texture_pool,
            rect_batch_limit,
            bitmap_batch_limit,
        );

        for chunk in chunks {
            match chunk {
                Chunk::Draw(chunk, needs_stencil, transform_buffers) => {
                    transform_buffers.copy_to(
                        staging_belt,
                        &descriptors.device,
                        draw_encoder,
                        &dynamic_transforms.buffer,
                    );
                    let mut render_pass =
                        draw_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: create_debug_label!(
                                "Chunked draw calls {}",
                                if needs_stencil {
                                    "(with stencil)"
                                } else {
                                    "(Stencilless)"
                                }
                            )
                            .as_deref(),
                            color_attachments: &[target.color_attachments()],
                            depth_stencil_attachment: if needs_stencil {
                                target.stencil_attachment(descriptors, texture_pool)
                            } else {
                                None
                            },
                            ..Default::default()
                        });
                    render_pass.set_bind_group(0, target.globals().bind_group(), &[]);
                    let mut renderer = CommandRenderer::new(
                        &self.pipelines,
                        descriptors,
                        dynamic_transforms,
                        render_pass,
                        num_masks,
                        mask_state,
                        needs_stencil,
                    );

                    for command in &chunk {
                        renderer.execute(command);
                    }

                    num_masks = renderer.num_masks();
                    mask_state = renderer.mask_state();
                    frame_draw_calls += renderer.draw_call_count();
                }
                Chunk::Blend(texture, ChunkBlendMode::Shader(shader), needs_stencil) => {
                    assert!(
                        !needs_stencil,
                        "Shader blend should not need stencil buffer"
                    );
                    let parent_blend_buffer =
                        target.update_blend_buffer(descriptors, texture_pool, draw_encoder);
                    run_pixelbender_shader_impl(
                        descriptors,
                        shader,
                        ShaderMode::Filter,
                        &[
                            PixelBenderShaderArgument::ImageInput {
                                index: 0,
                                channels: 0xFF,
                                name: "background".to_string(),
                                texture: Some(ImageInputTexture::TextureRef(
                                    parent_blend_buffer.texture(),
                                )),
                            },
                            PixelBenderShaderArgument::ImageInput {
                                index: 1,
                                channels: 0xff,
                                name: "foreground".to_string(),
                                texture: Some(ImageInputTexture::TextureRef(texture.texture())),
                            },
                        ],
                        parent_blend_buffer.texture(),
                        draw_encoder,
                        target.color_attachments(),
                        target.sample_count(),
                        &FilterSource::for_entire_texture(texture.texture()),
                    )
                    .expect("Failed to run PixelBender blend mode");
                    // run_pixelbender_shader_impl issues one draw_indexed call internally.
                    frame_draw_calls += 1;
                }
                Chunk::Blend(texture, ChunkBlendMode::Complex(blend_mode), needs_stencil) => {
                    let parent = match blend_mode {
                        ComplexBlend::Alpha | ComplexBlend::Erase => {
                            match nearest_layer {
                                LayerRef::None => {
                                    // An Alpha or Erase with no Layer above it should be ignored
                                    continue;
                                }
                                LayerRef::Current => &target,
                                LayerRef::Parent(layer) => layer,
                            }
                        }
                        _ => &target,
                    };

                    let parent_blend_buffer =
                        parent.update_blend_buffer(descriptors, texture_pool, draw_encoder);

                    let blend_bind_group =
                        descriptors
                            .device
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: create_debug_label!(
                                    "Complex blend binds {:?} {}",
                                    blend_mode,
                                    if needs_stencil {
                                        "(with stencil)"
                                    } else {
                                        "(Stencilless)"
                                    }
                                )
                                .as_deref(),
                                layout: &descriptors.bind_layouts.blend,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(
                                            parent_blend_buffer.view(),
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::TextureView(
                                            texture.view(),
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: wgpu::BindingResource::Sampler(
                                            descriptors.bitmap_samplers.get_sampler(false, false),
                                        ),
                                    },
                                ],
                            });

                    let mut render_pass =
                        draw_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: create_debug_label!(
                                "Complex blend {:?} {}",
                                blend_mode,
                                if needs_stencil {
                                    "(with stencil)"
                                } else {
                                    "(Stencilless)"
                                }
                            )
                            .as_deref(),
                            color_attachments: &[target.color_attachments()],
                            depth_stencil_attachment: if needs_stencil {
                                target.stencil_attachment(descriptors, texture_pool)
                            } else {
                                None
                            },
                            ..Default::default()
                        });
                    render_pass.set_bind_group(0, target.globals().bind_group(), &[]);

                    if needs_stencil {
                        match mask_state {
                            MaskState::NoMask => {}
                            MaskState::DrawMaskStencil => {
                                render_pass.set_stencil_reference(num_masks - 1);
                            }
                            MaskState::DrawMaskedContent => {
                                render_pass.set_stencil_reference(num_masks);
                            }
                            MaskState::ClearMaskStencil => {
                                render_pass.set_stencil_reference(num_masks);
                            }
                        }
                        render_pass.set_pipeline(
                            self.pipelines.complex_blends[blend_mode].pipeline_for(mask_state),
                        );
                    } else {
                        render_pass.set_pipeline(
                            self.pipelines.complex_blends[blend_mode].stencilless_pipeline(),
                        );
                    }

                    render_pass.set_bind_group(1, target.whole_frame_bind_group(descriptors), &[0]);
                    render_pass.set_bind_group(2, &blend_bind_group, &[]);

                    render_pass.set_vertex_buffer(0, descriptors.quad.vertices_pos.slice(..));
                    render_pass.set_index_buffer(
                        descriptors.quad.indices.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );

                    render_pass.draw_indexed(0..6, 0, 0..1);
                    frame_draw_calls += 1;
                }
            }
        }

        // If nothing happened, ensure it's cleared so we don't operate on garbage data
        target.ensure_cleared(draw_encoder);

        self.last_draw_call_count = frame_draw_calls;
        target
    }

    pub fn quality(&self) -> StageQuality {
        self.quality
    }

    pub fn post_process_quality(&self) -> PostProcessQuality {
        self.post_process_quality
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn size(&self) -> wgpu::Extent3d {
        self.size
    }

    /// Total GPU `draw_indexed` calls from the most recent `draw_commands` call.
    /// Includes calls from `Chunk::Draw` (via `CommandRenderer`), `Chunk::Blend(Complex)`,
    /// and `Chunk::Blend(Shader)` passes.  Post-process copy passes (e.g. the MSAA
    /// resolve/copy in `draw_commands_and_copy_to`) are not included.
    pub fn draw_call_count(&self) -> u32 {
        self.last_draw_call_count
    }
}
