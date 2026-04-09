use crate::backend::RenderTargetMode;
use crate::blend::TrivialBlend;
use crate::blend::{BlendType, ComplexBlend};
use crate::buffer_builder::BufferBuilder;
use crate::buffer_pool::TexturePool;
use crate::dynamic_transforms::DynamicTransforms;
use crate::mesh::{DrawType, Mesh, as_mesh};
use crate::surface::Surface;
use crate::surface::target::CommandTarget;
use crate::{Descriptors, MaskState, Pipelines, RectInstance, Transforms, as_texture};
use ruffle_render::backend::ShapeHandle;
use ruffle_render::bitmap::{BitmapHandle, PixelSnapping};
use ruffle_render::commands::{CommandHandler, CommandList, RenderBlendMode};
use ruffle_render::lines::{emulate_line, emulate_line_rect};
use ruffle_render::matrix::Matrix;
use ruffle_render::pixel_bender::PixelBenderShaderHandle;
use ruffle_render::quality::StageQuality;
use ruffle_render::transform::Transform;
use std::mem;
use swf::{BlendMode, Color, ColorTransform, Twips};
use wgpu::Backend;
use wgpu::util::DeviceExt;

use super::target::PoolOrArcTexture;

/// Maximum number of rectangles accumulated in a single `DrawRectInstanced`
/// batch before it is automatically flushed.  Each rect occupies one
/// `RectInstance` (40 bytes), so 16 384 rects ≈ 640 KiB.
const MAX_BATCH_RECTS: usize = 16_384;

/// Normalise a [`swf::Color`] to premultiplied linear RGBA in `[0.0, 1.0]`.
#[inline]
fn color_to_premult_rgba(c: Color) -> [f32; 4] {
    let r = f32::from(c.r) / 255.0;
    let g = f32::from(c.g) / 255.0;
    let b = f32::from(c.b) / 255.0;
    let a = f32::from(c.a) / 255.0;
    [r * a, g * a, b * a, a]
}

pub struct CommandRenderer<'pass, 'frame: 'pass, 'global: 'frame> {
    pipelines: &'frame Pipelines,
    descriptors: &'global Descriptors,
    num_masks: u32,
    mask_state: MaskState,
    render_pass: wgpu::RenderPass<'pass>,
    needs_stencil: bool,
    dynamic_transforms: &'global DynamicTransforms,
    current_pipeline: Option<*const wgpu::RenderPipeline>,
    current_bind_group_2: Option<*const wgpu::BindGroup>,
    current_transform_offset: Option<wgpu::DynamicOffset>,
    current_stencil_reference: Option<u32>,
    /// Cache key for the currently-bound vertex buffer. Stored as a raw pointer
    /// used for identity comparison only; it is never dereferenced.
    current_vertex_buffer: Option<*const wgpu::Buffer>,
    /// Cache key for the currently-bound index buffer. Stored as a raw pointer
    /// used for identity comparison only; it is never dereferenced.
    current_index_buffer: Option<*const wgpu::Buffer>,
}

impl<'pass, 'frame: 'pass, 'global: 'frame> CommandRenderer<'pass, 'frame, 'global> {
    pub fn new(
        pipelines: &'frame Pipelines,
        descriptors: &'global Descriptors,
        dynamic_transforms: &'global DynamicTransforms,
        render_pass: wgpu::RenderPass<'pass>,
        num_masks: u32,
        mask_state: MaskState,
        needs_stencil: bool,
    ) -> Self {
        Self {
            pipelines,
            num_masks,
            mask_state,
            render_pass,
            descriptors,
            needs_stencil,
            dynamic_transforms,
            current_pipeline: None,
            current_bind_group_2: None,
            current_transform_offset: None,
            current_stencil_reference: None,
            current_vertex_buffer: None,
            current_index_buffer: None,
        }
    }

    fn set_pipeline_cached(&mut self, pipeline: &'frame wgpu::RenderPipeline) {
        let ptr = pipeline as *const wgpu::RenderPipeline;
        if self.current_pipeline == Some(ptr) {
            return;
        }
        self.render_pass.set_pipeline(pipeline);
        self.current_pipeline = Some(ptr);
    }

    fn set_bind_group_2_cached(&mut self, bind_group: &'pass wgpu::BindGroup) {
        let ptr = bind_group as *const wgpu::BindGroup;
        if self.current_bind_group_2 == Some(ptr) {
            return;
        }
        self.render_pass.set_bind_group(2, bind_group, &[]);
        self.current_bind_group_2 = Some(ptr);
    }

    fn set_transform_bind_group(&mut self, transform_buffer: wgpu::DynamicOffset) {
        if self.current_transform_offset == Some(transform_buffer) {
            return;
        }
        self.render_pass.set_bind_group(
            1,
            &self.dynamic_transforms.bind_group,
            &[transform_buffer],
        );
        self.current_transform_offset = Some(transform_buffer);
    }

    fn set_stencil_reference_cached(&mut self, stencil_reference: u32) {
        if self.current_stencil_reference == Some(stencil_reference) {
            return;
        }
        self.render_pass.set_stencil_reference(stencil_reference);
        self.current_stencil_reference = Some(stencil_reference);
    }

    /// Set the vertex buffer, skipping the GPU command if it is already bound.
    fn set_vertex_buffer_cached(
        &mut self,
        key: *const wgpu::Buffer,
        slice: wgpu::BufferSlice<'pass>,
    ) {
        if self.current_vertex_buffer != Some(key) {
            self.render_pass.set_vertex_buffer(0, slice);
            self.current_vertex_buffer = Some(key);
        }
    }

    /// Set the index buffer, skipping the GPU command if it is already bound.
    fn set_index_buffer_cached(
        &mut self,
        key: *const wgpu::Buffer,
        slice: wgpu::BufferSlice<'pass>,
    ) {
        if self.current_index_buffer != Some(key) {
            self.render_pass
                .set_index_buffer(slice, wgpu::IndexFormat::Uint32);
            self.current_index_buffer = Some(key);
        }
    }

    /// Returns the raw-pointer cache key and full-range `BufferSlice` for
    /// the given buffer in a single step. The key is used only for pointer
    /// equality comparisons and is never dereferenced.
    #[inline]
    fn buf_key_slice(buf: &'pass wgpu::Buffer) -> (*const wgpu::Buffer, wgpu::BufferSlice<'pass>) {
        (buf as *const wgpu::Buffer, buf.slice(..))
    }

    pub fn execute(&mut self, command: &'frame DrawCommand) {
        if self.needs_stencil {
            let stencil_reference = match self.mask_state {
                MaskState::NoMask => None,
                MaskState::DrawMaskStencil => Some(self.num_masks - 1),
                MaskState::DrawMaskedContent | MaskState::ClearMaskStencil => Some(self.num_masks),
            };
            if let Some(stencil_reference) = stencil_reference {
                self.set_stencil_reference_cached(stencil_reference);
            }
        }

        match command {
            DrawCommand::RenderBitmap {
                bitmap,
                transform_buffer,
                smoothing,
                blend_mode,
                render_stage3d,
            } => self.render_bitmap(
                bitmap,
                *transform_buffer,
                *smoothing,
                *blend_mode,
                *render_stage3d,
            ),
            DrawCommand::RenderTexture {
                _texture,
                binds,
                transform_buffer,
                blend_mode,
            } => self.render_texture(*transform_buffer, binds, *blend_mode),
            DrawCommand::RenderShape {
                shape,
                transform_buffer,
            } => self.render_shape(shape, *transform_buffer),
            DrawCommand::DrawRect { transform_buffer } => self.draw_rect(*transform_buffer),
            DrawCommand::DrawLine { transform_buffer } => {
                self.draw_lines::<false>(*transform_buffer)
            }
            DrawCommand::DrawLineRect { transform_buffer } => {
                self.draw_lines::<true>(*transform_buffer)
            }
            DrawCommand::DrawRectInstanced {
                instances,
                num_instances,
            } => self.draw_rect_instanced(instances, *num_instances),
            DrawCommand::PushMask => self.push_mask(),
            DrawCommand::ActivateMask => self.activate_mask(),
            DrawCommand::DeactivateMask => self.deactivate_mask(),
            DrawCommand::PopMask => self.pop_mask(),
            DrawCommand::RenderAlphaMask {
                maskee,
                mask,
                binds,
                transform_buffer,
            } => self.render_alpha_mask(maskee, mask, binds, *transform_buffer),
        }
    }

    pub fn prep_color(&mut self) {
        if self.needs_stencil {
            self.set_pipeline_cached(self.pipelines.color.pipeline_for(self.mask_state));
        } else {
            self.set_pipeline_cached(self.pipelines.color.stencilless_pipeline());
        }
    }

    pub fn prep_lines(&mut self) {
        if self.needs_stencil {
            self.set_pipeline_cached(self.pipelines.lines.pipeline_for(self.mask_state));
        } else {
            self.set_pipeline_cached(self.pipelines.lines.stencilless_pipeline());
        }
    }

    pub fn prep_gradient(&mut self, bind_group: &'pass wgpu::BindGroup) {
        if self.needs_stencil {
            self.set_pipeline_cached(self.pipelines.gradients.pipeline_for(self.mask_state));
        } else {
            self.set_pipeline_cached(self.pipelines.gradients.stencilless_pipeline());
        }

        self.set_bind_group_2_cached(bind_group);
    }

    pub fn prep_bitmap(
        &mut self,
        bind_group: &'pass wgpu::BindGroup,
        blend_mode: TrivialBlend,
        render_stage3d: bool,
    ) {
        match (self.needs_stencil, render_stage3d) {
            (true, true) => {
                self.set_pipeline_cached(&self.pipelines.bitmap_opaque_dummy_stencil);
            }
            (true, false) => {
                self.set_pipeline_cached(
                    self.pipelines.bitmap[blend_mode].pipeline_for(self.mask_state),
                );
            }
            (false, true) => {
                self.set_pipeline_cached(&self.pipelines.bitmap_opaque);
            }
            (false, false) => {
                self.set_pipeline_cached(self.pipelines.bitmap[blend_mode].stencilless_pipeline());
            }
        }

        self.set_bind_group_2_cached(bind_group);
    }

    pub fn prep_alpha_mask(&mut self, bind_group: &'pass wgpu::BindGroup) {
        if self.needs_stencil {
            self.set_pipeline_cached(self.pipelines.alpha_mask.pipeline_for(self.mask_state));
        } else {
            self.set_pipeline_cached(self.pipelines.alpha_mask.stencilless_pipeline());
        }

        self.set_bind_group_2_cached(bind_group);
    }

    pub fn draw(
        &mut self,
        vertices: wgpu::BufferSlice<'pass>,
        indices: wgpu::BufferSlice<'pass>,
        num_indices: u32,
    ) {
        self.render_pass.set_vertex_buffer(0, vertices);
        self.render_pass
            .set_index_buffer(indices, wgpu::IndexFormat::Uint32);

        self.render_pass.draw_indexed(0..num_indices, 0, 0..1);
        // Mesh draws use partial buffer slices with varying ranges; invalidate
        // the cached state so the next draw correctly re-binds its buffers.
        self.current_vertex_buffer = None;
        self.current_index_buffer = None;
    }

    pub fn render_bitmap(
        &mut self,
        bitmap: &'frame BitmapHandle,
        transform_buffer: wgpu::DynamicOffset,
        smoothing: bool,
        blend_mode: TrivialBlend,
        render_stage3d: bool,
    ) {
        if cfg!(feature = "render_debug_labels") {
            self.render_pass
                .push_debug_group(&format!("render_bitmap {:?}", bitmap.0));
        }
        let texture = as_texture(bitmap);

        let descriptors = self.descriptors;
        let bind = texture.bind_group(
            smoothing,
            &descriptors.device,
            &descriptors.bind_layouts.bitmap,
            &descriptors.quad,
            bitmap.clone(),
            &descriptors.bitmap_samplers,
        );
        self.prep_bitmap(&bind.bind_group, blend_mode, render_stage3d);
        self.set_transform_bind_group(transform_buffer);

        let (vb_key, vb_slice) = Self::buf_key_slice(&self.descriptors.quad.vertices_pos);
        let (ib_key, ib_slice) = Self::buf_key_slice(&self.descriptors.quad.indices);
        self.set_vertex_buffer_cached(vb_key, vb_slice);
        self.set_index_buffer_cached(ib_key, ib_slice);
        self.render_pass.draw_indexed(0..6, 0, 0..1);
        if cfg!(feature = "render_debug_labels") {
            self.render_pass.pop_debug_group();
        }
    }

    pub fn render_texture(
        &mut self,
        transform_buffer: wgpu::DynamicOffset,
        bind_group: &'frame wgpu::BindGroup,
        blend_mode: TrivialBlend,
    ) {
        if cfg!(feature = "render_debug_labels") {
            self.render_pass.push_debug_group("render_texture");
        }
        self.prep_bitmap(bind_group, blend_mode, false);
        self.set_transform_bind_group(transform_buffer);

        let (vb_key, vb_slice) = Self::buf_key_slice(&self.descriptors.quad.vertices_pos);
        let (ib_key, ib_slice) = Self::buf_key_slice(&self.descriptors.quad.indices);
        self.set_vertex_buffer_cached(vb_key, vb_slice);
        self.set_index_buffer_cached(ib_key, ib_slice);
        self.render_pass.draw_indexed(0..6, 0, 0..1);
        if cfg!(feature = "render_debug_labels") {
            self.render_pass.pop_debug_group();
        }
    }

    pub fn render_shape(
        &mut self,
        shape: &'frame ShapeHandle,
        transform_buffer: wgpu::DynamicOffset,
    ) {
        if cfg!(feature = "render_debug_labels") {
            self.render_pass.push_debug_group("render_shape");
        }

        let mesh = as_mesh(shape);
        self.set_transform_bind_group(transform_buffer);
        for draw in &mesh.draws {
            let num_indices = if self.mask_state != MaskState::DrawMaskStencil
                && self.mask_state != MaskState::ClearMaskStencil
            {
                draw.num_indices
            } else {
                // Omit strokes when drawing a mask stencil.
                draw.num_mask_indices
            };
            if num_indices == 0 {
                continue;
            }

            match &draw.draw_type {
                DrawType::Color => {
                    self.prep_color();
                }
                DrawType::Gradient { bind_group, .. } => {
                    self.prep_gradient(bind_group);
                }
                DrawType::Bitmap { binds, .. } => {
                    self.prep_bitmap(&binds.bind_group, TrivialBlend::Normal, false);
                }
            }
            self.draw(
                mesh.vertex_buffer.slice(draw.vertices.clone()),
                mesh.index_buffer.slice(draw.indices.clone()),
                num_indices,
            );
        }
        if cfg!(feature = "render_debug_labels") {
            self.render_pass.pop_debug_group();
        }
    }

    pub fn render_alpha_mask(
        &mut self,
        _maskee: &PoolOrArcTexture,
        _mask: &PoolOrArcTexture,
        bind_group: &'frame wgpu::BindGroup,
        transform_buffer: wgpu::DynamicOffset,
    ) {
        if cfg!(feature = "render_debug_labels") {
            self.render_pass.push_debug_group("render_alpha_mask");
        }

        self.prep_alpha_mask(bind_group);
        self.set_transform_bind_group(transform_buffer);

        let (vb_key, vb_slice) = Self::buf_key_slice(&self.descriptors.quad.vertices_pos);
        let (ib_key, ib_slice) = Self::buf_key_slice(&self.descriptors.quad.indices);
        self.set_vertex_buffer_cached(vb_key, vb_slice);
        self.set_index_buffer_cached(ib_key, ib_slice);
        self.render_pass.draw_indexed(0..6, 0, 0..1);

        if cfg!(feature = "render_debug_labels") {
            self.render_pass.pop_debug_group();
        }
    }

    pub fn draw_rect(&mut self, transform_buffer: wgpu::DynamicOffset) {
        if cfg!(feature = "render_debug_labels") {
            self.render_pass.push_debug_group("draw_rect");
        }
        self.prep_color();
        self.set_transform_bind_group(transform_buffer);

        let (vb_key, vb_slice) = Self::buf_key_slice(&self.descriptors.quad.vertices_pos_color);
        let (ib_key, ib_slice) = Self::buf_key_slice(&self.descriptors.quad.indices);
        self.set_vertex_buffer_cached(vb_key, vb_slice);
        self.set_index_buffer_cached(ib_key, ib_slice);
        self.render_pass.draw_indexed(0..6, 0, 0..1);
        if cfg!(feature = "render_debug_labels") {
            self.render_pass.pop_debug_group();
        }
    }

    /// Select the instanced solid-colour rect pipeline, which uses the shared
    /// unit-quad vertex buffer (slot 0) and a per-instance buffer (slot 1),
    /// and therefore needs only the global bind group (group 0).
    pub fn prep_rect_instanced(&mut self) {
        if self.needs_stencil {
            self.set_pipeline_cached(
                self.pipelines.rect_instanced.pipeline_for(self.mask_state),
            );
        } else {
            self.set_pipeline_cached(self.pipelines.rect_instanced.stencilless_pipeline());
        }
    }

    /// Draw a batch of solid-colour rectangles using GPU instancing.
    ///
    /// `instances` holds one `RectInstance` per rectangle (affine transform +
    /// premultiplied colour).  The shared unit-quad geometry is read from
    /// `descriptors.quad.vertices_pos` and `descriptors.quad.indices`.
    /// A single `draw_indexed` call renders all instances at once.
    pub fn draw_rect_instanced(
        &mut self,
        instances: &'pass wgpu::Buffer,
        num_instances: u32,
    ) {
        if cfg!(feature = "render_debug_labels") {
            self.render_pass.push_debug_group("draw_rect_instanced");
        }
        self.prep_rect_instanced();
        // Slot 0: shared unit-quad positions (four corners [0,1]×[0,1]).
        self.render_pass
            .set_vertex_buffer(0, self.descriptors.quad.vertices_pos.slice(..));
        // Slot 1: per-rect instance data (varies per batch).
        self.render_pass.set_vertex_buffer(1, instances.slice(..));
        // Shared quad index buffer [0, 1, 2, 0, 2, 3] — two triangles.
        self.render_pass.set_index_buffer(
            self.descriptors.quad.indices.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        self.render_pass.draw_indexed(0..6, 0, 0..num_instances);
        // Invalidate the slot-0 and index-buffer caches so that the next
        // non-instanced draw correctly re-binds its own buffers.
        self.current_vertex_buffer = None;
        self.current_index_buffer = None;
        if cfg!(feature = "render_debug_labels") {
            self.render_pass.pop_debug_group();
        }
    }

    pub fn draw_lines<const RECT: bool>(&mut self, transform_buffer: wgpu::DynamicOffset) {
        if cfg!(feature = "render_debug_labels") {
            self.render_pass.push_debug_group("draw_lines");
        }
        self.prep_lines();
        self.set_transform_bind_group(transform_buffer);

        let (vb_key, vb_slice) = Self::buf_key_slice(&self.descriptors.quad.vertices_pos_color);
        let (ib_key, ib_slice, num_indices) = if RECT {
            let (k, s) = Self::buf_key_slice(&self.descriptors.quad.indices_line_rect);
            (k, s, 5)
        } else {
            let (k, s) = Self::buf_key_slice(&self.descriptors.quad.indices_line);
            (k, s, 2)
        };
        self.set_vertex_buffer_cached(vb_key, vb_slice);
        self.set_index_buffer_cached(ib_key, ib_slice);
        self.render_pass.draw_indexed(0..num_indices, 0, 0..1);
        if cfg!(feature = "render_debug_labels") {
            self.render_pass.pop_debug_group();
        }
    }

    pub fn push_mask(&mut self) {
        debug_assert!(
            self.mask_state == MaskState::NoMask || self.mask_state == MaskState::DrawMaskedContent
        );
        self.num_masks += 1;
        self.mask_state = MaskState::DrawMaskStencil;
        self.set_stencil_reference_cached(self.num_masks - 1);
    }

    pub fn activate_mask(&mut self) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::DrawMaskStencil);
        self.mask_state = MaskState::DrawMaskedContent;
        self.set_stencil_reference_cached(self.num_masks);
    }

    pub fn deactivate_mask(&mut self) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::DrawMaskedContent);
        self.mask_state = MaskState::ClearMaskStencil;
        self.set_stencil_reference_cached(self.num_masks);
    }

    pub fn pop_mask(&mut self) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::ClearMaskStencil);
        self.num_masks -= 1;
        self.set_stencil_reference_cached(self.num_masks);
        if self.num_masks == 0 {
            self.mask_state = MaskState::NoMask;
        } else {
            self.mask_state = MaskState::DrawMaskedContent;
        };
    }

    pub fn num_masks(&self) -> u32 {
        self.num_masks
    }

    pub fn mask_state(&self) -> MaskState {
        self.mask_state
    }
}

pub enum Chunk {
    Draw(Vec<DrawCommand>, bool, BufferBuilder),
    Blend(PoolOrArcTexture, ChunkBlendMode, bool),
}

#[derive(Debug)]
pub enum ChunkBlendMode {
    Complex(ComplexBlend),
    Shader(PixelBenderShaderHandle),
}

#[derive(Debug)]
pub enum DrawCommand {
    RenderBitmap {
        bitmap: BitmapHandle,
        transform_buffer: wgpu::DynamicOffset,
        smoothing: bool,
        blend_mode: TrivialBlend,
        render_stage3d: bool,
    },
    RenderTexture {
        _texture: PoolOrArcTexture,
        binds: wgpu::BindGroup,
        transform_buffer: wgpu::DynamicOffset,
        blend_mode: TrivialBlend,
    },
    RenderAlphaMask {
        maskee: PoolOrArcTexture,
        mask: PoolOrArcTexture,
        binds: wgpu::BindGroup,
        transform_buffer: wgpu::DynamicOffset,
    },
    RenderShape {
        shape: ShapeHandle,
        transform_buffer: wgpu::DynamicOffset,
    },
    DrawLine {
        transform_buffer: wgpu::DynamicOffset,
    },
    DrawLineRect {
        transform_buffer: wgpu::DynamicOffset,
    },
    /// A batch of solid-colour rectangles rendered with a single instanced
    /// draw call.  The `instances` buffer contains one `RectInstance` per
    /// rectangle: a 2D affine transform (ab, cd, txty) and a premultiplied
    /// RGBA colour.  The GPU applies the transform to the shared unit-quad
    /// geometry, so no index buffer needs to be supplied per batch.
    DrawRectInstanced {
        instances: wgpu::Buffer,
        num_instances: u32,
    },
    /// Single pre-transformed colored rectangle (legacy path, kept as a safe
    /// fallback; not currently generated but matched in `execute`).
    #[allow(dead_code)]
    DrawRect {
        transform_buffer: wgpu::DynamicOffset,
    },
    PushMask,
    ActivateMask,
    DeactivateMask,
    PopMask,
}

#[derive(Copy, Clone)]
pub enum LayerRef<'a> {
    None,
    Current,
    Parent(&'a CommandTarget),
}

/// Replaces every blend with a RenderBitmap, with the subcommands rendered out to a temporary texture
/// Every complex blend will be its own item, but every other draw will be chunked together
#[expect(clippy::too_many_arguments)]
pub fn chunk_blends<'a>(
    commands: CommandList,
    descriptors: &'a Descriptors,
    staging_belt: &'a mut wgpu::util::StagingBelt,
    dynamic_transforms: &'a DynamicTransforms,
    draw_encoder: &mut wgpu::CommandEncoder,
    meshes: &'a Vec<Mesh>,
    quality: StageQuality,
    width: u32,
    height: u32,
    nearest_layer: LayerRef,
    texture_pool: &mut TexturePool,
) -> Vec<Chunk> {
    WgpuCommandHandler::new(
        descriptors,
        staging_belt,
        dynamic_transforms,
        draw_encoder,
        meshes,
        quality,
        width,
        height,
        nearest_layer,
        texture_pool,
    )
    .chunk_blends(commands)
}

struct WgpuCommandHandler<'a> {
    descriptors: &'a Descriptors,
    quality: StageQuality,
    width: u32,
    height: u32,
    nearest_layer: LayerRef<'a>,
    meshes: &'a Vec<Mesh>,
    staging_belt: &'a mut wgpu::util::StagingBelt,
    dynamic_transforms: &'a DynamicTransforms,
    draw_encoder: &'a mut wgpu::CommandEncoder,
    texture_pool: &'a mut TexturePool,
    emulate_lines: bool,

    result: Vec<Chunk>,
    current: Vec<DrawCommand>,
    transforms: BufferBuilder,
    last_transform: Option<(Transforms, wgpu::DynamicOffset)>,
    needs_stencil: bool,
    num_masks: i32,

    /// Accumulated per-rect instance data for the in-progress instanced batch.
    rect_instances: Vec<RectInstance>,
}

impl<'a> WgpuCommandHandler<'a> {
    #[expect(clippy::too_many_arguments)]
    fn new(
        descriptors: &'a Descriptors,
        staging_belt: &'a mut wgpu::util::StagingBelt,
        dynamic_transforms: &'a DynamicTransforms,
        draw_encoder: &'a mut wgpu::CommandEncoder,
        meshes: &'a Vec<Mesh>,
        quality: StageQuality,
        width: u32,
        height: u32,
        nearest_layer: LayerRef<'a>,
        texture_pool: &'a mut TexturePool,
    ) -> Self {
        let transforms = Self::new_transforms(descriptors, dynamic_transforms);

        // DirectX does support drawing lines, but it's very inconsistent.
        // With MSAA, lines have 1.4px thickness, which makes them too thick.
        // Without MSAA, lines have 1px thickness, but their placement is sometimes off.
        let emulate_lines = descriptors.backend == Backend::Dx12;

        Self {
            descriptors,
            quality,
            width,
            height,
            nearest_layer,
            meshes,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            texture_pool,
            emulate_lines,

            result: vec![],
            current: vec![],
            transforms,
            last_transform: None,
            needs_stencil: false,
            num_masks: 0,

            rect_instances: Vec::new(),
        }
    }

    fn new_transforms(
        descriptors: &'a Descriptors,
        dynamic_transforms: &'a DynamicTransforms,
    ) -> BufferBuilder {
        let mut transforms = BufferBuilder::new_for_uniform(&descriptors.limits);
        transforms.set_buffer_limit(dynamic_transforms.buffer.size());
        transforms
    }

    /// Flush the accumulated instanced rect batch (if any) as a single
    /// `DrawRectInstanced` command in `self.current`.  Clears the accumulator.
    ///
    /// Call this before emitting any command that is not a `draw_rect` so that
    /// draw order between batched rects and other draw calls is preserved.
    fn flush_rect_batch(&mut self) {
        if self.rect_instances.is_empty() {
            return;
        }

        let num_instances = self.rect_instances.len() as u32;
        let instances = self
            .descriptors
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: create_debug_label!("Rect instanced buffer").as_deref(),
                contents: bytemuck::cast_slice(&self.rect_instances),
                usage: wgpu::BufferUsages::VERTEX,
            });

        self.rect_instances.clear();

        self.current.push(DrawCommand::DrawRectInstanced {
            instances,
            num_instances,
        });
    }

    /// Replaces every blend with a RenderBitmap, with the subcommands rendered out to a temporary texture
    /// Every complex blend will be its own item, but every other draw will be chunked together
    fn chunk_blends(&mut self, commands: CommandList) -> Vec<Chunk> {
        commands.execute(self);

        // Flush any remaining rect batch before finalising the chunk.
        self.flush_rect_batch();

        let current = mem::take(&mut self.current);
        let mut result = mem::take(&mut self.result);
        let needs_stencil = mem::take(&mut self.needs_stencil);
        let transforms = mem::replace(
            &mut self.transforms,
            Self::new_transforms(self.descriptors, self.dynamic_transforms),
        );

        if !current.is_empty() {
            result.push(Chunk::Draw(current, needs_stencil, transforms));
        }

        result
    }

    fn add_to_current(
        &mut self,
        matrix: Matrix,
        color_transform: ColorTransform,
        command_builder: impl FnOnce(wgpu::DynamicOffset) -> DrawCommand,
    ) {
        let transform = Transforms {
            world_matrix: [
                [matrix.a, matrix.b, 0.0, 0.0],
                [matrix.c, matrix.d, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [
                    matrix.tx.to_pixels() as f32,
                    matrix.ty.to_pixels() as f32,
                    0.0,
                    1.0,
                ],
            ],
            mult_color: color_transform.mult_rgba_normalized(),
            add_color: color_transform.add_rgba_normalized(),
        };

        if let Some((last_transform, transform_offset)) = self.last_transform
            && bytemuck::bytes_of(&last_transform) == bytemuck::bytes_of(&transform)
        {
            self.current.push(command_builder(transform_offset));
            return;
        }

        if let Ok(transform_range) = self.transforms.add(&[transform]) {
            let transform_offset = transform_range.start as wgpu::DynamicOffset;
            self.last_transform = Some((transform, transform_offset));
            self.current.push(command_builder(transform_offset));
        } else {
            self.result.push(Chunk::Draw(
                mem::take(&mut self.current),
                self.needs_stencil,
                mem::replace(
                    &mut self.transforms,
                    BufferBuilder::new_for_uniform(&self.descriptors.limits),
                ),
            ));
            self.last_transform = None;
            self.transforms
                .set_buffer_limit(self.dynamic_transforms.buffer.size());
            let transform_range = self
                .transforms
                .add(&[transform])
                .expect("Buffer must be able to fit a new thing, it was just emptied");
            let transform_offset = transform_range.start as wgpu::DynamicOffset;
            self.last_transform = Some((transform, transform_offset));
            self.current.push(command_builder(transform_offset));
        }
    }
}

impl CommandHandler for WgpuCommandHandler<'_> {
    fn blend(&mut self, commands: CommandList, blend_mode: RenderBlendMode) {
        // Preserve draw order: flush any pending rect batch first.
        self.flush_rect_batch();

        let mut surface = Surface::new(
            self.descriptors,
            self.quality,
            self.width,
            self.height,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let target_layer = if let RenderBlendMode::Builtin(BlendMode::Layer) = &blend_mode {
            LayerRef::Current
        } else {
            self.nearest_layer
        };
        let blend_type = BlendType::from(blend_mode);
        let clear_color = blend_type.default_color();
        let target = surface.draw_commands(
            RenderTargetMode::FreshWithColor(clear_color),
            self.descriptors,
            self.meshes,
            commands,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
            target_layer,
            self.texture_pool,
        );
        target.ensure_cleared(self.draw_encoder);

        match blend_type {
            BlendType::Trivial(blend_mode) => {
                let transform = Transform {
                    matrix: Matrix::scale(target.width() as f32, target.height() as f32),
                    color_transform: Default::default(),
                    perspective_projection: None,
                };
                let texture = target.take_color_texture();
                let bind_group =
                    self.descriptors
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: &self.descriptors.bind_layouts.bitmap,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: self
                                        .descriptors
                                        .quad
                                        .texture_transforms
                                        .as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(texture.view()),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Sampler(
                                        self.descriptors.bitmap_samplers.get_sampler(false, false),
                                    ),
                                },
                            ],
                            label: None,
                        });
                self.add_to_current(
                    transform.matrix,
                    transform.color_transform,
                    |transform_buffer| DrawCommand::RenderTexture {
                        _texture: texture,
                        binds: bind_group,
                        transform_buffer,
                        blend_mode,
                    },
                );
            }
            blend_type => {
                if !self.current.is_empty() {
                    self.result.push(Chunk::Draw(
                        mem::take(&mut self.current),
                        self.needs_stencil,
                        mem::replace(
                            &mut self.transforms,
                            BufferBuilder::new_for_uniform(&self.descriptors.limits),
                        ),
                    ));
                    self.last_transform = None;
                }
                self.transforms
                    .set_buffer_limit(self.dynamic_transforms.buffer.size());
                let chunk_blend_mode = match blend_type {
                    BlendType::Complex(complex) => ChunkBlendMode::Complex(complex),
                    BlendType::Shader(shader) => ChunkBlendMode::Shader(shader),
                    _ => unreachable!(),
                };
                self.result.push(Chunk::Blend(
                    target.take_color_texture(),
                    chunk_blend_mode,
                    self.num_masks > 0,
                ));
                self.needs_stencil = self.num_masks > 0;
            }
        }
    }

    fn render_bitmap(
        &mut self,
        bitmap: BitmapHandle,
        transform: Transform,
        smoothing: bool,
        pixel_snapping: PixelSnapping,
    ) {
        self.flush_rect_batch();
        let mut matrix = transform.matrix;
        {
            let texture = as_texture(&bitmap);
            pixel_snapping.apply(&mut matrix);
            matrix *= Matrix::scale(
                texture.texture.width() as f32,
                texture.texture.height() as f32,
            );
        }
        self.add_to_current(matrix, transform.color_transform, |transform_buffer| {
            DrawCommand::RenderBitmap {
                bitmap,
                transform_buffer,
                smoothing,
                blend_mode: TrivialBlend::Normal,
                render_stage3d: false,
            }
        });
    }
    fn render_stage3d(&mut self, bitmap: BitmapHandle, transform: Transform) {
        self.flush_rect_batch();
        let mut matrix = transform.matrix;
        {
            let texture = as_texture(&bitmap);
            matrix *= Matrix::scale(
                texture.texture.width() as f32,
                texture.texture.height() as f32,
            );
        }
        self.add_to_current(matrix, transform.color_transform, |transform_buffer| {
            DrawCommand::RenderBitmap {
                bitmap,
                transform_buffer,
                smoothing: false,
                blend_mode: TrivialBlend::Normal,
                render_stage3d: true,
            }
        });
    }

    fn render_shape(&mut self, shape: ShapeHandle, transform: Transform) {
        self.flush_rect_batch();
        self.add_to_current(
            transform.matrix,
            transform.color_transform,
            |transform_buffer| DrawCommand::RenderShape {
                shape,
                transform_buffer,
            },
        );
    }

    fn draw_rect(&mut self, color: Color, matrix: Matrix) {
        // Build a RectInstance: pack the raw matrix components and
        // premultiplied colour directly, without computing world-space vertex
        // positions on the CPU.  The vertex shader applies the affine transform
        // to the shared unit-quad geometry for each instance.
        self.rect_instances.push(RectInstance {
            ab:   [matrix.a, matrix.b],
            cd:   [matrix.c, matrix.d],
            txty: [matrix.tx.to_pixels() as f32, matrix.ty.to_pixels() as f32],
            color: color_to_premult_rgba(color),
        });

        // Flush when the batch reaches the size limit.
        if self.rect_instances.len() >= MAX_BATCH_RECTS {
            self.flush_rect_batch();
        }
    }

    fn draw_line(&mut self, color: Color, mut matrix: Matrix) {
        self.flush_rect_batch();
        if self.emulate_lines {
            let mut cl = CommandList::new();
            emulate_line(&mut cl, color, matrix);
            cl.execute(self);
        } else {
            matrix.tx += Twips::HALF_PX;
            matrix.ty += Twips::HALF_PX;
            self.add_to_current(
                matrix,
                ColorTransform::multiply_from(color),
                |transform_buffer| DrawCommand::DrawLine { transform_buffer },
            );
        }
    }

    fn draw_line_rect(&mut self, color: Color, mut matrix: Matrix) {
        self.flush_rect_batch();
        if self.emulate_lines {
            let mut cl = CommandList::new();
            emulate_line_rect(&mut cl, color, matrix);
            cl.execute(self);
        } else {
            matrix.tx += Twips::HALF_PX;
            matrix.ty += Twips::HALF_PX;
            self.add_to_current(
                matrix,
                ColorTransform::multiply_from(color),
                |transform_buffer| DrawCommand::DrawLineRect { transform_buffer },
            );
        }
    }

    fn push_mask(&mut self) {
        self.flush_rect_batch();
        self.needs_stencil = true;
        self.num_masks += 1;
        self.current.push(DrawCommand::PushMask);
    }

    fn activate_mask(&mut self) {
        self.flush_rect_batch();
        self.needs_stencil = true;
        self.current.push(DrawCommand::ActivateMask);
    }

    fn deactivate_mask(&mut self) {
        self.flush_rect_batch();
        self.needs_stencil = true;
        self.current.push(DrawCommand::DeactivateMask);
    }

    fn pop_mask(&mut self) {
        self.flush_rect_batch();
        self.needs_stencil = true;
        self.num_masks -= 1;
        self.current.push(DrawCommand::PopMask);
    }

    fn render_alpha_mask(&mut self, maskee_commands: CommandList, mask_commands: CommandList) {
        self.flush_rect_batch();
        let mut surface = Surface::new(
            self.descriptors,
            self.quality,
            self.width,
            self.height,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let maskee = surface.draw_commands(
            RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
            self.descriptors,
            self.meshes,
            maskee_commands,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
            LayerRef::None,
            self.texture_pool,
        );
        maskee.ensure_cleared(self.draw_encoder);
        let matrix = Matrix::scale(maskee.width() as f32, maskee.height() as f32);
        let maskee = maskee.take_color_texture();

        let mask = surface.draw_commands(
            RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
            self.descriptors,
            self.meshes,
            mask_commands,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
            LayerRef::None,
            self.texture_pool,
        );
        mask.ensure_cleared(self.draw_encoder);
        let mask = mask.take_color_texture();

        let binds = self
            .descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.descriptors.bind_layouts.alpha_mask,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(maskee.view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(mask.view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(
                            self.descriptors.bitmap_samplers.get_sampler(false, false),
                        ),
                    },
                ],
                label: None,
            });

        self.add_to_current(matrix, Default::default(), |transform_buffer| {
            DrawCommand::RenderAlphaMask {
                maskee,
                mask,
                binds,
                transform_buffer,
            }
        });
    }
}
