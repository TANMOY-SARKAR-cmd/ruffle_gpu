use crate::backend::RenderTargetMode;
use crate::blend::TrivialBlend;
use crate::blend::{BlendType, ComplexBlend};
use crate::buffer_builder::BufferBuilder;
use crate::buffer_pool::{PooledVertexBuffer, TexturePool};
use crate::dynamic_transforms::DynamicTransforms;
use crate::mesh::{DrawType, Mesh, as_mesh};
use crate::surface::Surface;
use crate::surface::target::CommandTarget;
use crate::{Descriptors, MaskState, Pipelines, PostProcessQuality, BitmapInstance, RectInstance, Transforms, as_texture};
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

use super::target::PoolOrArcTexture;

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
    /// Cache key for the bind group currently bound at group 1 by the
    /// bitmap-instanced pipeline.  Set by `draw_bitmap_instanced`; cleared
    /// whenever `set_transform_bind_group` rebinds group 1 for a non-instanced
    /// draw, ensuring the next instanced draw always re-binds correctly.
    current_group1_bitmap_bind_group: Option<*const wgpu::BindGroup>,
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
            current_group1_bitmap_bind_group: None,
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
        // Group 1 now holds the transform uniform; any previously cached bitmap
        // bind group at group 1 is no longer current.
        self.current_group1_bitmap_bind_group = None;
    }

    /// Bind a bitmap's bind group at group 1, skipping the GPU command if the
    /// same bind group is already bound.  Also clears `current_transform_offset`
    /// so that the next non-instanced draw will correctly rebind group 1 with
    /// its own transform uniform.
    fn set_group1_bitmap_bind_group_cached(&mut self, bind_group: &'pass wgpu::BindGroup) {
        let ptr = bind_group as *const wgpu::BindGroup;
        if self.current_group1_bitmap_bind_group != Some(ptr) {
            self.render_pass.set_bind_group(1, bind_group, &[]);
            self.current_group1_bitmap_bind_group = Some(ptr);
        }
        // Invalidate the transform cache so the next non-instanced draw rebinds.
        self.current_transform_offset = None;
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
            DrawCommand::DrawBitmapInstanced {
                bitmap,
                instances,
                num_instances,
                smoothing,
                blend_mode,
            } => self.draw_bitmap_instanced(bitmap, &**instances, *num_instances, *smoothing, *blend_mode),
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
            } => self.draw_rect_instanced(&**instances, *num_instances),
            DrawCommand::PendingRectBatch(_) | DrawCommand::PendingBitmapBatch { .. } => {
                unreachable!("pending batch commands must be consumed by optimize_draw_commands before rendering")
            }
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
        // Use the cached helper so that consecutive instanced draw calls skip
        // the redundant rebind when the same quad buffer is already bound.
        let (vb0_key, vb0_slice) = Self::buf_key_slice(&self.descriptors.quad.vertices_pos);
        let (ib_key, ib_slice) = Self::buf_key_slice(&self.descriptors.quad.indices);
        self.set_vertex_buffer_cached(vb0_key, vb0_slice);
        // Slot 1: per-rect instance data (varies per batch; always rebound).
        self.render_pass.set_vertex_buffer(1, instances.slice(..));
        // Shared quad index buffer [0, 1, 2, 0, 2, 3] — two triangles.
        self.set_index_buffer_cached(ib_key, ib_slice);
        self.render_pass.draw_indexed(0..6, 0, 0..num_instances);
        // Keep `current_vertex_buffer` pointing at `vertices_pos` and
        // `current_index_buffer` pointing at `indices`.  Non-instanced draws
        // that follow use the same cached helpers, so they will rebind slot 0
        // only when their buffer differs (e.g. `vertices_pos_color` for rects).
        // Shape draws (`draw`) always rebind unconditionally, so they are
        // unaffected by this cached state.
        if cfg!(feature = "render_debug_labels") {
            self.render_pass.pop_debug_group();
        }
    }

    /// Draw a batch of bitmap instances using GPU instancing.
    ///
    /// `instances` holds one `BitmapInstance` per bitmap (affine transform +
    /// color transforms).  All instances share the same texture (`bitmap`,
    /// `smoothing`) and `blend_mode`.  The shared unit-quad geometry is read
    /// from `descriptors.quad.vertices_pos` and `descriptors.quad.indices`.
    /// A single `draw_indexed` call renders all instances at once.
    pub fn draw_bitmap_instanced(
        &mut self,
        bitmap: &'frame BitmapHandle,
        instances: &'pass wgpu::Buffer,
        num_instances: u32,
        smoothing: bool,
        blend_mode: TrivialBlend,
    ) {
        if cfg!(feature = "render_debug_labels") {
            self.render_pass
                .push_debug_group(&format!("draw_bitmap_instanced {:?}", bitmap.0));
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

        // Select the bitmap-instanced pipeline.
        // The pipeline layout is [globals(0), bitmap(1)] — no per-object
        // transform uniform at group 1.
        if self.needs_stencil {
            self.set_pipeline_cached(
                self.pipelines.bitmap_instanced[blend_mode].pipeline_for(self.mask_state),
            );
        } else {
            self.set_pipeline_cached(
                self.pipelines.bitmap_instanced[blend_mode].stencilless_pipeline(),
            );
        }

        // Bind the bitmap (texture + sampler + texture_transforms) at group 1.
        // Uses pointer-equality caching to skip the GPU command when the same
        // texture batch runs consecutively.  `current_transform_offset` is
        // always cleared so the next non-instanced draw will rebind group 1.
        self.set_group1_bitmap_bind_group_cached(&bind.bind_group);

        // Slot 0: shared unit-quad positions (four corners [0,1]×[0,1]).
        // Use the cached helper so that consecutive instanced draw calls skip
        // the redundant rebind when the same quad buffer is already bound.
        let (vb0_key, vb0_slice) = Self::buf_key_slice(&self.descriptors.quad.vertices_pos);
        let (ib_key, ib_slice) = Self::buf_key_slice(&self.descriptors.quad.indices);
        self.set_vertex_buffer_cached(vb0_key, vb0_slice);
        // Slot 1: per-bitmap instance data (varies per batch; always rebound).
        self.render_pass.set_vertex_buffer(1, instances.slice(..));
        // Shared quad index buffer [0, 1, 2, 0, 2, 3] — two triangles.
        self.set_index_buffer_cached(ib_key, ib_slice);
        self.render_pass.draw_indexed(0..6, 0, 0..num_instances);

        // Keep `current_vertex_buffer` pointing at `vertices_pos` and
        // `current_index_buffer` pointing at `indices`.  Non-instanced draws
        // that follow use the same cached helpers, so they will rebind slot 0
        // only when their buffer differs (e.g. `vertices_pos_color` for rects).
        // Shape draws (`draw`) always rebind unconditionally, so they are
        // unaffected by this cached state.

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
    /// A batch of bitmap instances rendered with a single instanced draw call.
    /// The `instances` buffer contains one `BitmapInstance` per bitmap: a 2D
    /// affine transform and a color transform.  All instances share the same
    /// texture (identified by `bitmap`), sampler (`smoothing`), and `blend_mode`.
    ///
    /// Produced by `optimize_draw_commands` after merging `PendingBitmapBatch`
    /// commands.  Never appears as a pre-optimization intermediate.
    DrawBitmapInstanced {
        bitmap: BitmapHandle,
        instances: PooledVertexBuffer,
        num_instances: u32,
        smoothing: bool,
        blend_mode: TrivialBlend,
    },
    /// A batch of solid-colour rectangles rendered with a single instanced
    /// draw call.  The `instances` buffer contains one `RectInstance` per
    /// rectangle: a 2D affine transform (ab, cd, txty) and a premultiplied
    /// RGBA colour.  The GPU applies the transform to the shared unit-quad
    /// geometry, so no index buffer needs to be supplied per batch.
    ///
    /// Produced by `optimize_draw_commands` after merging `PendingRectBatch`
    /// commands.  Never appears as a pre-optimization intermediate.
    DrawRectInstanced {
        instances: PooledVertexBuffer,
        num_instances: u32,
    },
    /// Accumulated rect instances that have not yet been GPU-materialized.
    /// Emitted by `flush_rect_batch`; consumed (and transformed into
    /// `DrawRectInstanced`) by `optimize_draw_commands`.  Never reaches the
    /// renderer.
    PendingRectBatch(Vec<RectInstance>),
    /// Accumulated bitmap instances that have not yet been GPU-materialized.
    /// Emitted by `flush_bitmap_batch`; consumed (and transformed into
    /// `DrawBitmapInstanced`) by `optimize_draw_commands`.  Never reaches the
    /// renderer.
    PendingBitmapBatch {
        bitmap: BitmapHandle,
        bitmaps: Vec<BitmapInstance>,
        smoothing: bool,
        blend_mode: TrivialBlend,
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

/// Post-process a `Vec<DrawCommand>`, merging consecutive compatible pending
/// batch commands into a single GPU-backed draw call, then materialise the
/// GPU `wgpu::Buffer` for each merged batch exactly once.
///
/// **Safe merges:**
/// * Consecutive `PendingRectBatch` commands — same pipeline, no per-batch
///   GPU state; their instance data can be concatenated while preserving order.
///   Merges stop once the combined instance count would exceed `rect_batch_limit`.
/// * Consecutive `PendingBitmapBatch` commands that share the same
///   `(bitmap, smoothing, blend_mode)` key — same texture and pipeline.
///   Merges stop once the combined instance count would exceed `bitmap_batch_limit`.
///
/// Both limits come from the per-type adaptive values in `FrameMetrics`,
/// ensuring the flush threshold is respected end-to-end.
///
/// GPU buffers are created **once per final merged run**, eliminating the
/// quadratic reallocation of the old pairwise approach.
fn optimize_draw_commands(
    commands: Vec<DrawCommand>,
    descriptors: &Descriptors,
    rect_batch_limit: usize,
    bitmap_batch_limit: usize,
) -> Vec<DrawCommand> {
    // ── helpers ───────────────────────────────────────────────────────────
    fn emit_rect_run(
        run: &mut Vec<RectInstance>,
        out: &mut Vec<DrawCommand>,
        descriptors: &Descriptors,
    ) {
        if run.is_empty() {
            return;
        }
        let num_instances = run.len() as u32;
        let byte_data: &[u8] = bytemuck::cast_slice(run);
        let instances = descriptors
            .vertex_instance_pool
            .acquire(&descriptors.device, byte_data.len() as u64);
        descriptors.queue.write_buffer(&instances, 0, byte_data);
        run.clear();
        out.push(DrawCommand::DrawRectInstanced {
            instances,
            num_instances,
        });
    }

    fn emit_bitmap_run(
        run: &mut Option<(BitmapHandle, bool, TrivialBlend, Vec<BitmapInstance>)>,
        out: &mut Vec<DrawCommand>,
        descriptors: &Descriptors,
    ) {
        let Some((bitmap, smoothing, blend_mode, bitmaps)) = run.take() else {
            return;
        };
        // `flush_bitmap_batch` never creates an empty batch, so `bitmaps` is
        // always non-empty here.
        debug_assert!(!bitmaps.is_empty());
        let num_instances = bitmaps.len() as u32;
        let byte_data: &[u8] = bytemuck::cast_slice(&bitmaps);
        let instances = descriptors
            .vertex_instance_pool
            .acquire(&descriptors.device, byte_data.len() as u64);
        descriptors.queue.write_buffer(&instances, 0, byte_data);
        out.push(DrawCommand::DrawBitmapInstanced {
            bitmap,
            instances,
            num_instances,
            smoothing,
            blend_mode,
        });
    }
    // ── main pass ─────────────────────────────────────────────────────────
    let mut out: Vec<DrawCommand> = Vec::with_capacity(commands.len());
    let mut rect_run: Vec<RectInstance> = Vec::new();
    let mut bitmap_run: Option<(BitmapHandle, bool, TrivialBlend, Vec<BitmapInstance>)> = None;

    for cmd in commands {
        match cmd {
            // ── Rect batches ──────────────────────────────────────────────
            DrawCommand::PendingRectBatch(mut new_rects) => {
                // Different type: flush any in-progress bitmap run.
                emit_bitmap_run(&mut bitmap_run, &mut out, descriptors);

                // If adding new_rects would exceed the limit, flush the
                // current rect run first.
                if !rect_run.is_empty()
                    && rect_run.len() + new_rects.len() > rect_batch_limit
                {
                    emit_rect_run(&mut rect_run, &mut out, descriptors);
                }
                rect_run.append(&mut new_rects);
                // Emit immediately if we hit the cap exactly.
                if rect_run.len() >= rect_batch_limit {
                    emit_rect_run(&mut rect_run, &mut out, descriptors);
                }
            }

            // ── Bitmap batches ────────────────────────────────────────────
            DrawCommand::PendingBitmapBatch {
                bitmap,
                mut bitmaps,
                smoothing,
                blend_mode,
            } => {
                // Different type: flush any in-progress rect run.
                emit_rect_run(&mut rect_run, &mut out, descriptors);

                let same_key = bitmap_run.as_ref().map_or(false, |(pb, ps, pbl, _)| {
                    *pb == bitmap
                        && *ps == smoothing
                        && *pbl == blend_mode
                });

                if !same_key {
                    // New key: commit the previous bitmap run (if any) and
                    // start a fresh one.
                    emit_bitmap_run(&mut bitmap_run, &mut out, descriptors);
                    bitmap_run = Some((bitmap, smoothing, blend_mode, bitmaps));
                } else {
                    // Same key: try to extend the current run.
                    let (_, _, _, pending) = bitmap_run.as_mut().unwrap();
                    if !pending.is_empty() && pending.len() + bitmaps.len() > bitmap_batch_limit {
                        // Would exceed the limit: emit the current run and
                        // start a new one with this batch.
                        emit_bitmap_run(&mut bitmap_run, &mut out, descriptors);
                        bitmap_run = Some((bitmap, smoothing, blend_mode, bitmaps));
                    } else {
                        pending.append(&mut bitmaps);
                        if pending.len() >= bitmap_batch_limit {
                            emit_bitmap_run(&mut bitmap_run, &mut out, descriptors);
                        }
                    }
                }
            }

            // ── Everything else ───────────────────────────────────────────
            other => {
                emit_rect_run(&mut rect_run, &mut out, descriptors);
                emit_bitmap_run(&mut bitmap_run, &mut out, descriptors);
                out.push(other);
            }
        }
    }

    // Flush any trailing runs.
    emit_rect_run(&mut rect_run, &mut out, descriptors);
    emit_bitmap_run(&mut bitmap_run, &mut out, descriptors);

    out
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
    rect_batch_limit: usize,
    bitmap_batch_limit: usize,
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
        rect_batch_limit,
        bitmap_batch_limit,
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

    /// Adaptive batch size limit for `DrawRectInstanced` calls.
    /// Controls the maximum number of rect instances packed into a single
    /// draw call before an automatic flush.  Derived from `FrameMetrics`;
    /// does **not** affect which commands are issued or the final rendered image.
    rect_batch_limit: usize,

    /// Adaptive batch size limit for `DrawBitmapInstanced` calls.
    /// Controls the maximum number of bitmap instances packed into a single
    /// draw call before an automatic flush.  Derived from `FrameMetrics`;
    /// does **not** affect which commands are issued or the final rendered image.
    bitmap_batch_limit: usize,

    /// Accumulated per-rect instance data for the in-progress instanced batch.
    rect_instances: Vec<RectInstance>,

    /// Accumulated per-bitmap instance data for the in-progress instanced bitmap batch.
    bitmap_instances: Vec<BitmapInstance>,
    /// Key identifying the current bitmap batch: (texture handle, smoothing, blend mode).
    /// `None` means no bitmap batch is in progress.
    bitmap_batch_key: Option<(BitmapHandle, bool, TrivialBlend)>,
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
        rect_batch_limit: usize,
        bitmap_batch_limit: usize,
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

            rect_batch_limit,
            bitmap_batch_limit,

            rect_instances: Vec::new(),

            bitmap_instances: Vec::new(),
            bitmap_batch_key: None,
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

    /// Flush the accumulated instanced rect batch (if any) as a
    /// `PendingRectBatch` command in `self.current`.  The GPU buffer is
    /// **not** created here; `optimize_draw_commands` materialises it later,
    /// once after any merges, so the data is only copied once.
    ///
    /// Call this before emitting any command that is not a `draw_rect` so that
    /// draw order between batched rects and other draw calls is preserved.
    fn flush_rect_batch(&mut self) {
        if self.rect_instances.is_empty() {
            return;
        }
        let rects = mem::take(&mut self.rect_instances);
        self.current.push(DrawCommand::PendingRectBatch(rects));
    }

    /// Flush the accumulated instanced bitmap batch (if any) as a
    /// `PendingBitmapBatch` command in `self.current`.  The GPU buffer is
    /// **not** created here; `optimize_draw_commands` materialises it later,
    /// once after any merges, so the data is only copied once.
    ///
    /// Call this before emitting any command that is not a `render_bitmap` so
    /// that draw order between batched bitmaps and other draw calls is preserved.
    fn flush_bitmap_batch(&mut self) {
        let Some((bitmap, smoothing, blend_mode)) = self.bitmap_batch_key.take() else {
            return;
        };
        if self.bitmap_instances.is_empty() {
            return;
        }
        let bitmaps = mem::take(&mut self.bitmap_instances);
        self.current.push(DrawCommand::PendingBitmapBatch {
            bitmap,
            bitmaps,
            smoothing,
            blend_mode,
        });
    }

    /// Replaces every blend with a RenderBitmap, with the subcommands rendered out to a temporary texture
    /// Every complex blend will be its own item, but every other draw will be chunked together
    fn chunk_blends(&mut self, commands: CommandList) -> Vec<Chunk> {
        commands.execute(self);

        // Flush any remaining batches before finalising the chunk.
        self.flush_rect_batch();
        self.flush_bitmap_batch();

        let needs_stencil = mem::take(&mut self.needs_stencil);
        let transforms = mem::replace(
            &mut self.transforms,
            Self::new_transforms(self.descriptors, self.dynamic_transforms),
        );

        // Optimise and push the final chunk (if non-empty).
        if !self.current.is_empty() {
            let cmds = optimize_draw_commands(
                mem::take(&mut self.current),
                self.descriptors,
                self.rect_batch_limit,
                self.bitmap_batch_limit,
            );
            self.result.push(Chunk::Draw(cmds, needs_stencil, transforms));
        }

        mem::take(&mut self.result)
    }

    /// Move `self.current` into a finalized `Chunk::Draw` and push it to
    /// `self.result`.  The command list is passed through `optimize_draw_commands`
    /// first to merge any consecutive compatible batches.
    ///
    /// Resets `self.last_transform` so the next chunk starts fresh.
    fn push_current_chunk(&mut self, transforms: BufferBuilder) {
        if self.current.is_empty() {
            return;
        }
        let cmds = optimize_draw_commands(
            mem::take(&mut self.current),
            self.descriptors,
            self.rect_batch_limit,
            self.bitmap_batch_limit,
        );
        self.result
            .push(Chunk::Draw(cmds, self.needs_stencil, transforms));
        self.last_transform = None;
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
            let old_transforms = mem::replace(
                &mut self.transforms,
                BufferBuilder::new_for_uniform(&self.descriptors.limits),
            );
            self.push_current_chunk(old_transforms);
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
        // Preserve draw order: flush any pending batches first.
        self.flush_rect_batch();
        self.flush_bitmap_batch();

        let mut surface = Surface::new(
            self.descriptors,
            self.quality,
            PostProcessQuality::High,
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
            self.rect_batch_limit,
            self.bitmap_batch_limit,
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
                    let old_transforms = mem::replace(
                        &mut self.transforms,
                        BufferBuilder::new_for_uniform(&self.descriptors.limits),
                    );
                    self.push_current_chunk(old_transforms);
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
        // Flush any pending rect batch to preserve draw order.
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

        // The instanced bitmap pipeline always uses normal (premultiplied-alpha) blending.
        let blend_mode = TrivialBlend::Normal;

        // Flush the bitmap batch if the batch key changes (different texture,
        // smoothing setting, or blend mode).
        // `BitmapHandle::PartialEq` uses `Arc::ptr_eq`, so equality holds between
        // any two handles that were cloned from the same original — i.e. they
        // share the same underlying GPU texture allocation.
        if let Some((ref key_bitmap, key_smoothing, key_blend)) = self.bitmap_batch_key {
            if *key_bitmap != bitmap
                || key_smoothing != smoothing
                || key_blend != blend_mode
            {
                self.flush_bitmap_batch();
            }
        }
        self.bitmap_batch_key = Some((bitmap, smoothing, blend_mode));
        self.bitmap_instances.push(BitmapInstance {
            x_axis:      [matrix.a, matrix.b],
            y_axis:      [matrix.c, matrix.d],
            translation: [matrix.tx.to_pixels() as f32, matrix.ty.to_pixels() as f32],
            mult_color:  transform.color_transform.mult_rgba_normalized(),
            add_color:   transform.color_transform.add_rgba_normalized(),
            // Full texture: UV maps [0,1]×[0,1] to [0,0]–[1,1].
            uv_rect:     [0.0, 0.0, 1.0, 1.0],
        });

        // Flush when the batch reaches the size limit.
        if self.bitmap_instances.len() >= self.bitmap_batch_limit {
            self.flush_bitmap_batch();
        }
    }

    fn render_stage3d(&mut self, bitmap: BitmapHandle, transform: Transform) {
        // Stage3D renders are rare and always opaque; keep the simple per-draw path.
        self.flush_rect_batch();
        self.flush_bitmap_batch();
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
        self.flush_bitmap_batch();
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
        // Flush any pending bitmap batch to preserve draw order.
        self.flush_bitmap_batch();
        // Build a RectInstance: pack the raw matrix components and
        // premultiplied colour directly, without computing world-space vertex
        // positions on the CPU.  The vertex shader applies the affine transform
        // to the shared unit-quad geometry for each instance.
        self.rect_instances.push(RectInstance {
            x_axis:      [matrix.a, matrix.b],
            y_axis:      [matrix.c, matrix.d],
            translation: [matrix.tx.to_pixels() as f32, matrix.ty.to_pixels() as f32],
            color: color_to_premult_rgba(color),
        });

        // Flush when the batch reaches the size limit.
        if self.rect_instances.len() >= self.rect_batch_limit {
            self.flush_rect_batch();
        }
    }

    fn draw_line(&mut self, color: Color, mut matrix: Matrix) {
        self.flush_rect_batch();
        self.flush_bitmap_batch();
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
        self.flush_bitmap_batch();
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
        self.flush_bitmap_batch();
        self.needs_stencil = true;
        self.num_masks += 1;
        self.current.push(DrawCommand::PushMask);
    }

    fn activate_mask(&mut self) {
        self.flush_rect_batch();
        self.flush_bitmap_batch();
        self.needs_stencil = true;
        self.current.push(DrawCommand::ActivateMask);
    }

    fn deactivate_mask(&mut self) {
        self.flush_rect_batch();
        self.flush_bitmap_batch();
        self.needs_stencil = true;
        self.current.push(DrawCommand::DeactivateMask);
    }

    fn pop_mask(&mut self) {
        self.flush_rect_batch();
        self.flush_bitmap_batch();
        self.needs_stencil = true;
        self.num_masks -= 1;
        self.current.push(DrawCommand::PopMask);
    }

    fn render_alpha_mask(&mut self, maskee_commands: CommandList, mask_commands: CommandList) {
        self.flush_rect_batch();
        self.flush_bitmap_batch();
        let mut surface = Surface::new(
            self.descriptors,
            self.quality,
            PostProcessQuality::High,
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
            self.rect_batch_limit,
            self.bitmap_batch_limit,
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
            self.rect_batch_limit,
            self.bitmap_batch_limit,
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
