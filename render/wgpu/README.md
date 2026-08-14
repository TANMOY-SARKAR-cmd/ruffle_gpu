# ruffle_render_wgpu

The [wgpu](https://wgpu.rs/)-based rendering backend for
[Ruffle](https://ruffle.rs).

## Features

| Feature | Default | Description |
| --- | --- | --- |
| `clap` | no | Exposes command-line options (requires the `clap` crate dependency). |
| `render_debug_labels` | no | Adds debug labels to GPU resources for profiling tools. |
| `webgl` | no | Enables the WebGL backend inside wgpu (for `wasm` targets). |
| `profile-with-tracy` | no | Integrates the Tracy profiler. |
| `gpu_post_process` | **no** | See below. |

### GPU post-processing pipeline (`gpu_post_process`)

An optional full-screen pass applied to the **final scene-to-swapchain copy
only**. When enabled, the plain copy shader is replaced by a post-process
shader that samples the rendered scene with a bilinear (linear) sampler and
applies, in the fragment stage:

1. **FXAA** (fast approximate anti-aliasing)
2. A **sharpening** kernel to counter FXAA softening
3. A subtle **contrast / colour correction**

Intermediate copies (e.g. bitmap-cache entries with filters) are never
affected — only what the user actually sees on screen.

> **This feature is off by default.** Ruffle's primary goal is pixel-faithful
> Flash reproduction, and a post-processing pass alters every displayed
> pixel. Enable it deliberately (e.g. as an end-user visual-quality option):
>
> ```toml
> ruffle_render_wgpu = { features = ["gpu_post_process"] }
> ```

### Interaction with filters, bitmap caches, and Context3D

The post-process pass only intercepts the **last** copy (the rendered offscreen scene → swapchain). Every intermediate render target keeps using the plain copy pipeline, so game-logic pixels are never altered:

| Subsystem | Render path | Post-process effect |
| --- | --- | --- |
| Normal display list | Offscreen frame → swapchain | **Affected** (the final pass) |
| Bitmap cache / filter targets | Separate offscreen textures via `run_copy_pipeline` | **Unaffected** |
| Context3D (Stage3D) framebuffers | Own framebuffers | **Unaffected** — but Stage3D output *composited into the main scene* is processed, because it becomes part of the final scene texture |
| Mask stencils / blend intermediates | Internal pipeline state | **Unaffected** |

Filters (blur, glow, drop shadow, …) and cached bitmaps are rasterised, cached, and composited exactly as upstream Ruffle does; only the resulting picture that reaches the screen may receive FXAA / sharpen / colour correction. The `StageQuality::Low` skip means Flash's own no-anti-aliasing mode disables the pass entirely, and the `post_process_srgb.wgsl` variant handles platforms whose swapchain format differs from the internal render format (consistent with the existing `copy_srgb` pipeline).

Two implementation notes worth knowing when extending this pass:

- The pass respects `StageQuality`: it is skipped at `StageQuality::Low`,
  mirroring Flash's no-anti-aliasing behaviour.
- If the swapchain surface format differs from the internal render format
  (a platform supporting only an sRGB surface), the `post_process_srgb`
  shader variant performs the colour-space conversion — consistent with the
  existing `copy_srgb` pipeline.

## Build

```sh
cargo check -p ruffle_render_wgpu                      # default (no post-process)
cargo check -p ruffle_render_wgpu --features gpu_post_process  # with the pass
```
