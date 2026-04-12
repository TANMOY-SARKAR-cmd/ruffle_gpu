# Changes from Upstream (ruffle-rs/ruffle)

This file documents all the custom changes made in this fork (`TANMOY-SARKAR-cmd/ruffle_gpu`)
compared to the upstream [`ruffle-rs/ruffle`](https://github.com/ruffle-rs/ruffle) repository.
It is intended for future reference when re-applying these changes after syncing with upstream.

---

## 1. GPU Rendering Optimizations (Primary Goal)

The main purpose of this fork is to significantly improve GPU rendering performance.

### 1.1 GPU-Instanced Rendering for Solid Color Rectangles

- **New shader:** `render/wgpu/shaders/rect_instanced.wgsl`
  - Vertex slot 0 (step_mode=Vertex): unit-quad position (`vec2<f32>`)
  - Vertex slot 1 (step_mode=Instance): per-rect `RectInstance` data
- **`RectInstance` struct** (in `render/wgpu/src/lib.rs`): `x_axis`, `y_axis`, `translation`, `color` (premultiplied RGBA)
- `draw_rect()` now accumulates instances into `rect_instances: Vec<RectInstance>`
- `flush_rect_batch()` creates a GPU VERTEX buffer and emits `DrawCommand::DrawRectInstanced`
- `draw_rect_instanced()` binds slot 0 (unit quad) + slot 1 (instance buffer), calls `draw_indexed(0..6, 0, 0..N)` via the `rect_instanced` pipeline

### 1.2 GPU-Instanced Rendering for Bitmaps

- **New shader:** `render/wgpu/shaders/bitmap_instanced.wgsl`
  - Vertex slot 0 (step_mode=Vertex): unit-quad position (`vec2<f32>`)
  - Vertex slot 1 (step_mode=Instance): per-bitmap `BitmapInstance` data
- **`BitmapInstance` struct** (72 bytes): `x_axis`, `y_axis`, `translation`, `mult_color`, `add_color`, `uv_rect=[0,0,1,1 for full tex]`
  - Location 6 = `Float32x4` for `uv_rect` in vertex layout
  - Shader computes UV as `uv_rect.xy + pos * uv_rect.zw`
- Pipeline layout: `[globals, bitmap]` at groups 0 and 1
- `CommandRenderer` caches group-1 bitmap bind group to avoid redundant `set_bind_group` calls
- Batches are flushed when texture/smoothing/blend_mode changes

### 1.3 Adaptive EMA-Based Batch-Limit Tuner

- **`FrameMetrics` struct** in `render/wgpu/src/backend.rs`:
  - Measures per-frame wall-clock time using an Exponential Moving Average (EMA)
  - Maintains separate adaptive batch limits for `DrawRectInstanced` and `DrawBitmapInstanced`
  - Uses asymmetric decay: fast ramp-down, slow ramp-up (standard adaptive-rate controller)
  - Features cooldown period and lerp smoothing for stability
  - Disabled for determinism (limits stored but ignored in test/deterministic modes)
- `FrameMetrics` is threaded through `Surface`/`chunk_blends` to `optimize_draw_commands()`
- Per-type adaptive limits derived each frame

### 1.4 Command Optimization: Merging Consecutive Instanced Draws

- `optimize_draw_commands()` in `render/wgpu/src/surface/commands.rs`:
  - Merges consecutive `DrawRectInstanced` commands
  - Merges consecutive `DrawBitmapInstanced` commands with same bitmap/smoothing/blend_mode
  - `instance_bytes: Box<[u8]>` stored alongside GPU buffer for CPU-side merging

### 1.5 Vertex/Index Buffer State Caching

- `draw_rect_instanced()` and `draw_bitmap_instanced()` use `set_vertex_buffer_cached` / `set_index_buffer_cached` for slot 0 (vertices_pos) and the index buffer
- Avoids unconditional rebinds; they no longer clear `current_vertex_buffer`/`current_index_buffer` after drawing

### 1.6 Pre-transformed Rect Batching (Earlier Approach)

- **New shader:** `render/wgpu/shaders/color_pretransformed.wgsl`
  - CPU-pretransformed world-space positions + premultiplied RGBA per vertex
  - Only the view_matrix from globals (group 0) applied at draw time; no per-object group 1 uniform
- `DrawCommand::DrawRectBatch` variant backed by `color_pretransformed.wgsl` + `batched_color` `ShapePipeline`
  - *Note: superseded by the GPU-instanced approach (§1.1) but shader file remains*

### 1.7 Larger Transform Pre-Allocation

- Dynamic transform buffer in `render/wgpu/src/dynamic_transforms.rs` pre-allocates a larger initial size to reduce reallocations

---

## 2. wgpu Validation / Compatibility Fixes

### 2.1 ExistingWithColor COLOR_TARGET/RESOURCE Conflict (PR #59)

- **File:** `render/wgpu/src/surface/target.rs`
- wgpu panics when a texture appears as both `COLOR_TARGET` and `RESOURCE` in the same render pass
- Fix: when `RenderTargetMode::ExistingWithColor`, use a pooled frame buffer instead of the texture directly

### 2.2 Pooled Resolve Buffer for MSAA ExistingWithColor (PR #60)

- **File:** `render/wgpu/src/surface/target.rs`
- wgpu 27 panics when an MSAA resolve target is also sampled as `RESOURCE` in the same pass
- Fix: `ExistingWithColor` + MSAA now uses a **pooled resolve buffer** in addition to the pooled frame buffer
- After rendering, the resolve buffer is blitted back to the original CAB texture
- A `ResolveBuffer` field (`resolve_buffer: Option<ResolveBuffer>`) was added to the render target

### 2.3 TIMESTAMP_QUERY_INSIDE_ENCODERS on Unsupported Devices (PR #54)

- **File:** `render/wgpu/src/backend.rs` (and possibly descriptors)
- Fix: don't request `TIMESTAMP_QUERY_INSIDE_ENCODERS` feature on devices that don't support it
- Prevents validation panics on Vulkan/OpenGL devices lacking the feature

### 2.4 `.any()` on `Vec<wgpu::Adapter>` Compile Error (PR #57)

- **File:** `render/wgpu/src/backend.rs`
- Fix: iterator `.any()` call on `Vec<wgpu::Adapter>` (incorrect API usage that caused compile failure)

---

## 3. sRGB Swapchain Handling

- **File:** `render/wgpu/shaders/copy_srgb.wgsl`
  - Applies `common__srgb_to_linear` on sampled colors before writing to the sRGB swapchain
  - Counteracts the hardware linear→sRGB encode performed by the swapchain

---

## 4. Backend / Device Selection Fixes

- **File:** `desktop/src/gui/controller.rs` (or nearby)
- Fixed `prioritized_backends` logic to correctly prefer Vulkan/Metal/DX12
- Added explicit Vulkan label handling (PR #14)

---

## 5. Security Fixes

### 5.1 URL Substring Sanitization

- **File:** Related to `url_rewrite_rules`
- Fixed incomplete URL substring sanitization that could allow unexpected URL rewriting (CodeQL alert #1, PRs #2/#3)

### 5.2 Path Traversal in openh264

- **Files:** openh264-related source
- Fixed path traversal vulnerability (CodeQL alert #3, PRs #4/#5, #20)

### 5.3 Log Injection in desktop/build.rs

- **File:** `desktop/build.rs`
- Sanitized cargo output to prevent log injection (CodeQL alert #8, PRs #6/#7)

### 5.4 SSRF via Firefox Extension Submission

- **File:** `web/` extension build scripts (`submit_xpi.ts`)
- Sanitized `extensionId` in all URL paths to prevent SSRF (CodeQL alert #15, PRs #9/#10/#11/#12)
- Added file extension validation for paths; use original `movieSrc` URL

### 5.5 TOCTOU Race Condition and DOM Attribute Injection

- Fixed TOCTOU and DOM attribute injection issues flagged by CodeQL (PR #15)

### 5.6 npm Dependency Patches

- `serialize-javascript`: patched RCE and DoS via npm overrides (PR #21)
- `axios`: upgraded to 1.15.0 to patch NO_PROXY SSRF bypass (PRs #28/#34)
- `basic-ftp`: bumped from 5.2.0 to 5.2.1 (PR #8)

---

## 6. CI/CD Workflows (New — Not in Upstream)

The following files do not exist in upstream ruffle-rs/ruffle and were added entirely in this fork:

- **`.github/workflows/auto-tag.yml`**: Automatically creates a new patch-version tag on every push to `main`. Handles duplicate tags and push races.
- **`.github/workflows/release.yml`**: Builds the `ruffle_desktop` binary for Linux x86_64, macOS x86_64, and Windows x86_64 on tag push and on manual `workflow_dispatch`. Publishes a GitHub Release with pre-built binaries attached.
- **`.github/workflows/rust-clippy.yml`**: Runs `cargo clippy` on the render/wgpu crate.

---

## 7. Performance Optimizations for Desktop Player (PR #13)

- Various desktop-specific performance improvements (exact file list: `desktop/src/`)
- Explicit Vulkan preference in backend selection

---

## Files Modified vs Upstream (Key List)

| File | Change |
|------|--------|
| `render/wgpu/src/backend.rs` | FrameMetrics, adaptive limits, instanced rendering integration |
| `render/wgpu/src/surface/commands.rs` | Instanced draw commands, caching, optimization |
| `render/wgpu/src/surface/target.rs` | ExistingWithColor fixes, pooled resolve buffer |
| `render/wgpu/src/surface.rs` | FrameMetrics threading |
| `render/wgpu/src/lib.rs` | RectInstance, BitmapInstance structs |
| `render/wgpu/src/pipelines.rs` | rect_instanced, bitmap_instanced, batched_color pipelines |
| `render/wgpu/src/blend.rs` | TrivialBlend derives PartialEq+Eq for direct comparison |
| `render/wgpu/shaders/rect_instanced.wgsl` | **New file** — GPU instanced rect shader |
| `render/wgpu/shaders/bitmap_instanced.wgsl` | **New file** — GPU instanced bitmap shader |
| `render/wgpu/shaders/color_pretransformed.wgsl` | **New file** — CPU-pretransformed color batch shader |
| `render/wgpu/shaders/copy_srgb.wgsl` | sRGB swapchain correction |
| `render/wgpu/src/shaders.rs` | Registers new shaders |
| `desktop/src/gui/controller.rs` | Backend selection fixes |
| `desktop/build.rs` | Log injection fix |
| Various security-related files | See §5 above |
| `.github/workflows/auto-tag.yml` | **New file** — CI |
| `.github/workflows/release.yml` | **New file** — CI |
| `.github/workflows/rust-clippy.yml` | **New file** — CI |
