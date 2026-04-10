# Ruffle Architecture & Risk Analysis Report

## 1. Architecture Overview

### Core Modules
Based on the `use` and `mod` dependencies extracted in `analysis_report.txt`, the Ruffle architecture can be grouped into several key domains:
- **`core/`**: The heart of the implementation containing AVM1 (`core/src/avm1`), AVM2 (`core/src/avm2`), and the display hierarchy (`core/src/display_object/`). This module handles all ActionScript execution, DOM modeling, shape modeling, and bitmap processing (in `core/src/bitmap`).
- **`render/`**: Handles the rendering side with generic abstractions (`render/src/backend.rs`, `render/src/commands.rs`) as well as multiple backend implementations including `render/wgpu` (WebGPU/Vulkan/Metal/DX12), `render/canvas` (HTML5 Canvas), and specific components for shaders (`render/src/shader_source.rs`, `render/naga-agal`, `render/pixel_bender`).
- **`video/` & `flv/`**: For video decoding (e.g. h263, VP6 via `nihav`) and FLV container parsing.
- **`scanner/`, `exporter/`, `stub-report/`**: Tooling for scanning SWFs, exporting frames, and generating compatibility reports.

### Rendering Pipeline Flow
The rendering pipeline operates across clearly defined boundaries:
- **SWF Data parsing** builds tags which map into DisplayObjects (`core/src/display_object/`).
- The **`core/`** calculates the view and generates drawing commands (defined in `render/src/commands.rs`), utilizing filters, blends, matrices (`render/src/matrix.rs`, `render/src/matrix3d.rs`), and tessellation (`render/src/tessellator.rs`).
- These generic commands are handed off to a backend such as `render/wgpu/src/backend.rs`.
- The `wgpu` backend translates these into specific GPU descriptors, binding groups (`render/wgpu/src/descriptors.rs`), and executes draw calls on a `wgpu` RenderPass (`render/wgpu/src/surface.rs`, `render/wgpu/src/mesh.rs`, `render/wgpu/src/pipelines.rs`).

### Command Flow (CPU → GPU)
- ActionScript logic mutates DisplayObjects on the CPU (`core`).
- Each frame, `core` traverses the DisplayList to issue rendering commands via the `RenderBackend` trait.
- The GPU backends (like `wgpu`) receive these commands, batch mesh generation (`render/wgpu/src/buffer_builder.rs`, `render/wgpu/src/buffer_pool.rs`), and map Flash constructs (bitmaps, blends, text, filters) to shader permutations (`render/wgpu/src/shaders.rs`, `render/wgpu/src/context3d/`).
- Actual execution goes through the `wgpu` surface/target implementations.

---

## 2. Risk Areas

Based on dependency links and component structures, several risk areas stand out:
- **`render/wgpu/src/buffer_pool.rs` & `render/wgpu/src/buffer_builder.rs` (Buffer Lifecycle Issues):**
  - **Risk:** Flash applications can generate extremely dynamic and variable geometry (vector graphics). Poor buffer pooling or reallocation can lead to GPU memory fragmentation and CPU stalls if buffers are resized or created on every frame.
- **`render/src/tessellator.rs` (Timing-Sensitive Code & Performance):**
  - **Risk:** Tessellating vector graphics into triangles on the CPU is a major bottleneck. If not cached correctly, complex SWF shapes animated each frame will cause significant frame delays.
- **`render/wgpu/src/filters/*.rs` and `render/src/blend.rs` (Rendering Order Dependencies & Batching Logic):**
  - **Risk:** Flash relies heavily on specific blending modes (e.g., Layer, Erase) and filters (Blur, Drop Shadow). These often require intermediate render targets or specific draw orders. This breaks batching. If many small objects have different blend modes or filters, the command flow could experience massive overhead due to constant pipeline state switches and render pass interruptions.
- **`render/pixel_bender` and `render/naga-agal`:**
  - **Risk:** Converting Pixel Bender or AGAL (Flash 3D shaders) on-the-fly into WGSL. Improper conversion or failure to cache translated shaders can introduce runtime compilation stutter.
- **Possible Race Conditions:**
  - `scanner/src/ser_bridge.rs` uses `rayon::prelude::` for parallel processing. Depending on how global states or parsed data are shared, multithreading across large SWF collections can be risky if internal caches aren't properly synchronized.

---

## 3. Performance Analysis (Opportunities)

- **Unnecessary Allocations & Repeated Buffer Creation:** The rendering of dynamic vector graphics (which change shape/size) often leads to fresh allocations if `buffer_pool` isn't aggressively reusing memory. Pre-allocating larger arenas or using ring buffers for dynamic geometry could reduce allocations.
- **Inefficient Loops (Tessellation):** `lyon_tessellation` is used for rendering. There may be missed caching opportunities for shapes that undergo only affine transformations (translation/rotation) but are re-tessellated instead of having their existing mesh transformed by a matrix on the GPU.
- **Batching Opportunities:** Grouping objects that share the same texture/bitmap and the same pipeline state (no blend changes) before sending to `wgpu` could drastically reduce the number of draw calls.

---

## 4. Safety Check

- **Logic Moved to GPU Incorrectly:** Flash expects strict drawing order (painters algorithm) and exact pixel matching for certain blending. Offloading complex blending (like "Erase" or "Layer") to the GPU requires careful handling (often via intermediate textures) to ensure non-deterministic GPU rasterization differences do not break the expected visual output.
- **Frame Delay Introduced:** Transpiling shaders (AGAL to WGSL) or complex tessellation must not delay the frame. Shader compilation needs to be done ahead of time or handled without blocking the main render loop where possible.
- **Non-deterministic Behavior:** The use of floating-point math across different GPU drivers (`wgpu`) can cause slight rendering differences. Care must be taken to ensure logic dependent on precise bounds (e.g., hit-testing in `core`) remains purely CPU-based and deterministic.

---

## 5. Safe Improvements

- **CPU Optimizations:**
  - Enhance shape tessellation caching. If a shape's vertices haven't structurally changed, only its transform matrix should be updated.
  - Implement a more aggressive object pool for AVM1/AVM2 objects to reduce garbage collection pressure on the CPU.
- **Safe GPU Improvements & Memory Reuse:**
  - Utilize persistent mapped buffers or sub-allocation strategies within `render/wgpu/src/buffer_pool.rs` for dynamic vertex data to avoid mapping overhead.
  - Optimize the use of Render Targets for filters. Reuse a pool of intermediate textures rather than creating new ones per filtered object.
- **Batching Improvements (Without Changing Behavior):**
  - Implement a deferred state update mechanism in `render/src/commands.rs` to aggregate draw calls of the same type (e.g., solid color fills or same-bitmap drawing) if they are adjacent in the display list.

---

## 6. Feature Suggestions (Non-Breaking)

- **Performance Metrics (HUD):**
  - Add an optional overlay displaying FPS, draw call counts, tessellation time, and AVM execution time. This will not affect Flash behavior but greatly aid developers.
- **Debug Tools:**
  - Introduce a "Display List Explorer" tool or wireframe rendering mode (`wgpu` supports `PolygonMode::Line`) to help debug complex SWF layering issues safely.
- **Visual Enhancements:**
  - Optional multisample anti-aliasing (MSAA) pass. Flash natively had low/medium/high quality settings. Implementing high-quality MSAA at the `wgpu` level can smoothly scale vector graphics on modern high-DPI displays without changing internal logical bounds.
