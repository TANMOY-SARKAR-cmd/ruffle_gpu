# GPU Enhancements in Ruffle GPU

Design notes for the GPU-oriented additions in this fork. These changes are layered on top of stock Ruffle and are documented here so that the behaviour, the constant choices, and the interaction with the rest of the renderer are explicit and easy to audit or retune.

## 1. Adaptive batch-limit controller

**Location:** `render/wgpu/src/backend.rs` (`FrameMetrics`), used by `submit_frame` in both the main presentation path and the bitmap-cache render path.

**Unit tests:** the same module contains a test suite (`cargo test -p ruffle_render_wgpu --lib`) covering warm-up suppression, pressure reduction, minimum flooring, recovery after pressure, and steady-state stability.

### Problem

Flash content varies enormously in draw density: a simple UI scene issues a few hundred instanced primitives, while a busy Stage3D scene can issue tens of thousands. Ruffle packs instances into GPU draw calls up to a fixed cap, flushing when the cap is reached. A static cap creates two opposite failure modes on diverse hardware:

| Situation | Symptom |
| --- | --- |
| Cap too large for a slow GPU | Very large draw calls stall the GPU, frame time balloons, and the player appears to freeze; the GPU never recovers because the next frames are equally heavy. |
| Cap too small for a fast GPU | Draw-call overhead dominates: the CPU issues many tiny batches, and the GPU is never fully utilised. |

The original fixed cap (`16 384`) is a reasonable ceiling but gives no room to back off under pressure.

### Solution

A lightweight, output-invariant controller measures the **wall-clock time between consecutive frame submissions**, smooths it with an exponential moving average (EMA), and adjusts the per-type batch limits (`DrawRectInstanced` and `DrawBitmapInstanced`) toward a smaller limit under pressure and toward the original cap when headroom is detected. The controller changes only **how many** instances are packed before an automatic flush; it never changes **which** commands are issued, **when** a frame is presented, or the ActionScript execution path. No frames are skipped or merged.

### Algorithm

Each frame end updates the EMA and, subject to a warm-up gate and a cooldown gate, applies a standard linear-interpolation step to each limit independently:

```text
smoothed_ms  ← α · elapsed_ms + (1 − α) · smoothed_ms          // EMA
if smoothed_ms > PRESSURE:   limit ← lerp(limit, MIN, LERP_DOWN)   // back off
if smoothed_ms < RELIEF:     limit ← lerp(limit, MAX, LERP_UP)     // recover
```

### Constant choices and rationale

All values live as documented constants at the top of `render/wgpu/src/backend.rs` and can be retuned by editing that one block:

| Constant | Value | Rationale |
| --- | --- | --- |
| `EMA_ALPHA` | `0.1` | Weighs the newest frame at 10 % and history at 90 %. Sustained load changes are incorporated within roughly 10 frames, but single-frame spikes (shader compilation, asset stalls) are dampened to a 10 % influence. |
| `PRESSURE_THRESHOLD_MS` | `33.0 ms` (≈ 30 FPS) | A smoothed frame time above 33 ms means the player is visibly below 30 FPS — the standard threshold at which stutter becomes objectionable, so backing off is justified. |
| `RELIEF_THRESHOLD_MS` | `20.0 ms` (≈ 50 FPS) | Well below the 60 FPS target (16.67 ms is the ideal). Recovering only below 20 ms keeps a comfortable margin so the controller does not sit permanently on the boundary and ping-pong. |
| `WARMUP_FRAMES` | `8` (~133 ms at 60 FPS) | The first frames of a movie carry start-up overhead (pipeline compilation, asset decoding). Eight frames let the EMA absorb several real samples before any adjustment is permitted. |
| `COOLDOWN_FRAMES` | `10` | Limits the adjustment rate to at most once per ten frames, preventing oscillation when the smoothed time hovers near a threshold. The cooldown is anchored to *actual* limit changes, so idle hovering costs nothing. |
| `LERP_STEP_DOWN` | `0.25` | Closes a quarter of the remaining gap per reduction. Under real pressure, fast back-off is more valuable than smoothness. |
| `LERP_STEP_UP` | `0.10` | Recovery is deliberately slower than reduction: a fast ramp-up after a pressure event would immediately re-trigger reduction (sawtooth). Slow, steady recovery is the classic asymmetric-control shape. |
| `MIN_BATCH_LIMIT` | `256` | A floor that keeps batching useful even in the worst case — below this, draw-call overhead per instance dominates and the controller's goal (fewer, larger calls) is defeated. |
| `MAX_BATCH_LIMIT` | `16 384` | The original hard cap; the controller never exceeds the behaviour of stock Ruffle. |

The `lerp` step uses `floor` on reduction (guaranteeing strict progress toward the floor, reaching `MIN_BATCH_LIMIT` exactly once the remaining gap drops below 1.0) and `ceil` on recovery (the symmetric guarantee for `MAX_BATCH_LIMIT`, avoiding an infinite near-ceiling stall).

### Correctness guarantees

1. **Output invariance.** Batch limits only determine where automatic flushes occur in the command stream; the recorded command list, rasterisation, and presentation timing are unchanged. ActionScript execution is untouched.
2. **Boundedness.** Limits are clamped to `[MIN_BATCH_LIMIT, MAX_BATCH_LIMIT]` after every step.
3. **No oscillation in steady state.** The cooldown gate plus the hysteresis gap (`RELIEF` < any comfortable frame time < `PRESSURE`) means a stable workload provokes zero adjustments after the EMA converges. The test `sustained_sub_pressure_frames_stabilize_the_limits` asserts exactly this over hundreds of frames.

### Observed impact

Measured in the sandbox on software Vulkan (llvmpipe) with the consolidation build (`bc46e488`) while running a Stage3D-heavy title (Strike Force Heroes): under sustained software-render load the controller drove both limits from 16 384 down toward 256 within a few seconds of 60 ms frames, and after the load normalised the limits recovered step-by-step toward the cap. Because the sandbox has no real GPU, these numbers characterise the *controller's behaviour*, not the renderer's throughput — real-hardware measurements (GPU frame time, draw-call counts, batch sizes before/after) should be taken by the owner before claiming quantitative speed-ups. On the software backend the controller is expected to be neutral-to-negative (reducing batch sizes removes GPU parallelism that llvmpipe was already struggling with), which is exactly why it reacts to measured pressure rather than assuming faster is always better.

### Future work

Candidate refinements, in order of expected value: a per-frame-type split (separate pressure tracking for the bitmap-cache path versus the main presentation path), hysteresis on the EMA itself (separate fast/slow EMAs), an option to log the adjusted limits via `debug_info()`, and exposing the constants through a player config rather than source edits.

## 2. Optional GPU post-processing pipeline

**Location:** `render/wgpu/src/surface.rs` (presentation path), `render/wgpu/src/utils.rs` (`run_post_process_pipeline`), shaders `render/wgpu/shaders/post_process.wgsl` and `post_process_srgb.wgsl`. Feature flag: `gpu_post_process` in `render/wgpu/Cargo.toml` (off by default).

A single-pass, zero-latency, full-screen fragment shader applied to the **final scene-to-swapchain copy only**: bilinear sampling, FXAA edge smoothing, a mild sharpening kernel (centre × 1.25 minus the four cardinal neighbours × 0.0625, energy-preserving), and a subtle contrast curve (factor 1.02) in straight-alpha space. Flat regions (luma range below 1/32) skip FXAA and sharpen entirely, protecting pixel art and flat UI.

### Interaction with filters, bitmap caches, and Context3D

The post-process pass only intercepts the last copy from the rendered offscreen scene to the swapchain. Every intermediate render target is unaffected:

| Ruffle subsystem | Render path | Post-process effect |
| --- | --- | --- |
| Normal display list | Rendered to the offscreen frame, copied to the swapchain | **Affected** (this is the pass) |
| Bitmap cache / filter render targets | Separate offscreen textures, copied via the plain `run_copy_pipeline` | **Unaffected** |
| Context3D (Stage3D) textures | Rendered into their own framebuffers | **Unaffected** — Stage3D output *that is composited into the main scene* is processed, because it becomes part of the final scene texture |
| Mask stencils, blend intermediates | Internal pipeline state | **Unaffected** |

The pass respects `StageQuality::Low` (Flash's no-anti-aliasing mode skips it entirely) and switches to the sRGB variant when the surface format differs from the internal render format, mirroring the existing `copy_srgb` pipeline. Three quality modes exist (`Off` = nearest copy, zero overhead; `Low` = bilinear copy; `High` = full pipeline), with `Off` implemented as a CPU-side fallback to the original copy pipeline so the feature imposes no cost when disabled.

## 3. Instanced drawing improvements

**Location:** `render/wgpu` batching code. Rectangles and bitmaps are packed into instanced draw calls up to the per-type batch limit described in section 1, which is what makes the adaptive controller act on the two most common batch types. The change reduces per-draw overhead (descriptor/vertex submission and command-buffer bookkeeping) on scenes with many repeated primitives; it does not alter rasterisation.

## Tuning the controller

To retune for a specific hardware target, edit the documented constant block at the top of `render/wgpu/src/backend.rs` and re-run the controller test suite:

```sh
cargo test -p ruffle_render_wgpu --lib backend::tests
```

Good starting points: for a consistently fast desktop GPU, raise `RELIEF_THRESHOLD_MS` toward 16.67 and raise `EMA_ALPHA` slightly for faster reaction; for a thermally-throttled laptop, lower `PRESSURE_THRESHOLD_MS` toward 20–25 ms and raise `LERP_STEP_DOWN` to 0.5 for faster back-off.
