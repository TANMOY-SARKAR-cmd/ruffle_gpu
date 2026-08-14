<p align="center">
  <a href="https://ruffle.rs"><img alt="Ruffle" src="https://ruffle.rs/logo.svg" /></a>
</p>
<p align="center">
  <a href="https://github.com/TANMOY-SARKAR-cmd/ruffle_gpu/actions"><img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/TANMOY-SARKAR-cmd/ruffle_gpu/build_and_release.yml?label=Build&logo=github&branch=main" /></a>
  <br />
  <strong><a href="https://ruffle.rs">website</a> | <a href="https://github.com/TANMOY-SARKAR-cmd/ruffle_gpu">repository</a> | <a href="https://github.com/TANMOY-SARKAR-cmd/ruffle_gpu/releases">releases</a> | <a href="GPU_ENHANCEMENTS.md">GPU enhancements (design notes)</a></strong>
</p>

# Ruffle GPU

**Ruffle GPU** is a personal, actively-developed fork of [Ruffle](https://ruffle.rs) — the open-source Adobe Flash Player emulator written in Rust. It tracks the [official Ruffle repository](https://github.com/ruffle-rs/ruffle) and layers a small set of GPU-oriented enhancements on top: an **adaptive batch-limit controller** for the wgpu renderer, an **optional full-screen post-processing pipeline** (FXAA anti-aliasing, sharpening, and colour correction), and **instanced-draw batching improvements**. The goal is to keep Ruffle's pixel-faithful Flash reproduction as the default behaviour while offering performance-smoothing and visual-quality options for real hardware.

Everything else in this repository is stock Ruffle: the ActionScript 1/2 and 3 virtual machines, the audio/video subsystems, the web and desktop frontends, and the complete test infrastructure all come from upstream unchanged.

## Enhancements in this fork

| Enhancement | Crate | Default | What it does |
| --- | --- | --- | --- |
| Adaptive batch-limit controller | `render/wgpu` | **on** | Monitors wall-clock frame time with an exponential moving average and automatically reduces or recovers the instanced batch-size limits (`DrawRectInstanced`, `DrawBitmapInstanced`) when the GPU is under pressure. Purely a performance hint: it never changes which commands are issued, never skips or merges frames, and never touches ActionScript execution. See `GPU_ENHANCEMENTS.md` for the design rationale and the constants used. |
| GPU post-processing pipeline | `render/wgpu` (`gpu_post_process` feature) | **off** (compile-time opt-in) | An optional final fullscreen pass that applies FXAA, a sharpening kernel, and a mild colour correction to the scene-to-swapchain copy only. Intermediate render targets (bitmap caches, filters, Context3D textures) are never affected. Documentation: [`render/wgpu/README.md`](render/wgpu/README.md). |
| Instanced drawing | `render/wgpu` | **on** | Batched instanced draws for rectangles and bitmaps, reducing draw-call overhead on scenes with many repeated primitives. |

The adaptive controller's constants (EMA smoothing factor, pressure/relief thresholds, warm-up window, cooldown, per-step lerp rates, and min/max batch limits) are each documented with a one-line rationale next to their definition in [`render/wgpu/src/backend.rs`](render/wgpu/src/backend.rs) and in the design note. They are easy to retune for a particular GPU or workload; see `GPU_ENHANCEMENTS.md`.

## Relationship to upstream Ruffle

This fork rebases regularly on the official [ruffle-rs/ruffle](https://github.com/ruffle-rs/ruffle) `master` branch. Two of the three enhancements — the adaptive batch-limit controller and the instanced-drawing batching — are designed to be **upstreamable**: they do not change rendered output and are strictly performance improvements. The third — the post-processing pipeline — is deliberately kept as an **optional, off-by-default compile-time feature** precisely so it can coexist with Ruffle's pixel-faithfulness philosophy; it is not proposed for upstream.

If you want a vanilla Ruffle experience, build and use upstream Ruffle instead of this fork. If you want these GPU extras, track this repository and rebase on upstream releases in the usual way.

## Enabling the GPU post-processing pipeline

The post-process pass is behind the `gpu_post_process` cargo feature and is off by default. Build with:

```sh
# Check the default (no post-process) build
cargo check -p ruffle_render_wgpu

# Check / build with the post-processing pipeline enabled
cargo check -p ruffle_render_wgpu --features gpu_post_process
cargo build --release --features gpu_post_process

# Run the desktop player with the full FXAA + sharpen + colour pass
./target/release/ruffle_desktop movie.swf
```

Quality can be selected at the scene-presentation level: `PostProcessQuality::High` (default when the feature is on) runs the full pipeline, `Low` runs a bilinear copy only, and `Off` falls back to the original nearest-sampler copy with zero overhead. The pass is automatically skipped at `StageQuality::Low`, mirroring Flash's no-anti-aliasing behaviour, and the sRGB surface variant (`post_process_srgb.wgsl`) handles platforms whose swapchain format differs from the internal render format.

## Known fidelity trade-offs

> The post-processing pass (when enabled) alters **every displayed pixel**. It is an intentional visual-quality enhancement, not a bug fix. Pixel-art content and crisp UI may look softer or "enhanced" rather than Flash-faithful. Keep the feature disabled whenever you need bit-exact reproduction, and enabled only as a deliberate end-user quality option.

Everything outside the post-process feature is output-identical to upstream Ruffle: the adaptive batch controller changes only how many instances are packed per draw call, and the instanced drawing improvements do not alter rasterisation.

## Building from source

The build prerequisites are identical to upstream Ruffle. In short, on Linux you need the latest stable channel of [Rust](https://www.rust-lang.org/tools/install) and Java on your `PATH` as `java` (required to build the ActionScript 3 builtin class library), plus these typical system dependencies:

```sh
# Ubuntu/Debian
sudo apt install pkg-config libasound2-dev libudev-dev default-jre-headless g++
# Fedora/RHEL
sudo dnf install pkgconf-pkg-config alsa-lib-devel systemd-devel java-latest-openjdk-headless gcc-c++
```

Then:

```sh
cargo build --release                              # default: stock Ruffle + adaptive batching
cargo build --release --features gpu_post_process  # + post-processing pipeline
```

## Structure

Inherited from upstream: the workspace is made up of `core` (the ActionScript VMs and player), `desktop`, `web`, `render` and its backends (`wgpu`, `webgl`, `canvas`), `swf` (file format parsing), `flv`, `video`, `naga-agal`/`naga-pixelbender` (shader translators), `exporter`, `scanner`, and `frontend-utils`. The fork's own additions live in `render/wgpu` (`backend.rs` adaptive controller, `surface.rs` post-process presentation path, `utils.rs` post-process pipeline helpers, `shaders/post_process*.wgsl`) and in `GPU_ENHANCEMENTS.md` at the repository root.

## License

Inherited from upstream Ruffle: the code in this repository is licensed under either [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) or [MIT License](https://opensource.org/licenses/MIT), at your option. Ruffle depends on third-party libraries under compatible licenses; see [LICENSE.md](LICENSE.md) for full information.

### Contributing

Contributions to the GPU-specific code (`render/wgpu`) are welcome: performance improvements, controller tuning, or new optional visual features that respect the "off or opt-in by default" rule. Upstream-eligible improvements (output-identical performance work) are encouraged. Please keep the workspace green under both feature configurations before opening a change:

```sh
cargo check --workspace
cargo check -p ruffle_render_wgpu --features gpu_post_process
cargo clippy --workspace
cargo test -p ruffle_render_wgpu --lib
```
