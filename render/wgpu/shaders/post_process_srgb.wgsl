/// Post-processing shader (sRGB surface variant, High quality): bilinear
/// sampling, FXAA, sharpen, colour correction, and sRGB conversion.
/// Applied as the final fullscreen pass when the internal render format and the
/// swapchain surface format differ (e.g. Bgra8Unorm → Bgra8UnormSrgb).
/// All post-processing steps are identical to post_process.wgsl; only the
/// final output applies `common__srgb_to_linear` so that the sRGB-aware
/// surface encodes values correctly.

// NOTE: The `common.wgsl` source is prepended to this before compilation.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(1) @binding(0) var<uniform> transforms: common__Transforms;
@group(2) @binding(0) var<uniform> textureTransforms: common__TextureTransforms;
@group(2) @binding(1) var src_texture: texture_2d<f32>;
@group(2) @binding(2) var src_sampler: sampler;

@vertex
fn main_vertex(in: common__VertexInput) -> VertexOutput {
    let matrix_ = textureTransforms.texture_matrix;
    let uv = (mat3x3<f32>(matrix_[0].xyz, matrix_[1].xyz, matrix_[2].xyz) * vec3<f32>(in.position, 1.0)).xy;
    let pos = common__globals.view_matrix * transforms.world_matrix * vec4<f32>(in.position.x, in.position.y, 0.0, 1.0);
    return VertexOutput(pos, uv);
}

// --- helpers ----------------------------------------------------------------

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

// --- FXAA -------------------------------------------------------------------

fn fxaa(uv: vec2<f32>, px: vec2<f32>) -> vec4<f32> {
    let center = textureSample(src_texture, src_sampler, uv);

    let cn = textureSample(src_texture, src_sampler, uv + vec2<f32>( 0.0,  -px.y));
    let cs = textureSample(src_texture, src_sampler, uv + vec2<f32>( 0.0,   px.y));
    let ce = textureSample(src_texture, src_sampler, uv + vec2<f32>( px.x,  0.0 ));
    let cw = textureSample(src_texture, src_sampler, uv + vec2<f32>(-px.x,  0.0 ));

    let lc  = luma(center.rgb);
    let ln  = luma(cn.rgb);
    let ls  = luma(cs.rgb);
    let le  = luma(ce.rgb);
    let lw  = luma(cw.rgb);

    let luma_min = min(lc, min(min(ln, ls), min(le, lw)));
    let luma_max = max(lc, max(max(ln, ls), max(le, lw)));
    let luma_range = luma_max - luma_min;

    if luma_range < max(0.0625, luma_max * 0.125) {
        return center;
    }

    // Additional guard (Step 4): absolute minimum below which FXAA is skipped
    // to preserve pixel art and flat UI sharpness.
    let threshold_low = 0.03125; // 1/32
    if luma_range < threshold_low {
        return center;
    }

    let lnw = luma(textureSample(src_texture, src_sampler, uv + vec2<f32>(-px.x, -px.y)).rgb);
    let lne = luma(textureSample(src_texture, src_sampler, uv + vec2<f32>( px.x, -px.y)).rgb);
    let lsw = luma(textureSample(src_texture, src_sampler, uv + vec2<f32>(-px.x,  px.y)).rgb);
    let lse = luma(textureSample(src_texture, src_sampler, uv + vec2<f32>( px.x,  px.y)).rgb);

    let edge_h = abs(ln + ls - 2.0 * lc) * 2.0
               + abs(lne + lse - 2.0 * le)
               + abs(lnw + lsw - 2.0 * lw);
    let edge_v = abs(le + lw - 2.0 * lc) * 2.0
               + abs(lne + lnw - 2.0 * ln)
               + abs(lse + lsw - 2.0 * ls);

    let is_horz = edge_h >= edge_v;

    let blend = clamp(
        (abs(ln + ls + le + lw - 4.0 * lc) / 4.0
       + abs(lne + lnw + lse + lsw - 4.0 * lc) / 4.0) / luma_range,
        0.0, 1.0
    );
    let blend_sq = blend * blend;

    let offset = select(
        vec2<f32>(0.0,      blend_sq * px.y),
        vec2<f32>(blend_sq * px.x, 0.0),
        is_horz
    );
    return textureSample(src_texture, src_sampler, uv + offset);
}

// --- Sharpen ----------------------------------------------------------------

fn sharpen(uv: vec2<f32>, px: vec2<f32>, aa_center: vec4<f32>) -> vec4<f32> {
    let n = textureSample(src_texture, src_sampler, uv + vec2<f32>( 0.0,  -px.y));
    let s = textureSample(src_texture, src_sampler, uv + vec2<f32>( 0.0,   px.y));
    let e = textureSample(src_texture, src_sampler, uv + vec2<f32>( px.x,  0.0 ));
    let w = textureSample(src_texture, src_sampler, uv + vec2<f32>(-px.x,  0.0 ));

    // Safer kernel: centre × 1.25 − (N+S+E+W) × 0.0625
    // Net weight = 1.25 − 4 × 0.0625 = 1.0 (energy-preserving; no clipping artifacts)
    let sharpened = aa_center * 1.25 - (n + s + e + w) * 0.0625;
    return vec4<f32>(clamp(sharpened.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), aa_center.a);
}

// --- Colour correction ------------------------------------------------------

fn color_correct(c: vec4<f32>) -> vec4<f32> {
    var rgb = c.rgb;
    if c.a > 0.0 {
        rgb = rgb / c.a;
    }
    // Contrast factor reduced 1.05 → 1.02 to avoid aggressive contrast boost
    rgb = clamp((rgb - vec3<f32>(0.5)) * 1.02 + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(rgb * c.a, c.a);
}

// --- Main -------------------------------------------------------------------

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let px = 1.0 / vec2<f32>(textureDimensions(src_texture));

    // Early exit: detect flat regions (low luma variance) and skip both
    // FXAA and sharpen entirely, reducing GPU cost on uniform areas (Step 7).
    let center = textureSample(src_texture, src_sampler, in.uv);
    let ln_pre = luma(textureSample(src_texture, src_sampler, in.uv + vec2<f32>( 0.0,  -px.y)).rgb);
    let ls_pre = luma(textureSample(src_texture, src_sampler, in.uv + vec2<f32>( 0.0,   px.y)).rgb);
    let le_pre = luma(textureSample(src_texture, src_sampler, in.uv + vec2<f32>( px.x,  0.0 )).rgb);
    let lw_pre = luma(textureSample(src_texture, src_sampler, in.uv + vec2<f32>(-px.x,  0.0 )).rgb);
    let lc_pre = luma(center.rgb);

    let luma_max_pre = max(lc_pre, max(max(ln_pre, ls_pre), max(le_pre, lw_pre)));
    let luma_range_pre = luma_max_pre - min(lc_pre, min(min(ln_pre, ls_pre), min(le_pre, lw_pre)));

    let threshold_low = 0.03125; // 1/32
    if luma_range_pre < threshold_low {
        let corrected = color_correct(center);
        return common__srgb_to_linear(corrected);
    }

    // 1. FXAA
    let aa = fxaa(in.uv, px);

    // 2. Sharpen
    let sharp = sharpen(in.uv, px, aa);

    // 3. Colour correction
    let corrected = color_correct(sharp);

    // 4. sRGB conversion – identical to copy_srgb.wgsl – needed because the
    //    surface is an sRGB format and wgpu will apply the linear→sRGB
    //    hardware encode when writing to it.
    return common__srgb_to_linear(corrected);
}
