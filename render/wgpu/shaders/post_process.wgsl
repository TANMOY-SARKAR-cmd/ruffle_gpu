/// Post-processing shader: bilinear sampling, FXAA, sharpen, and colour correction.
/// Applied as the final fullscreen pass (offscreen scene texture → swapchain).
/// The internal texture format equals the surface format, so no sRGB conversion
/// is performed here (see post_process_srgb.wgsl for the sRGB variant).

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

/// Perceptual luminance used for FXAA luma estimation.
/// Works in premultiplied-alpha space; the result is used only for contrast
/// ratios, so the premultiplied scaling cancels out.
fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

// --- FXAA -------------------------------------------------------------------

/// Lightweight FXAA operating entirely in premultiplied-alpha colour space.
/// Detects aliased edges via luma contrast, then blends each pixel toward its
/// neighbour along the edge direction.  The algorithm is a single-pass, single-
/// texture-read-per-tap version of FXAA 3.11's "console" preset:
///   • Minimum cost: no extra render targets, no multi-pass, no extra uniforms.
///   • Works per-pixel; no geometry changes.
fn fxaa(uv: vec2<f32>, px: vec2<f32>) -> vec4<f32> {
    let center = textureSample(src_texture, src_sampler, uv);

    // Cardinal neighbours
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

    // Early-out for flat (non-edge) regions – avoids unnecessary blurring
    if luma_range < max(0.0625, luma_max * 0.125) {
        return center;
    }

    // Diagonal samples (needed for edge-direction estimation only)
    let lnw = luma(textureSample(src_texture, src_sampler, uv + vec2<f32>(-px.x, -px.y)).rgb);
    let lne = luma(textureSample(src_texture, src_sampler, uv + vec2<f32>( px.x, -px.y)).rgb);
    let lsw = luma(textureSample(src_texture, src_sampler, uv + vec2<f32>(-px.x,  px.y)).rgb);
    let lse = luma(textureSample(src_texture, src_sampler, uv + vec2<f32>( px.x,  px.y)).rgb);

    // Horizontal vs. vertical edge strength (Sobel-style)
    let edge_h = abs(ln + ls - 2.0 * lc) * 2.0
               + abs(lne + lse - 2.0 * le)
               + abs(lnw + lsw - 2.0 * lw);
    let edge_v = abs(le + lw - 2.0 * lc) * 2.0
               + abs(lne + lnw - 2.0 * ln)
               + abs(lse + lsw - 2.0 * ls);

    let is_horz = edge_h >= edge_v;

    // Sub-pixel blend factor – quadratic falloff keeps the effect subtle
    let blend = clamp(
        (abs(ln + ls + le + lw - 4.0 * lc) / 4.0
       + abs(lne + lnw + lse + lsw - 4.0 * lc) / 4.0) / luma_range,
        0.0, 1.0
    );
    let blend_sq = blend * blend;

    // Step toward the detected edge perpendicular direction
    let offset = select(
        vec2<f32>(0.0,      blend_sq * px.y),
        vec2<f32>(blend_sq * px.x, 0.0),
        is_horz
    );
    return textureSample(src_texture, src_sampler, uv + offset);
}

// --- Sharpen ----------------------------------------------------------------

/// 4-tap cross sharpening kernel applied after FXAA.
/// Uses the FXAA-processed value as the centre and samples the original
/// texture for the four neighbours, producing a mild unsharp-mask effect
/// that restores perceived crispness without reintroducing aliasing.
///
/// Kernel weights: centre × 1.5  −  (N+S+E+W) × 0.125
/// Net weight = 1.5 − 4 × 0.125 = 1.0 (energy-preserving).
fn sharpen(uv: vec2<f32>, px: vec2<f32>, aa_center: vec4<f32>) -> vec4<f32> {
    let n = textureSample(src_texture, src_sampler, uv + vec2<f32>( 0.0,  -px.y));
    let s = textureSample(src_texture, src_sampler, uv + vec2<f32>( 0.0,   px.y));
    let e = textureSample(src_texture, src_sampler, uv + vec2<f32>( px.x,  0.0 ));
    let w = textureSample(src_texture, src_sampler, uv + vec2<f32>(-px.x,  0.0 ));

    let sharpened = aa_center * 1.5 - (n + s + e + w) * 0.125;
    // Clamp RGB to [0,1]; preserve the (premultiplied) alpha unchanged
    return vec4<f32>(clamp(sharpened.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), aa_center.a);
}

// --- Colour correction ------------------------------------------------------

/// Subtle contrast boost in straight-alpha space.
/// Values are deliberately mild (contrast factor 1.05) to avoid visible
/// distortion while improving perceived depth.
fn color_correct(c: vec4<f32>) -> vec4<f32> {
    var rgb = c.rgb;
    // Unmultiply to straight alpha so the contrast curve is applied uniformly
    if c.a > 0.0 {
        rgb = rgb / c.a;
    }
    // Contrast: (x − 0.5) × 1.05 + 0.5
    rgb = clamp((rgb - vec3<f32>(0.5)) * 1.05 + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));
    // Re-multiply
    return vec4<f32>(rgb * c.a, c.a);
}

// --- Main -------------------------------------------------------------------

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let px = 1.0 / vec2<f32>(textureDimensions(src_texture));

    // 1. FXAA – smooth aliased edges
    let aa = fxaa(in.uv, px);

    // 2. Sharpen – restore crispness using the AA result as the centre
    let sharp = sharpen(in.uv, px, aa);

    // 3. Colour correction
    return color_correct(sharp);
}
