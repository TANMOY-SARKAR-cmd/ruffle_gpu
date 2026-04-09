/// Instanced rendering shader for bitmaps.
///
/// * Vertex slot 0 (step_mode = Vertex):   unit-quad position (`vec2<f32>`)
/// * Vertex slot 1 (step_mode = Instance): per-bitmap affine transform + color transforms
///
/// The per-instance data encodes the 2D affine transform mapping the unit quad
/// corners [0,1]×[0,1] to world space, plus multiplicative and additive color
/// transforms:
///   world.x = x_axis.x * pos.x + y_axis.x * pos.y + translation.x
///   world.y = x_axis.y * pos.x + y_axis.y * pos.y + translation.y
///
/// The view_matrix from globals (group 0) maps world to clip space.
/// The texture and its UV transform live in group 1 (no per-object transform
/// uniform at group 1; the pipeline layout is [globals, bitmap]).
///
/// NOTE: The `common.wgsl` source is prepended to this before compilation.

struct VertexInput {
    @location(0) position: vec2<f32>,
};

struct InstanceInput {
    @location(1) x_axis:      vec2<f32>,
    @location(2) y_axis:      vec2<f32>,
    @location(3) translation: vec2<f32>,
    @location(4) mult_color:  vec4<f32>,
    @location(5) add_color:   vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv:         vec2<f32>,
    @location(1) mult_color: vec4<f32>,
    @location(2) add_color:  vec4<f32>,
};

@group(1) @binding(0) var<uniform> textureTransforms: common__TextureTransforms;
@group(1) @binding(1) var texture: texture_2d<f32>;
@group(1) @binding(2) var texture_sampler: sampler;

@vertex
fn main_vertex(vert: VertexInput, inst: InstanceInput) -> VertexOutput {
    let world = vec2<f32>(
        inst.x_axis.x * vert.position.x + inst.y_axis.x * vert.position.y + inst.translation.x,
        inst.x_axis.y * vert.position.x + inst.y_axis.y * vert.position.y + inst.translation.y,
    );
    let pos = common__globals.view_matrix * vec4<f32>(world, 0.0, 1.0);
    let matrix_ = textureTransforms.texture_matrix;
    let uv = (mat3x3<f32>(matrix_[0].xyz, matrix_[1].xyz, matrix_[2].xyz) * vec3<f32>(vert.position, 1.0)).xy;
    return VertexOutput(pos, uv, inst.mult_color, inst.add_color);
}

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    var color: vec4<f32> = textureSample(texture, texture_sampler, in.uv);
    // Texture is premultiplied by alpha.
    // Unmultiply alpha, apply color transform, remultiply alpha.
    if (color.a > 0.0) {
        color = vec4<f32>(color.rgb / color.a, color.a);
        color = color * in.mult_color + in.add_color;
        color = saturate(color);
        color = vec4<f32>(color.rgb * color.a, color.a);
    }
    return color;
}
