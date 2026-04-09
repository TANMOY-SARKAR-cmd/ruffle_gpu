/// Shader for batched pre-transformed solid color fills.
///
/// Vertex positions are already in world space (the world_matrix has been
/// applied on the CPU). Only the view_matrix from the global uniforms
/// (group 0) is applied at draw time. Colors are stored as pre-computed
/// premultiplied RGBA values in the vertex buffer, so no per-object
/// transform uniform (group 1) is needed.
///
/// NOTE: The `common.wgsl` source is prepended to this before compilation.

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn main_vertex(in: VertexInput) -> VertexOutput {
    let pos = common__globals.view_matrix * vec4<f32>(in.position, 0.0, 1.0);
    return VertexOutput(pos, in.color);
}

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
