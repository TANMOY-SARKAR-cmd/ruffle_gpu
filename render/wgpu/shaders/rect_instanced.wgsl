/// Instanced rendering shader for solid-colour rectangles.
///
/// * Vertex slot 0 (step_mode = Vertex):   unit-quad position (`vec2<f32>`)
/// * Vertex slot 1 (step_mode = Instance): per-rect 2D affine transform + premultiplied colour
///
/// The per-instance data encodes the affine transform that maps the unit quad
/// corners [0,1]×[0,1] to world space:
///   world.x = ab.x * pos.x + cd.x * pos.y + txty.x
///   world.y = ab.y * pos.x + cd.y * pos.y + txty.y
///
/// The view_matrix from globals (group 0) is then applied to produce clip
/// coordinates.  No per-object transform uniform (group 1) is needed.
///
/// NOTE: The `common.wgsl` source is prepended to this before compilation.

struct VertexInput {
    @location(0) position: vec2<f32>,
};

struct InstanceInput {
    @location(1) ab:    vec2<f32>,   // world-matrix: [a, b]  (x-axis scale/shear)
    @location(2) cd:    vec2<f32>,   // world-matrix: [c, d]  (y-axis shear/scale)
    @location(3) txty:  vec2<f32>,   // world-space translation
    @location(4) color: vec4<f32>,   // premultiplied RGBA in [0, 1]
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn main_vertex(vert: VertexInput, inst: InstanceInput) -> VertexOutput {
    let world = vec2<f32>(
        inst.ab.x * vert.position.x + inst.cd.x * vert.position.y + inst.txty.x,
        inst.ab.y * vert.position.x + inst.cd.y * vert.position.y + inst.txty.y,
    );
    let pos = common__globals.view_matrix * vec4<f32>(world, 0.0, 1.0);
    return VertexOutput(pos, inst.color);
}

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
