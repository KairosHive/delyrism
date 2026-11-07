// glsl_tiles — lay the background mosaic under the particles.
//
// input 0: the tone-mapped particle image (display space)
// input 1: distance-to-graph field (glsl_carve) — kept wired for reference
// input 2: render_tiles — the snowflake mosaic, rendered as instances
//          (tiles_layout.py → glsl_tilefade → geo_tiles), premultiplied, already
//          coloured, faded per whole figure around the graph and crossfaded
//
// The mosaic only shows where the particles leave room: out = fg + bg·(1 − cover).

out vec4 fragColor;

void main()
{
	vec3 fg = texture(sTD2DInputs[0], vUV.st).rgb;
	vec3 bg = texture(sTD2DInputs[2], vUV.st).rgb;
	float cover = clamp(max(max(fg.r, fg.g), fg.b), 0.0, 1.0);
	fragColor = TDOutputSwizzle(vec4(fg + bg * (1.0 - cover), 1.0));
}
