// glsl_pool — max-pool the particle density into a coarse grid (output 1/8 res).
// Max, not average, so a single sparse particle still marks its cell — the
// distance field (glsl_carve) is computed on this grid.

out vec4 fragColor;

void main()
{
	vec2 inRes = uTD2DInfos[0].res.zw;
	vec2 outRes = uTDOutputInfo.res.zw;
	ivec2 f = ivec2(floor(inRes / outRes));
	ivec2 o = ivec2(floor(vUV.st * outRes)) * f;
	float m = 0.0;
	for (int y = 0; y < f.y; y++)
		for (int x = 0; x < f.x; x++) {
			vec4 d = texelFetch(sTD2DInputs[0], o + ivec2(x, y), 0);
			m = max(m, max(max(d.r, d.g), d.b));
		}
	fragColor = TDOutputSwizzle(vec4(m, m, m, 1.0));
}
