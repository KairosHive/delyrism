// glsl_carve — distance from the graph, eased over time.
//
// input 0: particle density at 1/8 resolution (blur_carve, preshrunk)
// input 1: previous result (feedback_carve)
// out.r  : distance to the nearest graph pixel (density > threshold), as a
//          fraction of the screen height, capped at uCarve.w; eased toward the
//          current value with time constant uCarve.z so background symbols
//          fade smoothly as the graph moves.
//
// Brute-force search within the cap radius — cheap at this resolution
// (≈160×90 cells, radius ≤ 48 cells).
//
// uCarve = (density threshold, dt, ease seconds, max distance as fraction of height)

uniform vec4 uCarve;

out vec4 fragColor;

void main()
{
	vec2 res = uTD2DInfos[0].res.zw;
	ivec2 size = ivec2(res);
	ivec2 c = ivec2(vUV.st * res);
	float maxd = max(1.0, uCarve.w * res.y);
	int R = int(min(48.0, ceil(maxd)));
	float best = maxd;
	for (int dy = -R; dy <= R; dy++) {
		int y = c.y + dy;
		if (y < 0 || y >= size.y) continue;
		for (int dx = -R; dx <= R; dx++) {
			int x = c.x + dx;
			if (x < 0 || x >= size.x) continue;
			float r = length(vec2(dx, dy));
			if (r >= best) continue;
			vec4 d = texelFetch(sTD2DInputs[0], ivec2(x, y), 0);
			if (max(max(d.r, d.g), d.b) > uCarve.x) best = r;
		}
	}
	float dist = best / res.y;
	float prev = texture(sTD2DInputs[1], vUV.st).r;
	float k = 1.0 - exp(-uCarve.y / max(1e-3, uCarve.z));
	float v = mix(prev, dist, k);
	fragColor = TDOutputSwizzle(vec4(v, v, v, 1.0));
}
