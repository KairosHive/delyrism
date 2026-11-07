// glsl_tilefade — per-figure drift, colour and fade for the background mosaic.
//
// One point per figure (from tiles_layout.py via choptoPOP): P = position in
// pixels from the bottom-left, Sy = size in pixels, Layer = 0 current mosaic /
// 1 the one fading out.  Writes P (drifted) and Color (straight alpha; geo_tiles premultiplies).
//
// Never cut a figure: the distance-to-graph field (glsl_carve, fraction of the
// screen height) is read at 9 points spread over the figure's footprint; the
// smallest distance d sets the opacity of the whole figure, eased across the
// space between the graph's padding and the screen outline:
//     t = (d − pad) / ((d − pad) + e)     e = distance from the figure to the nearest screen edge
//     opacity = ease(t)^curve
//
// Uniforms (auto-declared): sCarve (distance field)
//   uTile (tile size px, opacity, drift x px, drift y px)
//   uRes  (width px, height px, current-mosaic weight, previous-mosaic weight)
//   uFade (padding as fraction of the height, fade curve, 0, 0)
//   uCol  (r, g, b, 0)
//   uPer  repeat period of the pattern: p1 (x, y), p2 (x, y) in px — the drift
//         is wrapped in these lattice coordinates so the pattern never runs out

void main()
{
	const uint id = TDIndex();
	if (id >= TDNumElements())
		return;
	vec3 p = TDIn_P();
	vec2 res = uRes.xy;
	mat2 per = mat2(uPer.xy, uPer.zw);
	if (abs(determinant(per)) < 1.0)
		per = mat2(vec2(max(1.0, uTile.x), 0.0), vec2(0.0, max(1.0, uTile.x)));
	vec2 px = p.xy + per * fract(inverse(per) * uTile.zw);
	float s = abs(TDIn_Sy());

	float d = 1e9;
	for (int a = -1; a <= 1; a++)
		for (int b = -1; b <= 1; b++) {
			vec2 q = px + 0.42 * s * vec2(a, b);
			d = min(d, texture(sCarve, clamp(q / res, 0.0, 1.0)).r);
		}
	d -= uFade.x;
	float vis = 0.0;
	if (d > 0.0) {
		float e = max(0.0, min(min(px.x, res.x - px.x), min(px.y, res.y - px.y))) / res.y;
		float t = clamp(d / max(1e-4, d + e), 0.0, 1.0);
		t = t * t * (3.0 - 2.0 * t);
		vis = pow(t, max(0.05, uFade.y));
	}
	float w = (TDIn_Layer() < 0.5) ? uRes.z : uRes.w;
	float alpha = clamp(vis * w * uTile.y, 0.0, 1.0);

	P[id] = vec3(px, 0.0);
	Color[id] = vec4(uCol.rgb, alpha);              // TD premultiplies in the shader (instancecolorpremult)
}
