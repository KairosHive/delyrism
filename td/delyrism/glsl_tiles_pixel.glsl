// glsl_tiles — Lakota symbol tiling behind the particles, carved by the graph.
//
// input 0: the tone-mapped particle image (display space)
// input 1: carve field (glsl_carve): padded, eased graph footprint
// input 2: symbol atlas (td/assets/lakota/atlas.png, 4 × 3 cells, white + alpha)
//
// The tiling is drawn procedurally from the lattice found by
// td/tools/lakota_prepare.py: copies at  i·v1 + j·v2 + offset_k  (canvas units,
// one canvas = one atlas cell, y up), copy k drawn with transform t_k.
//
// Never cut a symbol: for the pixel's covering instance the distance-to-graph
// field (glsl_carve) is read at 9 points spread over that instance's whole
// footprint; the smallest distance d sets the opacity of the entire symbol.
// Opacity is eased across the whole space between the graph's padding and the
// screen outline:
//     t = (d − pad) / ((d − pad) + e)     e = distance from the symbol to the nearest screen edge
//     opacity = ease(t)^curve             0 at the padding, 1 at the screen edge
// Each symbol therefore fades as a unit — never sliced by the graph.
//
// Uniforms (vec4):
//   uCurA (cellx, celly, ncopies, t1)   uCurB (v1x, v1y, v2x, v2y)   uCurC (off1x, off1y, weight, 0)
//   uPrvA / uPrvB / uPrvC               same for the symbol being faded out
//   uTile (tile size px, opacity, drift x px, drift y px)
//   uTileCol (r, g, b, 0)
//   uAtlas (cols, rows, 0, 0)
//   uFade (padding as fraction of the screen height, fade curve exponent, 0, 0)

uniform vec4 uCurA;
uniform vec4 uCurB;
uniform vec4 uCurC;
uniform vec4 uPrvA;
uniform vec4 uPrvB;
uniform vec4 uPrvC;
uniform vec4 uTile;
uniform vec4 uTileCol;
uniform vec4 uAtlas;
uniform vec4 uFade;

out vec4 fragColor;

const float FOOT = 0.42;   // footprint half-size sampled for the carve test (canvas units)

vec2 untransform(vec2 q, float t)
{
	if (t > 2.5) return vec2(q.x, -q.y);    // flip y
	if (t > 1.5) return vec2(-q.x, q.y);    // flip x
	if (t > 0.5) return -q;                 // half turn
	return q;
}

float atlasAlpha(vec2 q, vec2 cell)
{
	if (abs(q.x) > 0.5 || abs(q.y) > 0.5) return 0.0;
	vec2 uv = vec2((cell.x + q.x + 0.5) / uAtlas.x, (uAtlas.y - cell.y - 0.5 + q.y) / uAtlas.y);
	return texture(sTD2DInputs[2], uv).a;
}

// screen pixel of a canvas-space point
vec2 toPixel(vec2 c, vec2 res) { return c * uTile.x + 0.5 * res + uTile.zw; }

// distance from the instance's footprint to the graph (fraction of screen height)
float distanceAt(vec2 centre, vec2 res)
{
	float m = 1e9;
	for (int a = -1; a <= 1; a++)
		for (int b = -1; b <= 1; b++) {
			vec2 px = toPixel(centre + FOOT * vec2(a, b), res);
			m = min(m, texture(sTD2DInputs[1], clamp(px / res, 0.0, 1.0)).r);
		}
	return m;
}

// whole-symbol opacity: eased between the graph padding (0) and the screen outline (1)
float visibility(vec2 centre, vec2 res)
{
	float d = distanceAt(centre, res) - uFade.x;
	if (d <= 0.0) return 0.0;
	vec2 px = toPixel(centre, res);
	float e = max(0.0, min(min(px.x, res.x - px.x), min(px.y, res.y - px.y))) / res.y;
	float t = clamp(d / max(1e-4, d + e), 0.0, 1.0);
	t = t * t * (3.0 - 2.0 * t);                              // ease in-out
	return pow(t, max(0.05, uFade.y));
}

// coverage of one symbol tiling at canvas point p
float tiling(vec2 p, vec4 A, vec4 B, vec4 C, vec2 res)
{
	if (C.z <= 0.0 || B.x <= 0.0) return 0.0;
	vec2 v1 = B.xy, v2 = B.zw;
	int ncop = int(A.z + 0.5);
	for (int k = 0; k < 2; k++) {
		if (k >= ncop) break;
		vec2 off = (k == 0) ? vec2(0.0) : C.xy;
		float t = (k == 0) ? 0.0 : A.w;
		vec2 r = p - off;
		float j0 = floor(r.y / v2.y + 0.5);
		for (int dj = -2; dj <= 2; dj++) {
			float j = j0 + float(dj);
			float i0 = floor((r.x - j * v2.x) / v1.x + 0.5);
			for (int di = -1; di <= 1; di++) {
				float i = i0 + float(di);
				vec2 centre = off + i * v1 + j * v2;
				float a = atlasAlpha(untransform(p - centre, t), A.xy);
				if (a > 0.002) {
					float vis = visibility(centre, res);
					return a * vis * C.z;
				}
			}
		}
	}
	return 0.0;
}

void main()
{
	vec2 res = uTD2DInfos[0].res.zw;
	vec2 frag = vUV.st * res;
	vec2 p = (frag - 0.5 * res - uTile.zw) / max(1.0, uTile.x);
	float bg = tiling(p, uCurA, uCurB, uCurC, res) + tiling(p, uPrvA, uPrvB, uPrvC, res);
	bg = clamp(bg, 0.0, 1.0) * uTile.y;

	vec3 fg = texture(sTD2DInputs[0], vUV.st).rgb;
	float cover = clamp(max(max(fg.r, fg.g), fg.b), 0.0, 1.0);
	vec3 outc = fg + uTileCol.rgb * bg * (1.0 - cover);
	fragColor = TDOutputSwizzle(vec4(outc, 1.0));
}
