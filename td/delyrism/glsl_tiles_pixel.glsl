// glsl_tiles — mosaic of the activated Lakota symbols behind the particles,
// carved by the graph.
//
// input 0: the tone-mapped particle image (display space)
// input 1: distance-to-graph field (glsl_carve), fraction of the screen height
// input 2: symbol atlas (td/assets/lakota/atlas.png, 4 × 3 cells, white + alpha)
//
// Mosaic: one symbol per lattice cell at  i·v1 + j·v2  (canvas units, one
// canvas = one atlas cell, y up).  The lattice is the densest one in which any
// symbol of the current set fits any cell (td/tools/lakota_prepare.py
// --mosaics).  Cells repeat in 4 × 4 blocks; cell (i mod 4, j mod 4) has a
// Bayer rank r and shows the symbol in slot r (symbol_select.py fills the 16
// slots in proportion to each symbol's activation, spread evenly over the ranks).
//
// Never cut a symbol: for the pixel's covering instance the distance field is
// read at 9 points spread over that instance's whole footprint; the smallest
// distance d sets the opacity of the entire symbol, eased across the space
// between the graph's padding and the screen outline:
//     t = (d − pad) / ((d − pad) + e)     e = distance from the symbol to the nearest screen edge
//     opacity = ease(t)^curve             0 at the padding, 1 at the screen edge
//
// Uniforms (vec4):
//   uCurL (v1x, v1y, v2x, v2y)   uCurW (weight, 0, 0, 0)   uCurS0..S3  slots 0–15 (atlas index, −1 empty)
//   uPrvL / uPrvW / uPrvS0..S3   the mosaic being faded out
//   uTile (tile size px, opacity, drift x px, drift y px)
//   uTileCol (r, g, b, 0)
//   uAtlas (cols, rows, 0, 0)
//   uFade (padding as fraction of the screen height, fade curve exponent, 0, 0)

uniform vec4 uCurL;
uniform vec4 uCurW;
uniform vec4 uCurS0;
uniform vec4 uCurS1;
uniform vec4 uCurS2;
uniform vec4 uCurS3;
uniform vec4 uPrvL;
uniform vec4 uPrvW;
uniform vec4 uPrvS0;
uniform vec4 uPrvS1;
uniform vec4 uPrvS2;
uniform vec4 uPrvS3;
uniform vec4 uTile;
uniform vec4 uTileCol;
uniform vec4 uAtlas;
uniform vec4 uFade;

out vec4 fragColor;

const float FOOT = 0.42;   // footprint half-size sampled for the carve test (canvas units)
const int BAYER[16] = int[16](0, 8, 2, 10,  12, 4, 14, 6,  3, 11, 1, 9,  15, 7, 13, 5);

float atlasAlpha(vec2 q, float idx)
{
	if (idx < -0.5 || abs(q.x) > 0.5 || abs(q.y) > 0.5) return 0.0;
	int k = int(idx + 0.5);
	vec2 cell = vec2(float(k % int(uAtlas.x)), float(k / int(uAtlas.x)));
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

float slotOf(int r, vec4 S0, vec4 S1, vec4 S2, vec4 S3)
{
	vec4 v = (r < 4) ? S0 : (r < 8) ? S1 : (r < 12) ? S2 : S3;
	return v[r % 4];
}

// coverage of one mosaic at canvas point p
float mosaic(vec2 p, vec4 L, float w, vec4 S0, vec4 S1, vec4 S2, vec4 S3, vec2 res)
{
	if (w <= 0.0 || L.x <= 0.0) return 0.0;
	vec2 v1 = L.xy, v2 = L.zw;
	float j0 = floor(p.y / v2.y + 0.5);
	for (int dj = -2; dj <= 2; dj++) {
		float j = j0 + float(dj);
		float i0 = floor((p.x - j * v2.x) / v1.x + 0.5);
		for (int di = -1; di <= 1; di++) {
			float i = i0 + float(di);
			vec2 centre = i * v1 + j * v2;
			int im = int(mod(i, 4.0)), jm = int(mod(j, 4.0));
			float idx = slotOf(BAYER[jm * 4 + im], S0, S1, S2, S3);
			float a = atlasAlpha(p - centre, idx);
			if (a > 0.002)
				return a * visibility(centre, res) * w;
		}
	}
	return 0.0;
}

void main()
{
	vec2 res = uTD2DInfos[0].res.zw;
	vec2 frag = vUV.st * res;
	vec2 p = (frag - 0.5 * res - uTile.zw) / max(1.0, uTile.x);
	float bg = mosaic(p, uCurL, uCurW.x, uCurS0, uCurS1, uCurS2, uCurS3, res)
	         + mosaic(p, uPrvL, uPrvW.x, uPrvS0, uPrvS1, uPrvS2, uPrvS3, res);
	bg = clamp(bg, 0.0, 1.0) * uTile.y;

	vec3 fg = texture(sTD2DInputs[0], vUV.st).rgb;
	float cover = clamp(max(max(fg.r, fg.g), fg.b), 0.0, 1.0);
	vec3 outc = fg + uTileCol.rgb * bg * (1.0 - cover);
	fragColor = TDOutputSwizzle(vec4(outc, 1.0));
}
