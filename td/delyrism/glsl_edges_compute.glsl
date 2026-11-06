// glsl_edges — edge-following force for the delyrism particle loop.
//
// Runs inside particlePOP's feedback loop (particle_sim → glsl_edges → null_sim).
// Each particle carries Cur (uid of the node it left) and Tgt (uid it heads to,
// -1 = none yet / drifting).  A newborn picks an edge of its node with
// probability ∝ edge strength; two forces, both ∝ the live strength of that
// edge, carry it along: attraction toward Tgt and a lateral spring onto the
// Cur→Tgt line.  On arrival it hops on (another strength-weighted edge) with
// probability uFlow.x, otherwise its lifespan is cut short so it fades out.
// If its edge disappears from the graph it drifts and fades.
//
// Uniforms (auto-declared from the Vectors / Samplers pages):
//   sGraph  graph texture (see graph_tex.py)
//   uSim    (dt, frame, alpha, pointsize)
//   uForce  (attraction, lateral, drag, turbulence)
//   uFlow   (hop probability, arrival radius, turbulence scale, noise time)
//   uLook   (size jitter, fade-in seconds, fade-out fraction of life, 0)
//   uSal    (influence radius, radius growth with node size, falloff distance, reference)
//   uSal2   (near scale, far scale, alpha coupling, 0)
//
// Saliency: a particle is larger inside the sphere of influence of its source
// node (Src = the emitting node, or with reference = 1 the node it last left)
// and shrinks to the far scale over the falloff distance beyond it.  The
// sphere's radius grows with the node's size (Σ|Δ|) and shrinks as the node
// fades out of the graph.
//
// Query: alpha is also multiplied by the emitting node's query opacity
// (graph texture row 1 .w, see graph_tex.py) — particles from nodes the query
// doesn't touch dim the way those nodes dim in the browser.

#define ROW_NODE 0
#define ROW_ADJ 1
#define ROW_CSR 2
#define ROW_HEAD 3
#define MAX_ADJ 128

uint hash(uint x) {
	x ^= x >> 16; x *= 0x7feb352dU;
	x ^= x >> 15; x *= 0x846ca68bU;
	x ^= x >> 16; return x;
}
float rnd(inout uint s) { s = hash(s); return float(s) * (1.0 / 4294967296.0); }

vec4 G(int x, int y) { return texelFetch(sGraph, ivec2(x, y), 0); }

int gNumUid;

bool nodeLive(int u) { return u >= 0 && u < gNumUid && G(u, ROW_NODE).z > 0.0; }

// next node from u, chosen ∝ edge strength (-1 if u has no edge)
int chooseEdge(int u, inout uint s) {
	if (u < 0 || u >= gNumUid) return -1;
	vec4 adj = G(u, ROW_ADJ);
	int start = int(adj.x), cnt = min(int(adj.y), MAX_ADJ);
	float tot = adj.z;
	if (cnt <= 0 || tot <= 0.0) return -1;
	float r = rnd(s) * tot, acc = 0.0;
	for (int i = 0; i < cnt; i++) {
		vec4 e = G(start + i, ROW_CSR);
		acc += e.y;
		if (r < acc) return int(e.x + 0.5);
	}
	return int(G(start + cnt - 1, ROW_CSR).x + 0.5);
}

// live strength of the u–t edge (0 if it no longer exists)
float legStrength(int u, int t) {
	if (u < 0 || u >= gNumUid || t < 0) return 0.0;
	vec4 adj = G(u, ROW_ADJ);
	int start = int(adj.x), cnt = min(int(adj.y), MAX_ADJ);
	for (int i = 0; i < cnt; i++) {
		vec4 e = G(start + i, ROW_CSR);
		if (int(e.x + 0.5) == t) return e.y;
	}
	return 0.0;
}

void main() {
	const uint id = TDIndex();
	if (id >= TDNumElements())
		return;

	vec3 p = TDIn_P();
	vec3 v = TDIn_PartVel();
	float age = TDIn_PartAge();
	float life = TDIn_PartLifeSpan();
	float mass = max(1e-4, TDIn_PartMass());
	vec4 col = TDIn_Color();
	int u = int(floor(TDIn_Cur() + 0.5));
	int t = int(floor(TDIn_Tgt() + 0.5));
	uint pid = TDIn_PartId();

	float dt = uSim.x;
	uint s = hash(pid * 2654435761U ^ hash(uint(uSim.y) + 0x9e3779b9U));
	gNumUid = int(G(0, ROW_HEAD).x + 0.5);

	// newborn: pick the first edge out of its source node
	if (t < 0 && age <= 1.5 * dt + 1e-4)
		t = chooseEdge(u, s);

	vec2 pos = p.xy;
	vec2 acc = vec2(0.0);
	float w = legStrength(u, t);
	bool has = t >= 0 && w > 0.0 && nodeLive(t);
	float d = 1e9;
	if (has) {
		vec2 T = G(t, ROW_NODE).xy;
		vec2 dv = T - pos;
		d = length(dv) + 1e-6;
		acc += uForce.x * w * dv / d;                       // attraction toward the target node
		if (nodeLive(u) || (u >= 0 && u < gNumUid)) {
			vec2 S = G(u, ROW_NODE).xy;
			vec2 e = T - S;
			float k = clamp(dot(pos - S, e) / (dot(e, e) + 1e-9), 0.0, 1.0);
			acc += uForce.y * w * (S + k * e - pos);         // lateral spring onto the edge
		}
	} else if (t >= 0 && age > 1.5 * dt) {
		t = -1;                                              // edge or node gone: drift + fade
		life = min(life, age + 0.6);
	}

	// smooth flow-field turbulence: curl of simplex noise
	if (uForce.w > 0.0) {
		float sc = max(1e-3, uFlow.z), h = 0.01;
		vec3 q = vec3(pos * sc, uFlow.w);
		float nx1 = TDSimplexNoise(q + vec3(0.0, h, 0.0)), nx2 = TDSimplexNoise(q - vec3(0.0, h, 0.0));
		float ny1 = TDSimplexNoise(q + vec3(h, 0.0, 0.0)), ny2 = TDSimplexNoise(q - vec3(h, 0.0, 0.0));
		acc += uForce.w * vec2(nx1 - nx2, -(ny1 - ny2)) / (2.0 * h * sc);
	}
	acc -= uForce.z * v.xy;                                  // linear drag

	// arrival: hop on along another strength-weighted edge, or fade out
	if (has && d < uFlow.y) {
		if (rnd(s) < uFlow.x) {
			int nt = chooseEdge(t, s);
			u = t;
			t = nt;
		} else {
			t = -1;
			life = min(life, age + 0.35);
		}
	}

	// saliency: proximity to the source node's sphere of influence
	int ref = (uSal.w > 0.5) ? u : int(floor(TDIn_Src() + 0.5));
	float near01 = 0.0;                                       // 1 inside the sphere, 0 far away
	if (ref >= 0 && ref < gNumUid) {
		vec4 nd = G(ref, ROW_NODE);
		float radius = uSal.x * (1.0 + uSal.y * nd.w) * clamp(nd.z, 0.0, 1.0);
		float dist = length(pos - nd.xy);
		float t = clamp((dist - radius) / max(1e-4, uSal.z), 0.0, 1.0);
		near01 = 1.0 - t * t * (3.0 - 2.0 * t);                  // smoothstep falloff
	}
	float salScale = mix(uSal2.y, uSal2.x, near01);
	float salAlpha = mix(1.0 - uSal2.z, 1.0, near01);

	// query: opacity of the emitting node, as the browser dims nodes off-query
	int srcUid = int(floor(TDIn_Src() + 0.5));
	float queryOpacity = (srcUid >= 0 && srcUid < textureSize(sGraph, 0).x) ? G(srcUid, ROW_ADJ).w : 1.0;

	float fadeIn = max(1e-3, uLook.y), fadeOut = max(1e-3, uLook.z * life);
	float alpha = clamp(age / fadeIn, 0.0, 1.0) * clamp((life - age) / fadeOut, 0.0, 1.0)
	            * uSim.z * salAlpha * queryOpacity;
	float r01 = float(hash(pid * 747796405U) & 0xffffU) / 65535.0;
	float jitter = max(0.0, 1.0 + uLook.x * (2.0 * r01 - 1.0)) * salScale;

	PartForce[id] = vec3(acc * mass, 0.0);
	PartLifeSpan[id] = life;
	Cur[id] = float(u);
	Tgt[id] = float(t);
	Color[id] = vec4(col.rgb, alpha);
	PointScale[id] = uSim.w * jitter;
}
