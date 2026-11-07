// A small, physically honest spring-mass layout for the live Δ-graph.
//
// Why not d3-force: its link force and Barnes-Hut repulsion are not
// conservative — the force a pair exerts on each other is not equal and
// opposite — so every tick injects a little net angular momentum, which no
// velocity damping can remove.  Here every pairwise force is symmetric
// (Newton's third law exactly), the only extra forces are central (toward
// the origin) and therefore torque-free, and the ONE non-conservative term
// is linear drag on node velocity:
//
//     a = F/m − c·v          v ← v·e^(−c·dt)  (exact for the drag part)
//
// Consequences: rest-length changes from the query drive the springs
// directly (no tweening), the settling time is set by stiffness k and the
// damping c alone, and any rotation or translation decays as e^(−c·t) like
// everything else.  When kinetic energy drops below a floor the simulation
// sleeps until something wakes it.

export interface SimNode {
  id: string;
  x: number; y: number;
  vx: number; vy: number;
  fixed: boolean;          // dragged
  rel: number;             // query relevance 0..1 (drives the radial target)
}

export interface SimLink {
  a: number; b: number;    // node indices
  rest: number;            // rest length (px)
  k: number;               // stiffness multiplier (1 = base)
}

export interface SimParams {
  stiffness: number;       // base spring stiffness k (1/s²)
  damping: number;         // drag c (1/s)
  repulsion: number;       // pair repulsion q (px³/s²), F = q / r²
  unlinked: number;        // extra repulsion between pairs that share NO edge (px³/s²) — disentangles
  center: number;          // central spring toward the origin (1/s²)
  radial: number;          // query radial spring strength (1/s²)
  radialOn: boolean;       // whether the radial target follows relevance
  minSep: number;          // collision: pairs closer than this (px) are pushed apart …
  collide: number;         // … by collide · (minSep − d)  (1/s²) — keeps related nodes from overlapping
}

/** Query radial target: relevant nodes settle on an inner ring, unrelated ones further out. */
export const RADIAL_INNER = 70, RADIAL_OUTER = 280;

export class SpringSim {
  nodes: SimNode[] = [];
  links: SimLink[] = [];
  params: SimParams = { stiffness: 40, damping: 8, repulsion: 9000, unlinked: 6000, center: 2, radial: 6, radialOn: false, minSep: 44, collide: 160 };
  awake = true;
  private sleepTimer = 0;
  private adjLinks: SimLink[] | null = null;   // links array the adjacency was built from
  private adjN = 0;
  private adj: Uint8Array = new Uint8Array(0);  // n×n, 1 = connected

  /** (Re)build the adjacency bitmap when the links array or node count changed. */
  private adjacency(n: number): Uint8Array {
    if (this.adjLinks === this.links && this.adjN === n && this.adj.length === n * n) return this.adj;
    const adj = new Uint8Array(n * n);
    for (const l of this.links) {
      if (l.a < n && l.b < n) { adj[l.a * n + l.b] = 1; adj[l.b * n + l.a] = 1; }
    }
    this.adj = adj; this.adjLinks = this.links; this.adjN = n;
    return adj;
  }
  private ax: Float64Array = new Float64Array(0);
  private ay: Float64Array = new Float64Array(0);

  wake() { this.awake = true; this.sleepTimer = 0; }

  /** Advance by `dt` seconds (sub-stepped for stability). Returns true if moved. */
  step(dt: number): boolean {
    if (!this.awake) return false;
    const n = this.nodes.length;
    if (n === 0) return false;
    if (this.ax.length !== n) { this.ax = new Float64Array(n); this.ay = new Float64Array(n); }
    const sub = Math.max(1, Math.ceil(dt / (1 / 240)));
    const h = dt / sub;
    const p = this.params;
    const drag = Math.exp(-p.damping * h);
    let energy = 0;

    for (let s = 0; s < sub; s++) {
      const ax = this.ax, ay = this.ay;
      ax.fill(0); ay.fill(0);

      // springs (symmetric)
      for (const l of this.links) {
        const A = this.nodes[l.a], B = this.nodes[l.b];
        let dx = B.x - A.x, dy = B.y - A.y;
        let d = Math.hypot(dx, dy);
        if (d < 1e-6) { dx = 1e-3; dy = 0; d = 1e-3; }
        const f = p.stiffness * l.k * (d - l.rest) / d;   // along the link
        ax[l.a] += f * dx; ay[l.a] += f * dy;
        ax[l.b] -= f * dx; ay[l.b] -= f * dy;
      }
      // repulsion (symmetric, all pairs — N is small).  Pairs that share no
      // edge get an extra, longer-ranged push (softened 1/r) so unrelated
      // clusters slide apart instead of resting on top of each other.
      const q = p.repulsion, qu = p.unlinked;
      const adj = qu > 0 ? this.adjacency(n) : null;
      for (let i = 0; i < n; i++) {
        const A = this.nodes[i];
        for (let j = i + 1; j < n; j++) {
          const B = this.nodes[j];
          let dx = B.x - A.x, dy = B.y - A.y;
          let d2 = dx * dx + dy * dy;
          if (d2 < 1e-6) { dx = (i % 2 ? 1 : -1) * 1e-2; dy = 1e-2; d2 = 2e-4; }
          const d = Math.sqrt(d2);
          let f = q / (d2 + 400) / d;                    // softened 1/r² (no blow-up at contact)
          if (adj && !adj[i * n + j]) f += qu / (d2 + 2500);   // unlinked: softened 1/r, acts further out
          if (d < p.minSep) f += p.collide * (p.minSep - d) / d; // collision: firm push below minSep
          ax[i] -= f * dx; ay[i] -= f * dy;
          ax[j] += f * dx; ay[j] += f * dy;
        }
      }
      // central forces (torque-free about the origin)
      for (let i = 0; i < n; i++) {
        const N = this.nodes[i];
        ax[i] -= p.center * N.x; ay[i] -= p.center * N.y;
        if (p.radialOn) {
          const r = Math.hypot(N.x, N.y) || 1e-6;
          const target = RADIAL_INNER + (1 - N.rel) * (RADIAL_OUTER - RADIAL_INNER);
          const f = p.radial * (target - r) / r;
          ax[i] += f * N.x; ay[i] += f * N.y;
        }
      }
      // integrate (semi-implicit Euler + exact linear drag)
      energy = 0;
      for (let i = 0; i < n; i++) {
        const N = this.nodes[i];
        if (N.fixed) { N.vx = 0; N.vy = 0; continue; }
        N.vx = (N.vx + ax[i] * h) * drag;
        N.vy = (N.vy + ay[i] * h) * drag;
        N.x += N.vx * h;
        N.y += N.vy * h;
        energy += N.vx * N.vx + N.vy * N.vy;
      }
    }
    // sleep when quiet for ~0.5 s (mean speed < 0.5 px/s)
    if (energy / Math.max(1, n) < 0.25) {
      this.sleepTimer += dt;
      if (this.sleepTimer > 0.5) { this.awake = false; for (const N of this.nodes) { N.vx = 0; N.vy = 0; } }
    } else {
      this.sleepTimer = 0;
    }
    return true;
  }
}
