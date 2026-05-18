// d3-force tuning for the network panels — implemented WITHOUT an external
// d3-force import so this works whether or not `npm install` was re-run after
// adding the dep to package.json.
//
// react-force-graph-2d ships these default forces under the hood:
//   * link    — pulls connected nodes together
//   * charge  — repels every pair
//   * center  — recenters the simulation barycenter on (0,0)
//
// We override `charge.strength` and `link.distance` to taste, and bolt on
// hand-rolled forceX / forceY (a single-axis pull toward x=0 or y=0 each
// tick) so disconnected components don't drift apart.

interface SimNode {
  x: number; y: number;
  vx: number; vy: number;
}
interface D3Force {
  (alpha: number): void;
  initialize?: (nodes: SimNode[]) => void;
}

function makeRadialForce(strength: number): D3Force {
  let nodes: SimNode[] = [];
  const force: D3Force = (alpha) => {
    const s = strength * alpha;
    for (const n of nodes) {
      n.vx -= n.x * s;
      n.vy -= n.y * s;
    }
  };
  force.initialize = (n) => { nodes = n; };
  return force;
}

export function tuneForces(fg: any, opts: { compact?: boolean; debug?: boolean } = {}) {
  if (!fg) {
    if (opts.debug) console.warn("[delyrism] tuneForces: fg ref is null");
    return;
  }
  const compact = !!opts.compact;
  try {
    const charge = fg.d3Force?.("charge");
    const link = fg.d3Force?.("link");
    if (opts.debug) {
      console.log("[delyrism] tuneForces:", {
        compact,
        d3ForceExists: typeof fg.d3Force === "function",
        chargeExists: !!charge,
        linkExists: !!link,
        zoomFn: typeof fg.zoomToFit,
        centerAtFn: typeof fg.centerAt,
      });
    }
    if (charge) charge.strength(compact ? -650 : -220);
    if (link) link.distance(compact ? 140 : 80).strength(0.5);

    fg.d3Force?.("radial", makeRadialForce(compact ? 0.01 : 0.03));
    fg.d3ReheatSimulation?.();
  } catch (err) {
    console.warn("[delyrism] tuneForces failed:", err);
  }
}

/** Recenter the camera then zoom-to-fit.  zoomToFit alone only scales — it
 *  doesn't pan, so when the simulation's barycenter drifts away from (0,0)
 *  the graph ends up wedged in a corner of the canvas. */
export function recenterAndFit(fg: any, padding: number, duration = 400) {
  if (!fg) return;
  try {
    fg.centerAt?.(0, 0, duration);
    fg.zoomToFit?.(duration, padding);
  } catch {
    fg?.zoomToFit?.(duration, padding);
  }
}
