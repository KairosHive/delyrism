"use client";
// Bridge component that turns a regular prop (`fwdRef`) into the actual
// `ref={}` on the underlying class component.
//
// Why: `next/dynamic(...)` returns a LoadableComponent that is itself a
// plain function component — it does NOT forward refs to whatever it
// eventually renders.  So `<ForceGraph2D ref={fgRef} />` where ForceGraph2D
// is dynamic-loaded silently drops the ref, leaving fgRef.current null and
// breaking every imperative call (tuneForces, zoomToFit, centerAt).
//
// Putting the ref on a *prop* instead of using the `ref` attribute sidesteps
// React's ref-forwarding rules entirely — the prop reaches us as plain data,
// we then bind it directly to the inner class component, which accepts refs
// natively because class components always do.
import ForceGraph2D from "react-force-graph-2d";
import * as React from "react";

interface Props {
  fwdRef?: React.MutableRefObject<any> | ((node: any) => void);
  [key: string]: any;
}

export default function ForceGraph2DInner({ fwdRef, ...rest }: Props) {
  return <ForceGraph2D ref={fwdRef as any} {...rest} />;
}
