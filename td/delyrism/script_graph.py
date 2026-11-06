# Script CHOP callbacks shared by script_frame / script_nodes / script_edges.
# Reads the last complete frame published by osc_callbacks.py and lays it out
# as one sample per node (or edge), one channel per attribute — the shape
# instancing and particle emitters want.  The group is chosen from the
# operator's name.

import numpy as np

# OSC attribute → CHOP channel name (x/y → tx/ty so instancing picks them up by default)
GROUPS = {
	'nodes': ('node', [('x', 'tx'), ('y', 'ty'), ('vx', 'vx'), ('vy', 'vy'), ('size', 'size'),
	                   ('rel', 'rel'), ('pull', 'pull'), ('r', 'r'), ('g', 'g'), ('b', 'b'),
	                   ('sym', 'sym'), ('alive', 'alive'), ('fade', 'fade'), ('uid', 'uid')]),
	'edges': ('edge', [('x0', 'x0'), ('y0', 'y0'), ('x1', 'x1'), ('y1', 'y1'), ('weight', 'weight'),
	                   ('delta', 'delta'), ('sign', 'sign'), ('overlap', 'overlap'), ('len', 'len'),
	                   ('alive', 'alive'), ('fade', 'fade'), ('src', 'src'), ('dst', 'dst')]),
}
FRAME = ['seq', 'time', 'nnodes', 'nedges', 'graphseq', 'queryactive', 'awake']
BOUNDS = ['xmin', 'ymin', 'xmax', 'ymax']


def onCook(scriptOp):
	scriptOp.clear()
	frame = parent().fetch('frame', None, search=False) or {}
	if scriptOp.name.endswith('frame'):
		_cook_frame(scriptOp, frame)
		return
	group = 'nodes' if scriptOp.name.endswith('nodes') else 'edges'
	prefix, attrs = GROUPS[group]
	n = len(frame.get(f'{prefix}/{attrs[0][0]}', ()))
	# a CHOP needs at least one sample: an empty graph is one invisible (fade 0) row
	scriptOp.numSamples = max(1, n)
	for src, name in attrs:
		vals = np.zeros(max(1, n), dtype=np.float32)
		a = frame.get(f'{prefix}/{src}', ())
		if n and len(a) == n:
			vals[:] = a
		scriptOp.appendChan(name).vals = vals


def _cook_frame(scriptOp, frame):
	scriptOp.numSamples = 1
	head = frame.get('frame', ())
	for i, name in enumerate(FRAME):
		scriptOp.appendChan(name)[0] = float(head[i]) if i < len(head) else 0.0
	b = frame.get('bounds', ())
	for i, name in enumerate(BOUNDS):
		scriptOp.appendChan(name)[0] = float(b[i]) if i < len(b) else 0.0


def onGetCookLevel(scriptOp):
	# cooked explicitly by osc_callbacks.py when a full frame lands
	return CookLevel.AUTOMATIC


def onSetupParameters(scriptOp):
	return


def onPulse(par):
	return
