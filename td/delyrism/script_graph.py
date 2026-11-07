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

# Particle palette: symbol i (/delyrism/node/sym) takes colour i mod 10, in
# place of the web app's palette (/delyrism/node/r|g|b).  Controls › Look ›
# "TD Palette" switches back to the web colours.
PALETTE_HEX = ['#D8961F', '#C56A3D', '#4D88D3', '#629843', '#1397AE',
               '#BC6A7F', '#C850B5', '#7B70F4', '#AB2966', '#FFBDE7']
PALETTE = np.array([[int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)] for h in PALETTE_HEX], dtype=np.float32)


def _use_palette():
	c = op.Controls if hasattr(op, 'Controls') else None
	return not (c is not None and hasattr(c.par, 'Tdpalette') and not c.par.Tdpalette.eval())
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
	cols = {}
	for src, name in attrs:
		vals = np.zeros(max(1, n), dtype=np.float32)
		a = frame.get(f'{prefix}/{src}', ())
		if n and len(a) == n:
			vals[:] = a
		cols[name] = vals
	if group == 'nodes' and _use_palette():
		rgb = PALETTE[np.mod(cols['sym'].astype(np.int64), len(PALETTE))]
		cols['r'], cols['g'], cols['b'] = rgb[:, 0].copy(), rgb[:, 1].copy(), rgb[:, 2].copy()
	for name, vals in cols.items():
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
