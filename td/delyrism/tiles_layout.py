# Script CHOP: the background mosaic as instances (one sample per figure).
#
# Reads the selection published by script_symbol ('mosaic' = {cur, prev, mix}),
# composes each mosaic as a symmetric snowflake wallpaper tile
# (mosaic_compose.py, seeded by the mosaic + Tilevariation, cached) and repeats
# the tile across the screen.  The tile has the symmetry of the square with
# mirror edges, so plain translated copies join seamlessly.
#
# Channels (pixels from the bottom-left, before drift):
#   tx ty  position · rz rotation (deg) · sx sy size (sx < 0 = mirrored)
#   sym    atlas index · layer 0 = current mosaic, 1 = the one fading out
# glsl_tilefade adds the drift and the whole-figure fade around the graph.

import numpy as np

C = mod('text_mosaic_compose')
_cache = {'infos': None, 'layouts': {}, 'key': None, 'out': None}
CHANS = ['tx', 'ty', 'rz', 'sx', 'sy', 'sym', 'layer']


def _infos():
	if _cache['infos'] is None:
		a = op('movie_atlas').numpyArray(delayed=False)[::-1, :, 3]      # row 0 at the top
		cols, rows = 4, 3
		ch, cw = a.shape[0] // rows, a.shape[1] // cols
		_cache['infos'] = {k: C.symbol_info(a[(k // cols) * ch:(k // cols + 1) * ch, (k % cols) * cw:(k % cols + 1) * cw])
						   for k in range(cols * rows - 1)}
	return _cache['infos']


def _layout(m, variation, density):
	key = (m['sig'], variation, round(density, 3))
	if key not in _cache['layouts']:
		seed = (hash(m['sig']) ^ (variation * 7919)) & 0x7fffffff
		res = C.compose(_infos(), m['symbols'], m['shares'], seed=seed, density=density)
		_cache['layouts'][key] = np.array([(k, x, y, s, phi, -1.0 if mir else 1.0)
										   for k, x, y, s, phi, mir in res['items']], np.float32).reshape(-1, 6)
		if len(_cache['layouts']) > 24:
			_cache['layouts'].pop(next(iter(_cache['layouts'])))
	return _cache['layouts'][key]


def _expand(L, layer, W, H, T, corner):
	"""Repeat tile items over the screen plus one tile of margin (for the drift)."""
	if not len(L):
		return np.zeros((0, len(CHANS)), np.float32)
	o = T / 2.0 if corner else 0.0                     # tile corner (or centre) at the screen centre
	i0, i1 = int(np.floor((-W / 2 - o) / T)) - 2, int(np.ceil((W / 2 - o) / T)) + 1
	j0, j1 = int(np.floor((-H / 2 - o) / T)) - 2, int(np.ceil((H / 2 - o) / T)) + 1
	out = []
	for i in range(i0, i1 + 1):
		for j in range(j0, j1 + 1):
			cx, cy = W / 2 + o + i * T, H / 2 + o + j * T
			blk = np.empty((len(L), len(CHANS)), np.float32)
			blk[:, 0] = cx + L[:, 1] * T
			blk[:, 1] = cy + L[:, 2] * T
			blk[:, 2] = L[:, 4]
			blk[:, 3] = L[:, 3] * T * L[:, 5]
			blk[:, 4] = L[:, 3] * T
			blk[:, 5] = L[:, 0]
			blk[:, 6] = layer
			out.append(blk)
	return np.concatenate(out)


def onCook(scriptOp):
	P = op.Controls.par
	r = op('render_particles')
	W, H = float(r.width), float(r.height)
	T = H / max(0.1, float(P.Tilecount))
	sel = op('script_symbol').fetch('mosaic', None, search=False) or {}
	variation, density = int(P.Tilevariation), float(P.Tiledensity)
	corner = P.Tileorigin.eval() == 'corner'
	key = (tuple(m['sig'] if m else None for m in (sel.get('cur'), sel.get('prev'))),
		   W, H, round(T, 2), variation, round(density, 3), corner)
	if key != _cache['key']:
		parts = [_expand(_layout(m, variation, density), layer, W, H, T, corner)
				 for layer, m in ((0, sel.get('cur')), (1, sel.get('prev'))) if m]
		data = np.concatenate(parts) if parts else np.zeros((0, len(CHANS)), np.float32)
		if not len(data):
			data = np.array([[-1e4, -1e4, 0, 0, 0, 0, 0]], np.float32)   # one invisible instance
		_cache['out'], _cache['key'] = np.ascontiguousarray(data.T), key
	scriptOp.clear()
	out = _cache['out']
	scriptOp.numSamples = out.shape[1]
	for name, row in zip(CHANS, out):
		scriptOp.appendChan(name).vals = row


def onGetCookLevel(scriptOp):
	return CookLevel.ALWAYS


def onSetupParameters(scriptOp):
	return


def onPulse(par):
	return
