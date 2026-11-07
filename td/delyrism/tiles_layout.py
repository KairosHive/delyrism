# Script CHOP: the background mosaic as instances (one sample per figure).
#
# Reads the selection published by script_symbol ('mosaic' = {cur, prev, mix})
# and lays out each mosaic in one of two modes (Controls › Tiling Mode):
#
#   lattice   — every symbol upright on one regular lattice, the densest one in
#               which any symbol of the set fits any cell (td/assets/lakota/
#               mosaics.tsv, td/tools/lakota_prepare.py --mosaics).  Cells repeat
#               in 4 × 4 blocks; the shares become cell counts (largest
#               remainder, ≥ 1 each), interleaved by smooth weighted round-robin
#               and laid onto the 4 × 4 Bayer ranking so each symbol is spread
#               evenly.  Size: Lattice Symbols per Screen Height, Lattice Spacing.
#   snowflake — one point-symmetric snowflake (D_n) grown around the screen
#               centre out to the corners (mosaic_compose.py, seeded by the
#               mosaic + Tilevariation, cached): arms, branches and
#               sub-branches, every symbol turned with its path.  Controls:
#               Arms, Symbol Size, Path Gap, Branching.  No drift (it would
#               move the centre off the graph).
#
# Channels (pixels from the bottom-left, before drift):
#   tx ty  position · rz rotation (deg) · sx sy size (sx < 0 = mirrored)
#   sym    atlas index · layer 0 = current mosaic, 1 = the one fading out
# The repeat period of the pattern (p1x p1y p2x p2y, px; zeros = no drift) is
# stored as 'period' so glsl_tilefade can wrap the drift; it also does the
# whole-figure fade.

import math
import os
import numpy as np

C = mod('text_mosaic_compose')
_cache = {'infos': None, 'layouts': {}, 'key': None, 'out': None, 'lattices': None}
CHANS = ['tx', 'ty', 'rz', 'sx', 'sy', 'sym', 'layer']
SLOTS = 16
BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]       # rank of cell (i mod 4, j mod 4) at [j*4 + i]


# ------------------------------------------------------------------ snowflake
def _infos():
	if _cache['infos'] is None:
		a = op('movie_atlas').numpyArray(delayed=False)[::-1, :, 3]      # row 0 at the top
		cols, rows = 4, 3
		ch, cw = a.shape[0] // rows, a.shape[1] // cols
		_cache['infos'] = {k: C.symbol_info(a[(k // cols) * ch:(k // cols + 1) * ch, (k % cols) * cw:(k % cols + 1) * cw])
						   for k in range(cols * rows - 1)}
	return _cache['infos']


def _snowflake(m, variation, aspect, opts):
	key = ('snow', m['sig'], variation, round(aspect, 3), opts)
	if key not in _cache['layouts']:
		seed = (hash(m['sig']) ^ (variation * 7919)) & 0x7fffffff
		arms, size, gap, branching = opts
		res = C.compose(_infos(), m['symbols'], m['shares'], seed=seed, aspect=aspect,
						arms=arms, size=size, gap=gap, branching=branching)
		_cache['layouts'][key] = np.array([(k, x, y, s, phi, -1.0 if mir else 1.0)
										   for k, x, y, s, phi, mir in res['items']], np.float32).reshape(-1, 6)
		if len(_cache['layouts']) > 24:
			_cache['layouts'].pop(next(iter(_cache['layouts'])))
	return _cache['layouts'][key]


def _place_flake(L, layer, W, H):
	"""Snowflake items (screen-height units, centre origin) → pixels."""
	out = np.empty((len(L), len(CHANS)), np.float32)
	if not len(L):
		return out
	out[:, 0] = W / 2 + L[:, 1] * H
	out[:, 1] = H / 2 + L[:, 2] * H
	out[:, 2] = L[:, 4]
	out[:, 3] = L[:, 3] * H * L[:, 5]
	out[:, 4] = L[:, 3] * H
	out[:, 5] = L[:, 0]
	out[:, 6] = layer
	return out


# ------------------------------------------------------------------ lattice
def _lattices():
	if _cache['lattices'] is None:
		path = os.path.join(project.folder, 'assets', 'lakota', 'mosaics.tsv')
		lat = {}
		with open(path, encoding='utf-8') as f:
			next(f)
			for line in f:
				p = line.rstrip('\n').split('\t')
				if len(p) == 5:
					lat[p[0]] = [float(x) for x in p[1:]]
		_cache['lattices'] = lat
	return _cache['lattices']


def _counts(shares):
	"""Largest-remainder cell counts over SLOTS, at least one per symbol."""
	n = min(len(shares), SLOTS)
	raw = np.asarray(shares[:n], float) * (SLOTS - n)
	c = np.floor(raw).astype(int) + 1
	for k in np.argsort(-(raw - np.floor(raw)))[: SLOTS - int(c.sum())]:
		c[k] += 1
	return c


def _sequence(counts):
	"""Smooth weighted round-robin: each symbol's turns spread evenly over the 16 ranks."""
	cw = np.zeros(len(counts))
	seq = []
	for _ in range(SLOTS):
		cw += counts
		k = int(np.argmax(cw))
		cw[k] -= SLOTS
		seq.append(k)
	return seq


def _lattice_vectors(m):
	key = '+'.join(sorted(m['names']))
	lat = _lattices().get(key)
	if lat is None:                                     # unknown set: the loosest lattice of its members
		cand = [_lattices().get(n) for n in m['names']]
		cand = [c for c in cand if c] or [[1.0, 0.0, 0.0, -1.0]]
		lat = max(cand, key=lambda v: abs(v[0] * v[3] - v[1] * v[2]))
	return lat


def _lattice(m, layer, W, H, T, spacing):
	"""Cells i·v1 + j·v2 (canvas units × T px) over the screen plus one block of margin."""
	v = np.asarray(_lattice_vectors(m), float) * spacing * T
	v1, v2 = v[:2], v[2:]
	seq = _sequence(_counts(m['shares']))
	slots = [m['symbols'][seq[r]] for r in range(SLOTS)]
	P1, P2 = 4 * v1, 4 * v2                             # the 4 × 4 block repeats exactly
	marg = np.abs(P1).sum() + np.abs(P2).sum() + T
	# solve for the (i, j) range covering the margin-padded screen
	Minv = np.linalg.inv(np.column_stack([v1, v2]))
	corners = np.array([[-marg, -marg], [W + marg, -marg], [-marg, H + marg], [W + marg, H + marg]]) - [W / 2, H / 2]
	ij = corners @ Minv.T
	i0, i1 = int(np.floor(ij[:, 0].min())), int(np.ceil(ij[:, 0].max()))
	j0, j1 = int(np.floor(ij[:, 1].min())), int(np.ceil(ij[:, 1].max()))
	I, J = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing='ij')
	I, J = I.ravel(), J.ravel()
	X = W / 2 + I * v1[0] + J * v2[0]
	Y = H / 2 + I * v1[1] + J * v2[1]
	ok = (X > -marg) & (X < W + marg) & (Y > -marg) & (Y < H + marg)
	I, J, X, Y = I[ok], J[ok], X[ok], Y[ok]
	rank = np.asarray(BAYER)[(J % 4) * 4 + (I % 4)]
	out = np.empty((len(X), len(CHANS)), np.float32)
	out[:, 0], out[:, 1], out[:, 2] = X, Y, 0.0
	out[:, 3] = out[:, 4] = T
	out[:, 5] = np.asarray(slots)[rank]
	out[:, 6] = layer
	return out, (P1[0], P1[1], P2[0], P2[1])


# ------------------------------------------------------------------ cook
def onCook(scriptOp):
	P = op.Controls.par
	r = op('render_particles')
	W, H = float(r.width), float(r.height)
	mode = P.Tilemode.eval()
	sel = op('script_symbol').fetch('mosaic', None, search=False) or {}
	layers = [(layer, m) for layer, m in ((0, sel.get('cur')), (1, sel.get('prev'))) if m]
	sigs = tuple(m['sig'] if m else None for m in (sel.get('cur'), sel.get('prev')))

	if mode == 'lattice':
		T = H / max(1.0, float(P.Latticecount))
		spacing = float(P.Latticespacing)
		key = ('lattice', sigs, W, H, round(T, 2), round(spacing, 3))
		if key != _cache['key']:
			parts, period = [], (T, 0.0, 0.0, T)
			for layer, m in layers:
				blk, per = _lattice(m, layer, W, H, T, spacing)
				parts.append(blk)
				if layer == 0 or len(layers) == 1:
					period = per
			_publish(scriptOp, parts, period, key)
	else:
		variation = int(P.Tilevariation)
		opts = (int(P.Snowarms), round(float(P.Snowsize), 4), round(float(P.Snowgap), 3),
				round(float(P.Snowbranch), 3))
		key = ('snow', sigs, W, H, variation, opts)
		if key != _cache['key']:
			parts = [_place_flake(_snowflake(m, variation, W / H, opts), layer, W, H) for layer, m in layers]
			_publish(scriptOp, parts, (0.0, 0.0, 0.0, 0.0), key)

	scriptOp.clear()
	out = _cache['out']
	scriptOp.numSamples = out.shape[1]
	for name, row in zip(CHANS, out):
		scriptOp.appendChan(name).vals = row


def _publish(scriptOp, parts, period, key):
	data = np.concatenate(parts) if parts else np.zeros((0, len(CHANS)), np.float32)
	if not len(data):
		data = np.array([[-1e4, -1e4, 0, 0, 0, 0, 0]], np.float32)   # one invisible instance
	_cache['out'], _cache['key'] = np.ascontiguousarray(data.T), key
	scriptOp.store('period', [float(x) for x in period])


def onGetCookLevel(scriptOp):
	return CookLevel.ALWAYS


def onSetupParameters(scriptOp):
	return


def onPulse(par):
	return
