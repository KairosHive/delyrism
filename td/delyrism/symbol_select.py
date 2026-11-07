# Script CHOP: which Lakota symbols tile the background, and in what mosaic.
#
# Activation of each symbol = Σ over its nodes in the live graph of
#     no query : node size  (Σ|Δ|, the browser's node radius)
#     query    : node relevance (the browser's node brightness)
# (nodes fading out don't count), eased with Symbollag.  Symbol Cutoff c
# drops the bottom c of the activated symbols: the top ceil((1 − c)·n), at
# least one, form the mosaic (50 % = top half, 100 % = only the top symbol,
# 0 % = all), each with a share proportional to its activation among them.
# Symbolpin forces one symbol.
#
# The selection (symbols + shares, shares rounded to 5 %) is published in this
# operator's storage as 'mosaic' = {cur, prev, mix}; tiles_layout.py composes
# each mosaic as a symmetric snowflake tiling (mosaic_compose.py).  A new
# mosaic is adopted once it has been stable for Symbolhold seconds and
# crossfades in over Symbolfade.  Output (one sample): cur_w / prev_w and
# act_<name> for monitoring.

import math
import numpy as np

_s = {'act': None, 'cur': None, 'prev': None, 'mix': 1.0, 'cand': None, 'cand_t': 0.0, 'shown': None}


def _table(name):
	t = op(name)
	if t is None or t.numRows < 2:
		return []
	head = [c.val for c in t.row(0)]
	return [{h: t[r, h].val for h in head} for r in range(1, t.numRows)]




def onCook(scriptOp):
	P = op.Controls.par
	scriptOp.clear()
	rows = [r for r in _table('table_tilings') if r['mode'] == 'upright']
	K = len(rows)
	dt = min(0.1, max(0.0, float(absTime.stepSeconds)))

	variant = P.Earthvariant.eval()
	idx_of = {}
	for k, r in enumerate(rows):
		if r['symbol'] == 'EARTH' and r['name'] != variant:
			continue
		idx_of[r['symbol']] = k

	D = op.Delyrism
	fr, N, meta = D.op('null_frame'), D.op('null_nodes'), D.op('table_meta')
	n = int(fr['nnodes'][0]) if fr is not None and fr.chan('nnodes') else 0
	qa = float(fr['queryactive'][0]) > 0.5 if fr is not None and fr.chan('queryactive') else False
	names = []
	if meta is not None and meta.numRows > 1 and meta['symbols', 1] is not None:
		names = [x.strip() for x in meta['symbols', 1].val.split('|')]

	act = np.zeros(K)
	if n and K:
		g = lambda c, d=0.0: N[c].numpyArray()[:n] if N.chan(c) else np.full(n, d)  # noqa: E731
		sym, size, rel, alive = g('sym', -1), g('size'), g('rel'), g('alive', 1)
		score = rel if qa else size
		for i in range(n):
			s = int(sym[i])
			k = idx_of.get(names[s]) if 0 <= s < len(names) else None
			if k is not None and alive[i] >= 0.5:
				act[k] += float(score[i])

	prev_act = _s['act']
	if prev_act is None or len(prev_act) != K:
		_s['act'] = act
	else:
		a = 1.0 - math.exp(-dt / max(1e-3, float(P.Symbollag)))
		_s['act'] = prev_act + (act - prev_act) * a
	sm = _s['act']

	# ---- the mosaic this graph asks for
	pin = P.Symbolpin.eval()
	if pin != 'auto':
		keep = [k for k, r in enumerate(rows) if r['name'] == pin][:1]
		shares = [1.0] * len(keep)
	else:
		active = [k for k in np.argsort(-sm) if sm[k] > 1e-6]
		cut = min(1.0, max(0.0, float(P.Symbolcutoff) / 100.0))
		keep = active[: max(1, math.ceil(len(active) * (1.0 - cut) - 1e-9))] if active else []
		tot = float(sum(sm[k] for k in keep)) or 1.0
		shares = [float(sm[k]) / tot for k in keep]
	want = None
	if keep:
		q = [max(1, int(round(x * 20))) for x in shares]          # shares in 5 % steps
		qs = [x / sum(q) for x in q]
		want = dict(symbols=[int(k) for k in keep], shares=qs, names=[rows[k]['name'] for k in keep],
					label=' · '.join(f"{rows[k]['name']} {round(100 * x)}%" for k, x in zip(keep, qs)),
					sig=(tuple(int(k) for k in keep), tuple(q)))

	# ---- adopt it once stable for Symbolhold seconds, then crossfade
	cur = _s['cur']
	if want is None:
		_s['cand'] = None
	elif cur is None or want['sig'] != cur['sig']:
		if _s['cand'] is None or _s['cand']['sig'] != want['sig']:
			_s['cand'], _s['cand_t'] = want, 0.0
		_s['cand_t'] += dt
		if cur is None or pin != 'auto' or _s['cand_t'] >= float(P.Symbolhold):
			_s['prev'], _s['cur'], _s['mix'], _s['cand'] = cur, want, 0.0, None
	else:
		_s['cand'] = None
	_s['mix'] = min(1.0, _s['mix'] + dt / max(1e-3, float(P.Symbolfade)))

	scriptOp.store('mosaic', {'cur': _s['cur'], 'prev': _s['prev'], 'mix': _s['mix']})
	scriptOp.numSamples = 1
	scriptOp.appendChan('cur_w')[0] = _s['mix'] if _s['cur'] else 0.0
	scriptOp.appendChan('prev_w')[0] = (1.0 - _s['mix']) if _s['prev'] else 0.0
	for k, r in enumerate(rows):
		scriptOp.appendChan('act_' + r['name'])[0] = float(sm[k]) if k < len(sm) else 0.0

	shown = _s['cur']['label'] if _s['cur'] else '—'
	if shown != _s['shown']:
		_s['shown'] = shown
		run("op.Controls.par.Currentsymbol = args[0]", shown, delayFrames=1)


def onGetCookLevel(scriptOp):
	return CookLevel.ALWAYS


def onSetupParameters(scriptOp):
	return


def onPulse(par):
	return
