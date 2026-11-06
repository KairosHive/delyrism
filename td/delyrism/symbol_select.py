# Script CHOP: which Lakota symbol tiles the background.
#
# Activation of each symbol from the live graph:
#     Σ over its nodes  (Actsizefloor + size) · fade · (query ? Actqueryfloor + (1−Actqueryfloor)·rel : 1)
# eased with Symbollag.  The top symbol wins, but only switches when it beats the
# current one by Symbolmargin and the current one has been up for Symbolhold
# seconds; switches crossfade over Symbolfade.  Symbolpin forces one symbol.
#
# Output (one sample): the tiling of the current and the previous symbol, as
# the uniforms glsl_tiles needs (lattice in canvas units, see
# td/tools/lakota_prepare.py), plus their crossfade weights and act_<name>
# channels for monitoring.

import math
import numpy as np

_s = {'act': None, 'cur': -1, 'prev': -1, 'mix': 1.0, 'hold': 1e9, 'shown': None}
FIELDS = ['cellx', 'celly', 'ncopies', 't1', 'v1x', 'v1y', 'v2x', 'v2y', 'off1x', 'off1y']


def _rows(mode):
	t = op('table_tilings')
	if t is None or t.numRows < 2:
		return []
	head = [c.val for c in t.row(0)]
	out = []
	for r in range(1, t.numRows):
		row = {h: t[r, h].val for h in head}
		if row['mode'] == mode:
			out.append(row)
	return out


def onCook(scriptOp):
	P = op.Controls.par
	scriptOp.clear()
	rows = _rows(P.Tilemode.eval())
	K = len(rows)
	dt = min(0.1, max(0.0, float(absTime.stepSeconds)))

	# graph symbol name → atlas row (EARTH: the chosen drawing)
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
		sym, size, fade, rel = g('sym', -1), g('size'), g('fade'), g('rel')
		qf = float(P.Actqueryfloor)
		w = (float(P.Actsizefloor) + size) * fade
		if qa:
			w = w * (qf + (1.0 - qf) * rel)
		for i in range(n):
			s = int(sym[i])
			k = idx_of.get(names[s]) if 0 <= s < len(names) else None
			if k is not None:
				act[k] += w[i]

	prev_act = _s['act']
	if prev_act is None or len(prev_act) != K:
		_s['act'] = act
	else:
		a = 1.0 - math.exp(-dt / max(1e-3, float(P.Symbollag)))
		_s['act'] = prev_act + (act - prev_act) * a
	sm = _s['act']

	pin = P.Symbolpin.eval()
	if pin != 'auto':
		target = next((k for k, r in enumerate(rows) if r['name'] == pin), -1)
	else:
		target = int(np.argmax(sm)) if K and sm.max() > 1e-6 else -1

	cur = _s['cur']
	_s['hold'] += dt
	if cur >= K:
		cur = _s['cur'] = -1
	if target >= 0 and target != cur:
		better = cur < 0 or pin != 'auto' or sm[target] > sm[cur] * (1.0 + float(P.Symbolmargin))
		if better and (cur < 0 or pin != 'auto' or _s['hold'] >= float(P.Symbolhold)):
			_s['prev'], _s['cur'], _s['mix'], _s['hold'] = cur, target, 0.0, 0.0
	_s['mix'] = min(1.0, _s['mix'] + dt / max(1e-3, float(P.Symbolfade)))

	scriptOp.numSamples = 1
	for tag, k, wgt in (('cur', _s['cur'], _s['mix']), ('prev', _s['prev'], 1.0 - _s['mix'])):
		r = rows[k] if 0 <= k < K else None
		for f in FIELDS:
			scriptOp.appendChan(f'{tag}_{f}')[0] = float(r[f]) if r else 0.0
		scriptOp.appendChan(f'{tag}_w')[0] = wgt if r else 0.0
	scriptOp.appendChan('cur_index')[0] = _s['cur']
	for k, r in enumerate(rows):
		scriptOp.appendChan('act_' + r['name'])[0] = float(sm[k]) if k < len(sm) else 0.0

	shown = rows[_s['cur']]['name'] if 0 <= _s['cur'] < K else '—'
	if shown != _s['shown']:
		_s['shown'] = shown
		run("op.Controls.par.Currentsymbol = args[0]", shown, delayFrames=1)


def onGetCookLevel(scriptOp):
	return CookLevel.ALWAYS


def onSetupParameters(scriptOp):
	return


def onPulse(par):
	return
