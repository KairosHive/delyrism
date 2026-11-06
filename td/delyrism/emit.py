# Script CHOP: live delyrism nodes → particle emitters (one sample per node).
#
# Feeds choptoPOP → particlePOP (Emission from: attribute "Birth").  The
# particlePOP emits int(Birth) particles per point per frame, so the per-second
# rate  Rate · (Sizefloor + size)^Sizegamma · fade · (1 + Relboost · rel)
# is turned into a per-frame count with stochastic rounding — the long-run
# rate is exact even when it is below one particle per frame.  Tuning: op.Controls.
#
# Channels: tx ty tz  r g b a  Cur (node uid, inherited by the particles)  Birth
#           Src (same uid; unlike Cur it never changes when a particle hops)

import numpy as np

_rng = np.random.default_rng()


def _chan(chop, name, n, default=0.0):
	c = chop.chan(name) if chop is not None else None
	if c is None:
		return np.full(n, default, np.float32)
	a = c.numpyArray()
	return a[:n].astype(np.float32) if len(a) >= n else np.full(n, default, np.float32)


def onCook(scriptOp):
	P = op.Controls.par  # all tuning lives on /project1/controls
	src = op.Delyrism
	fr, N = src.op('null_frame'), src.op('null_nodes')
	n = int(fr['nnodes'][0]) if fr is not None and fr.chan('nnodes') else 0
	qa = float(fr['queryactive'][0]) if fr is not None and fr.chan('queryactive') else 0.0
	dt = min(0.1, max(0.0, float(absTime.stepSeconds)))

	size, fade, alive = _chan(N, 'size', n), _chan(N, 'fade', n), _chan(N, 'alive', n)
	rel = _chan(N, 'rel', n)
	rate = float(P.Rate) * np.power(float(P.Sizefloor) + np.clip(size, 0, 1), float(P.Sizegamma))
	rate *= fade * (alive > 0.5) * (1.0 + float(P.Relboost) * qa * rel)
	expected = rate * dt
	birth = np.floor(expected + _rng.random(n)).astype(np.float32) if n else np.zeros(0, np.float32)

	k = max(1, n)
	cols = {
		'tx': _chan(N, 'tx', n), 'ty': _chan(N, 'ty', n), 'tz': np.zeros(n, np.float32),
		'r': _chan(N, 'r', n, 1), 'g': _chan(N, 'g', n, 1), 'b': _chan(N, 'b', n, 1),
		'a': np.ones(n, np.float32), 'Cur': _chan(N, 'uid', n, -1), 'Birth': birth,
		'Src': _chan(N, 'uid', n, -1),   # emitting node, kept for life (saliency reference)
	}
	scriptOp.clear()
	scriptOp.numSamples = k
	for name, vals in cols.items():
		v = np.zeros(k, np.float32)
		if n:
			v[:] = vals
		scriptOp.appendChan(name).vals = v


def onGetCookLevel(scriptOp):
	return CookLevel.ALWAYS


def onSetupParameters(scriptOp):
	return


def onPulse(par):
	return
