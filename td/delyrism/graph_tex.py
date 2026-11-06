# Script TOP: the live delyrism graph packed into a small rgba32float texture
# for the edge-force compute shader (glsl_edges).  Indexed by node uid, so the
# shader never needs a uid → row remap; particles keep following a node across
# graph updates as long as its uid lives.
#
#   row 0  node   (x, y, fade, size)            column = uid
#   row 1  adj    (start, count, total, query opacity)   column = uid — slice of row 2
#   row 2  csr    (target uid, strength, 0, 0)  strength-weighted adjacency list
#   row 3  header (n_uid, n_csr, 0, 0)          column 0
#
# strength = Edgefloor + (1 − Edgefloor) · weight^Edgegamma, × edge fade (so a
# dropped edge loses its pull smoothly instead of snapping).
#
# query opacity mirrors the browser's node opacity under a query:
#     active ? Querydim + (1 − Querydim) · rel : 1      (rel = focus-sharpened relevance)
# eased per node with time constant Querylag, so a new query fades in rather
# than snapping.  The value is kept for nodes that left the graph, so their
# remaining particles keep the opacity they had.

import math
import numpy as np

_qop = {}   # uid → eased query opacity


def _chan(chop, name, n, default=0.0):
	c = chop.chan(name) if chop is not None else None
	if c is None:
		return np.full(n, default, np.float32)
	a = c.numpyArray()
	return a[:n].astype(np.float32) if len(a) >= n else np.full(n, default, np.float32)


def onCook(scriptOp):
	P = op.Controls.par  # all tuning lives on /project1/controls
	src = op.Delyrism
	fr, N, E = src.op('null_frame'), src.op('null_nodes'), src.op('null_edges')
	n = int(fr['nnodes'][0]) if fr is not None and fr.chan('nnodes') else 0
	e = int(fr['nedges'][0]) if fr is not None and fr.chan('nedges') else 0

	uid = _chan(N, 'uid', n, -1).astype(np.int64)
	n_uid = int(uid.max()) + 1 if n else 1
	es = _chan(E, 'src', e, -1).astype(np.int64)
	ed = _chan(E, 'dst', e, -1).astype(np.int64)
	ok = (es >= 0) & (es < n) & (ed >= 0) & (ed < n)
	es, ed = es[ok], ed[ok]
	w = _chan(E, 'weight', e)[ok]
	efade = _chan(E, 'fade', e)[ok] * _chan(E, 'alive', e, 1)[ok]
	floor_, gam = float(P.Edgefloor), float(P.Edgegamma)
	strength = (floor_ + (1.0 - floor_) * np.power(np.clip(w, 0, 1), gam)) * np.clip(efade, 0, 1)

	# undirected → both directions, grouped by source uid
	a = np.concatenate([uid[es], uid[ed]]) if n else np.zeros(0, np.int64)
	b = np.concatenate([uid[ed], uid[es]]) if n else np.zeros(0, np.int64)
	s2 = np.concatenate([strength, strength])
	order = np.argsort(a, kind='stable')
	a, b, s2 = a[order], b[order], s2[order]
	n_csr = len(a)

	# query opacity per node, eased
	qa = float(fr['queryactive'][0]) if fr is not None and fr.chan('queryactive') else 0.0
	rel = np.clip(_chan(N, 'rel', n), 0, 1)
	dim = float(P.Querydim)
	target = dim + (1.0 - dim) * rel if qa > 0.5 else np.ones(n, np.float32)
	dt = min(0.1, max(0.0, float(absTime.stepSeconds)))
	tau = float(P.Querylag)
	k = 1.0 if tau <= 1e-4 else 1.0 - math.exp(-dt / tau)
	for u, t in zip(uid.tolist(), np.asarray(target, np.float64).tolist()):
		prev = _qop.get(u, t)
		_qop[u] = prev + (t - prev) * k
	max_q = max(_qop) + 1 if _qop else 0

	W = max(n_uid, n_csr, max_q, 1)
	tex = np.zeros((4, W, 4), np.float32)
	tex[1, :, 3] = 1.0
	if _qop:
		qu = np.fromiter(_qop.keys(), np.int64)
		tex[1, qu, 3] = np.fromiter(_qop.values(), np.float32)
	if n:
		tex[0, uid, 0] = _chan(N, 'tx', n)
		tex[0, uid, 1] = _chan(N, 'ty', n)
		tex[0, uid, 2] = _chan(N, 'fade', n)
		tex[0, uid, 3] = _chan(N, 'size', n)
	if n_csr:
		starts = np.searchsorted(a, np.arange(n_uid))
		counts = np.bincount(a, minlength=n_uid)[:n_uid]
		totals = np.bincount(a, weights=s2, minlength=n_uid)[:n_uid]
		tex[1, :n_uid, 0] = starts
		tex[1, :n_uid, 1] = counts
		tex[1, :n_uid, 2] = totals
		tex[2, :n_csr, 0] = b
		tex[2, :n_csr, 1] = s2
	tex[3, 0, 0] = n_uid
	tex[3, 0, 1] = n_csr
	scriptOp.copyNumpyArray(tex)


def onGetCookLevel(scriptOp):
	return CookLevel.ALWAYS


def onSetupParameters(scriptOp):
	return


def onPulse(par):
	return
