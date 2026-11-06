# OSC In DAT callbacks for the delyrism live Δ-graph stream.
#
# The backend (web/backend/app/osc_bridge.py) sends one OSC bundle per frame:
# /delyrism/frame first, then one message per attribute (/delyrism/node/x,
# /delyrism/edge/x0, …, one argument per node / edge) and /delyrism/end last.
# We collect the messages of the frame in progress and only publish them on
# /delyrism/end, so the CHOPs never see a half-received frame.
#
# Published to the parent COMP's storage as 'frame' (dict: 'node/x' → tuple
# of args, …), then the script CHOPs are cooked.  Labels / symbols /
# context / query arrive about once a second and on every node-order change;
# they go into table_labels / table_meta.

_ADDR = '/delyrism/'


def onReceiveOSC(dat, rowIndex, message, byteData, timeStamp, address, args, peer):
	if not address.startswith(_ADDR):
		return
	key = address[len(_ADDR):]
	pending = dat.storage.setdefault('pending', {})
	if key == 'frame':
		pending.clear()
	if key != 'end':
		pending[key] = args
		return

	frame = dict(pending)
	pending.clear()
	comp = parent()
	comp.store('frame', frame)
	for name in ('script_frame', 'script_nodes', 'script_edges'):
		o = op(name)
		if o is not None:
			o.cook(force=True)
	if 'node/label' in frame:
		_write_labels(frame)


def _write_labels(frame):
	labels = frame.get('node/label', ())
	uids = frame.get('node/uid', ())
	syms = frame.get('node/sym', ())
	names = frame.get('symbols', ())
	t = op('table_labels')
	if t is not None:
		rows = [['uid', 'label', 'symbol']]
		for i, lab in enumerate(labels):
			s = int(syms[i]) if i < len(syms) else -1
			rows.append([uids[i] if i < len(uids) else i, lab, names[s] if 0 <= s < len(names) else ''])
		t.clear()
		t.appendRows(rows)
	m = op('table_meta')
	if m is not None:
		q = frame.get('query', ('', ''))
		m.clear()
		m.appendRows([
			['key', 'value'],
			['context', (frame.get('context') or [''])[0]],
			['query', q[0] if len(q) > 0 else ''],
			['query_kind', q[1] if len(q) > 1 else ''],
			['symbols', ' | '.join(names)],
		])
