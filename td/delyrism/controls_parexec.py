# Parameter Execute DAT on /project1/controls (op.Controls).
#
#  - Resolution preset → Width / Height; typing a custom size flips the preset
#    to "custom" (or back to a preset when the numbers match one).
#  - Open / Close Output Window pulses drive /project1/window_output.
#  - Reset Particles + Trails re-initialises the particlePOP and the feedback.
#  - Voice page: Voice / Slowness / Pitch / Reverb Decay are rendered by the TTS
#    service (web/tts/tts_server.py) and pushed to its /config; Preview
#    re-synthesises the last spoken word with them, Replay plays it again.
#    HTTP runs on a worker thread; results are picked up on the main thread.

import json
import threading
import urllib.request

TTS = 'http://127.0.0.1:8010'
SERVER_PARS = {'Voicename': 'voice', 'Voicestretch': 'stretch', 'Voicepitch': 'pitch', 'Reverbdecay': 'decay'}

PRESETS = {
	'720p': (1280, 720),
	'1080p': (1920, 1080),
	'1440p': (2560, 1440),
	'2160p': (3840, 2160),
	'square': (1080, 1080),
	'portrait': (1080, 1920),
}


def _post(path, body, box=None):
	def go():
		try:
			req = urllib.request.Request(TTS + path, data=json.dumps(body).encode(),
										 headers={'Content-Type': 'application/json'})
			with urllib.request.urlopen(req, timeout=30) as r:
				res = json.loads(r.read())
		except Exception as ex:
			res = {'error': str(ex)}
		if box is not None:
			box['res'] = res
	threading.Thread(target=go, daemon=True).start()


def _server_config(c):
	return {k: c.par[n].eval() for n, k in SERVER_PARS.items()}


def _await_speak(box, tries=0):
	res = box.get('res')
	if res is None:
		if tries < 600:
			run(_await_speak, box, tries + 1, delayFrames=3)
		return
	if 'path' in res:
		a = op.Delyrism.op('audiofilein_speak')
		a.par.file = res['path']
		a.par.cuepulse.pulse()
	else:
		debug('voice preview failed:', res)


def onValueChange(par, prev):
	c = par.owner
	if par.name in SERVER_PARS:
		_post('/config', _server_config(c))
	elif par.name == 'Respreset':
		wh = PRESETS.get(par.eval())
		if wh and (c.par.Resw.eval(), c.par.Resh.eval()) != wh:
			c.par.Resw, c.par.Resh = wh
	elif par.name in ('Resw', 'Resh'):
		cur = (c.par.Resw.eval(), c.par.Resh.eval())
		match = next((k for k, v in PRESETS.items() if v == cur), 'custom')
		if c.par.Respreset.eval() != match:
			c.par.Respreset = match
		# the trail feedback holds its old size until reset
		run("op.Particles.op('feedback_trails').par.resetpulse.pulse()", delayFrames=2)


def onPulse(par):
	win = parent.Project.op('window_output')
	if par.name == 'Openoutput' and win is not None:
		win.par.winopen.pulse()
	elif par.name == 'Closeoutput' and win is not None:
		win.par.winclose.pulse()
	elif par.name == 'Resetsim':
		p = op.Particles
		sim = p.op('particle_sim')
		sim.par.initializepulse.pulse()
		run('args[0].par.startpulse.pulse()', sim, delayFrames=1)
		p.op('feedback_trails').par.resetpulse.pulse()
	elif par.name == 'Voicereplay':
		op.Delyrism.op('audiofilein_speak').par.cuepulse.pulse()
	elif par.name == 'Voicepreview':
		word = par.owner.par.Lastspoken.eval() or 'stillness'
		box = {}
		_post('/config', _server_config(par.owner))          # make sure the server has the current settings
		run(lambda: _post('/speak', {'text': word}, box), delayFrames=10)
		run(_await_speak, box, delayFrames=12)


def onValuesChanged(changes):
	return


def onExpressionChange(par, val, prev):
	return


def onExportChange(par, val, prev):
	return


def onEnableChange(par, val, prev):
	return


def onModeChange(par, val, prev):
	return
