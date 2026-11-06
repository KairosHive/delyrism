# Parameter Execute DAT on /project1/controls (op.Controls).
#
#  - Resolution preset → Width / Height; typing a custom size flips the preset
#    to "custom" (or back to a preset when the numbers match one).
#  - Open / Close Output Window pulses drive /project1/window_output.
#  - Reset Particles + Trails re-initialises the particlePOP and the feedback.

PRESETS = {
	'720p': (1280, 720),
	'1080p': (1920, 1080),
	'1440p': (2560, 1440),
	'2160p': (3840, 2160),
	'square': (1080, 1080),
	'portrait': (1080, 1920),
}


def onValueChange(par, prev):
	c = par.owner
	if par.name == 'Respreset':
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
