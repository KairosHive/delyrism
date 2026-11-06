// glsl_tonemap — hue-preserving tone map for the accumulated particle image.
//
// Particles are blended additively into a float buffer, so overlaps add real
// energy (density).  A plain clamp then saturates each channel separately and
// dense spots drift to white.  Here every pixel is split into
//     hue       = rgb / max(rgb)          (chroma — brightest channel = 1)
//     intensity = max(rgb) · exposure
// and only the intensity is compressed, with a soft knee:
//     mapped = 1 − exp(−aggregation · intensity)
// so a lone particle stays clearly visible, overlaps build toward the full,
// saturated version of their own colour, and nothing turns white unless the
// colours themselves mix to white.  "White core" optionally lets the very
// densest spots bloom toward white.
//
// uTone = (exposure, aggregation, white core, mode)   mode 0 = tone map, 1 = raw (clamp only)

uniform vec4 uTone;

out vec4 fragColor;

void main()
{
	vec4 c = texture(sTD2DInputs[0], vUV.st);
	vec3 rgb = max(c.rgb, vec3(0.0)) * uTone.x;

	if (uTone.w > 0.5) {
		fragColor = TDOutputSwizzle(vec4(clamp(rgb, 0.0, 1.0), 1.0));
		return;
	}

	float m = max(max(rgb.r, rgb.g), rgb.b);
	vec3 outc = vec3(0.0);
	if (m > 1e-6) {
		vec3 hue = rgb / m;
		float mapped = 1.0 - exp(-uTone.y * m);
		// optional bloom toward white, only where density is far past saturation
		float core = uTone.z * (1.0 - exp(-0.15 * max(0.0, uTone.y * m - 2.0)));
		outc = mix(hue, vec3(1.0), clamp(core, 0.0, 1.0)) * mapped;
	}
	fragColor = TDOutputSwizzle(vec4(outc, 1.0));
}
