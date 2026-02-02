import shlex

class SoxAllpassNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_allpass": ("BOOLEAN", {"default": True, "tooltip": "allpass frequency width[h|k|q|o]"}),
                "allpass_frequency": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "allpass_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization/Filtering/Tone Shaping"
    DESCRIPTION = "Allpass SoX effect node for chaining. dbg-text STRING: 'allpass params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_allpass=True, allpass_frequency=1000.0, allpass_width=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["allpass", str(allpass_frequency), f"{allpass_width}q"]
        debug_str = shlex.join(effect_params)
        if enable_allpass:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxBandNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_band": ("BOOLEAN", {"default": True, "tooltip": "band [-n] center [width[h|k|q|o]]"}),
                "band_narrow": ("BOOLEAN", {"default": False}),
                "band_center": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "band_width": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 10000.0, "step": 1.0}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization/Filtering/Tone Shaping"
    DESCRIPTION = "Band SoX effect node for chaining. dbg-text STRING: 'band params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_band=True, band_narrow=False, band_center=1000.0, band_width=100.0,
                prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["band"]
        if band_narrow:
            effect_params.append("-n")
        effect_params += [str(band_center), f"{band_width}Hz"]
        debug_str = shlex.join(effect_params)
        if enable_band:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxBandpassNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_bandpass": ("BOOLEAN",
                                    {"default": True, "tooltip": "bandpass [-c center] [width[h|k|q|o]] freq"}),
                "bandpass_frequency": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "bandpass_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization/Filtering/Tone Shaping"
    DESCRIPTION = "Bandpass SoX effect node for chaining. dbg-text STRING: 'bandpass params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_bandpass=True, bandpass_frequency=1000.0, bandpass_width=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["bandpass", str(bandpass_frequency), f"{bandpass_width}q"]
        debug_str = shlex.join(effect_params)
        if enable_bandpass:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxBandrejectNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_bandreject": ("BOOLEAN",
                                      {"default": True, "tooltip": "bandreject [-c center] [width[h|k|q|o]] freq"}),
                "bandreject_frequency": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "bandreject_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization/Filtering/Tone Shaping"
    DESCRIPTION = "Bandreject SoX effect node for chaining. dbg-text STRING: 'bandreject params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_bandreject=True, bandreject_frequency=1000.0, bandreject_width=1.0,
                prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["bandreject", str(bandreject_frequency), f"{bandreject_width}q"]
        debug_str = shlex.join(effect_params)
        if enable_bandreject:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxBiquadNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_biquad": ("BOOLEAN", {"default": True, "tooltip": "biquad frequency gain BW|Q|S [norm]"}),
                "biquad_frequency": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "biquad_gain": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "biquad_q": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "biquad_norm": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization/Filtering/Tone Shaping"
    DESCRIPTION = "Biquad SoX effect node for chaining. dbg-text STRING: 'biquad params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_biquad=True, biquad_frequency=1000.0, biquad_gain=0.0, biquad_q=1.0, biquad_norm=1,
                prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["biquad", str(biquad_frequency), str(biquad_gain), str(biquad_q), str(biquad_norm)]
        debug_str = shlex.join(effect_params)
        if enable_biquad:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxChannelsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_channels": ("BOOLEAN", {"default": True, "tooltip": "channels [number]"}),
                "channels_number": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Channel/Pan/Remix"
    DESCRIPTION = "Channels SoX effect node for chaining. dbg-text STRING: 'channels params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_channels=True, channels_number=2, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["channels", str(channels_number)]
        debug_str = shlex.join(effect_params)
        if enable_channels:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxContrastNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_contrast": ("BOOLEAN", {"default": True, "tooltip": "contrast [enhancement]"}),
                "contrast_enhancement": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Dynamics/Volume/Compression"
    DESCRIPTION = "Contrast SoX effect node for chaining. dbg-text STRING: 'contrast params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_contrast=True, contrast_enhancement=20.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["contrast", str(contrast_enhancement)]
        debug_str = shlex.join(effect_params)
        if enable_contrast:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxDcshiftNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_dcshift": ("BOOLEAN", {"default": True, "tooltip": "dcshift amount[%]"}),
                "dcshift_amount": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Noise/Dither/Restoration"
    DESCRIPTION = "Dcshift SoX effect node for chaining. dbg-text STRING: 'dcshift params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_dcshift=True, dcshift_amount=0.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["dcshift", str(dcshift_amount)]
        debug_str = shlex.join(effect_params)
        if enable_dcshift:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxDeemphNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_deemph": ("BOOLEAN", {"default": True, "tooltip": "deemph [profile]"}),
                "deemph_profile": (["ccir", "50us", "75us", "15khz"], {"default": "ccir"}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization/Filtering/Tone Shaping"
    DESCRIPTION = "Deemph SoX effect node for chaining. dbg-text STRING: 'deemph params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_deemph=True, deemph_profile="ccir", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["deemph", deemph_profile]
        debug_str = shlex.join(effect_params)
        if enable_deemph:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxDelayNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_delay": ("BOOLEAN", {"default": True, "tooltip": "delay length [pad]"}),
                "delay_length": ("FLOAT", {"default": 500.0, "min": 0.0, "max": 10000.0, "step": 1.0}),
                "delay_pad": ("FLOAT", {"default": 500.0, "min": 0.0, "max": 10000.0, "step": 1.0}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Other/Utility"
    DESCRIPTION = "Delay SoX effect node for chaining. dbg-text STRING: 'delay params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_delay=True, delay_length=500.0, delay_pad=500.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["delay", str(delay_length), str(delay_pad)]
        debug_str = shlex.join(effect_params)
        if enable_delay:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxDitherNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_dither": ("BOOLEAN", {"default": True, "tooltip": "dither [-s|-a|-h] [n]"}),
                "dither_type": (["s", "a", "h"], {"default": "s"}),
                "dither_depth": ("INT", {"default": 6, "min": 1, "max": 24, "step": 1}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Noise/Dither/Restoration"
    DESCRIPTION = "Dither SoX effect node for chaining. dbg-text STRING: 'dither params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_dither=True, dither_type="s", dither_depth=6, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["dither", "-" + dither_type, str(dither_depth)]
        debug_str = shlex.join(effect_params)
        if enable_dither:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxDownsampleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("audio",),
                "enable_downsample": ("boolean", {"default": True, "tooltip": "downsample factor"}),
                "downsample_factor": ("int", {"default": 2, "min": 1, "max": 32, "step": 1}),
            },
            "optional": {
                "prev_params": ("sox_params",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Other/Utility"
    DESCRIPTION = "Downsample SoX effect node for chaining. dbg-text `STRING`: 'downsample params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_downsample=True, downsample_factor=2, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["downsample", str(downsample_factor)]
        debug_str = shlex.join(effect_params)
        if enable_downsample:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxEarwaxNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("audio",),
                "enable_earwax": ("boolean", {"default": True, "tooltip": "earwax"}),
            },
            "optional": {
                "prev_params": ("sox_params",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Modulation/Special Effects"
    DESCRIPTION = "Earwax SoX effect node for chaining. dbg-text `STRING`: 'earwax params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_earwax=True, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["earwax"]
        debug_str = shlex.join(effect_params)
        if enable_earwax:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxFadeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("audio",),
                "enable_fade": ("boolean", {"default": True, "tooltip": "fade [t|h] length [in/out]"}),
                "fade_type": (["h", "t"], {"default": "h"}),
                "fade_in_length": ("float", {"default": 0.5, "min": 0.0, "max": 60.0, "step": 0.01}),
                "fade_out_length": ("float", {"default": 0.5, "min": 0.0, "max": 60.0, "step": 0.01}),
            },
            "optional": {
                "prev_params": ("sox_params",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Envelope/Fade/Silence"
    DESCRIPTION = "Fade SoX effect node for chaining. dbg-text `STRING`: 'fade params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_fade=True, fade_type="h", fade_in_length=0.5, fade_out_length=0.5,
                prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["fade", fade_type]
        if fade_in_length > 0:
            effect_params.append(str(fade_in_length))
        if fade_out_length > 0:
            effect_params.append(str(fade_out_length))
        debug_str = shlex.join(effect_params)
        if enable_fade:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxFirNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("audio",),
                "enable_fir": ("boolean", {"default": True, "tooltip": "fir [coefficients]"}),
                "fir_coefficients": ("string", {"multiline": True, "default": ""}),
            },
            "optional": {
                "prev_params": ("sox_params",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization/Filtering/Tone Shaping"
    DESCRIPTION = "Fir SoX effect node for chaining (provide coefficients). dbg-text `string`: 'fir params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_fir=True, fir_coefficients="", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if fir_coefficients.strip():
            coeffs = shlex.split(fir_coefficients.strip())
            effect_params = ["fir"] + coeffs
        else:
            effect_params = ["fir"]
        debug_str = shlex.join(effect_params)
        if enable_fir and fir_coefficients.strip():
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxGainNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("audio",),
                "enable_gain": ("boolean", {"default": True, "tooltip": "gain [-n] amount [dB]"}),
                "gain_normalize": ("boolean", {"default": False}),
                "gain_amount": ("float", {"default": 0.0, "min": -60.0, "max": 60.0, "step": 0.1}),
            },
            "optional": {
                "prev_params": ("sox_params",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Dynamics/Volume/Compression"
    DESCRIPTION = "Gain SoX effect node for chaining. dbg-text `string`: 'gain params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_gain=True, gain_normalize=False, gain_amount=0.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["gain"]
        if gain_normalize:
            effect_params.append("-n")
        effect_params.append(str(gain_amount))
        debug_str = shlex.join(effect_params)
        if enable_gain:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxHilbertNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_hilbert": ("BOOLEAN", {"default": True, "tooltip": "hilbert [window] [halflen]"}),
                "hilbert_window": ("INT", {"default": 64, "min": 8, "max": 1024, "step": 8}),
                "hilbert_halflen": ("INT", {"default": 16, "min": 4, "max": 256, "step": 4}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Other/Utility"
    DESCRIPTION = "Hilbert SoX effect node for chaining."

    def process(self, audio, enable_hilbert=True, hilbert_window=64, hilbert_halflen=16, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_hilbert:
            effect_params = ["hilbert", str(hilbert_window), str(hilbert_halflen)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxLadspaNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_ladspa": ("BOOLEAN", {"default": True, "tooltip": "ladspa plugin label [params...]"}),
                "ladspa_params": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Other/Utility"
    DESCRIPTION = "Ladspa SoX effect node for chaining (plugin label params)."

    def process(self, audio, enable_ladspa=True, ladspa_params="", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_ladspa and ladspa_params.strip():
            params = shlex.split(ladspa_params.strip())
            effect_params = ["ladspa"] + params
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxLoudnessNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_loudness": ("BOOLEAN", {"default": True, "tooltip": "loudness [gain [volume]]"}),
                "loudness_gain": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "loudness_volume": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Dynamics/Volume/Compression"
    DESCRIPTION = "Loudness SoX effect node for chaining."

    def process(self, audio, enable_loudness=True, loudness_gain=4.0, loudness_volume=12.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_loudness:
            effect_params = ["loudness", str(loudness_gain), str(loudness_volume)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxMcompandNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_mcompand": ("BOOLEAN", {"default": True, "tooltip": "mcompand [multi-band compand params]"}),
                "mcompand_params": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Dynamics/Volume/Compression"
    DESCRIPTION = "Mcompand SoX effect node for chaining (multi-band compand params)."

    def process(self, audio, enable_mcompand=True, mcompand_params="", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_mcompand and mcompand_params.strip():
            params = shlex.split(mcompand_params.strip())
            effect_params = ["mcompand"] + params
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxNoiseprofNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_noiseprof": ("BOOLEAN", {"default": True, "tooltip": "noiseprof [noise.wav]"}),
                "noiseprof_noise_file": ("STRING", {"default": "", "tooltip": "Optional noise profile file"}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Noise/Dither/Restoration"
    DESCRIPTION = "Noiseprof SoX effect node for chaining (generates noise profile)."

    def process(self, audio, enable_noiseprof=True, noiseprof_noise_file="", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_noiseprof:
            effect_params = ["noiseprof"]
            if noiseprof_noise_file:
                effect_params.append(noiseprof_noise_file)
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxNoiseredNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_noisered": ("BOOLEAN",
                                    {"default": True, "tooltip": "noisered [noise.prof] [amount [precision]]"}),
                "noisered_profile": ("STRING", {"default": "", "tooltip": "noise.prof file"}),
                "noisered_amount": ("FLOAT", {"default": 0.21, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noisered_precision": ("INT", {"default": 4, "min": 0, "max": 6, "step": 1}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Noise/Dither/Restoration"
    DESCRIPTION = "Noisered SoX effect node for chaining."

    def process(self, audio, enable_noisered=True, noisered_profile="", noisered_amount=0.21, noisered_precision=4,
                prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_noisered:
            effect_params = ["noisered"]
            if noisered_profile:
                effect_params.append(noisered_profile)
            effect_params += [str(noisered_amount), str(noisered_precision)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxNormNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_norm": ("BOOLEAN", {"default": True, "tooltip": "norm [-b|-r|-s] [level [precision]]"}),
                "norm_type": (["", "b", "r", "s"], {"default": ""}),
                "norm_level": ("FLOAT", {"default": -3.0, "min": -99.0, "max": 0.0, "step": 0.1}),
                "norm_precision": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Dynamics/Volume/Compression"
    DESCRIPTION = "Norm SoX effect node for chaining."

    def process(self, audio, enable_norm=True, norm_type="", norm_level=-3.0, norm_precision=0.1, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_norm:
            effect_params = ["norm"]
            if norm_type:
                effect_params.append("-" + norm_type)
            effect_params += [str(norm_level), str(norm_precision)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxOopsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_oops": ("BOOLEAN", {"default": True, "tooltip": "oops [threshold]"}),
                "oops_threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Other/Utility"
    DESCRIPTION = "Oops SoX effect node for chaining."

    def process(self, audio, enable_oops=True, oops_threshold=0.8, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_oops:
            effect_params = ["oops", str(oops_threshold)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxPadNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_pad": ("BOOLEAN", {"default": True, "tooltip": "pad intro [outro]"}),
                "pad_intro": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.01}),
                "pad_outro": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.01}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Envelope/Fade/Silence"
    DESCRIPTION = "Pad SoX effect node for chaining."

    def process(self, audio, enable_pad=True, pad_intro=0.0, pad_outro=0.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_pad:
            effect_params = ["pad", str(pad_intro), str(pad_outro)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxRateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_rate": ("BOOLEAN", {"default": True, "tooltip": "rate [-v] [-b low [high]] [q|h|v]"}),
                "rate_quality": (["q", "h", "v"], {"default": "q"}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Pitch/Speed/Tempo/Rate Manipulation"
    DESCRIPTION = "Rate SoX effect node for chaining."

    def process(self, audio, enable_rate=True, rate_quality="q", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_rate:
            effect_params = ["rate", "-v", rate_quality]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxRemixNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_remix": ("BOOLEAN", {"default": True, "tooltip": "remix [-m|--mix|--merge] gains"}),
                "remix_mode": (["", "m", "mix", "merge"], {"default": ""}),
                "remix_gains": ("STRING", {"multiline": True, "default": "1.0"}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Channel/Pan/Remix"
    DESCRIPTION = "Remix SoX effect node for chaining."

    def process(self, audio, enable_remix=True, remix_mode="", remix_gains="1.0", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_remix:
            gains = shlex.split(remix_gains.strip())
            effect_params = ["remix"]
            if remix_mode:
                effect_params.append("--" + remix_mode)
            effect_params += gains
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxRepeatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_repeat": ("BOOLEAN", {"default": True, "tooltip": "repeat count"}),
                "repeat_count": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Other/Utility"
    DESCRIPTION = "Repeat SoX effect node for chaining."

    def process(self, audio, enable_repeat=True, repeat_count=1, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_repeat:
            effect_params = ["repeat", str(repeat_count)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxReverseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_reverse": ("BOOLEAN", {"default": True, "tooltip": "reverse"}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Channel/Pan/Remix"
    DESCRIPTION = "Reverse SoX effect node for chaining."

    def process(self, audio, enable_reverse=True, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_reverse:
            effect_params = ["reverse"]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxRiaaNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_riaa": ("BOOLEAN", {"default": True, "tooltip": "riaa [pre]"}),
                "riaa_pre": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization/Filtering/Tone Shaping"
    DESCRIPTION = "Riaa SoX effect node for chaining."

    def process(self, audio, enable_riaa=True, riaa_pre=False, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_riaa:
            effect_params = ["riaa"]
            if riaa_pre:
                effect_params.append("pre")
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxSilenceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_silence": ("BOOLEAN", {"default": True, "tooltip": "silence [-l] above duration"}),
                "silence_above": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "silence_duration": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 60.0, "step": 0.01}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Envelope/Fade/Silence"
    DESCRIPTION = "Silence SoX effect node for chaining."

    def process(self, audio, enable_silence=True, silence_above=0.01, silence_duration=0.1, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_silence:
            effect_params = ["silence", "1", str(silence_duration), f"{silence_above}%"]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxSincNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_sinc": ("BOOLEAN", {"default": True, "tooltip": "sinc [-h] [-n|-t] [-k freq] freq"}),
                "sinc_frequency": ("FLOAT", {"default": 8000.0, "min": 0.0, "max": 96000.0, "step": 100.0}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization/Filtering/Tone Shaping"
    DESCRIPTION = "Sinc SoX effect node for chaining."

    def process(self, audio, enable_sinc=True, sinc_frequency=8000.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_sinc:
            effect_params = ["sinc", str(sinc_frequency)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxSpeedNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_speed": ("BOOLEAN", {"default": True, "tooltip": "speed factor"}),
                "speed_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Pitch/Speed/Tempo/Rate Manipulation"
    DESCRIPTION = "Speed SoX effect node for chaining."

    def process(self, audio, enable_speed=True, speed_factor=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_speed:
            effect_params = ["speed", str(speed_factor)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxSpliceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_splice": ("BOOLEAN", {"default": True, "tooltip": "splice start [duration]"}),
                "splice_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.01}),
                "splice_duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.01}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Envelope/Fade/Silence"
    DESCRIPTION = "Splice SoX effect node for chaining."

    def process(self, audio, enable_splice=True, splice_start=0.0, splice_duration=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_splice:
            effect_params = ["splice", str(splice_start), str(splice_duration)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxStatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_stat": ("BOOLEAN", {"default": True, "tooltip": "stat [tags]"}),
                "stat_tags": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Visualization/Analysis (non-destructive)"
    DESCRIPTION = "Stat SoX effect node for chaining (audio stats)."

    def process(self, audio, enable_stat=True, stat_tags="", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_stat:
            tags = shlex.split(stat_tags.strip())
            effect_params = ["stat"] + tags
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxStatsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_stats": ("BOOLEAN", {"default": True, "tooltip": "stats [tag]"}),
                "stats_tag": ("STRING", {"default": ""}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Visualization/Analysis (non-destructive)"
    DESCRIPTION = "Stats SoX effect node for chaining."

    def process(self, audio, enable_stats=True, stats_tag="", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_stats:
            effect_params = ["stats"]
            if stats_tag:
                effect_params.append(stats_tag)
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxStretchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_stretch": ("BOOLEAN", {"default": True, "tooltip": "stretch factor [fadelen]"}),
                "stretch_factor": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.01}),
                "stretch_fadelen": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Pitch/Speed/Tempo/Rate Manipulation"
    DESCRIPTION = "Stretch SoX effect node for chaining."

    def process(self, audio, enable_stretch=True, stretch_factor=1.0, stretch_fadelen=0.05, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_stretch:
            effect_params = ["stretch", str(stretch_factor), str(stretch_fadelen)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxSwapNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_swap": ("BOOLEAN", {"default": True, "tooltip": "swap [1|2|3|4]"}),
                "swap_operation": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Channel/Pan/Remix"
    DESCRIPTION = "Swap SoX effect node for chaining."

    def process(self, audio, enable_swap=True, swap_operation=1, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_swap:
            effect_params = ["swap", str(swap_operation)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxSynthNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_synth": ("BOOLEAN", {"default": True, "tooltip": "synth [len] TYPE freq"}),
                "synth_params": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Modulation/Special Effects"
    DESCRIPTION = "Synth SoX effect node for chaining (generator)."

    def process(self, audio, enable_synth=True, synth_params="", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_synth and synth_params.strip():
            params = shlex.split(synth_params.strip())
            effect_params = ["synth"] + params
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxTrimNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_trim": ("BOOLEAN", {"default": True, "tooltip": "trim start [end]"}),
                "trim_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.01}),
                "trim_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.01}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Envelope/Fade/Silence"
    DESCRIPTION = "Trim SoX effect node for chaining."

    def process(self, audio, enable_trim=True, trim_start=0.0, trim_end=0.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_trim:
            effect_params = ["trim"]
            effect_params.append(str(trim_start))
            if trim_end > 0:
                effect_params.append(str(trim_end))
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxUpsampleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_upsample": ("BOOLEAN", {"default": True, "tooltip": "upsample factor"}),
                "upsample_factor": ("INT", {"default": 2, "min": 1, "max": 32, "step": 1}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Other/Utility"
    DESCRIPTION = "Upsample SoX effect node for chaining."

    def process(self, audio, enable_upsample=True, upsample_factor=2, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_upsample:
            effect_params = ["upsample", str(upsample_factor)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxVadNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_vad": ("BOOLEAN", {"default": True,
                                           "tooltip": "Enable VAD (Voice Activity Detection) SoX effect: Trims silence before/after detected speech/audio activity. Usage: Chain early in workflow → SoxApplyEffectsNode for clean recordings (podcasts, vocals). Pairs with Vol for balance."}),
                "vad_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "VAD threshold (0.0-1.0): Energy level above which audio is considered 'voice'; trims leading/trailing silence. Higher values trim more aggressively."}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Envelope/Fade/Silence"
    DESCRIPTION = """SoxVadNode: Chains VAD (Voice Activity Detection) SoX effect to SOX_PARAMS.

**What it does**: Adds `vad -t <threshold>` param; trims leading/trailing silence by detecting energy above threshold (0.0-1.0).

**How to use**:
- Toggle `enable_vad`; adjust `vad_threshold` (0.5 default: balanced).
- Wire: AUDIO → SoxVadNode → [Vol/Bass/etc.] → SoxApplyEffectsNode → Output.
- Best early: Clean raw audio (podcasts/vocals) before mixing/gain/EQ.
- Output: Unchanged AUDIO + updated SOX_PARAMS."""

    def process(self, audio, enable_vad=True, vad_threshold=0.5, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_vad:
            effect_params = ["vad", f"-t {vad_threshold}"]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxVolNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_vol": ("BOOLEAN", {"default": True,
                                           "tooltip": "Enable Vol (Volume) SoX effect: Adjusts gain by dB. Usage: Chain for level matching → SoxApplyEffectsNode. Use post-VAD, pre-mix to prevent clipping."}),
                "vol_gain": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 60.0, "step": 0.1,
                                       "tooltip": "Volume gain in dB: Positive boosts amplitude, negative attenuates. 0dB=unity. Use for per-track balance pre-mix/effects."}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Dynamics/Volume/Compression"
    DESCRIPTION = """SoxVolNode: Chains Vol (Volume Gain) SoX effect to SOX_PARAMS.

**What it does**: Adds `vol <gain_dB>` param; amplifies/attenuates audio linearly in dB (-60/+60 range).

**How to use**:
- Toggle `enable_vol`; set `vol_gain` (0.0=unity; +boost, -=cut).
- Wire: AUDIO → [Vad/etc.] → SoxVolNode → [EQ/Effects] → SoxApplyEffectsNode → Output.
- Best mid-chain: Balance after trim (VAD), before compression/mix. Prevents overload with negative gain.
- Output: Unchanged AUDIO + updated SOX_PARAMS."""

    def process(self, audio, enable_vol=True, vol_gain=0.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_vol:
            effect_params = ["vol", str(vol_gain)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxBassNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_bass": ("BOOLEAN", {"default": True,
                                            "tooltip": """bass gain [frequency(100) [width[s|h|k|q|o]](0.5s)]"""}),
                "bass_gain": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "bass_frequency": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "bass_width": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01, }),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization/Filtering/Tone Shaping"
    DESCRIPTION = "Bass SoX effect node for chaining. dbg-text STRING: 'bass params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_bass=True, bass_gain=0.0, bass_frequency=100.0, bass_width=0.5, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["bass", str(bass_gain), str(bass_frequency), str(bass_width)]
        debug_str = shlex.join(effect_params)
        if enable_bass:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxBendNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_bend": ("BOOLEAN", {"default": True,
                                            "tooltip": """bend [-f frame-rate(25)] [-o over-sample(16)] {start,cents,end}"""}),
                "bend_frame_rate": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "bend_over_sample": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "bend_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "bend_cents": ("FLOAT", {"default": 0.0, "min": -1200.0, "max": 1200.0, "step": 1.0}),
                "bend_end_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.01}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Pitch/Speed/Tempo/Rate Manipulation"
    DESCRIPTION = "Bend SoX effect node for chaining. dbg-text STRING: 'bend params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_bend=True, bend_frame_rate=25, bend_over_sample=16, bend_start_time=0.0,
                bend_cents=0.0, bend_end_time=0.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["bend"]
        if bend_frame_rate != 25:
            effect_params += ["-f", str(bend_frame_rate)]
        if bend_over_sample != 16:
            effect_params += ["-o", str(bend_over_sample)]
        effect_params += [str(bend_start_time), str(bend_cents), str(bend_end_time)]
        debug_str = shlex.join(effect_params)
        if enable_bend:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxChorusNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_chorus": ("BOOLEAN", {"default": True,
                                              "tooltip": """chorus gain-in gain-out delay decay speed depth [ -s | -t ]"""}),
                "chorus_gain_in": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_gain_out": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_delay_1": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 2000.0, "step": 1.0}),
                "chorus_decay_1": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_speed_1": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 10.0, "step": 0.01}),
                "chorus_depth_1": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "chorus_shape_1": (["sin", "tri"], {"default": "sin"}),
                "chorus_delay_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2000.0, "step": 1.0}),
                "chorus_decay_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_speed_2": ("FLOAT", {"default": 0.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "chorus_depth_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "chorus_shape_2": (["sin", "tri"], {"default": "sin"}),
                "chorus_delay_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2000.0, "step": 1.0}),
                "chorus_decay_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_speed_3": ("FLOAT", {"default": 0.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "chorus_depth_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "chorus_shape_3": (["sin", "tri"], {"default": "sin"}),
                "chorus_delay_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2000.0, "step": 1.0}),
                "chorus_decay_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_speed_4": ("FLOAT", {"default": 0.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "chorus_depth_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "chorus_shape_4": (["sin", "tri"], {"default": "sin"}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Reverb/Delay/Echo Effects"
    DESCRIPTION = "Chorus SoX effect node for chaining. dbg-text STRING: 'chorus params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_chorus=True, chorus_gain_in=0.5, chorus_gain_out=0.5,
                chorus_delay_1=40.0, chorus_decay_1=0.8, chorus_speed_1=0.25, chorus_depth_1=2.0, chorus_shape_1="sin",
                chorus_delay_2=0.0, chorus_decay_2=0.0, chorus_speed_2=0.0, chorus_depth_2=0.0, chorus_shape_2="sin",
                chorus_delay_3=0.0, chorus_decay_3=0.0, chorus_speed_3=0.0, chorus_depth_3=0.0, chorus_shape_3="sin",
                chorus_delay_4=0.0, chorus_decay_4=0.0, chorus_speed_4=0.0, chorus_depth_4=0.0, chorus_shape_4="sin",
                prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["chorus", str(chorus_gain_in), str(chorus_gain_out)]
        taps = []
        for delay, decay, speed, depth, shape in [
            (chorus_delay_1, chorus_decay_1, chorus_speed_1, chorus_depth_1, chorus_shape_1),
            (chorus_delay_2, chorus_decay_2, chorus_speed_2, chorus_depth_2, chorus_shape_2),
            (chorus_delay_3, chorus_decay_3, chorus_speed_3, chorus_depth_3, chorus_shape_3),
            (chorus_delay_4, chorus_decay_4, chorus_speed_4, chorus_depth_4, chorus_shape_4),
        ]:
            if decay > 0.0:
                shape_str = "-s" if shape == "sin" else "-t"
                taps.extend([str(delay), str(decay), str(speed), str(depth), shape_str])
        effect_params += taps
        debug_str = shlex.join(effect_params)
        if enable_chorus:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxCompandNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_compand": ("BOOLEAN", {"default": True, "tooltip": """compand attack1,decay1{,attack2,decay2} [soft-knee-dB:]in-dB1[,out-dB1]{,in-dB2,out-dB2} [gain [initial-volume-dB [delay]]]
        where {} means optional and repeatable and [] means optional.
        dB values are floating point or -inf'; times are in seconds."""}),
                "compand_attack_1": ("FLOAT", {"default": 0.3, "min": -120.0, "max": 20.0, "step": 0.01}),
                "compand_decay_1": ("FLOAT", {"default": 1.0, "min": -120.0, "max": 20.0, "step": 0.01}),
                "compand_enable_ad_2": ("BOOLEAN", {"default": False}),
                "compand_attack_2": ("FLOAT", {"default": 0.0, "min": -120.0, "max": 20.0, "step": 0.01}),
                "compand_decay_2": ("FLOAT", {"default": 0.0, "min": -120.0, "max": 20.0, "step": 0.01}),
                "compand_enable_ad_3": ("BOOLEAN", {"default": False}),
                "compand_attack_3": ("FLOAT", {"default": 0.0, "min": -120.0, "max": 20.0, "step": 0.01}),
                "compand_decay_3": ("FLOAT", {"default": 0.0, "min": -120.0, "max": 20.0, "step": 0.01}),
                "compand_enable_soft_knee": ("BOOLEAN", {"default": True}),
                "compand_soft_knee": ("FLOAT", {"default": 6.0, "min": -120.0, "max": 20.0, "step": 0.1}),
                "compand_in_db_1": ("FLOAT", {"default": -70.0, "min": -120.0, "max": 20.0, "step": 1.0}),
                "compand_enable_out_db_1": ("BOOLEAN", {"default": True}),
                "compand_out_db_1": ("FLOAT", {"default": -60.0, "min": -120.0, "max": 20.0, "step": 1.0}),
                "compand_enable_in_out_2": ("BOOLEAN", {"default": True}),
                "compand_in_db_2": ("FLOAT", {"default": -20.0, "min": -120.0, "max": 20.0, "step": 1.0}),
                "compand_out_db_2": ("FLOAT", {"default": -5.0, "min": -120.0, "max": 20.0, "step": 1.0}),
                "compand_enable_gain": ("BOOLEAN", {"default": False}),
                "compand_gain": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "compand_enable_init_vol": ("BOOLEAN", {"default": False}),
                "compand_init_vol": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "compand_enable_delay": ("BOOLEAN", {"default": False}),
                "compand_delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Dynamics/Volume/Compression"
    DESCRIPTION = "Compand SoX effect node for chaining. dbg-text STRING: 'compand params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_compand=True,
                compand_attack_1=0.3, compand_decay_1=1.0,
                compand_enable_ad_2=False,
                compand_attack_2=0.0, compand_decay_2=0.0,
                compand_enable_ad_3=False,
                compand_attack_3=0.0, compand_decay_3=0.0,
                compand_enable_soft_knee=True,
                compand_soft_knee=6.0,
                compand_in_db_1=-70.0,
                compand_enable_out_db_1=True,
                compand_out_db_1=-60.0,
                compand_enable_in_out_2=True,
                compand_in_db_2=-20.0,
                compand_out_db_2=-5.0,
                compand_enable_gain=False,
                compand_gain=0.0,
                compand_enable_init_vol=False,
                compand_init_vol=0.0,
                compand_enable_delay=False,
                compand_delay=0.0,
                prev_params=None):
        # Clamp time parameters to >= 0.0 for valid SoX args
        compand_attack_1 = max(0.0, compand_attack_1)
        compand_decay_1 = max(0.0, compand_decay_1)
        compand_attack_2 = max(0.0, compand_attack_2)
        compand_decay_2 = max(0.0, compand_decay_2)
        compand_attack_3 = max(0.0, compand_attack_3)
        compand_decay_3 = max(0.0, compand_decay_3)
        compand_soft_knee = max(0.0, compand_soft_knee)
        compand_delay = max(0.0, compand_delay)
        current_params = prev_params["sox_params"] if prev_params is not None else []
        debug_str = "Compand disabled"
        if enable_compand:
            attack_decay_parts = [f"{compand_attack_1},{compand_decay_1}"]
            if compand_enable_ad_2:
                attack_decay_parts.append(f"{compand_attack_2},{compand_decay_2}")
            if compand_enable_ad_3:
                attack_decay_parts.append(f"{compand_attack_3},{compand_decay_3}")
            attack_decay = " ".join(attack_decay_parts)
            transfer_parts = [str(compand_in_db_1)]
            if compand_enable_out_db_1:
                transfer_parts[0] += f",{compand_out_db_1}"
            if compand_enable_in_out_2:
                transfer_parts.append(f"{compand_in_db_2},{compand_out_db_2}")
            transfer_str = " ".join(transfer_parts)
            if not transfer_parts:
                debug_str = "Compand skipped: no transfer pairs defined"
            else:
                if compand_enable_soft_knee:
                    knee_str = str(compand_soft_knee)
                    compand_str = f"{attack_decay} {knee_str}:{transfer_str}"
                else:
                    compand_str = f"{attack_decay} {transfer_str}"
                tail_parts = []
                if compand_enable_gain:
                    tail_parts.append(str(compand_gain))
                    if compand_enable_init_vol:
                        tail_parts.append(str(compand_init_vol))
                    if compand_enable_delay:
                        tail_parts.append(str(compand_delay))
                if tail_parts:
                    compand_str += " " + " ".join(tail_parts)
                compand_str = compand_str.strip()
                debug_str = compand_str
                effect_params = ["compand"] + shlex.split(compand_str)
                current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)
import subprocess
import tempfile
import os
import shlex
import torch
import torchaudio
import numpy as np
import uuid
import re
import shutil
from PIL import Image
from .sox_node_utils import SoxNodeUtils as sxu


class SoxEchoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_echo": ("BOOLEAN", {"default": True,
                                            "tooltip": """echo gain-in gain-out delay decay [ delay decay ... ]"""}),
                "echo_gain_in": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echo_gain_out": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echo_delay_1": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "echo_decay_1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echo_delay_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "echo_decay_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echo_delay_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "echo_decay_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echo_delay_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "echo_decay_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Reverb/Delay/Echo Effects"
    DESCRIPTION = "Echo SoX effect node for chaining. dbg-text STRING: 'echo gain-in gain-out [delay decay ...]' always (pre-extend, survives disable)."

    def process(self, audio, enable_echo=True, echo_gain_in=0.8, echo_gain_out=0.9,
                echo_delay_1=1000.0, echo_decay_1=0.5,
                echo_delay_2=0.0, echo_decay_2=0.0,
                echo_delay_3=0.0, echo_decay_3=0.0,
                echo_delay_4=0.0, echo_decay_4=0.0,
                prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        # Compute debug always
        taps = []
        for delay, decay in [(echo_delay_1, echo_decay_1),
                             (echo_delay_2, echo_decay_2),
                             (echo_delay_3, echo_decay_3),
                             (echo_delay_4, echo_decay_4)]:
            if decay > 0.0:
                taps.extend([str(delay), str(decay)])
        debug_str = shlex.join(["echo", str(echo_gain_in), str(echo_gain_out)] + taps)
        if enable_echo:
            debug_str = "** Enabled **\n" + debug_str
            if taps:
                current_params.extend(["echo", str(echo_gain_in), str(echo_gain_out)] + taps)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxEchosNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_echos": ("BOOLEAN", {"default": True,
                                             "tooltip": """echos gain-in gain-out delay decay [ delay decay ... ]"""}),
                "echos_gain_in": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echos_gain_out": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echos_delay_1": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "echos_decay_1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echos_delay_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "echos_decay_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echos_delay_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "echos_decay_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echos_delay_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "echos_decay_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Reverb/Delay/Echo Effects"
    DESCRIPTION = "Echos SoX effect node for chaining. dbg-text STRING: 'echos params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_echos=True, echos_gain_in=0.8, echos_gain_out=0.9,
                echos_delay_1=1000.0, echos_decay_1=0.5,
                echos_delay_2=0.0, echos_decay_2=0.0,
                echos_delay_3=0.0, echos_decay_3=0.0,
                echos_delay_4=0.0, echos_decay_4=0.0,
                prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        taps = []
        for delay, decay in [(echos_delay_1, echos_decay_1),
                             (echos_delay_2, echos_decay_2),
                             (echos_delay_3, echos_decay_3),
                             (echos_delay_4, echos_decay_4)]:
            if decay > 0.0:
                taps.extend([str(delay), str(decay)])
        effect_params = ["echos", str(echos_gain_in), str(echos_gain_out)] + taps
        debug_str = shlex.join(effect_params)
        if enable_echos and taps:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        elif enable_echos:
            debug_str = "** Enabled **\n" + debug_str
        return (audio, {"sox_params": current_params}, debug_str)


class SoxEqualizerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_equalizer": ("BOOLEAN",
                                     {"default": True, "tooltip": """equalizer frequency width[q|o|h|k] gain"""}),
                "equalizer_frequency": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "equalizer_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "equalizer_gain": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization/Filtering/Tone Shaping"
    DESCRIPTION = "Equalizer SoX effect node for chaining."

    def process(self, audio, enable_equalizer=True, equalizer_frequency=1000.0, equalizer_width=1.0, equalizer_gain=0.0,
                prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_equalizer:
            effect_params = ["equalizer", str(equalizer_frequency), str(equalizer_width) + "q", str(equalizer_gain)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxFlangerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_flanger": ("BOOLEAN", {"default": True, "tooltip": """flanger [delay depth regen width speed shape phase interp]
                  .
                 /|regen
                / |
            +--(  |------------+
            |   \\ |            |   .
           _V_   \\|  _______   |   |\\ width   ___
          |   |   ' |       |  |   | \\       |   |
      +-->| + |---->| DELAY |--+-->|  )----->|   |
      |   |___|     |_______|      | /       |   |
      |           delay : depth    |/        |   |
  In  |                 : interp   '         |   | Out
  --->+               __:__                  | + |--->
      |              |     |speed            |   |
      |              |  ~  |shape            |   |
      |              |_____|phase            |   |
      +------------------------------------->|   |
                                             |___|
       RANGE DEFAULT DESCRIPTION
delay   0 30    0    base delay in milliseconds
depth   0 10    2    added swept delay in milliseconds
regen -95 +95   0    percentage regeneration (delayed signal feedback)
width   0 100   71   percentage of delayed signal mixed with original
speed  0.1 10  0.5   sweeps per second (Hz)
shape    --    sin   swept wave shape: sine|triangle
phase   0 100   25   swept wave percentage phase-shift for multi-channel
                     (e.g. stereo) flange; 0 = 100 = same phase on each channel
interp   --    lin   delay-line interpolation: linear|quadratic"""}),
                "flanger_delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "flanger_depth": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "flanger_regen": ("FLOAT", {"default": 0.0, "min": -95.0, "max": 95.0, "step": 1.0}),
                "flanger_width": ("FLOAT", {"default": 71.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "flanger_speed": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "flanger_shape": (["sinusoidal", "triangular"], {"default": "sinusoidal"}),
                "flanger_phase": ("INT", {"default": 25, "min": 0, "max": 100, "step": 1}),
                "flanger_interp": (["linear", "quadratic"], {"default": "linear"}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Reverb/Delay/Echo Effects"
    DESCRIPTION = "Flanger SoX effect node for chaining. dbg-text `string`: 'flanger params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_flanger=True, flanger_delay=0.0, flanger_depth=2.0, flanger_regen=0.0,
                flanger_width=71.0, flanger_speed=0.5, flanger_shape="sinusoidal", flanger_phase=25,
                flanger_interp="linear", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        shape_str = "sin" if flanger_shape == "sinusoidal" else "tri"
        interp_str = "lin" if flanger_interp == "linear" else "quad"
        effect_params = ["flanger",
                         str(flanger_delay), str(flanger_depth), str(flanger_regen),
                         str(flanger_width), str(flanger_speed), shape_str, str(flanger_phase), interp_str
                         ]
        debug_str = shlex.join(effect_params)
        if enable_flanger:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxHighpassNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_highpass": ("BOOLEAN", {"default": True,
                                                "tooltip": """highpass [-1|-2] frequency [width[q|o|h|k](0.707q)]"""}),
                "highpass_poles": ("INT", {"default": 2, "min": 1, "max": 2, "step": 1}),
                "highpass_frequency": ("FLOAT", {"default": 3000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "highpass_width": ("FLOAT", {"default": 0.707, "min": 0.1, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization/Filtering/Tone Shaping"
    DESCRIPTION = "Highpass SoX effect node for chaining."

    def process(self, audio, enable_highpass=True, highpass_poles=2, highpass_frequency=3000.0, highpass_width=0.707,
                prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_highpass:
            poles = "-1" if highpass_poles == 1 else "-2"
            effect_params = ["highpass", poles, str(highpass_frequency), str(highpass_width) + "q"]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxLowpassNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_lowpass": ("BOOLEAN", {"default": True,
                                               "tooltip": """lowpass [-1|-2] frequency [width[q|o|h|k]](0.707q)"""}),
                "lowpass_poles": ("INT", {"default": 2, "min": 1, "max": 2, "step": 1}),
                "lowpass_frequency": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "lowpass_width": ("FLOAT", {"default": 0.707, "min": 0.1, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization/Filtering/Tone Shaping"
    DESCRIPTION = "Lowpass SoX effect node for chaining."

    def process(self, audio, enable_lowpass=True, lowpass_poles=2, lowpass_frequency=1000.0, lowpass_width=0.707,
                prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_lowpass:
            poles = "-1" if lowpass_poles == 1 else "-2"
            effect_params = ["lowpass", poles, str(lowpass_frequency), str(lowpass_width) + "q"]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxOverdriveNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_overdrive": ("BOOLEAN", {"default": True, "tooltip": """overdrive [gain [colour]]"""}),
                "overdrive_gain": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "overdrive_colour": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Reverb/Delay/Echo Effects"
    DESCRIPTION = "Overdrive SoX effect node for chaining."

    def process(self, audio, enable_overdrive=True, overdrive_gain=20.0, overdrive_colour=20.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_overdrive:
            effect_params = ["overdrive", str(overdrive_gain), str(overdrive_colour)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxPhaserNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_phaser": ("BOOLEAN", {"default": True,
                                              "tooltip": """phaser gain-in gain-out delay decay speed [ -s | -t ]"""}),
                "phaser_gain_in": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "phaser_gain_out": ("FLOAT", {"default": 0.74, "min": 0.0, "max": 1.0, "step": 0.01}),
                "phaser_delay": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "phaser_decay": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 0.5, "step": 0.01}),
                "phaser_speed": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1}),
                "phaser_mod": (["sinusoidal", "triangular"], {"default": "sinusoidal"}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Reverb/Delay/Echo Effects"
    DESCRIPTION = "Phaser SoX effect node for chaining."

    def process(self, audio, enable_phaser=True, phaser_gain_in=0.8, phaser_gain_out=0.74, phaser_delay=3.0,
                phaser_decay=0.4, phaser_speed=0.5, phaser_mod="sinusoidal", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_phaser:
            mod = "-s" if phaser_mod == "sinusoidal" else "-t"
            effect_params = ["phaser",
                             str(phaser_gain_in), str(phaser_gain_out),
                             str(phaser_delay), str(phaser_decay),
                             str(phaser_speed), mod
                             ]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxPitchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_pitch": ("BOOLEAN", {"default": True,
                                             "tooltip": """pitch [-q] shift-in-cents [segment-ms [search-ms [overlap-ms]]]"""}),
                "pitch_q": ("BOOLEAN", {"default": False}),
                "pitch_shift": ("INT", {"default": 0, "min": -1200, "max": 1200, "step": 1}),
                "pitch_segment": ("FLOAT", {"default": 82.0, "min": 10.0, "max": 200.0, "step": 1.0}),
                "pitch_search": ("FLOAT", {"default": 14.0, "min": 0.0, "max": 50.0, "step": 1.0}),
                "pitch_overlap": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 50.0, "step": 1.0}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Pitch/Speed/Tempo/Rate Manipulation"
    DESCRIPTION = "Pitch SoX effect node for chaining."

    def process(self, audio, enable_pitch=True, pitch_q=False, pitch_shift=0, pitch_segment=82.0, pitch_search=14.0,
                pitch_overlap=12.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_pitch:
            effect_params = ["pitch"]
            if pitch_q:
                effect_params += ["-q"]
            effect_params += [str(pitch_shift)]
            if pitch_segment != 82.0 or pitch_search != 14.0 or pitch_overlap != 12.0:
                effect_params += [str(pitch_segment), str(pitch_search), str(pitch_overlap)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxReverbNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_reverb": ("BOOLEAN", {"default": True,
                                              "tooltip": """reverb [-w|--wet-only] [reverberance (50%) [HF-damping (50%) [room-scale (100%) [stereo-depth (100%) [pre-delay (0ms) [wet-gain (0dB)]]]]]]"""}),
                "reverb_wet_only": ("BOOLEAN", {"default": False}),
                "reverb_reverberance": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "reverb_hf_damping": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "reverb_room_scale": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "reverb_stereo_depth": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "reverb_pre_delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 500.0, "step": 1.0}),
                "reverb_wet_gain": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Reverb/Delay/Echo Effects"
    DESCRIPTION = "Reverb SoX effect node for chaining."

    def process(self, audio, enable_reverb=True, reverb_wet_only=False, reverb_reverberance=50.0,
                reverb_hf_damping=50.0, reverb_room_scale=100.0, reverb_stereo_depth=100.0, reverb_pre_delay=0.0,
                reverb_wet_gain=0.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_reverb:
            effect_params = ["reverb"]
            if reverb_wet_only:
                effect_params += ["--wet-only"]
            effect_params += [
                str(reverb_reverberance), str(reverb_hf_damping),
                str(reverb_room_scale), str(reverb_stereo_depth),
                str(reverb_pre_delay), str(reverb_wet_gain)
            ]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxTempoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_tempo": ("BOOLEAN", {"default": True,
                                             "tooltip": """tempo [-q] [-m | -s | -l] factor [segment-ms [search-ms [overlap-ms]]]"""}),
                "tempo_q": ("BOOLEAN", {"default": False}),
                "tempo_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "tempo_segment": ("FLOAT", {"default": 82.0, "min": 10.0, "max": 200.0, "step": 1.0}),
                "tempo_search": ("FLOAT", {"default": 14.0, "min": 0.0, "max": 50.0, "step": 1.0}),
                "tempo_overlap": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 50.0, "step": 1.0}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Pitch/Speed/Tempo/Rate Manipulation"
    DESCRIPTION = "Tempo SoX effect node for chaining."

    def process(self, audio, enable_tempo=True, tempo_q=False, tempo_factor=1.0, tempo_segment=82.0, tempo_search=14.0,
                tempo_overlap=12.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_tempo:
            effect_params = ["tempo"]
            if tempo_q:
                effect_params += ["-q"]
            effect_params += [str(tempo_factor)]
            if tempo_segment != 82.0 or tempo_search != 14.0 or tempo_overlap != 12.0:
                effect_params += [str(tempo_segment), str(tempo_search), str(tempo_overlap)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxTrebleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_treble": ("BOOLEAN", {"default": True,
                                              "tooltip": """treble gain [frequency(3000) [width[s|h|k|q|o]](0.5s)]"""}),
                "treble_gain": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "treble_frequency": ("FLOAT", {"default": 3000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "treble_width": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization/Filtering/Tone Shaping"
    DESCRIPTION = "Treble SoX effect node for chaining. dbg-text `string`: 'treble params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_treble=True, treble_gain=0.0, treble_frequency=3000.0, treble_width=0.5,
                prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["treble", str(treble_gain), str(treble_frequency), str(treble_width)]
        debug_str = shlex.join(effect_params)
        if enable_treble:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxTremoloNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_tremolo": ("BOOLEAN", {"default": True, "tooltip": """tremolo speed_Hz [depth_percent]"""}),
                "tremolo_speed": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "tremolo_depth": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Modulation/Special Effects"
    DESCRIPTION = "Tremolo SoX effect node for chaining."

    def process(self, audio, enable_tremolo=True, tremolo_speed=0.5, tremolo_depth=40.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        effect_params = ["tremolo", str(tremolo_speed), str(tremolo_depth)]
        cmd_str = f"sox voice.wav vibrato.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_tremolo:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        else:
            dbg_text = "tremolo disabled"
        return (audio, {"sox_params": current_params}, dbg_text)

NODE_CLASS_MAPPINGS = {
    "SoxBass": SoxBassNode,
    "SoxBend": SoxBendNode,
    "SoxChorus": SoxChorusNode,
    "SoxCompand": SoxCompandNode,
    "SoxEcho": SoxEchoNode,
    "SoxEchos": SoxEchosNode,
    "SoxEqualizer": SoxEqualizerNode,
    "SoxFlanger": SoxFlangerNode,
    "SoxHighpass": SoxHighpassNode,
    "SoxLowpass": SoxLowpassNode,
    "SoxOverdrive": SoxOverdriveNode,
    "SoxPhaser": SoxPhaserNode,
    "SoxPitch": SoxPitchNode,
    "SoxReverb": SoxReverbNode,
    "SoxTempo": SoxTempoNode,
    "SoxTreble": SoxTrebleNode,
    "SoxAllpass": SoxAllpassNode,
    "SoxBand": SoxBandNode,
    "SoxBandpass": SoxBandpassNode,
    "SoxBandreject": SoxBandrejectNode,
    "SoxBiquad": SoxBiquadNode,
    "SoxChannels": SoxChannelsNode,
    "SoxContrast": SoxContrastNode,
    "SoxDcshift": SoxDcshiftNode,
    "SoxDeemph": SoxDeemphNode,
    "SoxDelay": SoxDelayNode,
    "SoxDither": SoxDitherNode,
    "SoxDownsample": SoxDownsampleNode,
    "SoxEarwax": SoxEarwaxNode,
    "SoxFade": SoxFadeNode,
    "SoxFir": SoxFirNode,
    "SoxGain": SoxGainNode,
    "SoxHilbert": SoxHilbertNode,
    "SoxLadspa": SoxLadspaNode,
    "SoxLoudness": SoxLoudnessNode,
    "SoxMcompand": SoxMcompandNode,
    "SoxNoiseprof": SoxNoiseprofNode,
    "SoxNoisered": SoxNoiseredNode,
    "SoxNorm": SoxNormNode,
    "SoxOops": SoxOopsNode,
    "SoxPad": SoxPadNode,
    "SoxRate": SoxRateNode,
    "SoxRemix": SoxRemixNode,
    "SoxRepeat": SoxRepeatNode,
    "SoxReverse": SoxReverseNode,
    "SoxRiaa": SoxRiaaNode,
    "SoxSilence": SoxSilenceNode,
    "SoxSinc": SoxSincNode,
    "SoxSpeed": SoxSpeedNode,
    "SoxSplice": SoxSpliceNode,
    "SoxStat": SoxStatNode,
    "SoxStats": SoxStatsNode,
    "SoxStretch": SoxStretchNode,
    "SoxSwap": SoxSwapNode,
    "SoxSynth": SoxSynthNode,
    "SoxTrim": SoxTrimNode,
    "SoxUpsample": SoxUpsampleNode,
    "SoxVad": SoxVadNode,
    "SoxVol": SoxVolNode,
    "SoxTremolo": SoxTremoloNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SoxBass": "SoX Bass",
    "SoxBend": "SoX Bend",
    "SoxChorus": "SoX Chorus",
    "SoxCompand": "SoX Compand",
    "SoxEcho": "SoX Echo",
    "SoxEchos": "SoX Echos",
    "SoxEqualizer": "SoX Equalizer",
    "SoxFlanger": "SoX Flanger",
    "SoxHighpass": "SoX Highpass",
    "SoxLowpass": "SoX Lowpass",
    "SoxOverdrive": "SoX Overdrive",
    "SoxPhaser": "SoX Phaser",
    "SoxPitch": "SoX Pitch",
    "SoxReverb": "SoX Reverb",
    "SoxTempo": "SoX Tempo",
    "SoxTreble": "SoX Treble",
    "SoxAllpass": "SoX Allpass",
    "SoxBand": "SoX Band",
    "SoxBandpass": "SoX Bandpass",
    "SoxBandreject": "SoX Bandreject",
    "SoxBiquad": "SoX Biquad",
    "SoxChannels": "SoX Channels",
    "SoxContrast": "SoX Contrast",
    "SoxDcshift": "SoX Dcshift",
    "SoxDeemph": "SoX Deemph",
    "SoxDelay": "SoX Delay",
    "SoxDither": "SoX Dither",
    "SoxDownsample": "SoX Downsample",
    "SoxEarwax": "SoX Earwax",
    "SoxFade": "SoX Fade",
    "SoxFir": "SoX Fir",
    "SoxGain": "SoX Gain",
    "SoxHilbert": "SoX Hilbert",
    "SoxLadspa": "SoX Ladspa",
    "SoxLoudness": "SoX Loudness",
    "SoxMcompand": "SoX Mcompand",
    "SoxNoiseprof": "SoX Noiseprof",
    "SoxNoisered": "SoX Noisered",
    "SoxNorm": "SoX Norm",
    "SoxOops": "SoX Oops",
    "SoxPad": "SoX Pad",
    "SoxRate": "SoX Rate",
    "SoxRemix": "SoX Remix",
    "SoxRepeat": "SoX Repeat",
    "SoxReverse": "SoX Reverse",
    "SoxRiaa": "SoX Riaa",
    "SoxSilence": "SoX Silence",
    "SoxSinc": "SoX Sinc",
    "SoxUtilSpectrogram": "SoX Spectrogram",
    "SoxSpeed": "SoX Speed",
    "SoxSplice": "SoX Splice",
    "SoxStat": "SoX Stat",
    "SoxStats": "SoX Stats",
    "SoxStretch": "SoX Stretch",
    "SoxSwap": "SoX Swap",
    "SoxSynth": "SoX Synth",
    "SoxTrim": "SoX Trim",
    "SoxUpsample": "SoX Upsample",
    "SoxVad": "SoX Vad",
    "SoxVol": "SoX Vol",
    "SoxTremolo": "SoX Tremolo",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
