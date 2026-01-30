import subprocess
import tempfile
import os
import shlex
import torch
import torchaudio
import numpy as np
import uuid
import re
from PIL import Image

class SoxApplyEffectsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_apply": ("BOOLEAN", {"default": True, "tooltip": "Enable application of SoX effects"}),
                "params": ("SOX_PARAMS",),
            },
        }
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "dbg-text")
    FUNCTION = "apply_effects"
    CATEGORY = "audio/SoX/Utilities"
    DESCRIPTION = "Applies the chained SoX effects parameters to the input audio. dbg-text `string`: full sox command always (pre-execute, survives errors/disable). Wire to PreviewTextNode."

    def apply_effects(self, audio, params, enable_apply=True):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        sox_cmd_params = params.get("sox_params", [])
        cmd_str = "sox input.wav output.wav " + shlex.join(sox_cmd_params) if sox_cmd_params else "No effects applied (audio passed through)."
        
        output_waveforms = []

        for i in range(waveform.shape[0]):
            single_waveform = waveform[i]
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
                torchaudio.save(temp_input.name, single_waveform, sample_rate)
                input_path = temp_input.name

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
                output_path = temp_output.name

            output_waveforms.append(single_waveform)
            
            if enable_apply and sox_cmd_params:
                cmd = ['sox', input_path, output_path] + sox_cmd_params
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    out_waveform, _ = torchaudio.load(output_path)
                    output_waveforms[-1] = out_waveform
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"SoX failed: {e.stderr}")
            
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)

        max_samples = max(w.shape[-1] for w in output_waveforms)
        padded_waveforms = []
        for w in output_waveforms:
            padding = max_samples - w.shape[-1]
            if padding > 0:
                w = torch.nn.functional.pad(w, (0, padding))
            padded_waveforms.append(w)
            
        stacked = torch.stack(padded_waveforms)
        
        return ({"waveform": stacked, "sample_rate": sample_rate}, cmd_str)

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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
    DESCRIPTION = "Band SoX effect node for chaining. dbg-text STRING: 'band params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_band=True, band_narrow=False, band_center=1000.0, band_width=100.0, prev_params=None):
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
                "enable_bandpass": ("BOOLEAN", {"default": True, "tooltip": "bandpass [-c center] [width[h|k|q|o]] freq"}),
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
    CATEGORY = "audio/SoX/Effects"
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
                "enable_bandreject": ("BOOLEAN", {"default": True, "tooltip": "bandreject [-c center] [width[h|k|q|o]] freq"}),
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
    CATEGORY = "audio/SoX/Effects"
    DESCRIPTION = "Bandreject SoX effect node for chaining. dbg-text STRING: 'bandreject params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_bandreject=True, bandreject_frequency=1000.0, bandreject_width=1.0, prev_params=None):
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
    CATEGORY = "audio/SoX/Effects"
    DESCRIPTION = "Biquad SoX effect node for chaining. dbg-text STRING: 'biquad params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_biquad=True, biquad_frequency=1000.0, biquad_gain=0.0, biquad_q=1.0, biquad_norm=1, prev_params=None):
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
    DESCRIPTION = "Fade SoX effect node for chaining. dbg-text `STRING`: 'fade params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_fade=True, fade_type="h", fade_in_length=0.5, fade_out_length=0.5, prev_params=None):
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
                "enable_noisered": ("BOOLEAN", {"default": True, "tooltip": "noisered [noise.prof] [amount [precision]]"}),
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
    CATEGORY = "audio/SoX/Effects"
    DESCRIPTION = "Noisered SoX effect node for chaining."

    def process(self, audio, enable_noisered=True, noisered_profile="", noisered_amount=0.21, noisered_precision=4, prev_params=None):
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
    DESCRIPTION = "Sinc SoX effect node for chaining."

    def process(self, audio, enable_sinc=True, sinc_frequency=8000.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_sinc:
            effect_params = ["sinc", str(sinc_frequency)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})

class SoxSpectrogramNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_spectrogram": ("BOOLEAN", {"default": True, "tooltip": "Enable spectrogram generation"}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
                "enable_x_pixels": ("BOOLEAN", {"default": True, "tooltip": "Enable -X option"}),
                "x_pixels": ("INT", {"default": 800, "min": 100, "max": 200000, "step": 10, "tooltip": "X-axis pixels (100-200000, def 800)"}),
                "enable_y_pixels": ("BOOLEAN", {"default": True, "tooltip": "Enable -Y option"}),
                "y_pixels": ("INT", {"default": 257, "min": 50, "max": 2000, "step": 1, "tooltip": "Y per channel (def 257=2^8+1)"}),
                "enable_Y_height": ("BOOLEAN", {"default": True, "tooltip": "Enable Y height resize"}),
                "Y_height": ("INT", {"default": 550, "min": 100, "max": 2000, "tooltip": "Total Y height (def 550)"}),
                "enable_z_range": ("BOOLEAN", {"default": True, "tooltip": "Enable -z option"}),
                "z_range": ("INT", {"default": 120, "min": 20, "max": 180, "tooltip": "Z dB range (def 120)"}),
                "enable_q_quant": ("BOOLEAN", {"default": True, "tooltip": "Enable -q option"}),
                "q_quant": ("INT", {"default": 249, "min": 2, "max": 256, "tooltip": "Colors (def 249)"}),
                "monochrome": ("BOOLEAN", {"default": False, "tooltip": "-m"}),
                "high_color": ("BOOLEAN", {"default": False, "tooltip": "-h"}),
                "light_bg": ("BOOLEAN", {"default": False, "tooltip": "-l"}),
                "no_axis": ("BOOLEAN", {"default": False, "tooltip": "-a"}),
                "raw_spec": ("BOOLEAN", {"default": False, "tooltip": "-r"}),
                "slack": ("BOOLEAN", {"default": False, "tooltip": "-s"}),
                "enable_window_type": ("BOOLEAN", {"default": True, "tooltip": "Enable -w window option"}),
                "window_type": (["hann", "hamming", "bartlett", "rectangular", "kaiser", "dolph"], {"default": "hann", "tooltip": "-w"}),
                "enable_window_adj": ("BOOLEAN", {"default": True, "tooltip": "Enable -W option"}),
                "window_adj": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1, "tooltip": "-W"}),
                "title_text": ("STRING", {"default": "", "tooltip": "-t"}),
                "comment_text": ("STRING", {"default": "", "tooltip": "-c"}),
                "png_prefix": ("STRING", {"default": "", "tooltip": "-o PNG filename prefix (saves to cwd with batch index if non-empty)"}),
            }
        }
    RETURN_TYPES = ("AUDIO", "IMAGE", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "image", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Utilities"
    DESCRIPTION = "Spectrogram node generates image from audio. dbg-text `string`: 'parameters: <spectrogram options>\\n\\ncommand: <simulated sox cmds>' always (pre-execute, survives errors/disabled)."

    def process(
        self,
        audio,
        enable_spectrogram=True,
        prev_params=None,
        enable_x_pixels=True,
        x_pixels=800,
        enable_y_pixels=True,
        y_pixels=257,
        enable_Y_height=True,
        Y_height=550,
        enable_z_range=True,
        z_range=120,
        enable_q_quant=True,
        q_quant=249,
        monochrome=False,
        high_color=False,
        light_bg=False,
        no_axis=False,
        raw_spec=False,
        slack=False,
        enable_window_type=True,
        window_type="hann",
        enable_window_adj=True,
        window_adj=0.0,
        title_text="",
        comment_text="",
        png_prefix="",
    ):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        sr = audio["sample_rate"]
        waveform = audio["waveform"]
        B = waveform.shape[0]

        # Build options
        options = []
        if enable_x_pixels:
            options += ["-x", str(x_pixels)]
        if enable_y_pixels:
            options += ["-y", str(y_pixels)]
        if enable_z_range:
            options += ["-z", str(z_range)]
        if enable_q_quant:
            options += ["-q", str(q_quant)]
        if monochrome:
            options += ["-m"]
        if high_color:
            options += ["-h"]
        if light_bg:
            options += ["-l"]
        if no_axis:
            options += ["-a"]
        if raw_spec:
            options += ["-r"]
        if slack:
            options += ["-s"]
        if enable_window_type:
            options += ["-w", window_type]
        if enable_window_adj:
            options += ["-W", str(window_adj)]
        if enable_Y_height:
            options += ["-Y", str(Y_height)]
        if title_text.strip():
            options += ["-t", title_text]
        if comment_text.strip():
            options += ["-c", comment_text]
        debug_str = shlex.join(["spectrogram"] + options)
        
        # Simulate potential sox commands for dbg-text (always)
        cmds = []
        for i in range(B):
            input_path = f"/tmp/input_{i:03d}.wav"
            proc_path = f"/tmp/processed_{i:03d}.wav"
            out_path = f"/tmp/output_{i:03d}.wav"
            png_path = f"{png_prefix.strip()}_{i:03d}.png" if png_prefix.strip() else f"temp_{i:03d}.png"
            if current_params:
                cmds.append(shlex.join(["sox", input_path, proc_path] + current_params))
            spec_input = proc_path if current_params else input_path
            cmds.append(shlex.join(["sox", spec_input, out_path, "spectrogram"] + options + ["-o", png_path]))
        cmd_str = "\n".join(cmds) if cmds else "No sox commands executed."
        dbg_text = f"parameters: {debug_str}\n\ncommand: {cmd_str}"
        
        # Prepare audio_out and image_out


        # Prepare audio_out and image_out
        if not enable_spectrogram:
            audio_out = audio
            H = Y_height
            W = x_pixels
            image_out = torch.zeros((B, H, W, 3), dtype=torch.float32)
        else:
            # 1. Apply prev_params to get processed_waveform
            processed_waveforms = []
            for i in range(B):
                single_waveform = waveform[i]
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
                    input_path = temp_input.name
                    torchaudio.save(input_path, single_waveform, sr)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
                    output_path = temp_output.name
                sox_cmd_params = current_params
                if sox_cmd_params:
                    cmd = ["sox", input_path, output_path] + sox_cmd_params
                    try:
                        subprocess.run(cmd, check=True, capture_output=True, text=True)
                        out_w, _ = torchaudio.load(output_path)
                        processed_waveforms.append(out_w.squeeze(0) if out_w.dim() > 1 else out_w)
                    except subprocess.CalledProcessError as e:
                        os.remove(input_path)
                        os.remove(output_path)
                        raise RuntimeError(f"SoX prev_params failed: {e.stderr}")
                else:
                    processed_waveforms.append(single_waveform)
                os.remove(input_path)
                if os.path.exists(output_path):
                    os.remove(output_path)
            max_samples = max(w.shape[-1] for w in processed_waveforms)
            padded_processed = [torch.nn.functional.pad(w, (0, max_samples - w.shape[-1])) for w in processed_waveforms]
            processed_waveform = torch.stack(padded_processed)

            # 2. Apply spectrogram to processed
            output_waveforms = []
            image_tensors = []
            for i in range(B):
                single_waveform = processed_waveform[i]
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
                    input_path = temp_input.name
                    torchaudio.save(input_path, single_waveform, sr)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
                    output_path = temp_output.name
                png_temp = True
                if png_prefix.strip():
                    png_path = f"{png_prefix.strip()}_{i:03d}.png"
                    png_temp = False
                else:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_png:
                        png_path = temp_png.name
                cmd = ["sox", input_path, output_path, "spectrogram"] + options + ["-o", png_path]
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    out_w, _ = torchaudio.load(output_path)
                    output_waveforms.append(out_w.squeeze(0) if out_w.dim() > 1 else out_w)
                    im = Image.open(png_path)
                    if enable_Y_height:
                        im = im.resize((x_pixels, Y_height), Image.Resampling.LANCZOS)
                    im = im.convert("RGB")
                    arr = np.array(im).astype(np.float32) / 255.0
                    img_t = torch.from_numpy(arr)
                    image_tensors.append(img_t)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Spectrogram sox failed: {e.stderr}")
                finally:
                    for p in [input_path, output_path]:
                        if os.path.exists(p):
                            os.remove(p)
                    if png_temp and os.path.exists(png_path):
                        os.remove(png_path)
            max_samples_out = max(w.shape[-1] for w in output_waveforms)
            padded_out = [torch.nn.functional.pad(w, (0, max_samples_out - w.shape[-1])) for w in output_waveforms]
            audio_out_waveform = torch.stack(padded_out)
            audio_out = {"waveform": audio_out_waveform, "sample_rate": sr}
            image_out = torch.stack(image_tensors)

        return (audio_out, image_out, {"sox_params": current_params}, dbg_text)

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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
    CATEGORY = "audio/SoX/Effects"
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
                "enable_vad": ("BOOLEAN", {"default": True, "tooltip": "Enable VAD (Voice Activity Detection) SoX effect: Trims silence before/after detected speech/audio activity. Usage: Chain early in workflow → SoxApplyEffectsNode for clean recordings (podcasts, vocals). Pairs with Vol for balance."}),
                "vad_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "VAD threshold (0.0-1.0): Energy level above which audio is considered 'voice'; trims leading/trailing silence. Higher values trim more aggressively."}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects"
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
                "enable_vol": ("BOOLEAN", {"default": True, "tooltip": "Enable Vol (Volume) SoX effect: Adjusts gain by dB. Usage: Chain for level matching → SoxApplyEffectsNode. Use post-VAD, pre-mix to prevent clipping."}),
                "vol_gain": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 60.0, "step": 0.1, "tooltip": "Volume gain in dB: Positive boosts amplitude, negative attenuates. 0dB=unity. Use for per-track balance pre-mix/effects."}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects"
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
                "enable_bass": ("BOOLEAN", {"default": True, "tooltip": """bass gain [frequency(100) [width[s|h|k|q|o]](0.5s)]"""} ),
                "bass_gain": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "bass_frequency": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "bass_width": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01,}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
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
                "enable_bend": ("BOOLEAN", {"default": True, "tooltip": """bend [-f frame-rate(25)] [-o over-sample(16)] {start,cents,end}"""} ),
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
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Bend SoX effect node for chaining. dbg-text STRING: 'bend params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_bend=True, bend_frame_rate=25, bend_over_sample=16, bend_start_time=0.0, bend_cents=0.0, bend_end_time=0.0, prev_params=None):
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
                "enable_chorus": ("BOOLEAN", {"default": True, "tooltip": """chorus gain-in gain-out delay decay speed depth [ -s | -t ]"""} ),
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
    CATEGORY = "audio/SoX/Voice"
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
        dB values are floating point or -inf'; times are in seconds."""} ),
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
    CATEGORY = "audio/SoX/Voice"
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

class SoxEchoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_echo": ("BOOLEAN", {"default": True, "tooltip": """echo gain-in gain-out delay decay [ delay decay ... ]"""} ),
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
    CATEGORY = "audio/SoX/Voice"
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
                "enable_echos": ("BOOLEAN", {"default": True, "tooltip": """echos gain-in gain-out delay decay [ delay decay ... ]"""} ),
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
    CATEGORY = "audio/SoX/Voice"
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
                "enable_equalizer": ("BOOLEAN", {"default": True, "tooltip": """equalizer frequency width[q|o|h|k] gain"""} ),
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
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Equalizer SoX effect node for chaining."

    def process(self, audio, enable_equalizer=True, equalizer_frequency=1000.0, equalizer_width=1.0, equalizer_gain=0.0, prev_params=None):
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
interp   --    lin   delay-line interpolation: linear|quadratic"""} ),
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
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Flanger SoX effect node for chaining. dbg-text `string`: 'flanger params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_flanger=True, flanger_delay=0.0, flanger_depth=2.0, flanger_regen=0.0, flanger_width=71.0, flanger_speed=0.5, flanger_shape="sinusoidal", flanger_phase=25, flanger_interp="linear", prev_params=None):
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
                "enable_highpass": ("BOOLEAN", {"default": True, "tooltip": """highpass [-1|-2] frequency [width[q|o|h|k](0.707q)]"""} ),
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
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Highpass SoX effect node for chaining."

    def process(self, audio, enable_highpass=True, highpass_poles=2, highpass_frequency=3000.0, highpass_width=0.707, prev_params=None):
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
                "enable_lowpass": ("BOOLEAN", {"default": True, "tooltip": """lowpass [-1|-2] frequency [width[q|o|h|k]](0.707q)"""} ),
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
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Lowpass SoX effect node for chaining."

    def process(self, audio, enable_lowpass=True, lowpass_poles=2, lowpass_frequency=1000.0, lowpass_width=0.707, prev_params=None):
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
                "enable_overdrive": ("BOOLEAN", {"default": True, "tooltip": """overdrive [gain [colour]]"""} ),
                "overdrive_gain": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "overdrive_colour": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
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
                "enable_phaser": ("BOOLEAN", {"default": True, "tooltip": """phaser gain-in gain-out delay decay speed [ -s | -t ]"""} ),
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
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Phaser SoX effect node for chaining."

    def process(self, audio, enable_phaser=True, phaser_gain_in=0.8, phaser_gain_out=0.74, phaser_delay=3.0, phaser_decay=0.4, phaser_speed=0.5, phaser_mod="sinusoidal", prev_params=None):
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
                "enable_pitch": ("BOOLEAN", {"default": True, "tooltip": """pitch [-q] shift-in-cents [segment-ms [search-ms [overlap-ms]]]"""} ),
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
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Pitch SoX effect node for chaining."

    def process(self, audio, enable_pitch=True, pitch_q=False, pitch_shift=0, pitch_segment=82.0, pitch_search=14.0, pitch_overlap=12.0, prev_params=None):
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
                "enable_reverb": ("BOOLEAN", {"default": True, "tooltip": """reverb [-w|--wet-only] [reverberance (50%) [HF-damping (50%) [room-scale (100%) [stereo-depth (100%) [pre-delay (0ms) [wet-gain (0dB)]]]]]]"""} ),
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
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Reverb SoX effect node for chaining."

    def process(self, audio, enable_reverb=True, reverb_wet_only=False, reverb_reverberance=50.0, reverb_hf_damping=50.0, reverb_room_scale=100.0, reverb_stereo_depth=100.0, reverb_pre_delay=0.0, reverb_wet_gain=0.0, prev_params=None):
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
                "enable_tempo": ("BOOLEAN", {"default": True, "tooltip": """tempo [-q] [-m | -s | -l] factor [segment-ms [search-ms [overlap-ms]]]"""} ),
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
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Tempo SoX effect node for chaining."

    def process(self, audio, enable_tempo=True, tempo_q=False, tempo_factor=1.0, tempo_segment=82.0, tempo_search=14.0, tempo_overlap=12.0, prev_params=None):
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
                "enable_treble": ("BOOLEAN", {"default": True, "tooltip": """treble gain [frequency(3000) [width[s|h|k|q|o]](0.5s)]"""} ),
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
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Treble SoX effect node for chaining. dbg-text `string`: 'treble params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_treble=True, treble_gain=0.0, treble_frequency=3000.0, treble_width=0.5, prev_params=None):
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
                "enable_tremolo": ("BOOLEAN", {"default": True, "tooltip": """tremolo speed_Hz [depth_percent]"""} ),
                "tremolo_speed": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "tremolo_depth": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            },
            "optional": {
                "prev_params": ("SOX_PARAMS",),
            }
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Tremolo SoX effect node for chaining."

    def process(self, audio, enable_tremolo=True, tremolo_speed=0.5, tremolo_depth=40.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params is not None else []
        if enable_tremolo:
            effect_params = ["tremolo", str(tremolo_speed), str(tremolo_depth)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})



class SoxUtilMultiInputText10:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(10):
            optional[f"text-in-{i}"] = ("STRING", {"multiline": True, "default": ""})
        return {"optional": optional}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text-out",)
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Utilities"
    DESCRIPTION = """Utility node that accepts 10 optional `text-in-<n>` (n=0-9) string inputs and outputs a single `text-out` string. For each non-empty input, prepends "=== text-in-<n> ===" followed by the text. Sections separated by double newlines."""

    def process(self, **kwargs):
        lines = []
        for i in range(10):
            text = kwargs.get(f"text-in-{i}", "").strip()
            if text:
                lines.append(f"=== text-in-{i} ===\n{text}")
        output = "\n\n".join(lines) if lines else "No text inputs provided."
        return (output,)



class SoxUtilMultiInputText5:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(5):
            optional[f"text-in-{i}"] = ("STRING", {"multiline": True, "default": ""})
        return {"optional": optional}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text-out",)
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Utilities"
    DESCRIPTION = """Utility node that accepts 5 optional `text-in-<n>` (n=0-4) string inputs and outputs a single `text-out` string. For each non-empty input, prepends "=== text-in-<n> ===" followed by the text. Sections separated by double newlines."""

    def process(self, **kwargs):
        lines = []
        for i in range(5):
            text = kwargs.get(f"text-in-{i}", "").strip()
            if text:
                lines.append(f"=== text-in-{i} ===\n{text}")
        output = "\n\n".join(lines) if lines else "No text inputs provided."
        return (output,)
        

class SoxUtilMultiInputAudio5_1:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        optional["resample"] = (["auto", "mono", "stereo"], {"default": "auto", "tooltip": """Channel resample mode:
• auto: if mix mono+stereo inputs, upmix mono→stereo (stereoize_headroom dB drop + stereoize_delay ms widen first); all-mono→mono, all-stereo→stereo.
• mono: force downmix all→mono (mean(dim=1, preserve RMS)).
• stereo: force →stereo (mono→stereoize [-5dB equiv], >2ch→slice first2)."""})
        optional["=== Stereoize Mono → Stereo ==="] = ("STRING", {"default": "", "tooltip": "Stereoization controls for mono→stereo upmix only (headroom + Haas delay widen)."})
        optional["stereoize_delay_ms"] = ("INT", {"default": 0, "min": 0, "max": 20, "step": 1, "tooltip": """Delay ms (0-20) applied **only** on mono→stereo upmix: Creates stereo width via Haas effect (delays right ch by N samples=round(N*sr/1000), left padded end to match len). 10-20ms sweet spot; 0=simple repeat."""})
        optional["stereoize_headroom"] = ("FLOAT", {"default": -3.0, "min": -7.0, "max": 0.0, "step": 0.1, "tooltip": """Headroom drop dB (-7 to 0) **only** on mono→stereo upmix: Attenuates mono before upmix/delay to prevent L/R sum clipping (~+3-6dB uncorrelated gain). Default -3dB ≈ orig -5dB fixed (now adjustable)."""})
        for i in range(5):
            optional[f"in-audio-{i}"] = ("AUDIO",)
        optional["=== Volume & Gain ==="] = ("STRING", {"default": ""})
        optional["  --- Master_Volume ---"] = ("STRING", {"default": ""})
        optional["master_gain_db"] = ("FLOAT", {"default": 0.0, "min": -60.0, "max": 12.0, "step": 0.1, "tooltip": "Global master gain dB applied post-mix to both outputs."})
        optional["=== Track Gain ==="] = ("STRING", {"default": "", "tooltip": "Group label before the per-input gain sliders."})
        for i in range(5):
            optional[f"input_gain_{i}"] = ("FLOAT", {"default": 0.0, "min": -60.0, "max": 12.0, "step": 0.1, "tooltip": f"Gain dB for in-audio-{i} (post-resample, pre-pad/mix)."})
        return {"optional": optional}

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("out-audio", "out-audio")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Utilities"
    DESCRIPTION = """Utility node: 5 optional `in-audio-0..4` → 2x `out-audio` (multi/target_ch mix + stereo).

`resample` (auto/mono/stereo) controls target_ch & per-input ch-resample (mono→stereo: stereoize_headroom dB drop + stereoize_delay ms widen first, pre-sum headroom):

• `auto`: mixed mono+stereo → target_ch=2 (upmix mono→stereoize); all-mono→1, all-stereo→2 (or max≤2).
• `mono`: target_ch=1 (downmix mean(dim=1, RMS preserve)).
• `stereo`: target_ch=2 (mono→stereoize, >2ch slice [:2]).

Stereoize (mono→stereo upmix only):
• `stereoize_headroom` (-7..0 dB): Gain drop pre-upmix (anti-clip sum headroom).
• `stereoize_delay` (0-20 ms): Right ch delay (Haas width); left end-pad to match len.

SR resample→first (post-ch/stereoize); zero-pad shorts→longest; stack→mean(dim=0)→mix multi [1,C,T].
- out-audio[0]: mix [1,C,T]
- out-audio[1]: stereo derive [1,2,T] (dup/slice)

Gains:
- input_gain_0-4: per-input **post-ch/SR-resample/stereoize pre-pad/mix**
- master_gain_db: post-mix *both*

Empty → dummy zero [1,C,1024]@44.1kHz (C per resample; stereoize if stereo; gains→zero)."""

    def process(self, **kwargs):
        master_gain_db = kwargs.get("master_gain_db", 0.0)
        input_gains_db = [kwargs.get(f"input_gain_{i}", 0.0) for i in range(5)]
        input_gains_lin = [10 ** (g / 20.0) for g in input_gains_db]
        master_lin = 10 ** (master_gain_db / 20.0)
        resample_mode = kwargs.get("resample", "auto")
        stereoize_delay = int(kwargs.get("stereoize_delay_ms", 0))
        stereoize_headroom_db = kwargs.get("stereoize_headroom", -3.0)
        active_audios = []
        for i in range(5):
            a = kwargs.get(f"in-audio-{i}")
            if a is not None:
                active_audios.append((i, a))
        if not active_audios:
            sr = 44100
            dtype = torch.float32
            if resample_mode == "mono":
                dummy_target_ch = 1
            elif resample_mode == "stereo":
                dummy_target_ch = 2
            else:  # auto
                dummy_target_ch = 1
            zero_base = torch.zeros((1, 1, 1024), dtype=dtype)
            if dummy_target_ch == 2:
                zero_base *= 10 ** (stereoize_headroom_db / 20.0)
                # Creates delayed stereo zero waveform for headroom
                if stereoize_delay > 0:
                    dummy_sr = 44100
                    delay_samples = int(round(stereoize_delay * dummy_sr / 1000.0))
                    left = torch.nn.functional.pad(zero_base, (0, delay_samples))
                    right = torch.nn.functional.pad(zero_base, (delay_samples, 0))
                    zero_multi = torch.cat([left[:, 0:1, :], right[:, 0:1, :]], dim=1)
                else:
                    zero_multi = zero_base.repeat(1, dummy_target_ch, 1)
            else:
                zero_multi = zero_base
            zero_multi *= master_lin
            if zero_multi.shape[1] == 1:
                zero_stereo = zero_multi.repeat(1, 2, 1)
            else:
                zero_stereo = zero_multi[:, :2, :]
            audio_mono = {"waveform": zero_multi, "sample_rate": sr}
            audio_stereo = {"waveform": zero_stereo, "sample_rate": sr}
            return (audio_mono, audio_stereo)

        first_audio = active_audios[0][1]
        target_sr = first_audio["sample_rate"]
        first_wave = first_audio["waveform"][0:1]
        dtype = first_wave.dtype
        device = first_wave.device
        if resample_mode == "mono":
            target_ch = 1
        elif resample_mode == "stereo":
            target_ch = 2
        else:  # "auto"
            has_mono = any(a["waveform"].shape[1] == 1 for _, a in active_audios)
            has_stereo = any(a["waveform"].shape[1] >= 2 for _, a in active_audios)
            if has_mono and has_stereo:
                target_ch = 2
            else:
                target_ch = 1
                for _, a in active_audios:
                    target_ch = max(target_ch, a["waveform"].shape[1])
                target_ch = min(2, target_ch)
        resampled_multis = []
        for slot_i, a in active_audios:
            w = a["waveform"][0:1]
            sr_i = a["sample_rate"]
            orig_c = w.shape[1]
            if orig_c != target_ch:
                if target_ch == 1:
                    w = torch.mean(w, dim=1, keepdim=True)
                elif orig_c == 1:
                    w *= 10 ** (stereoize_headroom_db / 20.0)
                    if stereoize_delay > 0 and target_ch == 2:
                        delay_samples = int(round(stereoize_delay * target_sr / 1000.0))
                        left = torch.nn.functional.pad(w, (0, delay_samples))
                        right = torch.nn.functional.pad(w, (delay_samples, 0))
                        w = torch.cat([left[:, 0:1, :], right[:, 0:1, :]], dim=1)
                    else:
                        w = w.repeat(1, target_ch, 1)
                else:  # slice if orig_c > target_ch
                    w = w[:, :target_ch, :]
            if sr_i != target_sr:
                resampler = torchaudio.transforms.Resample(sr_i, target_sr)
                w = resampler(w)
            w = w.to(device=device, dtype=dtype)
            resampled_multis.append((slot_i, w))

        max_len = max(w.shape[2] for _, w in resampled_multis)
        padded_ws = []
        for slot_i, w in resampled_multis:
            pad_len = max_len - w.shape[2]
            if pad_len > 0:
                w_padded = torch.nn.functional.pad(w, (0, pad_len))
            else:
                w_padded = w
            w_padded *= input_gains_lin[slot_i]
            padded_ws.append(w_padded)

        stacked = torch.stack(padded_ws, dim=0)
        mixed_multi = torch.mean(stacked, dim=0)
        mixed_mono = mixed_multi

        # Stereo: duplicate if mono
        if mixed_mono.shape[1] == 1:
            mixed_stereo = mixed_mono.repeat(1, 2, 1)
        else:
            mixed_stereo = mixed_mono[:, :2, :]

        mixed_mono *= master_lin
        mixed_stereo *= master_lin

        audio1 = {"waveform": mixed_mono, "sample_rate": target_sr}
        audio2 = {"waveform": mixed_stereo, "sample_rate": target_sr}
        return (audio1, audio2)

class SoxUtilMuxAudio5_1:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "enable_mux": ("BOOLEAN", {"default": True}),
            "mute_all": ("BOOLEAN", {"default": False}),
            "solo_channel": (["none", "1", "2", "3", "4", "5"], {"default": "none"}),
            "═══ REMIX OPTIONS ═══": ("STRING", {"default": "", "tooltip": "Resample mode and stereoize group for input channel handling."}),
            "resample": (["auto", "mono", "stereo"], {"default": "auto", "tooltip": """Channel resample mode:
• auto: if mix of mono+stereo inputs, upmix mono→stereo (w/ stereoize); uniform mono→mono, stereo→stereo (≤2ch).
• mono: downmix all to mono (mean dim=1).
• stereo: upmix to stereo (mono→stereoize headroom+delay, >2ch slice first2). 
Applied pre-SR resample, post-input; feeds auto_vol_balance."""}),
            "=== Stereoize Mono → Stereo ===": ("STRING", {"default": "", "tooltip": "Stereoization controls: headroom drop + Haas delay for mono→stereo upmix only."}),
            "stereoize_delay_ms": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1, "tooltip": """Haas delay ms (0-20): Right ch delayed by ~N samples (sr/1000); left end-pad match len. 0=repeat both ch; 10-20ms stereo width."""}),
            "stereoize_headroom": ("FLOAT", {"default": -3.0, "min": -7.0, "max": 0.0, "step": 0.1, "tooltip": """Pre-upmix gain drop dB (-7..0): Attenuate mono →stereo sum headroom (~+6dB incoherent). -3dB default safe."""}),
            "═══ MIX MODE ═══": ("STRING", {"default": "", "tooltip": "Mix mode and balance group"}),
            "mix_mode": (["linear_sum", "rms_power", "average", "max_amplitude"], {"default": "linear_sum"}),
            "mix_preset": (["none", "equal", "vocals_lead", "bass_heavy", "wide_stereo"], {"default": "none"}),
            "−−− AUTO VOL BALANCE −−−": ("STRING", {"default": "", "tooltip": "Auto volume balance group"}),
            "auto_vol_balance": ("BOOLEAN", {"default": False, "tooltip": "Auto-adjust `vol_n` dB to target metric (torch only, post-resample, after presets)"}),
            "target_rms_db": ("FLOAT", {"default": -18.0, "min": -60.0, "max": -6.0, "step": 0.5, "tooltip": "RMS target dB for rms_power mode"}),
            "target_peak_db": ("FLOAT", {"default": -6.0, "min": -20.0, "max": 0.0, "step": 0.5, "tooltip": "Peak target dBFS for max_amplitude mode (headroom)"}),
            "pad_mode": (["zero_fill", "loop_repeat", "fade_trim"], {"default": "zero_fill"}),
            "auto_normalize": ("BOOLEAN", {"default": True, "tooltip": "Post-mix peak normalize to -1dB headroom + clamp <=1.0 (default: on, prevents clipping/distortion)"}),
            "pre_mix_gain_db": ("FLOAT", {"default": -3.0, "min": -12.0, "max": 3.0, "step": 0.1, "tooltip": "Pre-mix gain reduction dB (headroom; negative reduces gain pre-mix/effects to prevent clipping)."}),
            "prev_params": ("SOX_PARAMS",),
            "═══ SAVE OPTIONS ═══": ("STRING", {"default": "", "tooltip": "Save options group"}),
            "enable_save": ("BOOLEAN", {"default": False}),
            "file_prefix": ("STRING", {"default": "output/audio/SoX_Effects", "multiline": False}),
            "save_format": (["wav", "flac", "mp3", "ogg"], {"default": "wav"}),
            "═══ TRACK CHANNELS 1-5 ═══": ("STRING", {"default": "", "tooltip": "enable_audio/vol/mute/in-audio for each channel 1-5"}),
        }
        for i in range(5):
            optional[f"−−−− Track {i+1} −−−−"] = ("STRING", {"default": "", "tooltip": f"Controls for Track [{i+1}]"})
            optional[f"enable_audio_{i+1}"] = ("BOOLEAN", {"default": True})
            optional[f"vol_{i+1}"] = ("FLOAT", {"default": 0.0, "min": -60.0, "max": 12.0, "step": 0.1})
            optional[f"mute_{i+1}"] = ("BOOLEAN", {"default": False})
            optional[f"in-audio-{i+1}"] = ("AUDIO",)
        return {"optional": optional}

    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING", "SOX_PARAMS")
    RETURN_NAMES = ("out-audio-mono", "out-audio-stereo", "mux-settings", "sox_params")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Utilities"
    DESCRIPTION = """AUDIO mux/mixer: 5 optional `in-audio-0..4` AUDIO → mono/stereo AUDIO.

Quick opts: 
   - mix_mode (linear_sum|rms_power|average|max_amplitude) 
   - pad_mode (zero_fill|loop_repeat|fade_trim), auto_normalize (-1dB peak), presets (override vols).
     - zero_fill: Shorter track are padded with silence to match longest track.
     - loop_repeat: Shorter track are repeated to match longest track.
     - fade_trim: Shorter track are faded in/out to match longest track.

Global: 
   - `mute_all`, 
   - `solo_channel` (exclusive)
   - mix_mode `torch` (default): 
     - torch mix: resample to first SR → mono (mean) → pad/vol → mix_mode → clamp/norm → mono/stereo AUDIO.
     - Not active/`enable_mux=off` → 1s silence dummy `[1,1,44100]@44100Hz`.
   - `enable_save` (off), `file_prefix` (`output/audio/SoX_Effects`), 
     - `save_format`: incremental saves `{prefix}_mono/stereo_{0001}.{fmt}` (abs paths logged).

Per-channel 1-5: 
   - `enable_audio_n` (default True)
   - `vol_n` dB (-60/+12, 0dB=unity) `mute_n`

#### Resample & Stereoize
`resample` (auto/mono/stereo) sets target_ch, adjusts per-input pre-SR:
• `auto`: mixed mono+stereo → target_ch=2 (mono→stereoize); all-mono→1, all-stereo→min(2,max_ch)
• `mono`: target_ch=1 (stereo→mean downmix)
• `stereo`: target_ch=2 (mono→stereoize, >2ch→[:2])

**Stereoize** (mono→stereo only):
• `stereoize_headroom` dB: pre-upmix attenuate (anti-clip)
• `stereoize_delay_ms`: Haas right-delay (pad left end); pads interpolate in resample/pad.

Post-adjust → SR resample → auto_vol_balance (if on) → pad/vol/mix.
Dummy: matches resample (stereo→stereoize zero).

#### rms_power Mixing Tips
  `rms_power` prioritizes **perceived loudness** (RMS average: `√(mean(x_i²))` power-conserving, no clip).
  `max_amplitude` prioritizes **peak loudness** (max absolute value).

**Target RMS levels** (rough, per-track pre-mix):
   - Kick/snare: -18 to -12 dB RMS
   - Bass: -20 to -14 dB RMS
   - Vocals lead: -18 to -14 dB; backing: -24 to -18 dB
   - Master bus: -20 to -16 dB RMS (headroom)

**Best practices**:
   - Meter RMS on tracks/buses/master (DAWs: Reaper/Logic built-in; free: Klanghelm VUMT, Youlean Loudness Meter free).
   - Gain stage + compress for dynamics → LUFS (-14/-9 streaming).
   - Great for pop/EDM/rock/podcasts (consistent loudness).

Use `vol_*` dB + presets for balancing; chain `SoxGainNode`/`SoxNormNode` post-mux.

#### Auto Vol Balance (torch only)
   - `auto_vol_balance` toggle: analyzes active channels (post-resample → mono), adds delta dB to `vol_n` (after presets) to hit target.
   - `rms_power`: RMS `target_rms_db` (-18dB def).
   - `max_amplitude`: Peak `target_peak_db` (-6dBFS def)
   - Logs measured/deltas to `mux-settings`.
"""

    def process(self, **kwargs):
        enable_mux = kwargs.get("enable_mux", True)
        mute_all = kwargs.get("mute_all", False)
        vols = [kwargs.get(f"vol_{i+1}", 0.0) for i in range(5)]
        mutes = [kwargs.get(f"mute_{i+1}", False) for i in range(5)]
        solo_channel = kwargs.get("solo_channel", "none")
        solos = [False] * 5
        if solo_channel != "none":
            solos[int(solo_channel) - 1] = True
        enables = [kwargs.get(f"enable_audio_{i+1}", True) for i in range(5)]
        any_solo = solo_channel != "none"
        audios = [kwargs.get(f"in-audio-{i+1}", None) for i in range(5)]
        mix_mode = kwargs.get("mix_mode", "linear_sum")
        pad_mode = kwargs.get("pad_mode", "zero_fill")
        auto_normalize = kwargs.get("auto_normalize", False)
        mix_preset = kwargs.get("mix_preset", "none")
        preset_vols = {
            "equal": [0.0, 0.0, 0.0, 0.0, 0.0],
            "vocals_lead": [3.5, -3.1, -3.1, -6.0, -6.0],
            "bass_heavy": [-2.0, 1.6, 0.0, -0.9, -0.9],
            "wide_stereo": [0.0, 0.0, -2.0, -2.0, 1.6],
        }
        if mix_preset != "none":
            vols = preset_vols[mix_preset]
        linear_vols = [10 ** (v / 20.0) for v in vols]
        prev_params = kwargs.get("prev_params", None)
        current_params = prev_params["sox_params"] if prev_params is not None else []
        enable_save = kwargs.get("enable_save", False)
        file_prefix = kwargs.get("file_prefix", "").strip()
        save_format = kwargs.get("save_format", "wav")
        pre_mix_gain_db = kwargs.get("pre_mix_gain_db", -3.0)
        auto_vol_balance = kwargs.get("auto_vol_balance", False)
        target_rms_db = kwargs.get("target_rms_db", -18.0)
        target_peak_db = kwargs.get("target_peak_db", -6.0)
        resample_mode = kwargs.get("resample", "auto")
        stereoize_delay_ms = int(kwargs.get("stereoize_delay_ms", 0))
        stereoize_headroom_db = kwargs.get("stereoize_headroom", -3.0)
        active_indices = []
        for i in range(5):
            audio = audios[i]
            if audio is not None and audio["waveform"].numel() > 0 and enables[i] and not mute_all and not mutes[i] and (not any_solo or solos[i]):
                active_indices.append(i)
        target_ch = 1
        if active_indices:
            if resample_mode == "mono":
                target_ch = 1
            elif resample_mode == "stereo":
                target_ch = 2
            else:  # auto
                has_mono = False
                has_stereo = False
                max_ch = 1
                for ii in active_indices:
                    ch_i = audios[ii]["waveform"].shape[1]
                    if ch_i == 1:
                        has_mono = True
                    if ch_i >= 2:
                        has_stereo = True
                    max_ch = max(max_ch, ch_i)
                if has_mono and has_stereo:
                    target_ch = 2
                else:
                    target_ch = min(2, max_ch)
        # dbg-text always
        dbg_parts = [
            f"enable_mux: {enable_mux}",
            f"mute_all: {mute_all}",
            f"mix_mode: {mix_mode}",
            f"pad_mode: {pad_mode}",
            f"auto_normalize: {auto_normalize}",
            f"mix_preset: {mix_preset}",
            f"Vols dB: [{', '.join(f'{v:.1f}' for v in vols)}]",
    f"Audio-Enabled: {enables}",
    f"Mutes: {mutes}",
            f"Solo channel: {solo_channel}",
            f"resample_mode: {resample_mode}",
            f"stereoize_delay_ms: {stereoize_delay_ms}, headroom_db: {stereoize_headroom_db:.1f}",
            f"Active indices: {active_indices}",
            f"Target ch: {target_ch}",
        ]
        for i in range(5):
            if audios[i] is not None:
                a = audios[i]
                info = f"sr={a['sample_rate']} C={a['waveform'].shape[1]} T={a['waveform'].shape[2]}"
            else:
                info = "None"
            dbg_parts.append(f"Audio{i+1}: {info}")
        base_dbg = "\n".join(dbg_parts)
        process_details = ""
        enabled_prefix = "** Enabled **\n" if enable_mux else ""
        if not enable_mux or not active_indices:
            sr = 44100
            dtype = torch.float32
            if resample_mode == "mono":
                dummy_target_ch = 1
            elif resample_mode == "stereo":
                dummy_target_ch = 2
            else:
                dummy_target_ch = 1
            zero_base = torch.zeros((1, 1, 44100), dtype=dtype)
            if dummy_target_ch == 2:
                zero_base *= 10 ** (stereoize_headroom_db / 20.0)
                if stereoize_delay_ms > 0:
                    dummy_sr = sr
                    delay_samples = int(round(stereoize_delay_ms * dummy_sr / 1000.0))
                    left = torch.nn.functional.pad(zero_base, (0, delay_samples))
                    right = torch.nn.functional.pad(zero_base, (delay_samples, 0))
                    zero_multi = torch.cat([left[:, 0:1, :], right[:, 0:1, :]], dim=1)
                else:
                    zero_multi = zero_base.repeat(1, dummy_target_ch, 1)
            else:
                zero_multi = zero_base
            if zero_multi.shape[1] == 1:
                zero_stereo = zero_multi.repeat(1, 2, 1)
            else:
                zero_stereo = zero_multi[:, :2, :]
            dummy_audio_mono = {"waveform": zero_multi, "sample_rate": sr}
            dummy_audio_stereo = {"waveform": zero_stereo, "sample_rate": sr}
            if enable_save and file_prefix:
                dir_path = os.path.dirname(os.path.abspath(f"{file_prefix}_mono_dummy.{save_format}")) or '.'
                os.makedirs(dir_path, exist_ok=True)
                pattern_mono = rf"^{re.escape(file_prefix)}_mono_(\d+)\.{re.escape(save_format)}$"
                pattern_stereo = rf"^{re.escape(file_prefix)}_stereo_(\d+)\.{re.escape(save_format)}$"
                nums_mono = []
                nums_stereo = []
                try:
                    for f in os.listdir(dir_path):
                        m = re.match(pattern_mono, f)
                        if m:
                            nums_mono.append(int(m.group(1)))
                        m = re.match(pattern_stereo, f)
                        if m:
                            nums_stereo.append(int(m.group(1)))
                except (OSError, PermissionError):
                    pass
                next_mono = max(nums_mono, default=0) + 1
                next_stereo = max(nums_stereo, default=0) + 1
                mono_fn = f"{file_prefix}_mono_{next_mono:04d}.{save_format}"
                stereo_fn = f"{file_prefix}_stereo_{next_stereo:04d}.{save_format}"
                w_mono = dummy_audio_mono["waveform"].squeeze(0)
                torchaudio.save(mono_fn, w_mono, dummy_audio_mono["sample_rate"], format=save_format)
                full_mono = os.path.abspath(mono_fn)
                w_stereo = dummy_audio_stereo["waveform"].squeeze(0)
                torchaudio.save(stereo_fn, w_stereo, dummy_audio_stereo["sample_rate"], format=save_format)
                full_stereo = os.path.abspath(stereo_fn)
                dbg_parts.append(f"Saved mono: {full_mono}")
                dbg_parts.append(f"Saved stereo: {full_stereo}")
            dbg_text = enabled_prefix + "\n".join(dbg_parts)
            return (dummy_audio_mono, dummy_audio_stereo, dbg_text, {"sox_params": current_params})
        # Get target_sr, dtype, device from first active
        first_i = active_indices[0]
        first_audio = audios[first_i]
        target_sr = first_audio["sample_rate"]
        first_wave = first_audio["waveform"][0:1]
        dtype = first_wave.dtype
        device = first_wave.device
        # Resample all active to multi-ch, upmix if needed, collect
        resampled_multis = []
        for i in active_indices:
            audio_i = audios[i]
            w = audio_i["waveform"][0:1]
            sr_i = audio_i["sample_rate"]
            orig_c = w.shape[1]
            if orig_c != target_ch:
                if target_ch == 1:
                    w = torch.mean(w, dim=1, keepdim=True)
                elif orig_c == 1:
                    w *= 10 ** (stereoize_headroom_db / 20.0)
                    if stereoize_delay_ms > 0 and target_ch == 2:
                        delay_samples = int(round(stereoize_delay_ms * target_sr / 1000.0))
                        left = torch.nn.functional.pad(w, (0, delay_samples))
                        right = torch.nn.functional.pad(w, (delay_samples, 0))
                        w = torch.cat([left[:, 0:1, :], right[:, 0:1, :]], dim=1)
                    else:
                        w = w.repeat(1, target_ch, 1)
                else:
                    w = w[:, :target_ch, :]
            if sr_i != target_sr:
                resampler = torchaudio.transforms.Resample(sr_i, target_sr)
                w = resampler(w)
            w = w.to(device=device, dtype=dtype)
            resampled_multis.append((i, w))
        auto_dbg = ""
        if auto_vol_balance and resampled_multis:
            measured = []
            deltas = []
            metric = None
            target = None
            current_linear_vols = [10 ** (v / 20.0) for v in vols]  # Post-preset snapshot
            if mix_mode == "rms_power":
                metric = "RMS"
                target = target_rms_db
                for j, (i_idx, w_multi) in enumerate(resampled_multis):
                    vol_lin = current_linear_vols[i_idx]
                    post_vol_multi = w_multi * vol_lin
                    rms = torch.sqrt(torch.mean(post_vol_multi ** 2))
                    measured_db = 20 * torch.log10(torch.clamp(rms, min=1e-8)).item()
                    delta_db = target - measured_db
                    vols[i_idx] += delta_db
                    measured.append("{:.1f}".format(measured_db))
                    deltas.append("{:.1f}".format(delta_db))
            elif mix_mode == "max_amplitude":
                metric = "Peak"
                target = target_peak_db
                for j, (i_idx, w_multi) in enumerate(resampled_multis):
                    vol_lin = current_linear_vols[i_idx]
                    post_vol_multi = w_multi * vol_lin
                    peak_val = torch.max(torch.abs(post_vol_multi))
                    measured_db = 20 * torch.log10(torch.clamp(peak_val, min=1e-8)).item()
                    delta_db = target - measured_db
                    vols[i_idx] += delta_db
                    measured.append("{:.1f}".format(measured_db))
                    deltas.append("{:.1f}".format(delta_db))
            if metric is not None:
                linear_vols = [10 ** (v / 20.0) for v in vols]  # Updated after deltas
                auto_dbg = f"Auto Vol Balance: True | {metric} Target: {target:.1f}dB | Measured (post-vol full) {metric} dB: [{', '.join(measured)}] | Deltas dB: [{', '.join(deltas)}]"
        # max_len after resample
        max_len = max(wm.shape[2] for _, wm in resampled_multis)
        # Pad, vol, list
        padded_list = []
        for i, w_multi in resampled_multis:
            vol_i = linear_vols[i]
            orig_len = w_multi.shape[2]
            pad_len = max_len - orig_len
            if pad_len <= 0:
                padded = w_multi[:, :, :max_len]
            else:
                if pad_mode == "zero_fill":
                    padded = torch.nn.functional.pad(w_multi, (0, pad_len))
                elif pad_mode == "loop_repeat":
                    if orig_len <= 0:
                        padded = torch.zeros((1, target_ch, max_len), dtype=dtype, device=device)
                    else:
                        n_repeat = (max_len + orig_len - 1) // orig_len
                        tiled = w_multi.repeat(1, 1, n_repeat)
                        padded = tiled[:, :, :max_len]
                elif pad_mode == "fade_trim":
                    last_val = w_multi[:, :, -1:]
                    decay_steps = torch.arange(pad_len, dtype=torch.float32, device=device)
                    decay = torch.exp(-5.0 * (decay_steps / pad_len))
                    faded_pad = last_val * decay.view(1, 1, -1)
                    padded = torch.cat((w_multi, faded_pad), dim=2)
            padded *= vol_i
            padded = padded.to(device=device, dtype=dtype)
            if padded.shape[2] != max_len:
                extra_pad = max_len - padded.shape[2]
                padded = torch.nn.functional.pad(padded, (0, extra_pad))
            padded_list.append(padded)
        headroom_lin = 10 ** (pre_mix_gain_db / 20.0)
        for padded in padded_list:
            padded *= headroom_lin
        # Stack and mix
        stacked = torch.stack(padded_list, dim=0)
        if mix_mode == "linear_sum":
            mixed_multi = torch.sum(stacked, dim=0)
        elif mix_mode == "rms_power":
            N = stacked.shape[0]
            mixed_multi = torch.sqrt(torch.sum(stacked ** 2, dim=0) / N)
        elif mix_mode == "average":
            mixed_multi = torch.mean(stacked, dim=0)
        elif mix_mode == "max_amplitude":
            mixed_multi = torch.max(stacked, dim=0)[0]
        peak = torch.max(torch.abs(mixed_multi)).item()
        if auto_normalize:
            if peak > 1e-8:
                mixed_multi *= (10 ** (-1 / 20)) / peak
        mixed_multi = torch.tanh(mixed_multi * 1.25) / 1.25
        post_peak = torch.max(torch.abs(mixed_multi)).item()
        process_details = auto_dbg + f"Used torch mix (resample={resample_mode}), Target SR: {target_sr}, ch: {target_ch}, Max len: {mixed_multi.shape[2]}, Peak pre:{peak:.3f} post:{post_peak:.3f} ({'norm+clamp' if auto_normalize else 'clamp'}:<=1.0)"
        mixed_mono = torch.mean(mixed_multi, dim=1, keepdim=True)
        if mixed_multi.shape[1] == 1:
            mixed_stereo = mixed_mono.repeat(1, 2, 1)
        else:
            mixed_stereo = mixed_multi[:, :2, :]
        audio_mono = {"waveform": mixed_mono, "sample_rate": target_sr}
        audio_stereo = {"waveform": mixed_stereo, "sample_rate": target_sr}
        if enable_save and file_prefix:
            uid = uuid.uuid4().hex[:8]
            mono_fn = f"{file_prefix}_mono_{uid}.{save_format}"
            stereo_fn = f"{file_prefix}_stereo_{uid}.{save_format}"
            w_mono = audio_mono["waveform"].squeeze(0)
            torchaudio.save(mono_fn, w_mono, audio_mono["sample_rate"], format=save_format)
            full_mono = os.path.abspath(mono_fn)
            w_stereo = audio_stereo["waveform"].squeeze(0)
            torchaudio.save(stereo_fn, w_stereo, audio_stereo["sample_rate"], format=save_format)
            full_stereo = os.path.abspath(stereo_fn)
            dbg_parts.append(f"Saved mono: {full_mono}")
            dbg_parts.append(f"Saved stereo: {full_stereo}")
        dbg_text = enabled_prefix + process_details + base_dbg
        return (audio_mono, audio_stereo, dbg_text, {"sox_params": current_params})

class SoxUtilMultiOutputAudio1_5:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "force_channels": (["auto", "mono", "stereo"], {"default": "auto", "tooltip": "Force output channels: auto=preserve input, mono=downmix to 1ch (mean), stereo=upmix to 2ch (repeat if mono, first 2ch if more)."}),
            "master_gain_db": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 12.0, "step": 0.1, "tooltip": "Global master gain dB applied to input before per-track gains."}),
            "=== Track Gain ===": ("STRING", {"default": "", "tooltip": "Group label before the per-output track gain sliders."}),
        }
        for i in range(5):
            optional[f"track_gain_{i}"] = ("FLOAT", {"default": 0.0, "min": -60.0, "max": 12.0, "step": 0.1, "tooltip": f"Gain dB for out-audio-{i} (multiplicative after master)."})
        return {"required": {"in-audio": ("AUDIO",)}, "optional": optional}

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("out-audio-0", "out-audio-1", "out-audio-2", "out-audio-3", "out-audio-4")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Utilities"
    DESCRIPTION = """Utility node: 1 `in-audio` → 5 `out-audio-0..4` AUDIO outputs (with optional force_channels mono/stereo/auto, master_gain_db, per-track track_gain_0-4 dB).

force_channels (auto/mono/stereo): auto=preserve input channels, mono=downmix to [1,1,T] via mean(dim=1), stereo=up/down to [1,2,T] (repeat mono, first 2ch if more).

Gains: master_gain_db (global dB → lin mul) then per-output track_gain_N dB (lin mul on copy).

Input None → 5 dummy zeros [1,C,1024]@44.1kHz (C=1 mono/2 stereo per force_channels), gains applied (zero→zero)."""

    def process(self, **kwargs):
        force_channels = kwargs.get("force_channels", "auto")
        master_gain_db = kwargs.get("master_gain_db", 0.0)
        track_gains_db = [kwargs.get(f"track_gain_{i}", 0.0) for i in range(5)]
        in_audio = kwargs.get("in-audio")
        if in_audio is None:
            sr = 44100
            dtype_ = torch.float32
            if force_channels == "mono":
                zero_w = torch.zeros((1, 1, 1024), dtype=dtype_)
            elif force_channels == "stereo":
                zero_mono = torch.zeros((1, 1, 1024), dtype=dtype_)
                zero_w = zero_mono.repeat(1, 2, 1)
            else:  # auto
                zero_w = torch.zeros((1, 1, 1024), dtype=dtype_)
            master_lin = 10 ** (master_gain_db / 20.0)
            w_master = zero_w * master_lin
            outs = []
            for i in range(5):
                track_lin = 10 ** (track_gains_db[i] / 20.0)
                out_w = w_master.clone() * track_lin
                out_audio = {"waveform": out_w, "sample_rate": sr}
                outs.append(out_audio)
            return tuple(outs)
        else:
            sr = in_audio["sample_rate"]
            w = in_audio["waveform"][0:1]
            device = w.device
            dtype_ = w.dtype
            orig_ch = w.shape[1]
            target_ch = orig_ch if force_channels == "auto" else 1 if force_channels == "mono" else 2
            if orig_ch != target_ch:
                if target_ch == 1:
                    w = torch.mean(w, dim=1, keepdim=True)
                elif target_ch == 2:
                    if orig_ch == 1:
                        w = w.repeat(1, 2, 1)
                    else:
                        w = w[:, :2, :]
            master_lin = 10 ** (master_gain_db / 20.0)
            w_master = w * master_lin
            outs = []
            for i in range(5):
                track_lin = 10 ** (track_gains_db[i] / 20.0)
                out_w = w_master.clone() * track_lin
                out_audio = {"waveform": out_w, "sample_rate": sr}
                outs.append(out_audio)
            return tuple(outs)

class SoxMuxWetDry:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wet_audio": ("AUDIO",),
                "dry_audio": ("AUDIO",),
            },
            "optional": {
                "enable_mix": ("BOOLEAN", {"default": True}),
                "process_mode": (["auto", "torch", "sox"], {"default": "auto"}),
                "mix": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "gain": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 60.0, "step": 0.1}),
                "sox_params": ("SOX_PARAMS",),
            }
        }
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("mono_mix", "stereo_mix", "wet_audio", "dry_audio", "sox_params", "dbg_text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Utilities"
    DESCRIPTION = """Wet/Dry AUDIO mixer: wet + dry → mono_mix/stereo_mix (mix% wet over dry, post-gain dB).
Passthrough wet_audio, dry_audio, sox_params. enable_mix off → dry mono/stereo passthrough.
process_mode: torch=tensor (default); sox=future. dbg_text: settings/mode/SR/len."""

    def process(self, wet_audio, dry_audio, enable_mix=True, process_mode="auto", mix=50.0, gain=0.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params else []
        sr = dry_audio["sample_rate"]
        device = dry_audio["waveform"].device
        dtype = dry_audio["waveform"].dtype
        dry_w = dry_audio["waveform"][0:1]

        wet_sr = wet_audio["sample_rate"]
        wet_w = wet_audio["waveform"][0:1].to(device=device, dtype=dtype)
        if wet_sr != sr:
            resampler = torchaudio.transforms.Resample(wet_sr, sr)
            wet_w_2d = wet_w.squeeze(0)
            wet_w = resampler(wet_w_2d).unsqueeze(0)

        max_t = max(wet_w.shape[2], dry_w.shape[2])
        pad_wet = torch.nn.functional.pad(wet_w, (0, max_t - wet_w.shape[2]))
        pad_dry = torch.nn.functional.pad(dry_w, (0, max_t - dry_w.shape[2]))

        wet_frac = mix / 100.0
        dry_frac = 1.0 - wet_frac

        if not enable_mix:
            mixed = pad_dry
            dbg_prefix = ""
        else:
            mixed = pad_dry * dry_frac + pad_wet * wet_frac
            dbg_prefix = "** Enabled **\n"

        gain_mult = 10 ** (gain / 20.0)
        mixed = torch.clamp(mixed * gain_mult, -1.0, 1.0)

        mixed_mono = mixed
        C = mixed.shape[1]
        if C == 1:
            mixed_stereo = mixed.repeat(1, 2, 1)
        else:
            mixed_stereo = mixed[:, :2, :]

        audio_mono = {"waveform": mixed_mono, "sample_rate": sr}
        audio_stereo = {"waveform": mixed_stereo, "sample_rate": sr}

        dbg_parts = [
            f"enable_mix: {enable_mix}",
            f"process_mode: {process_mode}",
            f"mix: {mix:.1f} ({wet_frac:.0%} wet + {dry_frac:.0%} dry)",
            f"gain: {gain:+.1f} dB",
            f"wet SR: {wet_sr} | dry SR: {sr}",
            f"len: {max_t}",
        ]
        dbg_prefix = "*** Enabled ***\n" if enable_mix else "--- Disabled ---\n"
        dbg_text = dbg_prefix + "\n".join(dbg_parts)

        return (audio_mono, audio_stereo, wet_audio, dry_audio, {"sox_params": current_params}, dbg_text)



NODE_CLASS_MAPPINGS = {
    "SoxApplyEffects": SoxApplyEffectsNode,
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
    "SoxSpectrogram": SoxSpectrogramNode,
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
    "SoxUtilMultiInputText10": SoxUtilMultiInputText10,
    "SoxUtilMultiInputText5": SoxUtilMultiInputText5,
    "SoxUtilMultiInputAudio5_1": SoxUtilMultiInputAudio5_1,
    "SoxUtilMuxAudio5_1": SoxUtilMuxAudio5_1,
    "SoxUtilMultiOutputAudio1_5": SoxUtilMultiOutputAudio1_5,
    "SoxMuxWetDry": SoxMuxWetDry,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SoxApplyEffects": "SoX Apply Effects",
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
    "SoxSpectrogram": "SoX Spectrogram",
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
    "SoxUtilMultiInputText10": "SoX Util Multi-Input Text 10",
    "SoxUtilMultiInputText5": "SoX Util Multi-Input Text 5",
    "SoxUtilMultiInputAudio5_1": "SoX Util Multi-Input Audio 5-1",
    "SoxUtilMuxAudio5_1": "SoX Util Mux Audio 5-1",
    "SoxUtilMultiOutputAudio1_5": "SoX Util Multi-Output Audio 1-5",
    "SoxMuxWetDry": "SoX Mux Wet/Dry",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
