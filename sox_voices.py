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
import torch

# All SoxVoice* nodes here - code copied from __init__.py lines 2286-3143

class SoxVeDeepOldManNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_deep_old_man": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "tooltip": "Scale bass gain."}),
                "bass_gain": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 30.0, "step": 1.0}),
                "bass_freq": ("FLOAT", {"default": 100.0, "min": 20.0, "max": 500.0}),
                "bass_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Deep/Old Man: bass +12 100 1q — Boosts lows for gravelly depth."

    def process(self, audio, enable_voice_deep_old_man=True, intensity=1.0, bass_gain=12.0, bass_freq=100.0, bass_width=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        scaled_gain = bass_gain * intensity
        effect_params = ["bass", f"+{scaled_gain}", str(bass_freq), f"{bass_width}q"]
        cmd_str = f"sox voice.wav deep_oldman.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_deep_old_man:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeChipmunkChildNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_chipmunk_child": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "pitch_shift": ("INT", {"default": 12, "min": 0, "max": 24}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Chipmunk/Child: pitch +12 — Octave up (semitones)."

    def process(self, audio, enable_voice_chipmunk_child=True, intensity=1.0, pitch_shift=12, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        scaled_shift = int(pitch_shift * intensity)
        effect_params = ["pitch", f"+{scaled_shift}"]
        cmd_str = f"sox voice.wav chipmunk_child.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_chipmunk_child:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeHeliumNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_helium": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "pitch_shift_hz": ("INT", {"default": 600, "min": 0, "max": 1200}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Helium: pitch +600h — +600Hz shift."

    def process(self, audio, enable_voice_helium=True, intensity=1.0, pitch_shift_hz=600, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        scaled_shift = int(pitch_shift_hz * intensity)
        effect_params = ["pitch", f"+{scaled_shift}h"]
        cmd_str = f"sox voice.wav helium.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_helium:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeRobotNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_robot": ("BOOLEAN", {"default": True}),
                "chorus_gain_in": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_gain_out": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_delay": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "chorus_decay": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_speed": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.01}),
                "chorus_depth": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "chorus_phase": ("INT", {"default": 2, "min": 0, "max": 10}),
                "chorus_wave": (["sin", "tri"], {"default": "tri"}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Robot/Vocoder-ish: chorus 0.5 0.9 50 0.4 0.25 2 -t — Thick metallic modulation."

    def process(self, audio, enable_voice_robot=True, chorus_gain_in=0.5, chorus_gain_out=0.9, chorus_delay=50.0, chorus_decay=0.4, chorus_speed=0.25, chorus_depth=2.0, chorus_phase=2, chorus_wave="tri", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        wave_flag = "-t" if chorus_wave == "tri" else "-s"
        effect_params = ["chorus", str(chorus_gain_in), str(chorus_gain_out), str(chorus_delay), str(chorus_decay), str(chorus_speed), str(chorus_depth), str(chorus_phase), wave_flag]
        cmd_str = f"sox voice.wav robot.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_robot:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeAlienNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_alien": ("BOOLEAN", {"default": True}),
                "flanger_delay": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.01}),
                "flanger_depth": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "flanger_regen": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "flanger_speed": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "flanger_shape": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "flanger_taps": ("INT", {"default": 2, "min": 1, "max": 10}),
                "flanger_wave": (["sin"], {"default": "sin"}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Alien: flanger 0.6 2 60 0.5 0.5 2 -s — Sweeping jet-like whoosh."

    def process(self, audio, enable_voice_alien=True, flanger_delay=0.6, flanger_depth=2.0, flanger_regen=60.0, flanger_speed=0.5, flanger_shape=0.5, flanger_taps=2, flanger_wave="sin", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        wave_flag = "-s"
        effect_params = ["flanger", str(flanger_delay), str(flanger_depth), str(flanger_regen), str(flanger_speed), str(flanger_shape), str(flanger_taps), wave_flag]
        cmd_str = f"sox voice.wav alien.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_alien:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeGhostNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_ghost": ("BOOLEAN", {"default": True}),
                "phaser_gain_in": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "phaser_delay": ("FLOAT", {"default": 0.74, "min": 0.0, "max": 2.0, "step": 0.01}),
                "phaser_decay": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "phaser_speed": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.01}),
                "phaser_mod": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Ghost: phaser 0.8 0.74 3 0.4 0.5 — Swirling otherworldly."

    def process(self, audio, enable_voice_ghost=True, phaser_gain_in=0.8, phaser_delay=0.74, phaser_decay=3.0, phaser_speed=0.4, phaser_mod=0.5, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        effect_params = ["phaser", str(phaser_gain_in), str(phaser_delay), str(phaser_decay), str(phaser_speed), str(phaser_mod)]
        cmd_str = f"sox voice.wav ghost.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_ghost:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeEchoCaveNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_echo_cave": ("BOOLEAN", {"default": True}),
                "echo_gain_in": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echo_gain_out": ("FLOAT", {"default": 0.88, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echo_delay_1": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "echo_decay_1": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Echo Cave: echo 0.8 0.88 6 0.6 — Spacious repeats."

    def process(self, audio, enable_voice_echo_cave=True, echo_gain_in=0.8, echo_gain_out=0.88, echo_delay_1=6.0, echo_decay_1=0.6, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        effect_params = ["echo", str(echo_gain_in), str(echo_gain_out), str(echo_delay_1), str(echo_decay_1)]
        cmd_str = f"sox voice.wav echo_cave.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_echo_cave:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeTelephoneNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_telephone": ("BOOLEAN", {"default": True}),
                "highpass_freq": ("FLOAT", {"default": 300.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "lowpass_freq": ("FLOAT", {"default": 3000.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Telephone: highpass 300 lowpass 3000 — Muffled band-pass."

    def process(self, audio, enable_voice_telephone=True, highpass_freq=300.0, lowpass_freq=3000.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        effect_params = ["highpass", str(highpass_freq), "lowpass", str(lowpass_freq)]
        cmd_str = f"sox voice.wav telephone.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_telephone:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeMonsterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_monster": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "overdrive_gain": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 50.0, "step": 1.0}),
                "overdrive_color": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 50.0, "step": 1.0}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Distorted Monster: overdrive 20 20 — Gritty clipping."

    def process(self, audio, enable_voice_monster=True, intensity=1.0, overdrive_gain=20.0, overdrive_color=20.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        scaled_gain = int(overdrive_gain * intensity)
        scaled_color = int(overdrive_color * intensity)
        effect_params = ["overdrive", str(scaled_gain), str(scaled_color)]
        cmd_str = f"sox voice.wav monster.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_monster:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeCompandRobotNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_compand_robot": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "compand_attack": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "compand_release": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "compand_points": ("STRING", {"default": "6:-70,-60,-20", "tooltip": "Transfer points e.g. 6:-70,-60,-20"}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Compressed Robot: compand 0.3,0.8 6:-70,-60,-20 — Punchy dynamic squeeze."

    def process(self, audio, enable_voice_compand_robot=True, intensity=1.0, compand_attack=0.3, compand_release=0.8, compand_points="6:-70,-60,-20", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        attack_str = f"{compand_attack * intensity},{compand_release * intensity}"
        compand_str = f"{attack_str} {compand_points}"
        effect_params = ["compand"] + shlex.split(compand_str)
        cmd_str = f"sox voice.wav compand_robot.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_compand_robot:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeBoomyDemonNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_boomy_demon": ("BOOLEAN", {"default": True}),
                "lowpass_rolloff": (["-1"], {"default": "-1"}),
                "lowpass_width": ("FLOAT", {"default": 200.0, "min": 20.0, "max": 2000.0, "step": 10.0}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Boomy Demon: lowpass -1 200 — Muddy rumble."

    def process(self, audio, enable_voice_boomy_demon=True, lowpass_rolloff="-1", lowpass_width=200.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        effect_params = ["lowpass", lowpass_rolloff, str(lowpass_width)]
        cmd_str = f"sox voice.wav boomy_demon.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_boomy_demon:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeWitchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_witch": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "treble_gain": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 40.0, "step": 1.0}),
                "treble_freq": ("FLOAT", {"default": 5000.0, "min": 1000.0, "max": 20000.0, "step": 100.0}),
                "treble_width": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Screechy Witch: treble +20 5000 0.5q — Harsh highs."

    def process(self, audio, enable_voice_witch=True, intensity=1.0, treble_gain=20.0, treble_freq=5000.0, treble_width=0.5, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        scaled_gain = treble_gain * intensity
        effect_params = ["treble", f"+{scaled_gain}", str(treble_freq), f"{treble_width}q"]
        cmd_str = f"sox voice.wav witch.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_witch:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeWarbleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_warble": ("BOOLEAN", {"default": True}),
                "bend_high": ("FLOAT", {"default": 4000.0, "min": 1000.0, "max": 8000.0, "step": 100.0}),
                "bend_low": ("FLOAT", {"default": 800.0, "min": 100.0, "max": 2000.0, "step": 100.0}),
                "wave_high": (["sin(0.3)"], {"default": "sin(0.3)"}),
                "wave_low": (["sin(1)"], {"default": "sin(1)"}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Warble: bend 4000 sin(0.3) 800 sin(1) — Pitch wobble."

    def process(self, audio, enable_voice_warble=True, bend_high=4000.0, bend_low=800.0, wave_high="sin(0.3)", wave_low="sin(1)", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        effect_params = ["bend", str(bend_high), wave_high, str(bend_low), wave_low]
        cmd_str = f"sox voice.wav warble.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_warble:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeTempleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_temple": ("BOOLEAN", {"default": True}),
                "reverb_reverberance": ("FLOAT", {"default": 80.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "reverb_hf": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "reverb_room": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "reverb_damp": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Temple Reverb: reverb 80 50 100 0 — Long decay hall."

    def process(self, audio, enable_voice_temple=True, reverb_reverberance=80.0, reverb_hf=50.0, reverb_room=100.0, reverb_damp=0.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        effect_params = ["reverb", str(reverb_reverberance), str(reverb_hf), str(reverb_room), str(reverb_damp)]
        cmd_str = f"sox voice.wav temple.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_temple:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeSquirrelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_squirrel": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "speed_factor": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 3.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Fast Squirrel: speed 1.5 — Chipmunk + faster."

    def process(self, audio, enable_voice_squirrel=True, intensity=1.0, speed_factor=1.5, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        scaled_speed = speed_factor * intensity
        effect_params = ["speed", str(scaled_speed)]
        cmd_str = f"sox voice.wav squirrel.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_squirrel:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeGiantNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_giant": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "tempo_factor": ("FLOAT", {"default": 0.8, "min": 0.3, "max": 2.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Slow Giant: tempo 0.8 — Time-stretch without pitch drop."

    def process(self, audio, enable_voice_giant=True, intensity=1.0, tempo_factor=0.8, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        scaled_tempo = tempo_factor * intensity
        effect_params = ["tempo", str(scaled_tempo)]
        cmd_str = f"sox voice.wav giant.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_giant:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeVibratoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_vibrato": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "tremolo_speed": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.01}),
                "tremolo_depth": ("FLOAT", {"default": 90.0, "min": 20.0, "max": 200.0, "step": 1.0}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Vibrato: tremolo 0.3 90 — Pitch modulation."

    def process(self, audio, enable_voice_vibrato=True, intensity=1.0, tremolo_speed=0.3, tremolo_depth=90.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        scaled_speed = tremolo_speed * intensity
        scaled_depth = tremolo_depth * intensity
        effect_params = ["tremolo", str(scaled_speed), str(scaled_depth)]
        cmd_str = f"sox voice.wav vibrato.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_vibrato:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeEvilDemonNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_evil_demon": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Evil Demon: pitch -8 bass +10 tremolo 0.15 80 — Low + rumble + shake."

    def process(self, audio, enable_voice_evil_demon=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        pitch_shift = int(-8 * intensity)
        bass_gain = 10 * intensity
        effect_params = ["pitch", f"{pitch_shift}", "bass", f"+{bass_gain}", "tremolo", "0.15", "80"]
        cmd_str = f"sox voice.wav evil_demon.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_evil_demon:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeCartoonDuckNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_cartoon_duck": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Cartoon Duck: pitch +5 chorus 0.4 0.8 40 0.3 0.2 3 speed 1.1 — Squawky wobble."

    def process(self, audio, enable_voice_cartoon_duck=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        pitch_shift = int(5 * intensity)
        effect_params = ["pitch", f"+{pitch_shift}", "chorus", "0.4", "0.8", "40", "0.3", "0.2", "3", "speed", "1.1"]
        cmd_str = f"sox voice.wav cartoon_duck.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_cartoon_duck:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeDarthVaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_darth_vader": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Darth Vader: pitch -7 lowpass -1 800 reverb 50 50 100 0 — Breathing mask reverb."

    def process(self, audio, enable_voice_darth_vader=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        pitch_shift = int(-7 * intensity)
        effect_params = ["pitch", f"{pitch_shift}", "lowpass", "-1", "800", "reverb", "50", "50", "100", "0"]
        cmd_str = f"sox voice.wav darth_vader.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_darth_vader:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeChipmunkNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_chipmunk": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Chipmunk: pitch +14 speed 1.3 rate 44100 — High/fast classic."

    def process(self, audio, enable_voice_chipmunk=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        pitch_shift = int(14 * intensity)
        speed_factor = 1.3 * intensity
        effect_params = ["pitch", f"+{pitch_shift}", "speed", str(speed_factor), "rate", "44100"]
        cmd_str = f"sox voice.wav chipmunk.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_chipmunk:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeOldWitchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_old_witch": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Old Witch: pitch -3 treble +12 phaser 0.6 0.5 2 0.3 0.4 — Cackly swirl."

    def process(self, audio, enable_voice_old_witch=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        pitch_shift = int(-3 * intensity)
        treble_gain = 12 * intensity
        effect_params = ["pitch", f"{pitch_shift}", "treble", f"+{treble_gain}", "phaser", "0.6", "0.5", "2", "0.3", "0.4"]
        cmd_str = f"sox voice.wav old_witch.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_old_witch:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeMinionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_minion": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Minion: pitch +7 chorus 0.3 0.95 30 0.5 0.15 5 — Bubbly chorus."

    def process(self, audio, enable_voice_minion=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        pitch_shift = int(7 * intensity)
        effect_params = ["pitch", f"+{pitch_shift}", "chorus", "0.3", "0.95", "30", "0.5", "0.15", "5"]
        cmd_str = f"sox voice.wav minion.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_minion:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeTerminatorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_terminator": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Terminator: pitch -5 flanger 0.4 0.5 20 0.6 0.8 2 overdrive 10 5 — Mechanical flange grit."

    def process(self, audio, enable_voice_terminator=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        pitch_shift = int(-5 * intensity)
        overdrive_gain = int(10 * intensity)
        effect_params = ["pitch", f"{pitch_shift}", "flanger", "0.4", "0.5", "20", "0.6", "0.8", "2", "overdrive", str(overdrive_gain), "5"]
        cmd_str = f"sox voice.wav terminator.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_terminator:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeFairyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_fairy": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Fairy: pitch +9 highpass 2000 reverb 30 90 80 10 — Airy sparkle."

    def process(self, audio, enable_voice_fairy=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        pitch_shift = int(9 * intensity)
        effect_params = ["pitch", f"+{pitch_shift}", "highpass", "2000", "reverb", "30", "90", "80", "10"]
        cmd_str = f"sox voice.wav fairy.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_fairy:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeZombieNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_zombie": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Zombie: pitch -10 echo 0.9 0.9 3 0.6 compand 0.2,0.8 6:-54,-30,-15 — Groan delay compress."

    def process(self, audio, enable_voice_zombie=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        pitch_shift = int(-10 * intensity)
        effect_params = ["pitch", f"{pitch_shift}", "echo", "0.9", "0.9", "3", "0.6", "compand", "0.2,0.8", "6:-54,-30,-15"]
        cmd_str = f"sox voice.wav zombie.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_zombie:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVePirateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_pirate": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Pirate: bass +6 lowpass 2500 tremolo 0.2 120 equalizer 200 0.8 +4 — Rumbling growl."

    def process(self, audio, enable_voice_pirate=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        bass_gain = 6 * intensity
        trem_speed = 0.2 * intensity
        effect_params = ["bass", f"+{bass_gain}", "lowpass", "2500", "tremolo", str(trem_speed), "120", "equalizer", "200", "0.8", "+4"]
        cmd_str = f"sox voice.wav pirate.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_pirate:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)


class SoxVeSuperheroNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_superhero": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voice"
    DESCRIPTION = "Superhero Echo: reverb 90 50 100 100 echo 0.7 0.8 10 0.3 pitch +2 — Epic boom."

    def process(self, audio, enable_voice_superhero=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        pitch_shift = int(2 * intensity)
        effect_params = ["reverb", "90", "50", "100", "100", "echo", "0.7", "0.8", "10", "0.3", "pitch", f"+{pitch_shift}"]
        cmd_str = f"sox voice.wav superhero.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_superhero:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, dbg_text)

NODE_CLASS_MAPPINGS = {
    "SoxVeDeepOldMan": SoxVeDeepOldManNode,
    "SoxVeChipmunkChild": SoxVeChipmunkChildNode,
    "SoxVeHelium": SoxVeHeliumNode,
    "SoxVeRobot": SoxVeRobotNode,
    "SoxVeAlien": SoxVeAlienNode,
    "SoxVeGhost": SoxVeGhostNode,
    "SoxVeEchoCave": SoxVeEchoCaveNode,
    "SoxVeTelephone": SoxVeTelephoneNode,
    "SoxVeMonster": SoxVeMonsterNode,
    "SoxVeCompandRobot": SoxVeCompandRobotNode,
    "SoxVeBoomyDemon": SoxVeBoomyDemonNode,
    "SoxVeWitch": SoxVeWitchNode,
    "SoxVeWarble": SoxVeWarbleNode,
    "SoxVeTemple": SoxVeTempleNode,
    "SoxVeSquirrel": SoxVeSquirrelNode,
    "SoxVeGiant": SoxVeGiantNode,
    "SoxVeVibrato": SoxVeVibratoNode,
    "SoxVeEvilDemon": SoxVeEvilDemonNode,
    "SoxVeCartoonDuck": SoxVeCartoonDuckNode,
    "SoxVeDarthVader": SoxVeDarthVaderNode,
    "SoxVeChipmunk": SoxVeChipmunkNode,
    "SoxVeOldWitch": SoxVeOldWitchNode,
    "SoxVeMinion": SoxVeMinionNode,
    "SoxVeTerminator": SoxVeTerminatorNode,
    "SoxVeFairy": SoxVeFairyNode,
    "SoxVeZombie": SoxVeZombieNode,
    "SoxVePirate": SoxVePirateNode,
    "SoxVeSuperhero": SoxVeSuperheroNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SoxVeDeepOldMan": "Voice: Deep/Old Man",
    "SoxVeChipmunkChild": "Voice: Chipmunk/Child",
    "SoxVeHelium": "Voice: Helium",
    "SoxVeRobot": "Voice: Robot",
    "SoxVeAlien": "Voice: Alien",
    "SoxVeGhost": "Voice: Ghost",
    "SoxVeEchoCave": "Voice: Echo Cave",
    "SoxVeTelephone": "Voice: Telephone",
    "SoxVeMonster": "Voice: Monster",
    "SoxVeCompandRobot": "Voice: Compand Robot",
    "SoxVeBoomyDemon": "Voice: Boomy Demon",
    "SoxVeWitch": "Voice: Witch",
    "SoxVeWarble": "Voice: Warble",
    "SoxVeTemple": "Voice: Temple Reverb",
    "SoxVeSquirrel": "Voice: Squirrel",
    "SoxVeGiant": "Voice: Giant",
    "SoxVeVibrato": "Voice: Vibrato",
    "SoxVeEvilDemon": "Voice: Evil Demon",
    "SoxVeCartoonDuck": "Voice: Cartoon Duck",
    "SoxVeDarthVader": "Voice: Darth Vader",
    "SoxVeChipmunk": "Voice: Chipmunk",
    "SoxVeOldWitch": "Voice: Old Witch",
    "SoxVeMinion": "Voice: Minion",
    "SoxVeTerminator": "Voice: Terminator",
    "SoxVeFairy": "Voice: Fairy",
    "SoxVeZombie": "Voice: Zombie",
    "SoxVePirate": "Voice: Pirate",
    "SoxVeSuperhero": "Voice: Superhero Echo",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
