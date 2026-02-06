import subprocess
import tempfile
import os
import shlex
import numpy as np
import struct
import shutil
import torch

# Check if the program is being run by bytest
if os.environ.get('PYTEST_VERSION'):
    from sox_node_utils import SoxNodeUtils
else:
    from .sox_node_utils import SoxNodeUtils


# All SoxVoice* nodes here - code copied from __init__.py lines 2286-3143

class SoxVeDeepOldManNode:
    # Tested: DGS v0.1.2
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_deep_old_man": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "tooltip": "Overall scale for adjustable params."}),
                "bass_gain": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 24.0, "step": 0.5, "tooltip": "Base bass boost (dB)."}),
                "pitch_shift": ("FLOAT", {"default": 1.8, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Pitch shift amount."}),
                "tremolo_speed": ("FLOAT", {"default": 4.8, "min": 0.1, "max": 20.0, "step": 0.1, "tooltip": "Tremolo modulation speed (Hz)."}),
                "tremolo_depth": ("FLOAT", {"default": 35.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Tremolo depth (%) scaled by intensity."}),
                "eq_freq": ("INT", {"default": 2500, "min": 20, "max": 20000, "step": 50, "tooltip": "Equalizer center frequency (Hz)."}),
                "eq_gain": ("FLOAT", {"default": -4.0, "min": -20.0, "max": 5.0, "step": 0.1, "tooltip": "EQ gain (dB, scaled)."}),
                "eq_width": ("INT", {"default": 1000, "min": 20, "max": 10000, "step": 50, "tooltip": "Equalizer width (Hz)."}),
                "lowpass_freq": ("INT", {"default": 2800, "min": 20, "max": 20000, "step": 50, "tooltip": "Lowpass filter cutoff (Hz)."}),
                "gain_adjust": ("FLOAT", {"default": -2.5, "min": -20.0, "max": 5.0, "step": 0.1, "tooltip": "Final gain adjustment (dB, scaled)."}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Old Deep Man: bass +12 100 1q pitch +1.8 tremolo 4.8 35 equalizer 2500 1000h -4 lowpass 2800 gain -2.5 — Deep thin shaky hoarse quiet old voice."
    @staticmethod
    def _save_wav(filename, tensor, sample_rate):
        channels = tensor.shape[0]
        if channels == 1:
            data = tensor[0].cpu().numpy().astype(np.float32)
        else:
            data = tensor.transpose(0, 1).contiguous().flatten().cpu().numpy().astype(np.float32)
        byte_rate = sample_rate * channels * 4
        block_align = channels * 4
        fmt_chunk_size = 16
        fmt_chunk = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
        data_chunk = data.tobytes()
        data_size = len(data) * 4
        riff_size = 36 + data_size
        header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", fmt_chunk_size) + fmt_chunk + struct.pack("<4sI", b"data", data_size)
        with open(filename, "wb") as f:
            f.write(header + data_chunk)
    @staticmethod
    def _load_wav(filename):
        with open(filename, "rb") as f:
            if f.read(4) != b'RIFF':
                raise ValueError("Not RIFF")
            riff_size = struct.unpack('<I', f.read(4))[0]
            if f.read(4) != b'WAVE':
                raise ValueError("Not WAVE")
            sr = None
            channels = None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("Truncated file")
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data_start = f.tell()
                if chunk_id == b'fmt ':
                    fmt_hdr = f.read(chunk_size)
                    if len(fmt_hdr) < 16:
                        raise ValueError("Short fmt")
                    fmt = struct.unpack('<HHIIHH', fmt_hdr[:16])
                    if fmt[0] != 3 or fmt[5] != 32:
                        raise ValueError(f"Not float32 (fmt={fmt[0]}, bits={fmt[5]})")
                    channels = fmt[1]
                    sr = fmt[2]
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    data_size = len(data_bytes)
                    data_bytes = data_bytes[:data_size // 4 * 4]
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    if len(data) % channels != 0:
                        data = data[:len(data) // channels * channels]
                    data = data.reshape(channels, -1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                    return waveform, sr
                else:
                    f.seek(chunk_size + (chunk_size % 2), 1)
            raise ValueError("No data chunk")

    def process(self, audio, enable_voice_deep_old_man=True, intensity=1.0, bass_gain=12.0, pitch_shift=1.8, tremolo_speed=4.8, tremolo_depth=35.0, eq_freq=2500, eq_gain=-4.0, eq_width=1000, lowpass_freq=2800, gain_adjust=-2.5, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        scaled_bass_gain = bass_gain * intensity
        scaled_pitch = pitch_shift * intensity
        scaled_tremolo_depth = tremolo_depth * intensity
        scaled_eq_gain = eq_gain * intensity
        scaled_gain = gain_adjust * intensity
        effect_params = [
            "bass", f"+{scaled_bass_gain}", "100", "1q",
            "pitch", f"+{scaled_pitch}",
            "tremolo", str(tremolo_speed), f"{scaled_tremolo_depth}",
            "equalizer", str(eq_freq), f"{eq_width}h", f"{scaled_eq_gain}",
            "lowpass", str(lowpass_freq),
            "gain", f"{scaled_gain}"
        ]
        cmd_str = f"sox input.wav output.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_deep_old_man:
            current_params.extend(effect_params)
            tmpdir = tempfile.mkdtemp(prefix='sox_')
            tmp_in = os.path.join(tmpdir, "input.wav")
            tmp_out = os.path.join(tmpdir, "output.wav")
            try:
                sr = int(audio["sample_rate"])
                waveform = audio["waveform"].squeeze(0)
                self._save_wav(tmp_in, waveform, sr)
                cmd = ["sox", tmp_in, tmp_out] + effect_params
                full_cmd_str = shlex.join(cmd)
                dbg_text = "** Enabled **\n" + full_cmd_str
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    dbg_text += f"\n*** SoX CLI failed (rc={result.returncode}):\n{result.stderr.strip() if result.stderr else 'No stderr'}"
                    processed_audio = audio
                else:
                    processed_waveform, processed_sr = self._load_wav(tmp_out)
                    processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\n*** SoX failed: {str(e)}"
                processed_audio = audio
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Chipmunk/Child: pitch +12 — Octave up (semitones)."

    @staticmethod
    def _save_wav(filename, tensor, sample_rate):
        channels = tensor.shape[0]
        if channels == 1:
            data = tensor[0].cpu().numpy().astype(np.float32)
        else:
            data = tensor.transpose(0, 1).contiguous().flatten().cpu().numpy().astype(np.float32)
        byte_rate = sample_rate * channels * 4
        block_align = channels * 4
        fmt_chunk_size = 16
        fmt_chunk = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
        data_chunk = data.tobytes()
        data_size = len(data) * 4
        riff_size = 36 + data_size
        header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", fmt_chunk_size) + fmt_chunk + struct.pack("<4sI", b"data", data_size)
        with open(filename, "wb") as f:
            f.write(header + data_chunk)

    @staticmethod
    def _load_wav(filename):
        with open(filename, "rb") as f:
            if f.read(4) != b'RIFF':
                raise ValueError("Not RIFF")
            riff_size = struct.unpack('<I', f.read(4))[0]
            if f.read(4) != b'WAVE':
                raise ValueError("Not WAVE")
            sr = None
            channels = None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("Truncated file")
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data_start = f.tell()
                if chunk_id == b'fmt ':
                    fmt_hdr = f.read(chunk_size)
                    if len(fmt_hdr) < 16:
                        raise ValueError("Short fmt")
                    fmt = struct.unpack('<HHIIHH', fmt_hdr[:16])
                    if fmt[0] != 3 or fmt[5] != 32:
                        raise ValueError(f"Not float32 (fmt={fmt[0]}, bits={fmt[5]})")
                    channels = fmt[1]
                    sr = fmt[2]
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    data_size = len(data_bytes)
                    data_bytes = data_bytes[:data_size // 4 * 4]
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    if len(data) % channels != 0:
                        data = data[:len(data) // channels * channels]
                    data = data.reshape(channels, -1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                    return waveform, sr
                else:
                    f.seek(chunk_size + (chunk_size % 2), 1)
            raise ValueError("No data chunk")

    def process(self, audio, enable_voice_chipmunk_child=True, intensity=1.0, pitch_shift=12, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        scaled_shift = int(pitch_shift * intensity)
        effect_params = ["pitch", f"+{scaled_shift}"]
        cmd_str = f"sox input.wav output.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_chipmunk_child:
            current_params.extend(effect_params)
            tmpdir = tempfile.mkdtemp(prefix='sox_')
            tmp_in = os.path.join(tmpdir, "input.wav")
            tmp_out = os.path.join(tmpdir, "output.wav")
            try:
                sr = int(audio["sample_rate"])
                waveform = audio["waveform"].squeeze(0)
                self._save_wav(tmp_in, waveform, sr)
                cmd = ["sox", tmp_in, tmp_out] + effect_params
                full_cmd_str = shlex.join(cmd)
                dbg_text = "** Enabled **\n" + full_cmd_str
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    dbg_text += f"\n*** SoX CLI failed (rc={result.returncode}):\n{result.stderr.strip() if result.stderr else 'No stderr'}"
                    processed_audio = audio
                else:
                    processed_waveform, processed_sr = self._load_wav(tmp_out)
                    processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\n*** SoX failed: {str(e)}"
                processed_audio = audio
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Helium: pitch +600h — +600Hz shift."

    @staticmethod
    def _save_wav(filename, tensor, sample_rate):
        channels = tensor.shape[0]
        if channels == 1:
            data = tensor[0].cpu().numpy().astype(np.float32)
        else:
            data = tensor.transpose(0, 1).contiguous().flatten().cpu().numpy().astype(np.float32)
        byte_rate = sample_rate * channels * 4
        block_align = channels * 4
        fmt_chunk_size = 16
        fmt_chunk = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
        data_chunk = data.tobytes()
        data_size = len(data) * 4
        riff_size = 36 + data_size
        header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", fmt_chunk_size) + fmt_chunk + struct.pack("<4sI", b"data", data_size)
        with open(filename, "wb") as f:
            f.write(header + data_chunk)

    @staticmethod
    def _load_wav(filename):
        with open(filename, "rb") as f:
            if f.read(4) != b'RIFF':
                raise ValueError("Not RIFF")
            riff_size = struct.unpack('<I', f.read(4))[0]
            if f.read(4) != b'WAVE':
                raise ValueError("Not WAVE")
            sr = None
            channels = None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("Truncated file")
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data_start = f.tell()
                if chunk_id == b'fmt ':
                    fmt_hdr = f.read(chunk_size)
                    if len(fmt_hdr) < 16:
                        raise ValueError("Short fmt")
                    fmt = struct.unpack('<HHIIHH', fmt_hdr[:16])
                    if fmt[0] != 3 or fmt[5] != 32:
                        raise ValueError(f"Not float32 (fmt={fmt[0]}, bits={fmt[5]})")
                    channels = fmt[1]
                    sr = fmt[2]
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    data_size = len(data_bytes)
                    data_bytes = data_bytes[:data_size // 4 * 4]
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    if len(data) % channels != 0:
                        data = data[:len(data) // channels * channels]
                    data = data.reshape(channels, -1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                    return waveform, sr
                else:
                    f.seek(chunk_size + (chunk_size % 2), 1)
            raise ValueError("No data chunk")

    def process(self, audio, enable_voice_helium=True, intensity=1.0, pitch_shift_hz=600, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        scaled_shift = int(pitch_shift_hz * intensity)
        effect_params = ["pitch", f"+{scaled_shift}h"]
        cmd_str = f"sox input.wav output.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_helium:
            current_params.extend(effect_params)
            tmpdir = tempfile.mkdtemp(prefix='sox_')
            tmp_in = os.path.join(tmpdir, "input.wav")
            tmp_out = os.path.join(tmpdir, "output.wav")
            try:
                sr = int(audio["sample_rate"])
                waveform = audio["waveform"].squeeze(0)
                self._save_wav(tmp_in, waveform, sr)
                cmd = ["sox", tmp_in, tmp_out] + effect_params
                full_cmd_str = shlex.join(cmd)
                dbg_text = "** Enabled **\n" + full_cmd_str
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    dbg_text += f"\n*** SoX CLI failed (rc={result.returncode}):\n{result.stderr.strip() if result.stderr else 'No stderr'}"
                    processed_audio = audio
                else:
                    processed_waveform, processed_sr = self._load_wav(tmp_out)
                    processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\n*** SoX failed: {str(e)}"
                processed_audio = audio
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Robot/Vocoder-ish: chorus 0.5 0.9 50 0.4 0.25 2 -t — Thick metallic modulation."

    @staticmethod
    def _save_wav(filename, tensor, sample_rate):
        channels = tensor.shape[0]
        if channels == 1:
            data = tensor[0].cpu().numpy().astype(np.float32)
        else:
            data = tensor.transpose(0, 1).contiguous().flatten().cpu().numpy().astype(np.float32)
        byte_rate = sample_rate * channels * 4
        block_align = channels * 4
        fmt_chunk_size = 16
        fmt_chunk = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
        data_chunk = data.tobytes()
        data_size = len(data) * 4
        riff_size = 36 + data_size
        header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", fmt_chunk_size) + fmt_chunk + struct.pack("<4sI", b"data", data_size)
        with open(filename, "wb") as f:
            f.write(header + data_chunk)

    @staticmethod
    def _load_wav(filename):
        with open(filename, "rb") as f:
            if f.read(4) != b'RIFF':
                raise ValueError("Not RIFF")
            riff_size = struct.unpack('<I', f.read(4))[0]
            if f.read(4) != b'WAVE':
                raise ValueError("Not WAVE")
            sr = None
            channels = None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("Truncated file")
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data_start = f.tell()
                if chunk_id == b'fmt ':
                    fmt_hdr = f.read(chunk_size)
                    if len(fmt_hdr) < 16:
                        raise ValueError("Short fmt")
                    fmt = struct.unpack('<HHIIHH', fmt_hdr[:16])
                    if fmt[0] != 3 or fmt[5] != 32:
                        raise ValueError(f"Not float32 (fmt={fmt[0]}, bits={fmt[5]})")
                    channels = fmt[1]
                    sr = fmt[2]
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    data_size = len(data_bytes)
                    data_bytes = data_bytes[:data_size // 4 * 4]
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    if len(data) % channels != 0:
                        data = data[:len(data) // channels * channels]
                    data = data.reshape(channels, -1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                    return waveform, sr
                else:
                    f.seek(chunk_size + (chunk_size % 2), 1)
            raise ValueError("No data chunk")

    def process(self, audio, enable_voice_robot=True, chorus_gain_in=0.5, chorus_gain_out=0.9, chorus_delay=50.0, chorus_decay=0.4, chorus_speed=0.25, chorus_depth=2.0, chorus_phase=2, chorus_wave="tri", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        wave_flag = "-t" if chorus_wave == "tri" else "-s"
        effect_params = ["chorus", str(chorus_gain_in), str(chorus_gain_out), str(chorus_delay), str(chorus_decay), str(chorus_speed), str(chorus_depth), str(chorus_phase), wave_flag]
        cmd_str = f"sox input.wav output.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_robot:
            current_params.extend(effect_params)
            tmpdir = tempfile.mkdtemp(prefix='sox_')
            tmp_in = os.path.join(tmpdir, "input.wav")
            tmp_out = os.path.join(tmpdir, "output.wav")
            try:
                sr = int(audio["sample_rate"])
                waveform = audio["waveform"].squeeze(0)
                self._save_wav(tmp_in, waveform, sr)
                cmd = ["sox", tmp_in, tmp_out] + effect_params
                full_cmd_str = shlex.join(cmd)
                dbg_text = "** Enabled **\n" + full_cmd_str
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    dbg_text += f"\n*** SoX CLI failed (rc={result.returncode}):\n{result.stderr.strip() if result.stderr else 'No stderr'}"
                    processed_audio = audio
                else:
                    processed_waveform, processed_sr = self._load_wav(tmp_out)
                    processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\n*** SoX failed: {str(e)}"
                processed_audio = audio
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


class SoxVeAlienJabaNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_alien_jaba": ("BOOLEAN", {"default": True}),
                "flanger_delay": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.01}),
                "flanger_depth": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "flanger_regen": ("FLOAT", {"default": 60.0, "min": -95.0, "max": 95.0, "step": 1.0}),
                "flanger_width": ("FLOAT", {"default": 71.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "flanger_speed": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0, "step": 0.01}),
                "flanger_shape": (["sin", "triangle"], {"default": "sin"}),
                "flanger_phase": ("INT", {"default": 25, "min": 0, "max": 100}),
                "flanger_interp": (["lin", "quadratic"], {"default": "lin"}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Alien Jaba: flanger 0.6 2.0 60 71 0.5 sin 25 lin — Sweeping jet-like whoosh."

    @staticmethod
    def _save_wav(filename, tensor, sample_rate):
        channels = tensor.shape[0]
        if channels == 1:
            data = tensor[0].cpu().numpy().astype(np.float32)
        else:
            data = tensor.transpose(0, 1).contiguous().flatten().cpu().numpy().astype(np.float32)
        byte_rate = sample_rate * channels * 4
        block_align = channels * 4
        fmt_chunk_size = 16
        fmt_chunk = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
        data_chunk = data.tobytes()
        data_size = len(data) * 4
        riff_size = 36 + data_size
        header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", fmt_chunk_size) + fmt_chunk + struct.pack("<4sI", b"data", data_size)
        with open(filename, "wb") as f:
            f.write(header + data_chunk)

    @staticmethod
    def _load_wav(filename):
        with open(filename, "rb") as f:
            if f.read(4) != b'RIFF':
                raise ValueError("Not RIFF")
            riff_size = struct.unpack('<I', f.read(4))[0]
            if f.read(4) != b'WAVE':
                raise ValueError("Not WAVE")
            sr = None
            channels = None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("Truncated file")
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data_start = f.tell()
                if chunk_id == b'fmt ':
                    fmt_hdr = f.read(chunk_size)
                    if len(fmt_hdr) < 16:
                        raise ValueError("Short fmt")
                    fmt = struct.unpack('<HHIIHH', fmt_hdr[:16])
                    if fmt[0] != 3 or fmt[5] != 32:
                        raise ValueError(f"Not float32 (fmt={fmt[0]}, bits={fmt[5]})")
                    channels = fmt[1]
                    sr = fmt[2]
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    data_size = len(data_bytes)
                    data_bytes = data_bytes[:data_size // 4 * 4]
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    if len(data) % channels != 0:
                        data = data[:len(data) // channels * channels]
                    data = data.reshape(channels, -1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                    return waveform, sr
                else:
                    f.seek(chunk_size + (chunk_size % 2), 1)
            raise ValueError("No data chunk")

    def process(self, audio, enable_voice_alien_jaba=True, flanger_delay=0.6, flanger_depth=2.0, flanger_regen=60.0, flanger_width=71.0, flanger_speed=0.5, flanger_shape="sin", flanger_phase=25, flanger_interp="lin", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        effect_params = ["flanger", str(flanger_delay), str(flanger_depth), str(flanger_regen), str(flanger_width), str(flanger_speed), flanger_shape, str(flanger_phase), flanger_interp]
        cmd_str = f"sox input.wav output.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_alien_jaba:
            current_params.extend(effect_params)
            tmpdir = tempfile.mkdtemp(prefix='sox_')
            tmp_in = os.path.join(tmpdir, "input.wav")
            tmp_out = os.path.join(tmpdir, "output.wav")
            try:
                sr = int(audio["sample_rate"])
                waveform = audio["waveform"].squeeze(0)
                self._save_wav(tmp_in, waveform, sr)
                cmd = ["sox", tmp_in, tmp_out] + effect_params
                full_cmd_str = shlex.join(cmd)
                dbg_text = "** Enabled **\n" + full_cmd_str
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    dbg_text += f"\n*** SoX CLI failed (rc={result.returncode}):\n{result.stderr.strip() if result.stderr else 'No stderr'}"
                    processed_audio = audio
                else:
                    processed_waveform, processed_sr = self._load_wav(tmp_out)
                    processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\n*** SoX failed: {str(e)}"
                processed_audio = audio
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


class SoxVeAlienHighNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_alien_high": ("BOOLEAN", {"default": True}),
                "pitch_shift": ("INT", {"default": 400, "min": 0, "max": 1200, "step": 10}),
                "tremolo_speed": ("FLOAT", {"default": 4.5, "min": 0.1, "max": 20.0, "step": 0.1}),
                "tremolo_depth": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "highpass_freq": ("INT", {"default": 200, "min": 20, "max": 1000, "step": 10}),
                "lowpass_freq": ("INT", {"default": 4500, "min": 20, "max": 20000, "step": 50}),
                "chorus_gain_in": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_gain_out": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_delay": ("FLOAT", {"default": 35.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "chorus_decay": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 0.99, "step": 0.01}),
                "chorus_speed": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 2.0, "step": 0.01}),
                "chorus_depth": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "chorus_wave": (["sin", "tri"], {"default": "tri"}),
                "reverb_reverberance": ("INT", {"default": 60, "min": 0, "max": 100, "step": 1}),
                "reverb_hf_damping": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "reverb_roomscale": ("INT", {"default": 80, "min": 0, "max": 100, "step": 1}),
                "reverb_stereo_depth": ("INT", {"default": 70, "min": 0, "max": 100, "step": 1}),
                "reverb_pre_delay": ("INT", {"default": 10, "min": 0, "max": 500, "step": 1}),
                "gain_adjust": ("FLOAT", {"default": -4.0, "min": -20.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Alien High: pitch +400 tremolo 4.5 40 highpass 200 lowpass 4500 chorus 0.6 0.9 35 0.4 0.3 2 -t reverb 60 50 80 70 10 0 gain -4 — High thin ethereal slightly unstable."

    @staticmethod
    def _save_wav(filename, tensor, sample_rate):
        channels = tensor.shape[0]
        if channels == 1:
            data = tensor[0].cpu().numpy().astype(np.float32)
        else:
            data = tensor.transpose(0, 1).contiguous().flatten().cpu().numpy().astype(np.float32)
        byte_rate = sample_rate * channels * 4
        block_align = channels * 4
        fmt_chunk_size = 16
        fmt_chunk = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
        data_chunk = data.tobytes()
        data_size = len(data) * 4
        riff_size = 36 + data_size
        header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", fmt_chunk_size) + fmt_chunk + struct.pack("<4sI", b"data", data_size)
        with open(filename, "wb") as f:
            f.write(header + data_chunk)

    @staticmethod
    def _load_wav(filename):
        with open(filename, "rb") as f:
            if f.read(4) != b'RIFF':
                raise ValueError("Not RIFF")
            riff_size = struct.unpack('<I', f.read(4))[0]
            if f.read(4) != b'WAVE':
                raise ValueError("Not WAVE")
            sr = None
            channels = None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("Truncated file")
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data_start = f.tell()
                if chunk_id == b'fmt ':
                    fmt_hdr = f.read(chunk_size)
                    if len(fmt_hdr) < 16:
                        raise ValueError("Short fmt")
                    fmt = struct.unpack('<HHIIHH', fmt_hdr[:16])
                    if fmt[0] != 3 or fmt[5] != 32:
                        raise ValueError(f"Not float32 (fmt={fmt[0]}, bits={fmt[5]})")
                    channels = fmt[1]
                    sr = fmt[2]
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    data_size = len(data_bytes)
                    data_bytes = data_bytes[:data_size // 4 * 4]
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    if len(data) % channels != 0:
                        data = data[:len(data) // channels * channels]
                    data = data.reshape(channels, -1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                    return waveform, sr
                else:
                    f.seek(chunk_size + (chunk_size % 2), 1)
            raise ValueError("No data chunk")

    def process(self, audio, enable_voice_alien_high=True,
                pitch_shift=400,
                tremolo_speed=4.5, tremolo_depth=40.0,
                highpass_freq=200, lowpass_freq=4500,
                chorus_gain_in=0.6, chorus_gain_out=0.9, chorus_delay=35.0,
                chorus_decay=0.4, chorus_speed=0.3, chorus_depth=2.0, chorus_wave="tri",
                reverb_reverberance=60, reverb_hf_damping=50, reverb_roomscale=80, reverb_stereo_depth=70, reverb_pre_delay=10,
                gain_adjust=-4.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        chorus_wave_flag = "-t" if chorus_wave == "tri" else "-s"
        effect_params = [
            "pitch", f"+{pitch_shift}",
            "tremolo", str(tremolo_speed), str(tremolo_depth),
            "highpass", str(highpass_freq),
            "lowpass", str(lowpass_freq),
            "chorus", str(chorus_gain_in), str(chorus_gain_out), str(chorus_delay), str(chorus_decay), str(chorus_speed), str(chorus_depth), chorus_wave_flag,
            "reverb", str(reverb_reverberance), str(reverb_hf_damping), str(reverb_roomscale), str(reverb_stereo_depth), str(reverb_pre_delay), "0",
            "gain", f"{gain_adjust}"
        ]
        cmd_str = f"sox input.wav output.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_alien_high:
            current_params.extend(effect_params)
            tmpdir = tempfile.mkdtemp(prefix='sox_')
            tmp_in = os.path.join(tmpdir, "input.wav")
            tmp_out = os.path.join(tmpdir, "output.wav")
            try:
                sr = int(audio["sample_rate"])
                waveform = audio["waveform"].squeeze(0)
                self._save_wav(tmp_in, waveform, sr)
                cmd = ["sox", tmp_in, tmp_out] + effect_params
                full_cmd_str = shlex.join(cmd)
                dbg_text = "** Enabled **\n executed: " + full_cmd_str
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    dbg_text += f"\n*** SoX CLI failed (rc={result.returncode}):\n{result.stderr.strip() if result.stderr else 'No stderr'}"
                    processed_audio = audio
                else:
                    processed_waveform, processed_sr = self._load_wav(tmp_out)
                    processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\n*** SoX failed: {str(e)}"
                processed_audio = audio
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


class SoxVeAlienRobotNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_alien_robot": ("BOOLEAN", {"default": True}),
                "pitch_shift": ("INT", {"default": -300, "min": -1200, "max": 1200, "step": 10}),
                "overdrive_gain": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "phaser_gain_in": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "phaser_gain_out": ("FLOAT", {"default": 0.74, "min": 0.0, "max": 1.0, "step": 0.01}),
                "phaser_delay": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "phaser_decay": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 0.99, "step": 0.01}),
                "phaser_speed": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1}),
                "phaser_wave": (["sin", "tri"], {"default": "sin"}),
                "eq1_freq": ("INT", {"default": 800, "min": 20, "max": 20000, "step": 50}),
                "eq1_gain": ("FLOAT", {"default": -6.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "eq1_q": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "eq2_freq": ("INT", {"default": 3000, "min": 20, "max": 20000, "step": 50}),
                "eq2_gain": ("FLOAT", {"default": 8.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "eq2_q": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "echo_gain_in": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echo_gain_out": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echo_delay1": ("INT", {"default": 200, "min": 1, "max": 1000, "step": 10}),
                "echo_decay1": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echo_delay2": ("INT", {"default": 400, "min": 1, "max": 1000, "step": 10}),
                "echo_decay2": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gain_adjust": ("FLOAT", {"default": -5.0, "min": -20.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Alien Robot: pitch -300 overdrive 10 phaser 0.8 0.74 3 0.4 0.5 -s equalizer 800 -6 1q equalizer 3000 8 1q echo 0.8 0.9 200 0.4 400 0.3 gain -5 — Harsh buzzing metallic Dalek-ish."

    @staticmethod
    def _save_wav(filename, tensor, sample_rate):
        channels = tensor.shape[0]
        if channels == 1:
            data = tensor[0].cpu().numpy().astype(np.float32)
        else:
            data = tensor.transpose(0, 1).contiguous().flatten().cpu().numpy().astype(np.float32)
        byte_rate = sample_rate * channels * 4
        block_align = channels * 4
        fmt_chunk_size = 16
        fmt_chunk = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
        data_chunk = data.tobytes()
        data_size = len(data) * 4
        riff_size = 36 + data_size
        header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", fmt_chunk_size) + fmt_chunk + struct.pack("<4sI", b"data", data_size)
        with open(filename, "wb") as f:
            f.write(header + data_chunk)

    @staticmethod
    def _load_wav(filename):
        with open(filename, "rb") as f:
            if f.read(4) != b'RIFF':
                raise ValueError("Not RIFF")
            riff_size = struct.unpack('<I', f.read(4))[0]
            if f.read(4) != b'WAVE':
                raise ValueError("Not WAVE")
            sr = None
            channels = None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("Truncated file")
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data_start = f.tell()
                if chunk_id == b'fmt ':
                    fmt_hdr = f.read(chunk_size)
                    if len(fmt_hdr) < 16:
                        raise ValueError("Short fmt")
                    fmt = struct.unpack('<HHIIHH', fmt_hdr[:16])
                    if fmt[0] != 3 or fmt[5] != 32:
                        raise ValueError(f"Not float32 (fmt={fmt[0]}, bits={fmt[5]})")
                    channels = fmt[1]
                    sr = fmt[2]
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    data_size = len(data_bytes)
                    data_bytes = data_bytes[:data_size // 4 * 4]
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    if len(data) % channels != 0:
                        data = data[:len(data) // channels * channels]
                    data = data.reshape(channels, -1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                    return waveform, sr
                else:
                    f.seek(chunk_size + (chunk_size % 2), 1)
            raise ValueError("No data chunk")

    def process(self, audio, enable_voice_alien_robot=True,
                pitch_shift=-300,
                overdrive_gain=10.0,
                phaser_gain_in=0.8, phaser_gain_out=0.74, phaser_delay=3.0,
                phaser_decay=0.4, phaser_speed=0.5, phaser_wave="sin",
                eq1_freq=800, eq1_gain=-6.0, eq1_q=1.0,
                eq2_freq=3000, eq2_gain=8.0, eq2_q=1.0,
                echo_gain_in=0.8, echo_gain_out=0.9,
                echo_delay1=200, echo_decay1=0.4,
                echo_delay2=400, echo_decay2=0.3,
                gain_adjust=-5.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        phaser_wave_flag = "-s" if phaser_wave == "sin" else "-t"
        effect_params = [
            "pitch", str(pitch_shift),
            "overdrive", str(overdrive_gain),
            "phaser", str(phaser_gain_in), str(phaser_gain_out), str(phaser_delay), str(phaser_decay), str(phaser_speed), phaser_wave_flag,
            "equalizer", str(eq1_freq), f"{eq1_q}q", str(eq1_gain),
            "equalizer", str(eq2_freq), f"{eq2_q}q", str(eq2_gain),
            "echo", str(echo_gain_in), str(echo_gain_out), str(echo_delay1), str(echo_decay1), str(echo_delay2), str(echo_decay2),
            "gain", f"{gain_adjust}"
        ]
        cmd_str = f"sox input.wav output.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_alien_robot:
            current_params.extend(effect_params)
            tmpdir = tempfile.mkdtemp(prefix='sox_')
            tmp_in = os.path.join(tmpdir, "input.wav")
            tmp_out = os.path.join(tmpdir, "output.wav")
            try:
                sr = int(audio["sample_rate"])
                waveform = audio["waveform"].squeeze(0)
                self._save_wav(tmp_in, waveform, sr)
                cmd = ["sox", tmp_in, tmp_out] + effect_params
                full_cmd_str = shlex.join(cmd)
                dbg_text = "** Enabled **\n executed: " + full_cmd_str
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    dbg_text += f"\n*** SoX CLI failed (rc={result.returncode}):\n{result.stderr.strip() if result.stderr else 'No stderr'}"
                    processed_audio = audio
                else:
                    processed_waveform, processed_sr = self._load_wav(tmp_out)
                    processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\n*** SoX failed: {str(e)}"
                processed_audio = audio
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


class SoxVeAlienWarbleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_alien_warble": ("BOOLEAN", {"default": True}),
                "pitch_shift": ("INT", {"default": 250, "min": 0, "max": 1200, "step": 10}),
                "flanger_delay": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "flanger_depth": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "flanger_regen": ("FLOAT", {"default": 0.6, "min": -95.0, "max": 95.0, "step": 1.0}),
                "flanger_width": ("FLOAT", {"default": 80.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "flanger_speed": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 10.0, "step": 0.1}),
                "flanger_shape": (["sine", "triangle"], {"default": "sine"}),
                "tremolo_speed": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "tremolo_depth": ("FLOAT", {"default": 35.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "lowpass_freq": ("INT", {"default": 3500, "min": 20, "max": 20000, "step": 50}),
                "reverb_reverberance": ("INT", {"default": 75, "min": 0, "max": 100, "step": 1}),
                "reverb_hf_damping": ("INT", {"default": 60, "min": 0, "max": 100, "step": 1}),
                "reverb_roomscale": ("INT", {"default": 90, "min": 0, "max": 100, "step": 1}),
                "reverb_stereo_depth": ("INT", {"default": 80, "min": 0, "max": 100, "step": 1}),
                "reverb_pre_delay": ("INT", {"default": 20, "min": 0, "max": 500, "step": 1}),
                "gain_adjust": ("FLOAT", {"default": -6.0, "min": -20.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Alien Warble: pitch +250 flanger 2 1.5 0.6 80 0.8 sine tremolo 6 35 lowpass 3500 reverb 75 60 90 80 20 0 gain -6 — Swirling unstable insect-like."

    @staticmethod
    def _save_wav(filename, tensor, sample_rate):
        channels = tensor.shape[0]
        if channels == 1:
            data = tensor[0].cpu().numpy().astype(np.float32)
        else:
            data = tensor.transpose(0, 1).contiguous().flatten().cpu().numpy().astype(np.float32)
        byte_rate = sample_rate * channels * 4
        block_align = channels * 4
        fmt_chunk_size = 16
        fmt_chunk = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
        data_chunk = data.tobytes()
        data_size = len(data) * 4
        riff_size = 36 + data_size
        header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", fmt_chunk_size) + fmt_chunk + struct.pack("<4sI", b"data", data_size)
        with open(filename, "wb") as f:
            f.write(header + data_chunk)

    @staticmethod
    def _load_wav(filename):
        with open(filename, "rb") as f:
            if f.read(4) != b'RIFF':
                raise ValueError("Not RIFF")
            riff_size = struct.unpack('<I', f.read(4))[0]
            if f.read(4) != b'WAVE':
                raise ValueError("Not WAVE")
            sr = None
            channels = None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("Truncated file")
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data_start = f.tell()
                if chunk_id == b'fmt ':
                    fmt_hdr = f.read(chunk_size)
                    if len(fmt_hdr) < 16:
                        raise ValueError("Short fmt")
                    fmt = struct.unpack('<HHIIHH', fmt_hdr[:16])
                    if fmt[0] != 3 or fmt[5] != 32:
                        raise ValueError(f"Not float32 (fmt={fmt[0]}, bits={fmt[5]})")
                    channels = fmt[1]
                    sr = fmt[2]
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    data_size = len(data_bytes)
                    data_bytes = data_bytes[:data_size // 4 * 4]
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    if len(data) % channels != 0:
                        data = data[:len(data) // channels * channels]
                    data = data.reshape(channels, -1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                    return waveform, sr
                else:
                    f.seek(chunk_size + (chunk_size % 2), 1)
            raise ValueError("No data chunk")

    def process(self, audio, enable_voice_alien_warble=True,
                pitch_shift=250,
                flanger_delay=2.0, flanger_depth=1.5, flanger_regen=0.6, flanger_width=80.0, flanger_speed=0.8, flanger_shape="sine",
                tremolo_speed=6.0, tremolo_depth=35.0,
                lowpass_freq=3500,
                reverb_reverberance=75, reverb_hf_damping=60, reverb_roomscale=90, reverb_stereo_depth=80, reverb_pre_delay=20,
                gain_adjust=-6.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        effect_params = [
            "pitch", f"+{pitch_shift}",
            "flanger", str(flanger_delay), str(flanger_depth), str(flanger_regen), str(flanger_width), str(flanger_speed), flanger_shape,
            "tremolo", str(tremolo_speed), str(tremolo_depth),
            "lowpass", str(lowpass_freq),
            "reverb", str(reverb_reverberance), str(reverb_hf_damping), str(reverb_roomscale), str(reverb_stereo_depth), str(reverb_pre_delay), "0",
            "gain", f"{gain_adjust}"
        ]
        cmd_str = f"sox input.wav output.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_alien_warble:
            current_params.extend(effect_params)
            tmpdir = tempfile.mkdtemp(prefix='sox_')
            tmp_in = os.path.join(tmpdir, "input.wav")
            tmp_out = os.path.join(tmpdir, "output.wav")
            try:
                sr = int(audio["sample_rate"])
                waveform = audio["waveform"].squeeze(0)
                self._save_wav(tmp_in, waveform, sr)
                cmd = ["sox", tmp_in, tmp_out] + effect_params
                full_cmd_str = shlex.join(cmd)
                dbg_text = "** Enabled **\n executed: " + full_cmd_str
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    dbg_text += f"\n*** SoX CLI failed (rc={result.returncode}):\n{result.stderr.strip() if result.stderr else 'No stderr'}"
                    processed_audio = audio
                else:
                    processed_waveform, processed_sr = self._load_wav(tmp_out)
                    processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\n*** SoX failed: {str(e)}"
                processed_audio = audio
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


class SoxVeAlienGhostNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_alien_ghost": ("BOOLEAN", {"default": True}),
                "pitch_shift": ("INT", {"default": 180, "min": 0, "max": 1200, "step": 10}),
                "chorus_gain_in": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_gain_out": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_delay": ("FLOAT", {"default": 40.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "chorus_decay": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 0.99, "step": 0.01}),
                "chorus_speed": ("FLOAT", {"default": 0.25, "min": 0.1, "max": 2.0, "step": 0.01}),
                "chorus_depth": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "chorus_wave": (["sin", "tri"], {"default": "tri"}),
                "highpass_freq": ("INT", {"default": 250, "min": 20, "max": 1000, "step": 10}),
                "gain_adjust": ("FLOAT", {"default": -7.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "reverb_reverberance": ("INT", {"default": 80, "min": 0, "max": 100, "step": 1}),
                "reverb_hf_damping": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "reverb_roomscale": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "reverb_stereo_depth": ("INT", {"default": 90, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Alien Ghost: Reverse-Reverb mix + pitch +180 chorus highpass gain -7 — Breathy distant otherworldly. Uses a single-pass reverse-reverb chain for preview."

    @staticmethod
    def _save_wav(filename, tensor, sample_rate):
        channels = tensor.shape[0]
        if channels == 1:
            data = tensor[0].cpu().numpy().astype(np.float32)
        else:
            data = tensor.transpose(0, 1).contiguous().flatten().cpu().numpy().astype(np.float32)
        byte_rate = sample_rate * channels * 4
        block_align = channels * 4
        fmt_chunk_size = 16
        fmt_chunk = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
        data_chunk = data.tobytes()
        data_size = len(data) * 4
        riff_size = 36 + data_size
        header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", fmt_chunk_size) + fmt_chunk + struct.pack("<4sI", b"data", data_size)
        with open(filename, "wb") as f:
            f.write(header + data_chunk)

    @staticmethod
    def _load_wav(filename):
        with open(filename, "rb") as f:
            if f.read(4) != b'RIFF':
                raise ValueError("Not RIFF")
            riff_size = struct.unpack('<I', f.read(4))[0]
            if f.read(4) != b'WAVE':
                raise ValueError("Not WAVE")
            sr = None
            channels = None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("Truncated file")
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data_start = f.tell()
                if chunk_id == b'fmt ':
                    fmt_hdr = f.read(chunk_size)
                    if len(fmt_hdr) < 16:
                        raise ValueError("Short fmt")
                    fmt = struct.unpack('<HHIIHH', fmt_hdr[:16])
                    if fmt[0] != 3 or fmt[5] != 32:
                        raise ValueError(f"Not float32 (fmt={fmt[0]}, bits={fmt[5]})")
                    channels = fmt[1]
                    sr = fmt[2]
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    data_size = len(data_bytes)
                    data_bytes = data_bytes[:data_size // 4 * 4]
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    if len(data) % channels != 0:
                        data = data[:len(data) // channels * channels]
                    data = data.reshape(channels, -1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                    return waveform, sr
                else:
                    f.seek(chunk_size + (chunk_size % 2), 1)
            raise ValueError("No data chunk")

    def process(self, audio, enable_voice_alien_ghost=True,
                pitch_shift=180,
                chorus_gain_in=0.5, chorus_gain_out=0.9, chorus_delay=40.0,
                chorus_decay=0.35, chorus_speed=0.25, chorus_depth=3.0, chorus_wave="tri",
                highpass_freq=250, gain_adjust=-7.0,
                reverb_reverberance=80, reverb_hf_damping=50, reverb_roomscale=100, reverb_stereo_depth=90,
                prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        chorus_wave_flag = "-t" if chorus_wave == "tri" else "-s"
        
        # Approximate the 2-step process with a single chain for preview simplification,
        # or implement it correctly by mixing. 
        # For preview, we'll do the requested: mix input + (reversed -> reverb -> reversed)
        # However, 'sox_params' for CLI usually refers to a single linear chain.
        # We will provide a linear approximation: reverse reverb reverse pitch chorus highpass gain.
        # This creates the "pre-echo" effect.
        effect_params = [
            "reverse",
            "reverb", str(reverb_reverberance), str(reverb_hf_damping), str(reverb_roomscale), str(reverb_stereo_depth), "0", "0",
            "reverse",
            "pitch", f"+{pitch_shift}",
            "chorus", str(chorus_gain_in), str(chorus_gain_out), str(chorus_delay), str(chorus_decay), str(chorus_speed), str(chorus_depth), chorus_wave_flag,
            "highpass", str(highpass_freq),
            "gain", f"{gain_adjust}"
        ]
        
        cmd_str = f"sox input.wav output.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_alien_ghost:
            current_params.extend(effect_params)
            tmpdir = tempfile.mkdtemp(prefix='sox_')
            tmp_in = os.path.join(tmpdir, "input.wav")
            tmp_out = os.path.join(tmpdir, "output.wav")
            try:
                sr = int(audio["sample_rate"])
                waveform = audio["waveform"].squeeze(0)
                self._save_wav(tmp_in, waveform, sr)
                
                # Executing the full mix logic for preview:
                # sox -m input.wav <(sox input.wav -p reverse reverb ... reverse) output.wav ...
                # Since shlex.join doesn't like process substitution, we'll use temp files.
                tmp_rev = os.path.join(tmpdir, "rev_reverb.wav")
                rev_cmd = ["sox", tmp_in, tmp_rev, "reverse", "reverb", str(reverb_reverberance), str(reverb_hf_damping), str(reverb_roomscale), str(reverb_stereo_depth), "0", "0", "reverse"]
                subprocess.run(rev_cmd, capture_output=True)
                
                mix_cmd = ["sox", "-m", tmp_in, tmp_rev, tmp_out, "pitch", f"+{pitch_shift}", "chorus", str(chorus_gain_in), str(chorus_gain_out), str(chorus_delay), str(chorus_decay), str(chorus_speed), str(chorus_depth), chorus_wave_flag, "highpass", str(highpass_freq), "gain", f"{gain_adjust}"]
                
                full_cmd_str = "Preview using: " + shlex.join(mix_cmd)
                dbg_text = "** Enabled **\n" + full_cmd_str
                result = subprocess.run(mix_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    dbg_text += f"\n*** SoX CLI failed (rc={result.returncode}):\n{result.stderr.strip() if result.stderr else 'No stderr'}"
                    processed_audio = audio
                else:
                    processed_waveform, processed_sr = self._load_wav(tmp_out)
                    processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\n*** SoX failed: {str(e)}"
                processed_audio = audio
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


class SoxVeAlienDemonNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_alien_demon": ("BOOLEAN", {"default": True}),
                "pitch_shift": ("INT", {"default": -500, "min": -1200, "max": 0, "step": 10}),
                "overdrive_gain": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "tremolo_speed": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "tremolo_depth": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "compand_attack": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "compand_release": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "compand_points": ("STRING", {"default": "6:-70,-60,-20,-19,0", "tooltip": "Transfer points e.g. 6:-70,-60,-20"}),
                "reverb_reverberance": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "reverb_hf_damping": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1}),
                "reverb_roomscale": ("INT", {"default": 70, "min": 0, "max": 100, "step": 1}),
                "reverb_stereo_depth": ("INT", {"default": 60, "min": 0, "max": 100, "step": 1}),
                "gain_adjust": ("FLOAT", {"default": -3.0, "min": -20.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Alien Demon Growl: pitch -500 overdrive 15 tremolo 3 50 compand reverb gain -3 — Low distorted vibrating aggressive."

    @staticmethod
    def _save_wav(filename, tensor, sample_rate):
        channels = tensor.shape[0]
        if channels == 1:
            data = tensor[0].cpu().numpy().astype(np.float32)
        else:
            data = tensor.transpose(0, 1).contiguous().flatten().cpu().numpy().astype(np.float32)
        byte_rate = sample_rate * channels * 4
        block_align = channels * 4
        fmt_chunk_size = 16
        fmt_chunk = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
        data_chunk = data.tobytes()
        data_size = len(data) * 4
        riff_size = 36 + data_size
        header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", fmt_chunk_size) + fmt_chunk + struct.pack("<4sI", b"data", data_size)
        with open(filename, "wb") as f:
            f.write(header + data_chunk)

    @staticmethod
    def _load_wav(filename):
        with open(filename, "rb") as f:
            if f.read(4) != b'RIFF':
                raise ValueError("Not RIFF")
            riff_size = struct.unpack('<I', f.read(4))[0]
            if f.read(4) != b'WAVE':
                raise ValueError("Not WAVE")
            sr = None
            channels = None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("Truncated file")
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data_start = f.tell()
                if chunk_id == b'fmt ':
                    fmt_hdr = f.read(chunk_size)
                    if len(fmt_hdr) < 16:
                        raise ValueError("Short fmt")
                    fmt = struct.unpack('<HHIIHH', fmt_hdr[:16])
                    if fmt[0] != 3 or fmt[5] != 32:
                        raise ValueError(f"Not float32 (fmt={fmt[0]}, bits={fmt[5]})")
                    channels = fmt[1]
                    sr = fmt[2]
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    data_size = len(data_bytes)
                    data_bytes = data_bytes[:data_size // 4 * 4]
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    if len(data) % channels != 0:
                        data = data[:len(data) // channels * channels]
                    data = data.reshape(channels, -1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                    return waveform, sr
                else:
                    f.seek(chunk_size + (chunk_size % 2), 1)
            raise ValueError("No data chunk")

    def process(self, audio, enable_voice_alien_demon=True,
                pitch_shift=-500,
                overdrive_gain=15.0,
                tremolo_speed=3.0, tremolo_depth=50.0,
                compand_attack=0.3, compand_release=0.8, compand_points="6:-70,-60,-20,-19,0",
                reverb_reverberance=50, reverb_hf_damping=40, reverb_roomscale=70, reverb_stereo_depth=60,
                gain_adjust=-3.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        compand_arg = f"{compand_attack},{compand_release} {compand_points}"
        effect_params = [
            "pitch", str(pitch_shift),
            "overdrive", str(overdrive_gain),
            "tremolo", str(tremolo_speed), str(tremolo_depth),
            "compand"
        ] + shlex.split(compand_arg) + [
            "reverb", str(reverb_reverberance), str(reverb_hf_damping), str(reverb_roomscale), str(reverb_stereo_depth), "0", "0",
            "gain", f"{gain_adjust}"
        ]
        cmd_str = f"sox input.wav output.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_alien_demon:
            current_params.extend(effect_params)
            tmpdir = tempfile.mkdtemp(prefix='sox_')
            tmp_in = os.path.join(tmpdir, "input.wav")
            tmp_out = os.path.join(tmpdir, "output.wav")
            try:
                sr = int(audio["sample_rate"])
                waveform = audio["waveform"].squeeze(0)
                self._save_wav(tmp_in, waveform, sr)
                cmd = ["sox", tmp_in, tmp_out] + effect_params
                full_cmd_str = shlex.join(cmd)
                dbg_text = "** Enabled **\n executed: " + full_cmd_str
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    dbg_text += f"\n*** SoX CLI failed (rc={result.returncode}):\n{result.stderr.strip() if result.stderr else 'No stderr'}"
                    processed_audio = audio
                else:
                    processed_waveform, processed_sr = self._load_wav(tmp_out)
                    processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\n*** SoX failed: {str(e)}"
                processed_audio = audio
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


class SoxVeGhostNode:
    # Tested: DGS v0.1.2
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_ghost": ("BOOLEAN", {"default": True}),
                "highpass_freq": ("INT", {"default": 200, "min": 20, "max": 1000, "step": 10}),
                "pitch_shift": ("INT", {"default": 350, "min": 0, "max": 1200, "step": 10}),
                "chorus_gain_in": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_gain_out": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_delay": ("FLOAT", {"default": 50.0, "min": 20.1, "max": 200.0, "step": 0.1}),
                "chorus_decay": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 0.99, "step": 0.01}),
                "chorus_speed": ("FLOAT", {"default": 0.25, "min": 0.1, "max": 2.0, "step": 0.01}),
                "chorus_depth": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "chorus_wave": (["sin", "tri"], {"default": "tri"}),
                "reverb_reverberance": ("INT", {"default": 85, "min": 0, "max": 100, "step": 1}),
                "reverb_hf_damping": ("INT", {"default": 70, "min": 50, "max": 100, "step": 1}),
                "reverb_roomscale": ("INT", {"default": 90, "min": 0, "max": 100, "step": 1}),
                "reverb_stereo_depth": ("INT", {"default": 90, "min": 0, "max": 100, "step": 1}),
                "reverb_pre_delay": ("INT", {"default": 20, "min": 0, "max": 500, "step": 1}),
                "echos_gain_in": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echos_gain_out": ("FLOAT", {"default": 0.88, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echos_delay1": ("INT", {"default": 1, "min": 1, "max": 2000, "step": 10}),
                "echos_decay1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echos_delay2": ("INT", {"default": 1, "min": 1, "max": 2000, "step": 1}),
                "echos_decay2": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echos_delay3": ("INT", {"default": 1, "min": 1, "max": 4000, "step": 1}),
                "echos_decay3": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lowpass_freq": ("INT", {"default": 4000, "min": 20, "max": 20000, "step": 50}),
                "gain_adjust": ("FLOAT", {"default": -8.0, "min": -20.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Scary Ghost: highpass 200 pitch +350 chorus 0.7 0.9 50 0.4 0.25 3 -t reverb 85 70 90 90 20 0 echos 0.8 0.88 300 0.5 600 0.3 1200 0.2 lowpass 4000 gain -8 — Eerie thin ghostly shimmer reverb multi-echoes (intensity scales pitch/depths/decays/etc.)."
    @staticmethod
    def _save_wav(filename, tensor, sample_rate):
        channels = tensor.shape[0]
        if channels == 1:
            data = tensor[0].cpu().numpy().astype(np.float32)
        else:
            data = tensor.transpose(0, 1).contiguous().flatten().cpu().numpy().astype(np.float32)
        byte_rate = sample_rate * channels * 4
        block_align = channels * 4
        fmt_chunk_size = 16
        fmt_chunk = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
        data_chunk = data.tobytes()
        data_size = len(data) * 4
        riff_size = 36 + data_size
        header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", fmt_chunk_size) + fmt_chunk + struct.pack("<4sI", b"data", data_size)
        with open(filename, "wb") as f:
            f.write(header + data_chunk)
    @staticmethod
    def _load_wav(filename):
        with open(filename, "rb") as f:
            if f.read(4) != b'RIFF':
                raise ValueError("Not RIFF")
            riff_size = struct.unpack('<I', f.read(4))[0]
            if f.read(4) != b'WAVE':
                raise ValueError("Not WAVE")
            sr = None
            channels = None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("Truncated file")
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data_start = f.tell()
                if chunk_id == b'fmt ':
                    fmt_hdr = f.read(chunk_size)
                    if len(fmt_hdr) < 16:
                        raise ValueError("Short fmt")
                    fmt = struct.unpack('<HHIIHH', fmt_hdr[:16])
                    if fmt[0] != 3 or fmt[5] != 32:
                        raise ValueError(f"Not float32 (fmt={fmt[0]}, bits={fmt[5]})")
                    channels = fmt[1]
                    sr = fmt[2]
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    data_size = len(data_bytes)
                    data_bytes = data_bytes[:data_size // 4 * 4]
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    if len(data) % channels != 0:
                        data = data[:len(data) // channels * channels]
                    data = data.reshape(channels, -1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                    return waveform, sr
                else:
                    f.seek(chunk_size + (chunk_size % 2), 1)
            raise ValueError("No data chunk")
    
    def process(self, audio, enable_voice_ghost=True,
                highpass_freq=200,
                pitch_shift=350,
                chorus_gain_in=0.7, chorus_gain_out=0.9, chorus_delay=21.0,
                chorus_decay=0.4, chorus_speed=0.25, chorus_depth=3.0, chorus_wave="tri",
                reverb_reverberance=85, reverb_hf_damping=70, reverb_roomscale=90, reverb_stereo_depth=90, reverb_pre_delay=20,
                echos_gain_in=0.8, echos_gain_out=0.88,
                echos_delay1=300, echos_decay1=0.5,
                echos_delay2=600, echos_decay2=0.3,
                echos_delay3=1200, echos_decay3=0.2,
                lowpass_freq=4000, gain_adjust=-8.0, prev_params=None):
        chorus_delay = min(chorus_delay, 21.0)
        current_params = prev_params["sox_params"] if prev_params else []
        lowpass_freq = min(lowpass_freq, 20000) # lowpass_freq max is 20000
        chorus_wave_flag = "-t" if chorus_wave == "tri" else "-s"
        effect_params = [
            "highpass", str(highpass_freq),
            "pitch", f"+{pitch_shift}",
            "chorus", str(chorus_gain_in), str(chorus_gain_out), str(chorus_delay), str(chorus_decay), str(chorus_speed), str(chorus_depth), chorus_wave_flag,
            "reverb", str(reverb_reverberance), str(reverb_hf_damping), str(reverb_roomscale), str(reverb_stereo_depth), str(reverb_pre_delay), "0",
            "echos", str(echos_gain_in), str(echos_gain_out), str(echos_delay1), str(echos_decay1), str(echos_delay2), str(echos_decay2), str(echos_delay3), str(echos_decay3),
            "lowpass", str(lowpass_freq),
            "gain", f"{gain_adjust}"
        ]
        cmd_str = f"sox input.wav output.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_ghost:
            current_params.extend(effect_params)
            tmpdir = tempfile.mkdtemp(prefix='sox_')
            tmp_in = os.path.join(tmpdir, "input.wav")
            tmp_out = os.path.join(tmpdir, "output.wav")
            try:
                sr = int(audio["sample_rate"])
                waveform = audio["waveform"].squeeze(0)
                self._save_wav(tmp_in, waveform, sr)
                cmd = ["sox", tmp_in, tmp_out] + effect_params
                full_cmd_str = shlex.join(cmd)
                dbg_text = "** Enabled **\n executed: " + full_cmd_str
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    dbg_text += f"\n*** SoX CLI failed (rc={result.returncode}):\n{result.stderr.strip() if result.stderr else 'No stderr'}"
                    processed_audio = audio
                else:
                    processed_waveform, processed_sr = self._load_wav(tmp_out)
                    processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\n*** SoX failed: {str(e)}"
                processed_audio = audio
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            dbg_text = "** Enabled **\n" + cmd_str
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Echo Cave: echo 0.8 0.88 6 0.6 — Spacious repeats."

    def process(self, audio, enable_voice_echo_cave=True, echo_gain_in=0.8, echo_gain_out=0.88, echo_delay_1=6.0, echo_decay_1=0.6, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        effect_params = ["echo", str(echo_gain_in), str(echo_gain_out), str(echo_delay_1), str(echo_decay_1)]
        if enable_voice_echo_cave:
            processed_waveform, processed_sr = torchaudio.sox_effects.apply_effects_tensor(
                audio["waveform"],
                int(audio["sample_rate"]),
                effect_params,
                channels_first=True,
            )
            processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
        else:
            processed_audio = audio
        cmd_str = f"sox voice.wav echo_cave.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_echo_cave:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Telephone: highpass 300 lowpass 3000 — Muffled band-pass."

    def process(self, audio, enable_voice_telephone=True, highpass_freq=300.0, lowpass_freq=3000.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        effect_params = ["highpass", str(highpass_freq), "lowpass", str(lowpass_freq)]
        if enable_voice_telephone:
            processed_waveform, processed_sr = torchaudio.sox_effects.apply_effects_tensor(
                audio["waveform"],
                int(audio["sample_rate"]),
                effect_params,
                channels_first=True,
            )
            processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
        else:
            processed_audio = audio
        cmd_str = f"sox voice.wav telephone.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_telephone:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Distorted Monster: overdrive 20 20 — Gritty clipping."

    def process(self, audio, enable_voice_monster=True, intensity=1.0, overdrive_gain=20.0, overdrive_color=20.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        scaled_gain = int(overdrive_gain * intensity)
        scaled_color = int(overdrive_color * intensity)
        effect_params = ["overdrive", str(scaled_gain), str(scaled_color)]
        if enable_voice_monster:
            processed_waveform, processed_sr = torchaudio.sox_effects.apply_effects_tensor(
                audio["waveform"],
                int(audio["sample_rate"]),
                effect_params,
                channels_first=True,
            )
            processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
        else:
            processed_audio = audio
        cmd_str = f"sox voice.wav monster.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_monster:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Compressed Robot: compand 0.3,0.8 6:-70,-60,-20 — Punchy dynamic squeeze."

    def process(self, audio, enable_voice_compand_robot=True, intensity=1.0, compand_attack=0.3, compand_release=0.8, compand_points="6:-70,-60,-20", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        attack_str = f"{compand_attack * intensity},{compand_release * intensity}"
        compand_str = f"{attack_str} {compand_points}"
        effect_params = ["compand"] + shlex.split(compand_str)
        if enable_voice_compand_robot:
            processed_waveform, processed_sr = torchaudio.sox_effects.apply_effects_tensor(
                audio["waveform"],
                int(audio["sample_rate"]),
                effect_params,
                channels_first=True,
            )
            processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
        else:
            processed_audio = audio
        cmd_str = f"sox voice.wav compand_robot.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_compand_robot:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Boomy Demon: lowpass -1 200 — Muddy rumble."

    def process(self, audio, enable_voice_boomy_demon=True, lowpass_rolloff="-1", lowpass_width=200.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        effect_params = ["lowpass", lowpass_rolloff, str(lowpass_width)]
        if enable_voice_boomy_demon:
            processed_waveform, processed_sr = torchaudio.sox_effects.apply_effects_tensor(
                audio["waveform"],
                int(audio["sample_rate"]),
                effect_params,
                channels_first=True,
            )
            processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
        else:
            processed_audio = audio
        cmd_str = f"sox voice.wav boomy_demon.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_boomy_demon:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Screechy Witch: treble +20 5000 0.5q — Harsh highs."

    def process(self, audio, enable_voice_witch=True, intensity=1.0, treble_gain=20.0, treble_freq=5000.0, treble_width=0.5, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        scaled_gain = treble_gain * intensity
        effect_params = ["treble", f"+{scaled_gain}", str(treble_freq), f"{treble_width}q"]
        if enable_voice_witch:
            processed_waveform, processed_sr = torchaudio.sox_effects.apply_effects_tensor(
                audio["waveform"],
                int(audio["sample_rate"]),
                effect_params,
                channels_first=True,
            )
            processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
        else:
            processed_audio = audio
        cmd_str = f"sox voice.wav witch.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_witch:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Warble: bend 4000 sin(0.3) 800 sin(1) — Pitch wobble."

    def process(self, audio, enable_voice_warble=True, bend_high=4000.0, bend_low=800.0, wave_high="sin(0.3)", wave_low="sin(1)", prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        effect_params = ["bend", str(bend_high), wave_high, str(bend_low), wave_low]
        if enable_voice_warble:
            processed_waveform, processed_sr = torchaudio.sox_effects.apply_effects_tensor(
                audio["waveform"],
                int(audio["sample_rate"]),
                effect_params,
                channels_first=True,
            )
            processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
        else:
            processed_audio = audio
        cmd_str = f"sox voice.wav warble.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_warble:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Temple Reverb: reverb 80 50 100 0 — Long decay hall."

    def process(self, audio, enable_voice_temple=True, reverb_reverberance=80.0, reverb_hf=50.0, reverb_room=100.0, reverb_damp=0.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        effect_params = ["reverb", str(reverb_reverberance), str(reverb_hf), str(reverb_room), str(reverb_damp)]
        if enable_voice_temple:
            processed_waveform, processed_sr = torchaudio.sox_effects.apply_effects_tensor(
                audio["waveform"],
                int(audio["sample_rate"]),
                effect_params,
                channels_first=True,
            )
            processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
        else:
            processed_audio = audio
        cmd_str = f"sox voice.wav temple.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_temple:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Fast Squirrel: speed 1.5 — Chipmunk + faster."

    def process(self, audio, enable_voice_squirrel=True, intensity=1.0, speed_factor=1.5, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        scaled_speed = speed_factor * intensity
        effect_params = ["speed", str(scaled_speed)]
        if enable_voice_squirrel:
            processed_waveform, processed_sr = torchaudio.sox_effects.apply_effects_tensor(
                audio["waveform"],
                int(audio["sample_rate"]),
                effect_params,
                channels_first=True,
            )
            processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
        else:
            processed_audio = audio
        cmd_str = f"sox voice.wav squirrel.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_squirrel:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Slow Giant: tempo 0.8 — Time-stretch without pitch drop."

    def process(self, audio, enable_voice_giant=True, intensity=1.0, tempo_factor=0.8, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        scaled_tempo = tempo_factor * intensity
        effect_params = ["tempo", str(scaled_tempo)]
        if enable_voice_giant:
            processed_waveform, processed_sr = torchaudio.sox_effects.apply_effects_tensor(
                audio["waveform"],
                int(audio["sample_rate"]),
                effect_params,
                channels_first=True,
            )
            processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
        else:
            processed_audio = audio
        cmd_str = f"sox voice.wav giant.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_giant:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Vibrato: tremolo 0.3 90 — Pitch modulation."

    def process(self, audio, enable_voice_vibrato=True, intensity=1.0, tremolo_speed=0.3, tremolo_depth=90.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        scaled_speed = tremolo_speed * intensity
        scaled_depth = tremolo_depth * intensity
        effect_params = ["tremolo", str(scaled_speed), str(scaled_depth)]
        if enable_voice_vibrato:
            processed_waveform, processed_sr = torchaudio.sox_effects.apply_effects_tensor(
                audio["waveform"],
                int(audio["sample_rate"]),
                effect_params,
                channels_first=True,
            )
            processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
        else:
            processed_audio = audio
        cmd_str = f"sox voice.wav vibrato.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_vibrato:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Evil Demon: pitch -8 bass +10 tremolo 0.15 80 — Low + rumble + shake."

    def process(self, audio, enable_voice_evil_demon=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        pitch_shift = int(-8 * intensity)
        bass_gain = 10 * intensity
        effect_params = ["pitch", f"{pitch_shift}", "bass", f"+{bass_gain}", "tremolo", "0.15", "80"]
        if enable_voice_evil_demon:
            processed_waveform, processed_sr = torchaudio.sox_effects.apply_effects_tensor(
                audio["waveform"],
                int(audio["sample_rate"]),
                effect_params,
                channels_first=True,
            )
            processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
        else:
            processed_audio = audio
        cmd_str = f"sox voice.wav evil_demon.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_evil_demon:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Cartoon Duck: pitch +5 chorus 0.4 0.8 40 0.3 0.2 3 speed 1.1 — Squawky wobble."

    def process(self, audio, enable_voice_cartoon_duck=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        pitch_shift = int(5 * intensity)
        effect_params = ["pitch", f"+{pitch_shift}", "chorus", "0.4", "0.8", "40", "0.3", "0.2", "3", "speed", "1.1"]
        if enable_voice_cartoon_duck:
            processed_waveform, processed_sr = torchaudio.sox_effects.apply_effects_tensor(
                audio["waveform"],
                int(audio["sample_rate"]),
                effect_params,
                channels_first=True,
            )
            processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
        else:
            processed_audio = audio
        cmd_str = f"sox voice.wav cartoon_duck.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_cartoon_duck:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Darth Vader: pitch -7 lowpass -1 800 reverb 50 50 100 0 — Breathing mask reverb."

    def process(self, audio, enable_voice_darth_vader=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        pitch_shift = int(-7 * intensity)
        effect_params = ["pitch", f"{pitch_shift}", "lowpass", "-1", "800", "reverb", "50", "50", "100", "0"]
        if enable_voice_darth_vader:
            processed_waveform, processed_sr = torchaudio.sox_effects.apply_effects_tensor(
                audio["waveform"],
                int(audio["sample_rate"]),
                effect_params,
                channels_first=True,
            )
            processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
        else:
            processed_audio = audio
        cmd_str = f"sox voice.wav darth_vader.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_darth_vader:
            dbg_text = "** Enabled **\\n" + cmd_str
            current_params.extend(effect_params)
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
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
            try:
                waveform = audio["waveform"].squeeze(0)
                processed_waveform, processed_sr = torchaudio.sox_effects.apply_effects_tensor(
                    waveform,
                    int(audio["sample_rate"]),
                    effect_params,
                    channels_first=True,
                )
                processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\\nPreview failed: {str(e)}"
                processed_audio = audio
        else:
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


class SoxVeMinionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_voice_minion": ("BOOLEAN", {"default": True}),
                "pitch_shift": ("INT", {"default": 7, "min": 0, "max": 24, "tooltip": "Pitch shift in semitones."}),
                "chorus_gain_in": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Chorus input gain, scaled by intensity."}),
                "chorus_gain_out": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Chorus output gain, scaled by intensity."}),
                "chorus_delay": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 100.0, "step": 0.1, "tooltip": "Chorus delay in ms, scaled by intensity."}),
                "chorus_decay": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Chorus decay, scaled by intensity."}),
                "chorus_speed": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Chorus speed in Hz, scaled by intensity."}),
                "chorus_depth": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Chorus depth, scaled by intensity."}),
                "chorus_wave": (["sin", "tri"], {"default": "tri", "tooltip": "Chorus waveform shape."}),
            },
            "optional": {"prev_params": ("SOX_PARAMS",)}
        }
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Minion: pitch +7 chorus 0.3 0.95 30 0.5 0.15 5 — Bubbly chorus."

    @staticmethod
    def _save_wav(filename, tensor, sample_rate):
        channels = tensor.shape[0]
        if channels == 1:
            data = tensor[0].cpu().numpy().astype(np.float32)
        else:
            data = tensor.transpose(0, 1).contiguous().flatten().cpu().numpy().astype(np.float32)
        byte_rate = sample_rate * channels * 4
        block_align = channels * 4
        fmt_chunk_size = 16
        fmt_chunk = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
        data_chunk = data.tobytes()
        data_size = len(data) * 4
        riff_size = 36 + data_size
        header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", fmt_chunk_size) + fmt_chunk + struct.pack("<4sI", b"data", data_size)
        with open(filename, "wb") as f:
            f.write(header + data_chunk)

    @staticmethod
    def _load_wav(filename):
        with open(filename, "rb") as f:
            if f.read(4) != b'RIFF':
                raise ValueError("Not RIFF")
            riff_size = struct.unpack('<I', f.read(4))[0]
            if f.read(4) != b'WAVE':
                raise ValueError("Not WAVE")
            sr = None
            channels = None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("Truncated file")
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data_start = f.tell()
                if chunk_id == b'fmt ':
                    fmt_hdr = f.read(chunk_size)
                    if len(fmt_hdr) < 16:
                        raise ValueError("Short fmt")
                    fmt = struct.unpack('<HHIIHH', fmt_hdr[:16])
                    if fmt[0] != 3 or fmt[5] != 32:
                        raise ValueError(f"Not float32 (fmt={fmt[0]}, bits={fmt[5]})")
                    channels = fmt[1]
                    sr = fmt[2]
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    data_size = len(data_bytes)
                    data_bytes = data_bytes[:data_size // 4 * 4]
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    if len(data) % channels != 0:
                        data = data[:len(data) // channels * channels]
                    data = data.reshape(channels, -1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                    return waveform, sr
                else:
                    f.seek(chunk_size + (chunk_size % 2), 1)
            raise ValueError("No data chunk")

    def process(self, audio, prev_params=None, enable_voice_minion=True, pitch_shift=7, chorus_gain_in=0.3, chorus_gain_out=0.95, chorus_delay=30.0, chorus_decay=0.5, chorus_speed=0.15, chorus_depth=5.0, chorus_wave="tri"):
        current_params = prev_params["sox_params"] if prev_params else []
        effect_params = ["pitch", f"+{pitch_shift}", "chorus", "0.3", "0.95", "30", "0.5", "0.15", "5", "-t"]
        cmd_str = f"sox input.wav output.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_minion:
            current_params.extend(effect_params)
            tmpdir = tempfile.mkdtemp(prefix='sox_')
            tmp_in = os.path.join(tmpdir, "input.wav")
            tmp_out = os.path.join(tmpdir, "output.wav")
            try:
                sr = int(audio["sample_rate"])
                waveform = audio["waveform"].squeeze(0)
                self._save_wav(tmp_in, waveform, sr)
                cmd = ["sox", tmp_in, tmp_out] + effect_params
                full_cmd_str = shlex.join(cmd)
                dbg_text = "** Enabled **\n" + full_cmd_str
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    dbg_text += f"\n*** SoX CLI failed (rc={result.returncode}):\n{result.stderr.strip() if result.stderr else 'No stderr'}"
                    processed_audio = audio
                else:
                    processed_waveform, processed_sr = self._load_wav(tmp_out)
                    processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\n*** SoX failed: {str(e)}"
                processed_audio = audio
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Terminator: pitch -5 flanger 0.4 0.5 20 0.6 0.8 2 overdrive 10 5 — Mechanical flange grit."

    @staticmethod
    def _save_wav(filename, tensor, sample_rate):
        channels = tensor.shape[0]
        if channels == 1:
            data = tensor[0].cpu().numpy().astype(np.float32)
        else:
            data = tensor.transpose(0, 1).contiguous().flatten().cpu().numpy().astype(np.float32)
        byte_rate = sample_rate * channels * 4
        block_align = channels * 4
        fmt_chunk_size = 16
        fmt_chunk = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
        data_chunk = data.tobytes()
        data_size = len(data) * 4
        riff_size = 36 + data_size
        header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", fmt_chunk_size) + fmt_chunk + struct.pack("<4sI", b"data", data_size)
        with open(filename, "wb") as f:
            f.write(header + data_chunk)

    @staticmethod
    def _load_wav(filename):
        with open(filename, "rb") as f:
            if f.read(4) != b'RIFF':
                raise ValueError("Not RIFF")
            riff_size = struct.unpack('<I', f.read(4))[0]
            if f.read(4) != b'WAVE':
                raise ValueError("Not WAVE")
            sr = None
            channels = None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("Truncated file")
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data_start = f.tell()
                if chunk_id == b'fmt ':
                    fmt_hdr = f.read(chunk_size)
                    if len(fmt_hdr) < 16:
                        raise ValueError("Short fmt")
                    fmt = struct.unpack('<HHIIHH', fmt_hdr[:16])
                    if fmt[0] != 3 or fmt[5] != 32:
                        raise ValueError(f"Not float32 (fmt={fmt[0]}, bits={fmt[5]})")
                    channels = fmt[1]
                    sr = fmt[2]
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    data_size = len(data_bytes)
                    data_bytes = data_bytes[:data_size // 4 * 4]
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    if len(data) % channels != 0:
                        data = data[:len(data) // channels * channels]
                    data = data.reshape(channels, -1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                    return waveform, sr
                else:
                    f.seek(chunk_size + (chunk_size % 2), 1)
            raise ValueError("No data chunk")

    def process(self, audio, enable_voice_terminator=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        pitch_shift = int(-5 * intensity)
        overdrive_gain = int(10 * intensity)
        effect_params = ["pitch", f"{pitch_shift}", "flanger", "0.4", "0.5", "20", "0.6", "0.8", "2", "overdrive", str(overdrive_gain), "5"]
        cmd_str = f"sox input.wav output.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_terminator:
            current_params.extend(effect_params)
            tmpdir = tempfile.mkdtemp(prefix='sox_')
            tmp_in = os.path.join(tmpdir, "input.wav")
            tmp_out = os.path.join(tmpdir, "output.wav")
            try:
                sr = int(audio["sample_rate"])
                waveform = audio["waveform"].squeeze(0)
                self._save_wav(tmp_in, waveform, sr)
                cmd = ["sox", tmp_in, tmp_out] + effect_params
                full_cmd_str = shlex.join(cmd)
                dbg_text = "** Enabled **\n" + full_cmd_str
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    dbg_text += f"\n*** SoX CLI failed (rc={result.returncode}):\n{result.stderr.strip() if result.stderr else 'No stderr'}"
                    processed_audio = audio
                else:
                    processed_waveform, processed_sr = self._load_wav(tmp_out)
                    processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\n*** SoX failed: {str(e)}"
                processed_audio = audio
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Fairy: pitch +9 highpass 2000 reverb 30 90 80 10 — Airy sparkle."

    @staticmethod
    def _save_wav(filename, tensor, sample_rate):
        channels = tensor.shape[0]
        if channels == 1:
            data = tensor[0].cpu().numpy().astype(np.float32)
        else:
            data = tensor.transpose(0, 1).contiguous().flatten().cpu().numpy().astype(np.float32)
        byte_rate = sample_rate * channels * 4
        block_align = channels * 4
        fmt_chunk_size = 16
        fmt_chunk = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
        data_chunk = data.tobytes()
        data_size = len(data) * 4
        riff_size = 36 + data_size
        header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", fmt_chunk_size) + fmt_chunk + struct.pack("<4sI", b"data", data_size)
        with open(filename, "wb") as f:
            f.write(header + data_chunk)

    @staticmethod
    def _load_wav(filename):
        with open(filename, "rb") as f:
            if f.read(4) != b'RIFF':
                raise ValueError("Not RIFF")
            riff_size = struct.unpack('<I', f.read(4))[0]
            if f.read(4) != b'WAVE':
                raise ValueError("Not WAVE")
            sr = None
            channels = None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("Truncated file")
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data_start = f.tell()
                if chunk_id == b'fmt ':
                    fmt_hdr = f.read(chunk_size)
                    if len(fmt_hdr) < 16:
                        raise ValueError("Short fmt")
                    fmt = struct.unpack('<HHIIHH', fmt_hdr[:16])
                    if fmt[0] != 3 or fmt[5] != 32:
                        raise ValueError(f"Not float32 (fmt={fmt[0]}, bits={fmt[5]})")
                    channels = fmt[1]
                    sr = fmt[2]
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    data_size = len(data_bytes)
                    data_bytes = data_bytes[:data_size // 4 * 4]
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    if len(data) % channels != 0:
                        data = data[:len(data) // channels * channels]
                    data = data.reshape(channels, -1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                    return waveform, sr
                else:
                    f.seek(chunk_size + (chunk_size % 2), 1)
            raise ValueError("No data chunk")

    def process(self, audio, enable_voice_fairy=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        pitch_shift = int(9 * intensity)
        effect_params = ["pitch", f"+{pitch_shift}", "highpass", "2000", "reverb", "30", "90", "80", "10"]
        cmd_str = f"sox input.wav output.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_fairy:
            current_params.extend(effect_params)
            tmpdir = tempfile.mkdtemp(prefix='sox_')
            tmp_in = os.path.join(tmpdir, "input.wav")
            tmp_out = os.path.join(tmpdir, "output.wav")
            try:
                sr = int(audio["sample_rate"])
                waveform = audio["waveform"].squeeze(0)
                self._save_wav(tmp_in, waveform, sr)
                cmd = ["sox", tmp_in, tmp_out] + effect_params
                full_cmd_str = shlex.join(cmd)
                dbg_text = "** Enabled **\n" + full_cmd_str
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    dbg_text += f"\n*** SoX CLI failed (rc={result.returncode}):\n{result.stderr.strip() if result.stderr else 'No stderr'}"
                    processed_audio = audio
                else:
                    processed_waveform, processed_sr = self._load_wav(tmp_out)
                    processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\n*** SoX failed: {str(e)}"
                processed_audio = audio
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Zombie: pitch -10 echo 0.9 0.9 3 0.6 compand 0.2,0.8 6:-54,-30,-15 — Groan delay compress."

    @staticmethod
    def _save_wav(filename, tensor, sample_rate):
        channels = tensor.shape[0]
        if channels == 1:
            data = tensor[0].cpu().numpy().astype(np.float32)
        else:
            data = tensor.transpose(0, 1).contiguous().flatten().cpu().numpy().astype(np.float32)
        byte_rate = sample_rate * channels * 4
        block_align = channels * 4
        fmt_chunk_size = 16
        fmt_chunk = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
        data_chunk = data.tobytes()
        data_size = len(data) * 4
        riff_size = 36 + data_size
        header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", fmt_chunk_size) + fmt_chunk + struct.pack("<4sI", b"data", data_size)
        with open(filename, "wb") as f:
            f.write(header + data_chunk)

    @staticmethod
    def _load_wav(filename):
        with open(filename, "rb") as f:
            if f.read(4) != b'RIFF':
                raise ValueError("Not RIFF")
            riff_size = struct.unpack('<I', f.read(4))[0]
            if f.read(4) != b'WAVE':
                raise ValueError("Not WAVE")
            sr = None
            channels = None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("Truncated file")
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data_start = f.tell()
                if chunk_id == b'fmt ':
                    fmt_hdr = f.read(chunk_size)
                    if len(fmt_hdr) < 16:
                        raise ValueError("Short fmt")
                    fmt = struct.unpack('<HHIIHH', fmt_hdr[:16])
                    if fmt[0] != 3 or fmt[5] != 32:
                        raise ValueError(f"Not float32 (fmt={fmt[0]}, bits={fmt[5]})")
                    channels = fmt[1]
                    sr = fmt[2]
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    data_size = len(data_bytes)
                    data_bytes = data_bytes[:data_size // 4 * 4]
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    if len(data) % channels != 0:
                        data = data[:len(data) // channels * channels]
                    data = data.reshape(channels, -1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                    return waveform, sr
                else:
                    f.seek(chunk_size + (chunk_size % 2), 1)
            raise ValueError("No data chunk")

    def process(self, audio, enable_voice_zombie=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        pitch_shift = int(-10 * intensity)
        effect_params = ["pitch", f"{pitch_shift}", "echo", "0.9", "0.9", "3", "0.6", "compand", "0.2,0.8", "6:-54,-30,-15"]
        cmd_str = f"sox input.wav output.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_zombie:
            current_params.extend(effect_params)
            tmpdir = tempfile.mkdtemp(prefix='sox_')
            tmp_in = os.path.join(tmpdir, "input.wav")
            tmp_out = os.path.join(tmpdir, "output.wav")
            try:
                sr = int(audio["sample_rate"])
                waveform = audio["waveform"].squeeze(0)
                self._save_wav(tmp_in, waveform, sr)
                cmd = ["sox", tmp_in, tmp_out] + effect_params
                full_cmd_str = shlex.join(cmd)
                dbg_text = "** Enabled **\n" + full_cmd_str
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    dbg_text += f"\n*** SoX CLI failed (rc={result.returncode}):\n{result.stderr.strip() if result.stderr else 'No stderr'}"
                    processed_audio = audio
                else:
                    processed_waveform, processed_sr = self._load_wav(tmp_out)
                    processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\n*** SoX failed: {str(e)}"
                processed_audio = audio
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Pirate: bass +6 lowpass 2500 tremolo 0.2 120 equalizer 200 0.8 +4 — Rumbling growl."

    @staticmethod
    def _save_wav(filename, tensor, sample_rate):
        channels = tensor.shape[0]
        if channels == 1:
            data = tensor[0].cpu().numpy().astype(np.float32)
        else:
            data = tensor.transpose(0, 1).contiguous().flatten().cpu().numpy().astype(np.float32)
        byte_rate = sample_rate * channels * 4
        block_align = channels * 4
        fmt_chunk_size = 16
        fmt_chunk = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
        data_chunk = data.tobytes()
        data_size = len(data) * 4
        riff_size = 36 + data_size
        header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", fmt_chunk_size) + fmt_chunk + struct.pack("<4sI", b"data", data_size)
        with open(filename, "wb") as f:
            f.write(header + data_chunk)

    @staticmethod
    def _load_wav(filename):
        with open(filename, "rb") as f:
            if f.read(4) != b'RIFF':
                raise ValueError("Not RIFF")
            riff_size = struct.unpack('<I', f.read(4))[0]
            if f.read(4) != b'WAVE':
                raise ValueError("Not WAVE")
            sr = None
            channels = None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("Truncated file")
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data_start = f.tell()
                if chunk_id == b'fmt ':
                    fmt_hdr = f.read(chunk_size)
                    if len(fmt_hdr) < 16:
                        raise ValueError("Short fmt")
                    fmt = struct.unpack('<HHIIHH', fmt_hdr[:16])
                    if fmt[0] != 3 or fmt[5] != 32:
                        raise ValueError(f"Not float32 (fmt={fmt[0]}, bits={fmt[5]})")
                    channels = fmt[1]
                    sr = fmt[2]
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    data_size = len(data_bytes)
                    data_bytes = data_bytes[:data_size // 4 * 4]
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    if len(data) % channels != 0:
                        data = data[:len(data) // channels * channels]
                    data = data.reshape(channels, -1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                    return waveform, sr
                else:
                    f.seek(chunk_size + (chunk_size % 2), 1)
            raise ValueError("No data chunk")

    def process(self, audio, enable_voice_pirate=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        bass_gain = 6 * intensity
        trem_speed = 0.2 * intensity
        effect_params = ["bass", f"+{bass_gain}", "lowpass", "2500", "tremolo", str(trem_speed), "120", "equalizer", "200", "0.8", "+4"]
        cmd_str = f"sox input.wav output.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_pirate:
            current_params.extend(effect_params)
            tmpdir = tempfile.mkdtemp(prefix='sox_')
            tmp_in = os.path.join(tmpdir, "input.wav")
            tmp_out = os.path.join(tmpdir, "output.wav")
            try:
                sr = int(audio["sample_rate"])
                waveform = audio["waveform"].squeeze(0)
                self._save_wav(tmp_in, waveform, sr)
                cmd = ["sox", tmp_in, tmp_out] + effect_params
                full_cmd_str = shlex.join(cmd)
                dbg_text = "** Enabled **\n" + full_cmd_str
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    dbg_text += f"\n*** SoX CLI failed (rc={result.returncode}):\n{result.stderr.strip() if result.stderr else 'No stderr'}"
                    processed_audio = audio
                else:
                    processed_waveform, processed_sr = self._load_wav(tmp_out)
                    processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\n*** SoX failed: {str(e)}"
                processed_audio = audio
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


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
    RETURN_TYPES = ("AUDIO", "AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("orig-audio", "audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Voices"
    DESCRIPTION = "Superhero Echo: reverb 90 50 100 100 echo 0.7 0.8 10 0.3 pitch +2 — Epic boom."

    @staticmethod
    def _save_wav(filename, tensor, sample_rate):
        channels = tensor.shape[0]
        if channels == 1:
            data = tensor[0].cpu().numpy().astype(np.float32)
        else:
            data = tensor.transpose(0, 1).contiguous().flatten().cpu().numpy().astype(np.float32)
        byte_rate = sample_rate * channels * 4
        block_align = channels * 4
        fmt_chunk_size = 16
        fmt_chunk = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
        data_chunk = data.tobytes()
        data_size = len(data) * 4
        riff_size = 36 + data_size
        header = struct.pack("<4sI4s4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", fmt_chunk_size) + fmt_chunk + struct.pack("<4sI", b"data", data_size)
        with open(filename, "wb") as f:
            f.write(header + data_chunk)

    @staticmethod
    def _load_wav(filename):
        with open(filename, "rb") as f:
            if f.read(4) != b'RIFF':
                raise ValueError("Not RIFF")
            riff_size = struct.unpack('<I', f.read(4))[0]
            if f.read(4) != b'WAVE':
                raise ValueError("Not WAVE")
            sr = None
            channels = None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("Truncated file")
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_data_start = f.tell()
                if chunk_id == b'fmt ':
                    fmt_hdr = f.read(chunk_size)
                    if len(fmt_hdr) < 16:
                        raise ValueError("Short fmt")
                    fmt = struct.unpack('<HHIIHH', fmt_hdr[:16])
                    if fmt[0] != 3 or fmt[5] != 32:
                        raise ValueError(f"Not float32 (fmt={fmt[0]}, bits={fmt[5]})")
                    channels = fmt[1]
                    sr = fmt[2]
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                elif chunk_id == b'data':
                    data_bytes = f.read(chunk_size)
                    data_size = len(data_bytes)
                    data_bytes = data_bytes[:data_size // 4 * 4]
                    data = np.frombuffer(data_bytes, dtype=np.float32)
                    if len(data) % channels != 0:
                        data = data[:len(data) // channels * channels]
                    data = data.reshape(channels, -1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    f.seek(chunk_data_start + chunk_size + (chunk_size % 2), 0)
                    return waveform, sr
                else:
                    f.seek(chunk_size + (chunk_size % 2), 1)
            raise ValueError("No data chunk")

    def process(self, audio, enable_voice_superhero=True, intensity=1.0, prev_params=None):
        current_params = prev_params["sox_params"] if prev_params else []
        pitch_shift = int(2 * intensity)
        effect_params = ["reverb", "90", "50", "100", "100", "echo", "0.7", "0.8", "10", "0.3", "pitch", f"+{pitch_shift}"]
        cmd_str = f"sox input.wav output.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_voice_superhero:
            current_params.extend(effect_params)
            tmpdir = tempfile.mkdtemp(prefix='sox_')
            tmp_in = os.path.join(tmpdir, "input.wav")
            tmp_out = os.path.join(tmpdir, "output.wav")
            try:
                sr = int(audio["sample_rate"])
                waveform = audio["waveform"].squeeze(0)
                self._save_wav(tmp_in, waveform, sr)
                cmd = ["sox", tmp_in, tmp_out] + effect_params
                full_cmd_str = shlex.join(cmd)
                dbg_text = "** Enabled **\n" + full_cmd_str
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    dbg_text += f"\n*** SoX CLI failed (rc={result.returncode}):\n{result.stderr.strip() if result.stderr else 'No stderr'}"
                    processed_audio = audio
                else:
                    processed_waveform, processed_sr = self._load_wav(tmp_out)
                    processed_audio = {"waveform": processed_waveform, "sample_rate": processed_sr}
            except Exception as e:
                dbg_text += f"\n*** SoX failed: {str(e)}"
                processed_audio = audio
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            processed_audio = audio
        return (audio, processed_audio, {"sox_params": current_params}, dbg_text)


NODE_CLASS_MAPPINGS = {
    "SoxVeAlienDemon": SoxVeAlienDemonNode,
    "SoxVeAlienDemon": SoxVeAlienDemonNode,
    "SoxVeAlienGhost": SoxVeAlienGhostNode,
    "SoxVeAlienGhost": SoxVeAlienGhostNode,
    "SoxVeAlienHigh": SoxVeAlienHighNode,
    "SoxVeAlienHigh": SoxVeAlienHighNode,
    "SoxVeAlienJaba": SoxVeAlienJabaNode,
    "SoxVeAlienRobot": SoxVeAlienRobotNode,
    "SoxVeAlienRobot": SoxVeAlienRobotNode,
    "SoxVeAlienWarble": SoxVeAlienWarbleNode,
    "SoxVeAlienWarble": SoxVeAlienWarbleNode,
    "SoxVeBoomyDemon": SoxVeBoomyDemonNode,
    "SoxVeCartoonDuck": SoxVeCartoonDuckNode,
    "SoxVeChipmunk": SoxVeChipmunkNode,
    "SoxVeChipmunkChild": SoxVeChipmunkChildNode,
    "SoxVeCompandRobot": SoxVeCompandRobotNode,
    "SoxVeDarthVader": SoxVeDarthVaderNode,
    "SoxVeDeepOldMan": SoxVeDeepOldManNode,
    "SoxVeEchoCave": SoxVeEchoCaveNode,
    "SoxVeEvilDemon": SoxVeEvilDemonNode,
    "SoxVeFairy": SoxVeFairyNode,
    "SoxVeGhost": SoxVeGhostNode,
    "SoxVeGiant": SoxVeGiantNode,
    "SoxVeHelium": SoxVeHeliumNode,
    "SoxVeMinion": SoxVeMinionNode,
    "SoxVeMonster": SoxVeMonsterNode,
    "SoxVeOldWitch": SoxVeOldWitchNode,
    "SoxVePirate": SoxVePirateNode,
    "SoxVeRobot": SoxVeRobotNode,
    "SoxVeSquirrel": SoxVeSquirrelNode,
    "SoxVeSuperhero": SoxVeSuperheroNode,
    "SoxVeTelephone": SoxVeTelephoneNode,
    "SoxVeTemple": SoxVeTempleNode,
    "SoxVeTerminator": SoxVeTerminatorNode,
    "SoxVeVibrato": SoxVeVibratoNode,
    "SoxVeWarble": SoxVeWarbleNode,
    "SoxVeWitch": SoxVeWitchNode,
    "SoxVeZombie": SoxVeZombieNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SoxVeAlienDemon": "SoX Ve Alien Demon",
    "SoxVeAlienDemon": "SoX Ve Alien Demon",
    "SoxVeAlienGhost": "SoX Ve Alien Ghost",
    "SoxVeAlienGhost": "SoX Ve Alien Ghost",
    "SoxVeAlienHigh": "SoX Ve Alien High",
    "SoxVeAlienHigh": "SoX Ve Alien High",
    "SoxVeAlienJaba": "SoX Ve Alien Jaba",
    "SoxVeAlienRobot": "SoX Ve Alien Robot",
    "SoxVeAlienRobot": "SoX Ve Alien Robot",
    "SoxVeAlienWarble": "SoX Ve Alien Warble",
    "SoxVeAlienWarble": "SoX Ve Alien Warble",
    "SoxVeBoomyDemon": "SoX Ve Boomy Demon",
    "SoxVeCartoonDuck": "SoX Ve Cartoon Duck",
    "SoxVeChipmunk": "SoX Ve Chipmunk",
    "SoxVeChipmunkChild": "SoX Ve Chipmunk Child",
    "SoxVeCompandRobot": "SoX Ve Compand Robot",
    "SoxVeDarthVader": "SoX Ve Darth Vader",
    "SoxVeDeepOldMan": "SoX Ve Deep Old Man",
    "SoxVeEchoCave": "SoX Ve Echo Cave",
    "SoxVeEvilDemon": "SoX Ve Evil Demon",
    "SoxVeFairy": "SoX Ve Fairy",
    "SoxVeGhost": "SoX Ve Ghost",
    "SoxVeGiant": "SoX Ve Giant",
    "SoxVeHelium": "SoX Ve Helium",
    "SoxVeMinion": "SoX Ve Minion",
    "SoxVeMonster": "SoX Ve Monster",
    "SoxVeOldWitch": "SoX Ve Old Witch",
    "SoxVePirate": "SoX Ve Pirate",
    "SoxVeRobot": "SoX Ve Robot",
    "SoxVeSquirrel": "SoX Ve Squirrel",
    "SoxVeSuperhero": "SoX Ve Superhero",
    "SoxVeTelephone": "SoX Ve Telephone",
    "SoxVeTemple": "SoX Ve Temple",
    "SoxVeTerminator": "SoX Ve Terminator",
    "SoxVeVibrato": "SoX Ve Vibrato",
    "SoxVeWarble": "SoX Ve Warble",
    "SoxVeWitch": "SoX Ve Witch",
    "SoxVeZombie": "SoX Ve Zombie",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
