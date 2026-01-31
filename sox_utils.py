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
        cmd_str = "sox input.wav output.wav " + shlex.join(
            sox_cmd_params) if sox_cmd_params else "No effects applied (audio passed through)."

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


class SoxUtilSpectrogramNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_spectrogram": ("BOOLEAN", {"default": True, "tooltip": "Enable spectrogram generation"}),
            },
            "optional": {
                "audio-0": ("AUDIO", {"tooltip": "Audio input 0 for spectrogram (mono auto→stereo)."}),
                "audio-1": ("AUDIO", {"tooltip": "Audio input 1 for spectrogram (mono auto→stereo)."}),
                "audio-2": ("AUDIO", {"tooltip": "Audio input 2 for spectrogram (mono auto→stereo)."}),
                "audio-3": ("AUDIO", {"tooltip": "Audio input 3 for spectrogram (mono auto→stereo)."}),
                "prev_params": ("SOX_PARAMS",),
                "enable_x_pixels": ("BOOLEAN", {"default": False,
                                                "tooltip": "Enable -x: X-axis size in pixels; default derived or 800"}),
                "---- Window Size ----": ("STRING", {"default": "",
                                                     "tooltip": "Options for Spectrogram window size."}),
                "x_pixels": ("INT", {"default": 800, "min": 100, "max": 200000, "step": 10,
                                     "tooltip": "X-axis size in pixels; default derived or 800"}),
                "enable_y_pixels": ("BOOLEAN", {"default": False,
                                                "tooltip": "Enable -y: Y-axis size in pixels (per channel); slow if not 1 + 2^n"}),
                "y_pixels": ("INT", {"default": 257, "min": 50, "max": 2000, "step": 1,
                                     "tooltip": "Y-axis size in pixels (per channel); slow if not 1 + 2^n"}),
                "enable_Y_height": ("BOOLEAN", {"default": False,
                                                "tooltip": "Enable -Y: Y-height total (i.e. not per channel); default 550"}),
                "Y_height": ("INT", {"default": 550, "min": 100, "max": 2000,
                                     "tooltip": "Y-height total (i.e. not per channel); default 550"}),
                "enable_z_range": ("BOOLEAN",
                                   {"default": False, "tooltip": "Enable -z: Z-axis range in dB; default 120"}),
                "z_range": ("INT",
                            {"default": 120, "min": 20, "max": 180, "tooltip": "Z-axis range in dB; default 120"}),
                "enable_q_quant": ("BOOLEAN", {"default": True,
                                               "tooltip": "Enable -q: Z-axis quantisation (0 - 249); default 249"}),
                "q_quant": ("INT", {"default": 249, "min": 2, "max": 256,
                                    "tooltip": "Z-axis quantisation (0 - 249); default 249"}),
                "---- Spectrogram Colors ----": ("STRING", {"default": "",
                                                            "tooltip": "Options for Spectrogram colors."}),
                "light_bg": ("BOOLEAN", {"default": False, "tooltip": "Light background (-l)"}),
                "monochrome": ("BOOLEAN", {"default": False, "tooltip": "Monochrome (-m)"}),
                "high_color": ("BOOLEAN", {"default": False, "tooltip": "High colour (-h)"}),
                "no_axis": ("BOOLEAN", {"default": False, "tooltip": "Suppress axis lines (-a)"}),
                "---- Axis Options ----": ("STRING", {"default": "",
                                                      "tooltip": "Options for Spectrogram Axis."}),
                "raw_spec": ("BOOLEAN", {"default": False, "tooltip": "Raw spectrogram; no axes or legends (-r)"}),
                "slack": ("BOOLEAN", {"default": False, "tooltip": "Slack overlap of windows (-s)"}),
                "---- Sectrogram Type ----": ("STRING", {"default": "",
                                                         "tooltip": "Options for Spectrogram Type."}),
                "enable_window_type": ("BOOLEAN", {"default": True,
                                                   "tooltip": "Enable -w: Window: Hann(default)/Hamming/Bartlett/Rectangular/Kaiser/Dolph"}),
                "window_type": (["hann", "hamming", "bartlett", "rectangular", "kaiser", "dolph"], {"default": "hann",
                                                                                                    "tooltip": "Window: Hann(default)/Hamming/Bartlett/Rectangular/Kaiser/Dolph"}),
                "window_adj": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,
                                         "tooltip": "Window adjust parameter (-10 - 10); applies only to Kaiser/Dolph"}),
                "enable_window_adj": ("BOOLEAN", {"default": True,
                                                  "tooltip": "Enable -W: Window adjust parameter (-10 - 10); applies only to Kaiser/Dolph"}),
                "---- Title and Comments ----": ("STRING", {"default": "",
                                                            "tooltip": "Options for Spectrogram Titles and Comments."}),
                "title_text": ("STRING", {"default": "", "tooltip": "Title text (-t)"}),
                "comment_text": ("STRING", {"default": "", "tooltip": "Comment text (-c)"}),
                " === Save Spectrogram File ===": ("STRING", {"default": "",
                                                              "tooltip": "Set the png_prefix to output/spectro-images/<your-file-name>"}),
                "png_prefix": ("STRING",
                               {"default": "", "tooltip": "Output file name; default `spectrogram.png' (-o)"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "IMAGE", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "image", "sox_params", "dbg-text")
    RETURN_NAMES = ("audio", "image", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Utilities"
    DESCRIPTION = """Spectrogram node: Native unaltered `IMAGE`(s) from up to 4 `audio-0`..`audio-3` (SoX PNG→torch uint8 IMAGE, no resize; multi: batched [N,H_native~257,W_native~512-800,3] grid preview).

- Per-input: mono auto→stereo remix; title `[Mono|Stereo Audio-n]`.

- dbg-text: params + sim cmds; **errors logged**.

- PNG: `{png_prefix}_audio-n_{b:03d}.png` cwd (prefix).

Passthrough first stereoized AUDIO + `SOX_PARAMS` (temp preview)."""

    def process(self, **kwargs):
        # Extract params from kwargs
        enable_spectrogram = kwargs.get("enable_spectrogram", True)
        prev_params = kwargs.get("prev_params", None)
        enable_x_pixels = kwargs.get("enable_x_pixels", True)
        x_pixels = kwargs.get("x_pixels", 800)
        enable_y_pixels = kwargs.get("enable_y_pixels", True)
        y_pixels = kwargs.get("y_pixels", 257)
        enable_Y_height = kwargs.get("enable_Y_height", True)
        Y_height = kwargs.get("Y_height", 550)
        enable_z_range = kwargs.get("enable_z_range", True)
        z_range = kwargs.get("z_range", 120)
        enable_q_quant = kwargs.get("enable_q_quant", True)
        q_quant = kwargs.get("q_quant", 249)
        monochrome = kwargs.get("monochrome", False)
        high_color = kwargs.get("high_color", False)
        light_bg = kwargs.get("light_bg", False)
        no_axis = kwargs.get("no_axis", False)
        raw_spec = kwargs.get("raw_spec", False)
        slack = kwargs.get("slack", False)
        enable_window_type = kwargs.get("enable_window_type", True)
        window_type = kwargs.get("window_type", "hann")
        enable_window_adj = kwargs.get("enable_window_adj", True)
        window_adj = kwargs.get("window_adj", 0.0)
        title_text = kwargs.get("title_text", "")
        comment_text = kwargs.get("comment_text", "")
        png_prefix = kwargs.get("png_prefix", "")
        current_params = prev_params["sox_params"] if prev_params is not None else []

        # Single audio input handling
        audios_for_spec = []
        audio_out = None
        for n_str in ["0", "1", "2", "3"]:
            audio_n = kwargs.get(f"audio-{n_str}", None)
            if audio_n is not None:
                if audio_out is None:
                    # First non-None for passthrough AUDIO out (stereoized)
                    w_out = audio_n["waveform"][0:1]
                    orig_ch_out = w_out.shape[1]
                    if orig_ch_out == 1:
                        w_out = w_out.repeat(1, 2, 1)
                    else:
                        w_out = w_out[:, :2, :]
                    audio_out = {"waveform": w_out, "sample_rate": audio_n["sample_rate"]}
                # Per-input spec prep (stereoized)
                w_spec = audio_n["waveform"][0:1]
                orig_ch_spec = w_spec.shape[1]
                ch_type = "Mono" if orig_ch_spec == 1 else "Stereo"
                if orig_ch_spec == 1:
                    w_spec = w_spec.repeat(1, 2, 1)
                else:
                    w_spec = w_spec[:, :2, :]
                label = f"audio-{n_str}"
                audios_for_spec.append(
                    (label, {"waveform": w_spec, "sample_rate": audio_n["sample_rate"]}, ch_type, n_str))
        if audio_out is None:
            dummy_w = torch.zeros((1, 2, 44100), dtype=torch.float32)  # stereo dummy
            audio_out = {"waveform": dummy_w, "sample_rate": 44100}

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
        if comment_text.strip():
            options += ["-c", comment_text]
        debug_str = shlex.join(["spectrogram"] + options)

        # Simulate cmds per input/batch (always)
        cmds_dbg = []
        for label, audio_sp, _, _ in audios_for_spec:
            waveform_sp = audio_sp["waveform"]
            B_sp = waveform_sp.shape[0]
            sr_sp = audio_sp["sample_rate"]
            for b in range(B_sp):
                input_p = f"/tmp/{label}_in_{b:03d}.wav"
                proc_p = f"/tmp/{label}_proc_{b:03d}.wav"
                png_p = f"{png_prefix.strip()}_{label}_{b:03d}.png" if png_prefix.strip() else f"/tmp/temp_{label}_{b:03d}.png"
                if current_params:
                    cmds_dbg.append(shlex.join(["sox", input_p, proc_p] + current_params))
                spec_input = proc_p if current_params else input_p
                cmds_dbg.append(shlex.join(["sox", spec_input, "-n", "spectrogram"] + options + ["-o", png_p]))
        dbg_text = f"parameters: {debug_str}\n" + '' if cmds_dbg else f"parameters: {debug_str}\n ** No audio for spectrogram. **\n"
        all_images = []
        saved_msgs = []
        errors = []

        # Prepare audio_out and image_out

        # Prepare audio_out and image_out
        if not enable_spectrogram or len(audios_for_spec) == 0:
            image_out = torch.zeros((1, 257, 800, 3), dtype=torch.uint8)
        else:
            # Spectrogram generation
            # 1. Apply prev_params to get processed_waveform
            # Multi-input spectrogram gen
            num_inputs = len(audios_for_spec)
            all_images = []
            saved_msgs = []
            for label, audio_sp, ch_type, n_str in audios_for_spec:
                waveform_sp = audio_sp["waveform"]
                sr_sp = audio_sp["sample_rate"]
                B_sp = waveform_sp.shape[0]
                batch_imgs = []
                for b in range(B_sp):
                    w_b_2d = waveform_sp[b:b + 1].squeeze(0)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
                        input_path = tmp_in.name
                        torchaudio.save(input_path, w_b_2d, sr_sp)
                    spec_input_path = input_path
                    output_path = None
                    if current_params:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                            output_path = tmp_out.name
                        cmd = ["sox", input_path, output_path] + current_params
                        subprocess.run(cmd, check=True, capture_output=True, text=True)
                        spec_input_path = output_path

                    png_path = f"/tmp/spec_{label}_{b:03d}_{uuid.uuid4().hex[:8]}.png"
                    spec_cmd = ["sox", spec_input_path, "-n", "spectrogram"] + options + ["-o", png_path]
                    dbg_text += "\ncli: " + " ".join(spec_cmd)
                    try:
                        subprocess.run(spec_cmd, check=True, capture_output=True, text=True)
                    except subprocess.CalledProcessError:
                        raise
                    except Exception as e:
                        errors.append(f"spec {label} b{b}: {str(e)}")
                        batch_imgs.append(torch.zeros((1, 257, 800, 3), dtype=torch.uint8))
                        continue  # skip this batch item

                    # PNG exists (/tmp/png_path), load IMAGE first
                    try:
                        pil_img = Image.open(png_path)
                        if pil_img.mode != "RGB":
                            pil_img = pil_img.convert("RGB")
                        img_np = np.array(pil_img)
                        img_mean = img_np.mean().item()
                        if img_mean < 10:
                            errors.append(f"Dark spectrogram {label} b{b}: mean={img_mean:.1f} (silent/short audio?)")
                        img_t = (torch.from_numpy(img_np).to(torch.float32) / 255.0).unsqueeze(0)
                        batch_imgs.append(img_t)
                        if png_prefix.strip():
                            # Incremental filename sequence to avoid overwrites
                            base_prefix = png_prefix.strip()
                            dir_path = os.path.dirname(os.path.abspath(f"{base_prefix}_{label}_000.png")) or '.'
                            pattern = rf"^{re.escape(base_prefix)}_{re.escape(label)}_(\\d{{3}})\\\.png$"
                            nums = []
                            try:
                                for f in os.listdir(dir_path):
                                    m = re.match(pattern, f)
                                    if m:
                                        nums.append(int(m.group(1)))
                            except OSError:
                                pass
                            next_seq = max(nums, default=0) + 1
                            while True:
                                save_png = f"{png_prefix.strip()}_{label}_{next_seq:03d}_{uuid.uuid4().hex[:8]}.png"
                                save_path = os.path.abspath(save_png)
                                if not os.path.exists(save_path):
                                    break
                                next_seq += 1
                            try:
                                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                shutil.copy2(png_path, save_path)
                                saved_msgs.append(f"Saved {save_path} (seq {next_seq:03d})")
                            except Exception as e:
                                errors.append(f"PNG copy fail {label} b{b}: {str(e)}")
                    except Exception as e:
                        errors.append(f"PIL {label} b{b}: {str(e)}")
                        batch_imgs.append(torch.zeros((1, 257, 800, 3), dtype=torch.uint8))
                    # Cleanup temps safely
                    try:
                        os.unlink(input_path)
                    except OSError:
                        pass
                    if output_path:
                        try:
                            os.unlink(output_path)
                        except OSError:
                            pass
                    if png_path.startswith("/tmp/"):
                        try:
                            os.unlink(png_path)
                        except OSError:
                            pass
                stacked_b = torch.cat(batch_imgs, dim=0) if batch_imgs else torch.zeros((1, 257, 800, 3),
                                                                                        dtype=torch.uint8)
                all_images.append(stacked_b)
                if errors:
                    dbg_text += "\n\n**SoX Errors:**\n" + "\n".join(errors)
        # Combine images as batch (native sizes, grid preview)
        if len(all_images) == 0:
            image_out = torch.zeros((1, 257, 800, 3), dtype=torch.uint8)
        elif len(all_images) == 1:
            image_out = all_images[0]
        else:
            image_out = torch.cat(all_images, dim=0)
        if saved_msgs:
            dbg_text += "\n" + "\n".join(saved_msgs)

        return (audio_out, image_out, {"sox_params": current_params}, dbg_text)


class SoxUtilTextMux5Node:
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


class SoxUtilTextMux10Node:
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


class SoxUtilAudioMux5Node:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        optional["resample"] = (["auto", "mono", "stereo"], {"default": "auto", "tooltip": """Channel resample mode:
• auto: if mix mono+stereo inputs, upmix mono→stereo (stereoize_headroom dB drop + stereoize_delay ms widen first); all-mono→mono, all-stereo→stereo.
• mono: force downmix all→mono (mean(dim=1, preserve RMS)).
• stereo: force →stereo (mono→stereoize [-5dB equiv], >2ch→slice first2)."""})
        optional["=== Stereoize Mono → Stereo ==="] = ("STRING", {"default": "",
                                                                  "tooltip": "Stereoization controls for mono→stereo upmix only (headroom + Haas delay widen)."})
        optional["stereoize_delay_ms"] = ("INT", {"default": 0, "min": 0, "max": 20, "step": 1,
                                                  "tooltip": """Delay ms (0-20) applied **only** on mono→stereo upmix: Creates stereo width via Haas effect (delays right ch by N samples=round(N*sr/1000), left padded end to match len). 10-20ms sweet spot; 0=simple repeat."""})
        optional["stereoize_headroom"] = ("FLOAT", {"default": -3.0, "min": -7.0, "max": 0.0, "step": 0.1,
                                                    "tooltip": """Headroom drop dB (-7 to 0) **only** on mono→stereo upmix: Attenuates mono before upmix/delay to prevent L/R sum clipping (~+3-6dB uncorrelated gain). Default -3dB ≈ orig -5dB fixed (now adjustable)."""})
        for i in range(5):
            optional[f"in-audio-{i}"] = ("AUDIO",)
        optional["=== Volume & Gain ==="] = ("STRING", {"default": ""})
        optional["  --- Master_Volume ---"] = ("STRING", {"default": ""})
        optional["master_gain_db"] = ("FLOAT", {"default": 0.0, "min": -60.0, "max": 12.0, "step": 0.1,
                                                "tooltip": "Global master gain dB applied post-mix to both outputs."})
        optional["=== Track Gain ==="] = ("STRING",
                                          {"default": "", "tooltip": "Group label before the per-input gain sliders."})
        for i in range(5):
            optional[f"input_gain_{i}"] = ("FLOAT", {"default": 0.0, "min": -60.0, "max": 12.0, "step": 0.1,
                                                     "tooltip": f"Gain dB for in-audio-{i} (post-resample, pre-pad/mix)."})
        return {"optional": optional}

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("mono-audio", "stereo-audio")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Utilities"
    DESCRIPTION = """Utility node: 5 optional `in-audio-0..4` → mono-audio + stereo-audio (multi/target_ch mix).

`resample` (auto/mono/stereo) controls target_ch & per-input ch-resample (mono→stereo: stereoize_headroom dB drop + stereoize_delay ms widen first, pre-sum headroom):

• `auto`: mixed mono+stereo → target_ch=2 (upmix mono→stereoize); all-mono→1, all-stereo→2 (or max≤2).
• `mono`: target_ch=1 (downmix mean(dim=1, RMS preserve)).
• `stereo`: target_ch=2 (mono→stereoize, >2ch slice [:2]).

Stereoize (mono→stereo upmix only):
• `stereoize_headroom` (-7..0 dB): Gain drop pre-upmix (anti-clip sum headroom).
• `stereoize_delay` (0-20 ms): Right ch delay (Haas width); left end-pad to match len.

SR resample→first (post-ch/stereoize); zero-pad shorts→longest; stack→mean(dim=0)→mix multi [1,C,T].
- mono-audio: true mono downmix mean(dim=1) [1,1,T]
- stereo-audio: stereo derive [1,2,T] (repeat/slice)

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
            zero_mono = torch.mean(zero_multi, dim=1, keepdim=True)
            if zero_multi.shape[1] == 1:
                zero_stereo = zero_mono.repeat(1, 2, 1)
            else:
                zero_stereo = zero_multi[:, :2, :]
            audio_mono = {"waveform": zero_mono, "sample_rate": sr}
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
        mixed_mono = torch.mean(mixed_multi, dim=1, keepdim=True)

        # Stereo derive from multi mix
        if mixed_multi.shape[1] == 1:
            mixed_stereo = mixed_mono.repeat(1, 2, 1)
        else:
            mixed_stereo = mixed_multi[:, :2, :]

        mixed_mono *= master_lin
        mixed_stereo *= master_lin

        audio1 = {"waveform": mixed_mono, "sample_rate": target_sr}
        audio2 = {"waveform": mixed_stereo, "sample_rate": target_sr}
        return (audio1, audio2)


class SoxUtilAudioMuxPro5Node:
    """
    TODO: AddUpdae tool-tips
    # Renamed to SoxUtilAudioMuxPro5 ✓
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "enable_mux": ("BOOLEAN", {"default": True}),
            "mute_all": ("BOOLEAN", {"default": False}),
            "solo_channel": (["none", "1", "2", "3", "4", "5"], {"default": "none"}),
            "═══ REMIX OPTIONS ═══": ("STRING", {"default": "",
                                                 "tooltip": "Resample mode and stereoize group for input channel handling."}),
            "resample": (["auto", "mono", "stereo"], {"default": "auto", "tooltip": """Channel resample mode:
• auto: if mix of mono+stereo inputs, upmix mono→stereo (w/ stereoize); uniform mono→mono, stereo→stereo (≤2ch).
• mono: downmix all to mono (mean dim=1).
• stereo: upmix to stereo (mono→stereoize headroom+delay, >2ch slice first2). 
Applied pre-SR resample, post-input; feeds auto_vol_balance."""}),
            "=== Stereoize Mono → Stereo ===": ("STRING", {"default": "",
                                                           "tooltip": "Stereoization controls: headroom drop + Haas delay for mono→stereo upmix only."}),
            "stereoize_delay_ms": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1,
                                           "tooltip": """Haas delay ms (0-20): Right ch delayed by ~N samples (sr/1000); left end-pad match len. 0=repeat both ch; 10-20ms stereo width."""}),
            "stereoize_headroom": ("FLOAT", {"default": -3.0, "min": -7.0, "max": 0.0, "step": 0.1,
                                             "tooltip": """Pre-upmix gain drop dB (-7..0): Attenuate mono →stereo sum headroom (~+6dB incoherent). -3dB default safe."""}),
            "═══ MIX MODE ═══": ("STRING", {"default": "", "tooltip": "Mix mode and balance group"}),
            "mix_mode": (["linear_sum", "average", "rms_power", "max_amplitude"], {"default": "linear_sum",
                                                                                   "tooltip": "linear_sum/rms_power/max_amplitude: coherent voltage SUM (fixed, real DAW mix, rec.); average: arithmetic mean."}),
            "mix_vol_preset_overrides": (["none", "equal", "vocals_lead", "bass_heavy", "wide_stereo"],
                                         {"default": "none",
                                          "tooltip": """Volume preset overrides (overrides vol_1-5 sliders):\nnone: use sliders\n equal: [0.0, 0.0, 0.0, 0.0, 0.0]\nvocals_lead: [3.5, -3.1, -3.1, -6.0, -6.0]\nbass_heavy: [-2.0, 1.6, 0.0, -0.9, -0.9]\nwide_stereo: [0.0, 0.0, -2.0, -2.0, 1.6]"""}),
            "sample_accurate_mix_strategy": (["max", "power_sum"], {"default": "power_sum",
                                                                    "tooltip": "Sample-accurate group mix: max (peak envelope, experimental); power_sum (√N-normalized coherent sum, musical fallback)."}),
            "short_duration_threshold": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 30.0, "step": 0.1,
                                                   "tooltip": "Auto-suggestion short_threshold seconds: crest>16dB AND duration<short → Sample-accurate (transients/sparse)."}),
            "−−− Track Characterisaction −−−": ("STRING", {"default": "", "tooltip": "Auto volume balance group"}),
            "characterize_tracks": ("BOOLEAN", {"default": False,
                                                "tooltip": "Automatically characterize tracks for ideal mix in rms_power and max_amplitude mixing modes"}),
            "target_rms_db": ("FLOAT", {"default": -20.0, "min": -60.0, "max": -6.0, "step": 0.5,
                                        "tooltip": "RMS target dBFS per-track pre-mix (-20dB rec. for headroom with multiple coherent tracks); rms_power mode."}),
            "target_peak_db": ("FLOAT", {"default": -9.0, "min": -20.0, "max": 0.0, "step": 0.5,
                                         "tooltip": "Peak target dBFS per-track pre-mix for max_amplitude mode (-9dB rec. headroom); now uses coherent voltage sum."}),
            "−−− Track Padding-Filling-Cutting −−−": ("STRING",
                                                      {"default": "", "tooltip": "Auto volume balance group"}),
            "track_length_master": (["0", "1", "2", "3", "4", "5"], {"default": "0", "tooltip": """Master track length reference (0=none): 
  - Trim longer tracks post-resample to this active track's resampled length (exact duration match at target SR).
  - Shorter tracks extended per pad_mode. Use to sync to shortest track (e.g. vocal)."""}),
            "pad_mode": (["zero_fill", "loop_repeat", "fade_trim"], {"default": "zero_fill"}),
            "auto_normalize": ("BOOLEAN", {"default": True,
                                           "tooltip": "Post-mix peak normalize to -1dB headroom + clamp <=1.0 (default: on, prevents clipping/distortion)"}),
            "pre_mix_gain_db": ("FLOAT", {"default": -3.0, "min": -12.0, "max": 3.0, "step": 0.1,
                                          "tooltip": "Pre-mix gain reduction dB (headroom; negative reduces gain pre-mix/effects to prevent clipping)."}),
            "prev_params": ("SOX_PARAMS",),
            "═══ SAVE OPTIONS ═══": ("STRING", {"default": "", "tooltip": "Save options group"}),
            "enable_save": ("BOOLEAN", {"default": False}),
            "file_prefix": ("STRING", {"default": "output/audio/SoX_Effects", "multiline": False}),
            "save_format": (["wav", "flac", "mp3", "ogg"], {"default": "wav"}),
            "=== Master Volume ===": ("STRING",
                                      {"default": "", "tooltip": "Master output volume adjustment (post-mix)."}),
            "master_vol_db": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 12.0, "step": 0.1,
                                        "tooltip": "Master volume dBFS (applied after mix, normalize, tanh-clamp)."}),
            "headroom_db": ("FLOAT", {"default": -3.0, "min": -12.0, "max": 0.0, "step": 0.1,
                                      "tooltip": "Post-mix auto_normalize target headroom dBFS (e.g. -3dB=0.707 linear); applied before softclip."}),
            "softclip_strength": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.1, "tooltip": """Tanh softclip strength: Controls how aggressively peaks are limited post-normalization (after headroom_db scaling).

Uses hyperbolic tangent (tanh): tanh(strength * x) / strength → smooth compression >0 dBFS, no hard clips.

- 1.0: No effective clipping (linear pass-through).
- 1.1 (default): Gentle peak rounding, transparent/mastering.
- 1.25: Moderate 'tape' saturation/warmth.
- ≥1.5: Aggressive distortion (creative/analog emulation).

Always ≤1.0 output. Low= clean limit; high= colored compression."""}),
            "═══ TRACK CHANNELS 1-5 ═══": ("STRING", {"default": "",
                                                      "tooltip": "enable_audio/vol/mute/in-audio for each channel 1-5"}),
        }
        for i in range(5):
            optional[f"−−−− Track {i + 1} −−−−"] = ("STRING",
                                                    {"default": "", "tooltip": f"Controls for Track [{i + 1}]"})
            optional[f"enable_audio_{i + 1}"] = ("BOOLEAN", {"default": i < 2})
            optional[f"vol_{i + 1}"] = ("FLOAT", {"default": 0.0, "min": -60.0, "max": 12.0, "step": 0.1})
            optional[f"mute_{i + 1}"] = ("BOOLEAN", {"default": False})
            optional[f"track_type_{i + 1}"] = (["Auto", "Standard", "Sample-accurate", "Absonic"],
                                               {"default": "Auto", "tooltip": """Track Type (rms_power or max_amplitude modes):
- Auto (req. auto_vol_balance=true): crest_db>16 AND dur<short_threshold=Sample-accurate, ch>=4=Absonic, else Standard
- Standard: coherent voltage sum (music/instrument/vocal stems)
- Sample-accurate: sample-wise MAX (experimental/glitch/granular/phase-critical)
- Absonic: coherent sum, sqrt(N)-weight if multiple (spatial/Ambisonics/immersive; sliced to stereo)
"""})

            optional[f"in-audio-{i + 1}"] = ("AUDIO",)
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
        vols = [kwargs.get(f"vol_{i + 1}", 0.0) for i in range(5)]
        mutes = [kwargs.get(f"mute_{i + 1}", False) for i in range(5)]
        solo_channel = kwargs.get("solo_channel", "none")
        solos = [False] * 5
        if solo_channel != "none":
            solos[int(solo_channel) - 1] = True
        enables = [kwargs.get(f"enable_audio_{i + 1}", True) for i in range(5)]
        track_types = [kwargs.get(f"track_type_{i + 1}", "Standard") for i in range(5)]
        any_solo = solo_channel != "none"
        audios = [kwargs.get(f"in-audio-{i + 1}", None) for i in range(5)]
        mix_mode = kwargs.get("mix_mode", "linear_sum")
        track_length_master = int(kwargs.get("track_length_master", "0"))
        pad_mode = kwargs.get("pad_mode", "zero_fill")
        auto_normalize = kwargs.get("auto_normalize", False)
        mix_vol_preset_overrides = kwargs.get("mix_vol_preset_overrides", "none")
        preset_vols = {
            "equal": [0.0, 0.0, 0.0, 0.0, 0.0],
            "vocals_lead": [3.5, -3.1, -3.1, -6.0, -6.0],
            "bass_heavy": [-2.0, 1.6, 0.0, -0.9, -0.9],
            "wide_stereo": [0.0, 0.0, -2.0, -2.0, 1.6],
        }
        if mix_vol_preset_overrides != "none":
            vols = preset_vols[mix_vol_preset_overrides]
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
        headroom_db = kwargs.get("headroom_db", -3.0)
        softclip_strength = kwargs.get("softclip_strength", 1.1)
        sample_accurate_mix_strategy = kwargs.get("sample_accurate_mix_strategy", "power_sum")
        short_duration_threshold = kwargs.get("short_duration_threshold", 3.0)
        active_indices = []
        for i in range(5):
            audio = audios[i]
            if audio is not None and audio["waveform"].numel() > 0 and enables[i] and not mute_all and not mutes[
                i] and (not any_solo or solos[i]):
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
            f"- enable_mux: {enable_mux}",
            f"- mute_all: {mute_all}",
            f"- mix_mode: {mix_mode}",
            f"- pad_mode: {pad_mode}",
            f"- auto_normalize: {auto_normalize}",
            f"- mix_vol_preset_overrides: {mix_vol_preset_overrides}",
            f"- Vols dB: [{', '.join(f'{v:.1f}' for v in vols)}]",
            f"- Audio-Enabled: {enables}",
            f"- Mutes: {mutes}",
            f"- Solo channel: {solo_channel}",
            f"- resample_mode: {resample_mode}",
            f"- stereoize_delay_ms: {stereoize_delay_ms}, headroom_db: {stereoize_headroom_db:.1f}",
            f"- Active indices: {active_indices}",
            f"- Final Track: {'mono' if target_ch == 1 else 'stereo'}",
            f"- Preproc params: rms_t={target_rms_db:.1f}dB peak_t={target_peak_db:.1f}dB pre_g={pre_mix_gain_db:.1f}dB len_m={track_length_master} pad={pad_mode}",
        ]
        for i in range(5):
            if audios[i] is not None:
                a = audios[i]
                info = f"sr={a['sample_rate']} C={a['waveform'].shape[1]} T={a['waveform'].shape[2]}"
            else:
                info = "None"
            dbg_parts.append(f"- Audio{i + 1}: {info}")
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

            master_vol_db = kwargs.get("master_vol_db", 0.0)
            master_lin = 10 ** (master_vol_db / 20.0)
            zero_multi *= master_lin
            zero_stereo *= master_lin
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
                dbg_parts.append(f"- Saved mono: {full_mono}")
                dbg_parts.append(f"- Saved stereo: {full_stereo}")
            dbg_text = enabled_prefix + "\n".join(dbg_parts)
            return (dummy_audio_mono, dummy_audio_stereo, dbg_text, {"sox_params": current_params})
        # Get target_sr, dtype, device from first active
        first_i = active_indices[0]
        first_audio = audios[first_i]
        target_sr = first_audio["sample_rate"]
        first_wave = first_audio["waveform"][0:1]
        dtype = first_wave.dtype
        device = first_wave.device
        orig_channels = [audios[i]["waveform"].shape[1] for i in active_indices]
        orig_srs = [audios[i]["sample_rate"] for i in active_indices]
        orig_lens = [audios[i]["waveform"].shape[2] for i in active_indices]
        # Resample all active to multi-ch, upmix if needed, collect
        resampled_multis = []
        resample_msgs = []
        for j, i in enumerate(active_indices):
            audio_i = audios[i]
            orig_sr = orig_srs[j]
            orig_ch = orig_channels[j]
            orig_len = orig_lens[j]
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

            new_len = w.shape[2]
            orig_dur_str = f"{orig_len / orig_sr:.1f}s" if orig_sr > 0 else "0s"
            new_dur_str = f"{new_len / target_sr:.1f}s"
            ch_action = ""
            if orig_ch != target_ch:
                if target_ch == 1:
                    ch_action = "stereo->mono"
                elif orig_ch == 1:
                    ch_action = "mono->stereo"
                else:
                    ch_action = ">2ch-slice"
            sr_action = " SR-resamp" if sr_i != target_sr else ""
            msg = f"- Track-{i + 1}: {orig_sr}/{orig_ch}/{orig_dur_str} -> {target_sr}/{target_ch}/{new_dur_str} ({ch_action}{sr_action})"
            resample_msgs.append(msg)
            w = w.to(device=device, dtype=dtype)
            resampled_multis.append((i, w))
        resample_dbg = "\n".join(resample_msgs) if resample_msgs else ""
        # Master length trim (post-resample, pre-balance/pad)
        master_len = None
        trim_msgs = []
        if track_length_master > 0 and active_indices:
            master_idx = track_length_master - 1
            if master_idx in active_indices:
                for orig_i_pos, w_pos in resampled_multis:
                    if orig_i_pos == master_idx:
                        master_len = w_pos.shape[2]
                        break
        trim_info = ""
        if master_len is not None:
            for j in range(len(resampled_multis)):
                orig_i, w = resampled_multis[j]
                if w.shape[2] > master_len:
                    resampled_multis[j] = (orig_i, w[:, :, :master_len])
                    trim_msgs.append(f"Track-{orig_i + 1}")
            if trim_msgs:
                trim_info = f"- Trim to track{track_length_master} ({master_len / target_sr:.1f}s): {', '.join(trim_msgs)}"
        auto_applied_str = ""
        auto_dbg = ""
        if auto_vol_balance and resampled_multis:
            measured = []
            deltas = []
            suggestions = []
            type_sugs = []
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
                    peak_val = torch.max(torch.abs(post_vol_multi))
                    peak_db = 20 * torch.log10(torch.clamp(peak_val, min=1e-8)).item()
                    crest_db = peak_db - measured_db
                    # NEW: Effective duration (non-zero samples; scale-invariant)
                    nonzero_samples = (torch.abs(post_vol_multi) > 1e-5).sum().item()
                    effective_duration = nonzero_samples / target_sr  # seconds
                    # IMPROVED: Raised threshold + duration check (conservative: extremes only)
                    if crest_db > 16 and effective_duration < short_duration_threshold:
                        sug = "Sample-accurate"  # Sparse transients/glitch
                    elif orig_channels[j] >= 4:
                        sug = "Absonic"  # Spatial/multi-ch priority
                    else:
                        sug = "Standard"  # Default: music/stems
                    suggestions.append(
                        f"track{i_idx + 1}:{sug} (crest:{crest_db:.1f}dB,dur:{effective_duration:.1f}s,ch:{orig_channels[j]})")
                    type_sugs.append(sug)
                    delta_db = target - measured_db
                    vols[i_idx] += delta_db
                    measured.append("{:.1f}".format(measured_db))
                    deltas.append("{:.1f}".format(delta_db))
            elif mix_mode == "max_amplitude":
                metric = "Peak"
                target = target_peak_db
                # Junie recommended - Grok - Not recommended...
                # No, this peak-based version does not make much sense as a
                # general-purpose method for reviewing / balancing / gain-staging
                # individual tracks before actual mixing — at least not in most
                # modern (and even classic) mixing workflows.
                # It is mathematically correct and runs without crashing,
                # but it usually gives poor musical results compared to
                # RMS / average-loudness / LUFS-style balancing.
                # for j, (i_idx, w_multi) in enumerate(resampled_multis):
                #     vol_lin = current_linear_vols[i_idx]
                #     post_vol_multi = w_multi * vol_lin
                #     peak_val = torch.max(torch.abs(post_vol_multi))
                #     measured_db = 20 * torch.log10(torch.clamp(peak_val, min=1e-8)).item()
                #     delta_db = target - measured_db
                #     vols[i_idx] += delta_db
                #     measured.append("{:.1f}".format(measured_db))
                #     deltas.append("{:.1f}".format(delta_db))

                # Minimal clean / safer variant (recommended)
                for j, (i_idx, w_multi) in enumerate(resampled_multis):
                    vol_lin = current_linear_vols[i_idx]
                    post_vol_multi = w_multi * vol_lin
                    rms = torch.sqrt(torch.mean(post_vol_multi ** 2))
                    rms_db = 20 * torch.log10(torch.clamp(rms, min=1e-8)).item()
                    peak_val = torch.max(torch.abs(post_vol_multi))
                    peak_db = 20 * torch.log10(torch.clamp(peak_val, min=1e-8)).item()
                    crest_db = peak_db - rms_db
                    # NEW: Effective duration (non-zero samples; scale-invariant)
                    nonzero_samples = (torch.abs(post_vol_multi) > 1e-5).sum().item()
                    effective_duration = nonzero_samples / target_sr  # seconds
                    # IMPROVED: Raised threshold + duration check (conservative: extremes only)
                    if crest_db > 16 and effective_duration < short_duration_threshold:
                        sug = "Sample-accurate"  # Sparse transients/glitch
                    elif orig_channels[j] >= 4:
                        sug = "Absonic"  # Spatial/multi-ch priority
                    else:
                        sug = "Standard"  # Default: music/stems
                    suggestions.append(
                        f"track{i_idx + 1}:{sug} (crest:{crest_db:.1f}dB,dur:{effective_duration:.1f}s,ch:{orig_channels[j]})")
                    type_sugs.append(sug)
                    delta_db = target - peak_db
                    vols[i_idx] += delta_db
                    measured.append(f"{peak_db:.1f}")
                    deltas.append(f"{delta_db:+.1f}")

                # Even stricter safety version (prevents huge boosts on near-silent tracks)
                # Python
                # for j, (i_idx, w_multi) in enumerate(resampled_multis):
                #     vol_lin = current_linear_vols[i_idx]
                #     post_vol_multi = w_multi * vol_lin
                #     peak_val = torch.max(torch.abs(post_vol_multi))
                #     peak_db = 20 * torch.log10(torch.clamp(peak_val, min=1e-8)).item()
                #     if peak_db < -60:  # very quiet → most likely shouldn't get boosted massively
                #         measured.append(f"{peak_db:.1f} (quiet)")
                #         deltas.append("—")
                #         continue
                #     delta_db = target - peak_db
                #     vols[i_idx] += delta_db
                #     measured.append(f"{peak_db:.1f}")
                #     deltas.append(f"{delta_db:+.1f}")

            if metric is not None:
                linear_vols = [10 ** (v / 20.0) for v in vols]  # Updated after deltas
                auto_dbg = f"- Auto Vol Balance: True | {metric} Target: {target:.1f}dB | Measured (post-vol full) {metric} dB: [{', '.join(measured)}] | Deltas dB: [{', '.join(deltas)}]" + (
                    f" | Type suggestions: [{', '.join(suggestions)}]" if suggestions else "")
                # Apply "Auto" track types using analysis (req. auto_vol_balance)
                auto_applied_count = 0
                if mix_mode in ["rms_power", "max_amplitude"]:
                    for j, orig_i in enumerate(active_indices):
                        if track_types[orig_i] == "Auto":
                            track_types[orig_i] = type_sugs[j]
                            auto_applied_count += 1
                auto_applied_str = f" (auto-applied:{auto_applied_count})" if auto_applied_count > 0 else ""
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
        # Fallback "Auto" to "Standard" if no analysis done
        for i_idx in active_indices:
            if track_types[i_idx] == "Auto":
                track_types[i_idx] = "Standard"
        active_types = [track_types[i] for i in active_indices]
        if mix_mode in ["rms_power", "max_amplitude"] and len(padded_list) > 0:
            standard = []
            absonic = []
            sample_acc = []
            for j, orig_i in enumerate(active_indices):
                typ = track_types[orig_i]
                if typ == "Standard":
                    standard.append(padded_list[j])
                elif typ == "Absonic":
                    absonic.append(padded_list[j])
                else:  # Sample-accurate
                    sample_acc.append(padded_list[j])
            target_ch = padded_list[0].shape[1]
            max_len = padded_list[0].shape[2]
            device = padded_list[0].device
            dtype_ = padded_list[0].dtype
            zero_shape = (1, target_ch, max_len)
            standard_sum = torch.sum(torch.stack(standard), dim=0) if standard else torch.zeros(zero_shape,
                                                                                                dtype=dtype_,
                                                                                                device=device)
            absonic_sum = torch.sum(torch.stack(absonic), dim=0) if absonic else torch.zeros(zero_shape, dtype=dtype_,
                                                                                             device=device)
            if len(absonic) > 1:
                absonic_sum /= len(absonic) ** 0.5

            # NEW REFINED Sample-accurate: hybrid strategy
            if not sample_acc:
                sample_mix = torch.zeros(zero_shape, dtype=dtype_, device=device)
            elif len(sample_acc) == 1:
                sample_mix = sample_acc[0]  # Identity: no op needed
            else:
                stack_sa = torch.stack(sample_acc)
                if sample_accurate_mix_strategy == "max":
                    sample_mix = torch.max(stack_sa, dim=0)[0]  # Peak envelope (experimental)
                else:  # power_sum (default, musical)
                    sample_mix = torch.sum(stack_sa, dim=0) / torch.sqrt(
                        torch.tensor(len(sample_acc), dtype=dtype_, device=device))
            mixed_multi = standard_sum + absonic_sum + sample_mix
            used_types_str = ", ".join(
                [f"track{active_indices[j] + 1}:{active_types[j]}" for j in range(len(active_types))])
            type_counts = f"{mix_mode} types: Std:{len(standard)} Ab:{len(absonic)} Sa:{len(sample_acc)}"
            absonic_note = f" (Absonic sliced to {target_ch}ch; full B-format needs multi-ch)" if absonic and target_ch < 4 else ""
            type_info = f"- Used: [{used_types_str}] | {type_counts}{absonic_note}{auto_applied_str}"
        else:
            stacked = torch.stack(padded_list, dim=0)
            if mix_mode == "linear_sum":
                mixed_multi = torch.sum(stacked, dim=0)
            elif mix_mode == "average":
                mixed_multi = torch.mean(stacked, dim=0)
            elif mix_mode == "rms_power":
                mixed_multi = torch.sum(stacked, dim=0)  # fallback coherent sum
            elif mix_mode == "max_amplitude":
                mixed_multi = torch.sum(stacked, dim=0)  # coherent sum
            type_info = ""
        peak = torch.max(torch.abs(mixed_multi)).item()
        norm_info = ""
        if auto_normalize and peak > 1e-8:
            headroom_lin = 10 ** (headroom_db / 20.0)
            scale_factor = headroom_lin / peak
            mixed_multi *= scale_factor
            norm_info = f"norm:{headroom_db:.1f}dB "

        # UPDATED: Configurable softclip (gentler default)
        mixed_multi = torch.tanh(mixed_multi * softclip_strength) / softclip_strength
        post_peak = torch.max(torch.abs(mixed_multi)).item()
        process_details = ((resample_dbg + "\n" if resample_dbg else "") + auto_dbg + (
            f"\n{trim_info}" if trim_info else "") + (
                               f"\n{type_info}" if type_info else "")) + f"\n- Used torch mix (resample={resample_mode}), Target SR: {target_sr}, ch: {target_ch}, Max len: {mixed_multi.shape[2]}, Peak pre:{peak:.3f} post:{post_peak:.3f} ({norm_info}tanh{softclip_strength:.1f}:<=1.0)"
        mixed_mono = torch.mean(mixed_multi, dim=1, keepdim=True)
        if mixed_multi.shape[1] == 1:
            mixed_stereo = mixed_mono.repeat(1, 2, 1)
        else:
            mixed_stereo = mixed_multi[:, :2, :]

        master_vol_db = kwargs.get("master_vol_db", 0.0)
        master_lin = 10 ** (master_vol_db / 20.0)
        mixed_mono *= master_lin
        mixed_stereo *= master_lin
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


class SoxUtilAudioSplit5Node:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "force_channels": (["auto", "mono", "stereo"], {"default": "auto",
                                                            "tooltip": "Force output channels: auto=preserve input, mono=downmix to 1ch (mean), stereo=upmix to 2ch (repeat if mono, first 2ch if more)."}),
            "master_gain_db": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 12.0, "step": 0.1,
                                         "tooltip": "Global master gain dB applied to input before per-track gains."}),
            "=== Track Gain ===": ("STRING",
                                   {"default": "", "tooltip": "Group label before the per-output track gain sliders."}),
        }
        for i in range(5):
            optional[f"track_gain_{i}"] = ("FLOAT", {"default": 0.0, "min": -60.0, "max": 12.0, "step": 0.1,
                                                     "tooltip": f"Gain dB for out-audio-{i} (multiplicative after master)."})
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
    "SoxUtilSpectrogram": SoxUtilSpectrogramNode,
    "SoxUtilTextMux10": SoxUtilTextMux10Node,
    "SoxUtilTextMux5": SoxUtilTextMux5Node,
    "SoxUtilAudioMux5": SoxUtilAudioMux5Node,
    "SoxUtilAudioMuxPro5": SoxUtilAudioMuxPro5Node,
    "SoxUtilAudioSpli5": SoxUtilAudioSplit5Node,
    "SoxMuxWetDry": SoxMuxWetDry,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SoxApplyEffects": "SoX Apply Effects",
    "SoxUtilTextMux10": "SoX Util Text Mux 10",
    "SoxUtilTextMux5": "SoX Util Text Mux 5",
    "SoxUtilAudioMux5": "SoX Util Audio Mux 5",
    "SoxUtilAudioMuxPro5": "SoX Util Audio Mux Pro 5",
    "SoxUtilAudioSplit5": "SoX Util Audio Split 5",
    "SoxMuxWetDry": "SoX Mux Wet/Dry",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
