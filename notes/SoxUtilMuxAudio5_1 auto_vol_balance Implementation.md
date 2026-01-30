# SoxUtilMuxAudio5_1 auto_vol_balance Implementation
SoxUtilMuxAudio5_1 nodes's auto_vol_balande Implementation for rms_power and max_amplitude.

## The source...

```python
class SoxUtilMuxAudio5_1:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "enable_mux": ("BOOLEAN", {"default": True}),
            "mute_all": ("BOOLEAN", {"default": False}),
            "solo_channel": (["none", "1", "2", "3", "4", "5"], {"default": "none"}),
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
        active_indices = []
        for i in range(5):
            audio = audios[i]
            if audio is not None and audio["waveform"].numel() > 0 and enables[i] and not mute_all and not mutes[i] and (not any_solo or solos[i]):
                active_indices.append(i)
        target_ch = 1
        if active_indices:
            max_ch = 1
            for ii in active_indices:
                max_ch = max(max_ch, audios[ii]["waveform"].shape[1])
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
            dummy_wave_mono = torch.zeros((1, 1, 44100), dtype=torch.float32)
            dummy_audio_mono = {"waveform": dummy_wave_mono, "sample_rate": 44100}
            dummy_wave_stereo = dummy_wave_mono.repeat(1, 2, 1)
            dummy_audio_stereo = {"waveform": dummy_wave_stereo, "sample_rate": 44100}
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
            if orig_c < target_ch:
                w = w.repeat(1, target_ch, 1)
            if sr_i != target_sr:
                resampler = torchaudio.transforms.Resample(sr_i, target_sr)
                w = resampler(w)
            resampled_multis.append((i, w))
        # =====================================================
        # AUTO VOL BALANCE: Pre-resample Adjustment (toggle: auto_vol_balance)
        # 
        # OVERVIEW: Pre-computes dB corrections to `vols` based on post-resample waveforms.
        #           Ensures each active channel hits target RMS/Peak PRE-MIX for balance.
        #           Reduces clipping risk by consistent levels + downstream headroom/norm/clip.
        # 
        # FLOW:
        # 1. Snapshot `current_linear_vols` = 10^(vols/20)  (post-preset, pre-adjust)
        # 2. IF rms_power: target=target_rms_db; measure RMS on (w_resampled * curr_vol_lin)
        # 3. IF max_amplitude: target=target_peak_db; measure Peak on same
        # 4. Per-channel: delta_db = target - measured_db; vols[i_idx] += delta_db
        # 5. Log PRE-adjust measured/deltas in `auto_dbg`
        # 6. Recompute `linear_vols` for pad/vol/mix (now balanced)
        # 
        # CLIPPING PREVENTION ROLE:
        # - Per-ch targets (-18dB RMS/-6dB Peak def) keep individuals conservative
        # - pre_mix_gain_db (-3dB) adds mix headroom
        # - auto_normalize scales mix peak to -1dBFS
        # - tanh(*1.25)/1.25 soft-limits to |x|<=1.0
        # 
        # Issue: Metric computed PRE-PAD:
        #   - zero_fill pads dilute RMS ~sqrt(orig_len/max_len) → actual post-pad RMS < target (under-boost shorts)
        #     Fix idea: Simulate pad in metric calc, or adjust target *= sqrt(max_len/orig_len)
        #   - Peak usually unchanged (zeros low); loop_repeat may increase
        # Issue: No multi-ch interaction → correlated peaks sum higher in mix (phase align)
        #        Use conservative targets, monitor with dbg peak pre/post
        # =====================================================
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
                # RMS_POWER: Adjust per-channel to target_rms_db (loudness balance pre-mix)
                for j, (i_idx, w_multi) in enumerate(resampled_multis):
                    vol_lin = current_linear_vols[i_idx]  # Pre-adjust lin-vol snapshot
                    post_vol_multi = w_multi * vol_lin  # Simulate post-vol PRE-PAD
                    rms = torch.sqrt(torch.mean(post_vol_multi ** 2))  # RMS linear amp
                    measured_db = 20 * torch.log10(torch.clamp(rms, min=1e-8)).item()  # dBFS
                    delta_db = target - measured_db
                    vols[i_idx] += delta_db  # dB correction (multiplicative lin scale)
                    measured.append("{:.1f}".format(measured_db))  # PRE-adjust log
                    deltas.append("{:.1f}".format(delta_db))
            elif mix_mode == "max_amplitude":
                metric = "Peak"
                target = target_peak_db
                # MAX_AMPLITUDE: Adjust per-channel peak to target_peak_db (headroom pre-mix)
                for j, (i_idx, w_multi) in enumerate(resampled_multis):
                    vol_lin = current_linear_vols[i_idx]  # Pre-adjust snapshot
                    post_vol_multi = w_multi * vol_lin  # Pre-pad sim
                    peak_val = torch.max(torch.abs(post_vol_multi))  # Linear peak amp
                    measured_db = 20 * torch.log10(torch.clamp(peak_val, min=1e-8)).item()
                    delta_db = target - measured_db
                    vols[i_idx] += delta_db
                    measured.append("{:.1f}".format(measured_db))
                    deltas.append("{:.1f}".format(delta_db))
            # Post-adjustment: vols now reflect targeted per-channel metric.
            if metric is not None:
                linear_vols = [10 ** (v / 20.0) for v in vols]  # Updated after deltas; for pad/vol
                auto_dbg = f"Auto Vol Balance: True | {metric} Target: {target:.1f}dB | Measured (post-vol full) {metric} dB: [{', '.join(measured)}] | Deltas dB: [{', '.join(deltas)}]"
        # max_len after resample (post-auto_balance)
        max_len = max(wm.shape[2] for _, wm in resampled_multis)
        # PAD & VOL: Pad resampled to max_len (mode-specific), THEN *= UPDATED linear_vols[i] (balanced)
        padded_list = []
        for i, w_multi in resampled_multis:
            vol_i = linear_vols[i]  # Post-auto_balance vol
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
        # PRE-MIX GAIN/HEADROOM: Global lin-gain from pre_mix_gain_db (def -3dB) 
        #                      post per-ch vol/pad, pre-stack. Prevents overload.
        headroom_lin = 10 ** (pre_mix_gain_db / 20.0)
        for padded in padded_list:
            padded *= headroom_lin
        # Stack and mix (mode-specific: linear_sum/rms_power/average/max_amplitude)
        stacked = torch.stack(padded_list, dim=0)
        if mix_mode == "linear_sum":
            mixed_multi = torch.sum(stacked, dim=0)
        elif mix_mode == "rms_power":
            # RMS_POWER mix: sqrt(sum(power)/N) = incoherent power average RMS
            #                With auto_balance RMS~target/ch, mix RMS~target
            N = stacked.shape[0]
            mixed_multi = torch.sqrt(torch.sum(stacked ** 2, dim=0) / N)
        elif mix_mode == "average":
            mixed_multi = torch.mean(stacked, dim=0)
        elif mix_mode == "max_amplitude":
            # MAX_AMPLITUDE mix: Max across channels → highest peak (auto_balance aids headroom)
            mixed_multi = torch.max(stacked, dim=0)[0]
        # POST-MIX CLIPPING PREVENTION (complements auto_vol_balance)
        peak = torch.max(torch.abs(mixed_multi)).item()  # Peak pre-norm/clamp
        if auto_normalize:
            if peak > 1e-8:
                mixed_multi *= (10 ** (-1 / 20)) / peak  # Peak-norm to -1dBFS headroom
        mixed_multi = torch.tanh(mixed_multi * 1.25) / 1.25  # Soft clip transients: |x| <=1.0 guaranteed
        post_peak = torch.max(torch.abs(mixed_multi)).item()
        process_details = auto_dbg + f"\
Used torch mix, Target SR: {target_sr}, ch: {target_ch}, Max len: {mixed_multi.shape[2]}, Peak pre:{peak:.3f} post:{post_peak:.3f} ({'norm+clamp' if auto_normalize else 'clamp'}:<=1.0)"
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

```

## Source Explained

This loop is part of an **auto volume balancing** mechanism in the `SoxUtilMuxAudio5_1.process` method. It adjusts per-channel volume levels (`vols`) in dB to make the **RMS (Root Mean Square)** power of each mixed audio channel match a `target` RMS level (in dB). It's executed when `mix_mode == "rms_power"`.

It performs a **single-pass adjustment**:
- Computes the current RMS (post-volume) for each resampled audio channel.
- Calculates the dB delta needed to reach the target.
- Updates the corresponding `vols[i_idx]` entry.
- Logs the measured RMS and adjustment delta for debugging.

### Key Assumptions (from Outer Context)
- `resampled_multis`: List of tuples `[(i_idx, w_multi), ...]`, where:
  - `i_idx`: Index into `vols` and `current_linear_vols` lists (e.g., channel index).
  - `w_multi`: PyTorch tensor of resampled waveform data, shape likely `(batch, channels, samples)` or similar.
- `current_linear_vols`: Snapshot of linear-scale volumes `[10 ** (v / 20.0) for v in vols]` **before** this loop (linear multipliers: 1.0 = 0 dB).
- `vols`: Mutable list of volume levels **in dB** (updated in-place).
- `target`: Target RMS level in dB (e.g., `target_rms_db`).
- `measured`, `deltas`: Lists for logging formatted values (used later in `auto_dbg` string).
- `j`: Unused loop index (from `enumerate`).

### Line-by-Line Breakdown

```python
for j, (i_idx, w_multi) in enumerate(resampled_multis):
```
- Iterates over `resampled_multis`. Unpacks each tuple into channel index `i_idx` and waveform tensor `w_multi`. `j` is the iteration counter (unused here).

```python
    vol_lin = current_linear_vols[i_idx]
```
- Retrieves the **current linear volume multiplier** for this channel (e.g., 0.5 = -6 dB).

```python
    post_vol_multi = w_multi * vol_lin
```
- Applies the current volume: scales the entire waveform tensor element-wise. Result is "post-volume" audio.

```python
    rms = torch.sqrt(torch.mean(post_vol_multi ** 2))
```
- Computes **RMS amplitude** (unitless, linear scale):
  - `post_vol_multi ** 2`: Squares all samples (power).
  - `torch.mean(...)`: Averages over all elements (typically across time/samples, possibly channels if flattened).
  - `torch.sqrt(...)`: RMS = √(mean power).

```python
    measured_db = 20 * torch.log10(torch.clamp(rms, min=1e-8)).item()
```
- Converts RMS to **dB SPL** (standard audio convention):
  - `torch.clamp(rms, min=1e-8)`: Avoids log(0) → -∞ by clamping tiny values.
  - `torch.log10(...)`: Log magnitude.
  - `20 * ...`: Voltage/dB factor (for amplitude).
  - `.item()`: Converts scalar tensor to Python float.

```python
    delta_db = target - measured_db
```
- **Adjustment needed** in dB: Positive = boost volume; negative = attenuate.

```python
    vols[i_idx] += delta_db
```
- **Updates volume in-place** in dB scale. Equivalent to multiplying linear volume by `10 ** (delta_db / 20.0)`.
  - Single-pass: Uses pre-loop snapshot (`current_linear_vols`), so no iteration/feedback.

```python
    measured.append("{:.1f}".format(measured_db))
    deltas.append("{:.1f}".format(delta_db))
```
- Logs **pre-adjustment** measured RMS (dB) and delta (dB) as 1-decimal strings for debug output (e.g., `"Auto Vol Balance: True | RMS Target: -20.0dB | Measured ... | Deltas ..."`).

### Behavior Notes
- **Single adjustment**: No loop convergence; assumes one delta is sufficient (common for preset balancing).
- **Later usage**: Updated `vols` → new `linear_vols` used for final padding/mixing/scaling.
- **Edge cases**: Clamping prevents NaN/Inf; works on batched tensors.
- **Similar in "max_amplitude" mode**: Uses `torch.max(torch.abs(...))` for peak instead of RMS.

This ensures balanced multi-channel mix (e.g., 5.1 audio) where each channel hits the target RMS after volume tweaks, preventing dominant/quiet channels.