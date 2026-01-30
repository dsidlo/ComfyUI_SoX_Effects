The project implements ~61 SoX effect nodes (extracted from `__init__.py` class definitions). Pure effect nodes (excluding utilities like `SoxApplyEffectsNode`, `SoxSpectrogramNode`, `SoxUtil*`):

#### Effects (non-voice):
- `allpass`, `band`, `bandpass`, `bandreject`, `biquad`, `channels`, `contrast`, `dcshift`, `deemph`, `delay`, `dither`, `downsample`, `earwax`, `fade`, `fir`, `gain`, `hilbert`, `ladspa`, `loudness`, `mcompand`, `noiseprof`, `noisered`, `norm`, `oops`, `pad`, `rate`, `remix`, `repeat`, `reverse`, `riaa`, `silence`, `sinc`, `speed`, `splice`, `stat`, `stats`, `stretch`, `swap`, `synth`, `trim`, `upsample`, `vad`, `vol`

#### Voice/Advanced:
- `bass`, `bend`, `chorus`, `compand`, `echo`, `echos`, `equalizer`, `flanger`, `highpass`, `lowpass`, `overdrive`, `phaser`, `pitch`, `reverb`, `tempo`, `treble`, `tremolo`

### `torchaudio.sox_effects` Supported Effects
`torchaudio.sox_effects.apply_effects(waveform, sample_rate, effects)` emulates ~35-40 common SoX effects **directly on tensors** (no CLI/files). Key supported (from docs/source):

- `band`, `bandpass`, `bandreject`
- `bass`, `treble`
- `biquad`
- `compand` (basic), `mcompand` (multi-band)
- `contrast`
- `dcshift`
- `deemph`
- `earwax`
- `equalizer`
- `fade` (in/out/h/q/t)
- `flanger`
- `gain`
- `highpass`, `lowpass`
- `mixer` (for `channels`/`pan`/`remix` partial)
- `noisered` (requires profile)
- `norm`
- `overdrive`
- `pad`
- `phaser`
- `pitch`
- `rate` (polyphase)
- `reverb`
- `silence`
- `speed`
- `stats`
- `swap`
- `synth`
- `tempo`
- `tremolo`
- `trim`
- `upsample`/`downsample` (via `rate`)
- `vad`
- `vol`

**Unsupported** (CLI-only): `allpass`, `bend`, `chorus`, `delay`/`echo`, `dither`, `fir` (full), `hilbert`, `ladspa`, `loudness`, `noiseprof`, `oops`, `riaa`, `repeat`, `reverse`, `sinc` (full), `splice`, `stat`, `stretch`, full `remix`/`channels`.

### Modules That **Can Be Re-Implemented** Using `torchaudio.sox_effects`
~30 modules have **direct equivalents** (params match closely; replace CLI param append → tensor apply). Re-implementation: New `TorchSox*Node` or hybrid `process()` using `torchaudio.sox_effects.apply_effects([effect, param1, ...])`.

| Re-implementable Module | torch.sox_effects Equivalent | Notes |
|-------------------------|------------------------------|-------|
| `SoxBandNode` | `band` | ✓ Full match. |
| `SoxBandpassNode` | `bandpass` | ✓ Freq/width. |
| `SoxBandrejectNode` | `bandreject` | ✓ Freq/width. |
| `SoxBassNode` | `bass` | ✓ Gain/freq/width. |
| `SoxBiquadNode` | `biquad` | ✓ Freq/gain/Q. |
| `SoxContrastNode` | `contrast` | ✓ Enhancement %. |
| `SoxDcshiftNode` | `dcshift` | ✓ Amount. |
| `SoxEarwaxNode` | `earwax` | ✓ Simple toggle. |
| `SoxEqualizerNode` | `equalizer` | ✓ Freq/width/gain. |
| `SoxFadeNode` | `fade` | ✓ Type/in/out len. |
| `SoxFlangerNode` | `flanger` | ✓ Delay/depth/regen/speed/shape. |
| `SoxGainNode` | `gain` | ✓ Amount/normalize. |
| `SoxHighpassNode` | `highpass` | ✓ Poles/freq/width. |
| `SoxLowpassNode` | `lowpass` | ✓ Poles/freq/width. |
| `SoxMcompandNode` | `mcompand` | ✓ Params string (parse/match). |
| `SoxNormNode` | `norm` | ✓ Level/precision. |
| `SoxOverdriveNode` | `overdrive` | ✓ Gain/colour. |
| `SoxPadNode` | `pad` | ✓ Intro/outro. |
| `SoxPhaserNode` | `phaser` | ✓ Gain/delay/decay/speed. |
| `SoxPitchNode` | `pitch` | ✓ Shift/segment/search/overlap. |
| `SoxRateNode` | `rate` | ✓ Quality. |
| `SoxReverbNode` | `reverb` | ✓ Reverberance/HF/room/stereo/pre/wet. |
| `SoxSilenceNode` | `silence` | ✓ Above/duration. |
| `SoxSpeedNode` | `speed` | ✓ Factor. |
| `SoxStatsNode` | `stats` | ✓ Tags. |
| `SoxSwapNode` | `swap` | ✓ Operation. |
| `SoxSynthNode` | `synth` | ✓ Params. |
| `SoxTempoNode` | `tempo` | ✓ Factor/segment/search/overlap. |
| `SoxTrebleNode` | `treble` | ✓ Gain/freq/width. |
| `SoxTremoloNode` | `tremolo` | ✓ Speed/depth. |
| `SoxTrimNode` | `trim` | ✓ Start/end. |
| `SoxVadNode` | `vad` | ✓ Threshold. |
| `SoxVolNode` | `vol` | ✓ Gain. |
| `SoxDeemphNode` | `deemph` | ✓ Profile (ccir/50t/75t). |

**Partial**:
- `SoxChannelsNode`/`SoxRemixNode`: `mixer`.
- `SoxNoiseredNode`: `noisered` (pair w/ `noiseprof` manual).
- `SoxDownsampleNode`/`SoxUpsampleNode`: `rate`.
- `SoxStatsNode`: `stats` (output text).

### Modules That **Cannot** Be Re-Implemented (SoX-Unique)
~25 modules require CLI-only effects:
- `SoxAllpassNode`, `SoxBendNode`, `SoxChorusNode`, `SoxCompandNode` (use `mcompand`), `SoxDelayNode`, `SoxEchoNode`/`SoxEchosNode`, `SoxDitherNode`, `SoxFirNode`, `SoxHilbertNode`, `SoxLadspaNode`, `SoxLoudnessNode`, `SoxNoiseprofNode`, `SoxOopsNode`, `SoxRepeatNode`, `SoxReverseNode`, `SoxRiaaNode`, `SoxSincNode`, `SoxSpliceNode`, `SoxStatNode`, `SoxStretchNode`.

Keep CLI for these (as per prior analysis: SoX excels in modulation/creative effects).

### Additional `torch.sox_effects` Nodes to Add
Few gaps (project covers most):
- **Dedicated `SoxCompandNode`**: Basic single-band (vs `mcompand`).
- **SoxMixerNode**: Explicit `mixer` for advanced `remix`/`pan`/`channels`.
- **SoxPolyphaseNode**: `polyphase` (resampling quality).

**Big Win: TorchApplyEffectsNode**
- New node parallel to `SoxApplyEffectsNode`: Chain `TORCH_PARAMS` → apply `torchaudio.sox_effects.apply_effects()` on tensor.
- **Benefits**: GPU-accelerated, no subprocess/tempfiles, faster/batchable, differentiable.
- **Fallback**: If unsupported effect → CLI or warn.
- Supports **all above** in one node; hybrid chain.

**Recommendation**: Prioritize re-implement simple/CPU-heavy (gain/vol/norm/pad/trim/fade) for speed. Keep complex/unique CLI. Test params parity (e.g., `sox input.wav gain -6 output.wav` vs `apply_effects(..., ["gain", "-6"])`).

**Status**: High overlap (~70%). `torch.sox_effects` viable for optimization/portability (no SoX CLI dep).

## SoX cli vs torchaudio.sox_effects Parameter Parity

**Note**: Parameter parity testing is crucial for seamless integration. Ensure CLI and `apply_effects()` produce identical results for common effects (e.g., gain, pitch, speed) to maintain consistency and reliability.

Yes, there are some **parameter differences** between **torchaudio.sox_effects** (via `apply_effects_tensor` / `apply_effects_file`) and the standalone CLI **sox** tool, even though both rely on the same underlying **libsox** library.

### Key Differences
1. **Automatic implicit effects**  
   - CLI sox often inserts additional effects automatically (e.g., `rate` after `speed`, `pitch`, `tempo`, or other rate-altering effects to make the output playable/sample-rate consistent).  
   - torchaudio **does not** do this — it applies **exactly** the effects you provide, in order.  
     → After `speed`, `pitch`, `tempo`, or similar, you must **explicitly add** a `rate` effect with your desired final sample rate, or the result may have an incorrect/inconsistent rate or fail.

2. **Effect parameter syntax and parsing**  
   - CLI sox: Parameters are space-separated after the effect name (e.g., `sox in.wav out.wav pitch 300 0.1`).  
   - torchaudio: Each effect is a list where the first element is the effect name (string), and subsequent elements are **individual string arguments** (even numbers must be strings).  
     → CLI: `pitch 300`  
     → torchaudio: `["pitch", "300"]`  
   - Most effects accept the **same numerical values and order**, but you must convert everything to strings and split properly.

3. **Effect option validation and errors**  
   - Some older torchaudio versions had bugs where certain parameter combinations triggered "Invalid effect option" even when valid in CLI sox (GitHub issue #1666 from 2021). These are mostly fixed in recent versions (torchaudio 2.0+), but rare edge cases might still differ slightly due to how arguments are passed through the C API.

4. **Other minor/behavioral differences**  
   - No automatic dithering, format handling, or output normalization unless explicitly added.  
   - torchaudio works only on CPU tensors (no CUDA acceleration for these effects).  
   - Some very CLI-specific flags (e.g., global options like `--buffer`, `--effects-chain`) don't apply.

### Table of Notable Parameter / Behavior Differences
For most effects, **parameters are identical** in meaning/order/value — just the passing format differs. The table highlights effects/behaviors with **known practical differences**:

| Effect / Aspect          | CLI sox Example                              | torchaudio Example                              | Key Difference / Note                                                                 |
|--------------------------|----------------------------------------------|-------------------------------------------------|---------------------------------------------------------------------------------------|
| speed                    | `speed 1.2`                                  | `["speed", "1.2"]`                              | CLI auto-adds `rate` after; torchaudio **requires explicit `["rate", "desired_sr"]`** after |
| pitch                    | `pitch 300`                                  | `["pitch", "300"]`                              | Same as above — explicit `rate` needed afterward                                      |
| tempo                    | `tempo 1.3`                                  | `["tempo", "1.3"]`                              | CLI auto-adds `rate`; torchaudio needs explicit `rate`                                |
| rate                     | `rate 44100`                                 | `["rate", "44100"]`                             | Often implicit in CLI chains; must be explicit in torchaudio if changing sr           |
| gain -n                  | `gain -n`                                    | `["gain", "-n"]`                                | Identical; normalizes to 0 dB                                                         |
| reverb                   | `reverb 50 50 60 ...` (many params)          | `["reverb", "50", "50", "60", ...]`             | Identical parameters; just string-split                                               |
| compand                  | `compand 0.3|0.8:6:-70/-60/-20`                | `["compand", "0.3|0.8:6:-70/-60/-20"]`          | CLI allows compact `|` / `:` syntax; torchaudio usually needs full string as one arg  |
| noisered                 | `noisered noise.prof 0.21`                   | `["noisered", "noise.prof", "0.21"]`            | Identical; but noise profile file must be prepared separately                         |
| General argument passing | Space-separated, mixed types                 | List of strings only                            | All args → str; no automatic type conversion                                          |
| Automatic rate handling  | Often inserted by sox                        | Never inserted                                  | Biggest practical difference — chains can sound wrong without explicit `rate`         |
| Effect availability      | ~60 (your install)                           | ~45–55 (depends on build)                       | Some (e.g. ladspa, firfit) often missing in torchaudio                                |

### Recommendation for Your Use Case (ComfyUI Node)
When writing your fallback code:
- Use the **same parameter values** as in SoX docs / CLI examples.
- Always convert them to strings in lists: e.g. for CLI `pitch 400 rate 22050` → `[["pitch", "400"], ["rate", "22050"]]`
- For effects that alter playback rate/duration (`speed`, `pitch`, `tempo`, `stretch`, etc.), **always append** a `rate` effect at your target sample rate — this is the most common gotcha when migrating from CLI to torchaudio.
- Test chains with `torchaudio.sox_effects.effect_names()` on your install to confirm support.

