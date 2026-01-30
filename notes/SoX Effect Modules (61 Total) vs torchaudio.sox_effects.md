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