# SoX --plot Usage

**It would be inappropriate (or at least pointless/unhelpful) to use the `--plot` global option in SoX in several common scenarios**, based on how the feature is designed and documented in the SoX man page and source behavior.

`--plot` (with `gnuplot` or `octave`) is a **special diagnostic mode** that:
- Only supports **transfer-function-based effects** (linear filters and some dynamics processors that have a well-defined frequency/phase or gain curve).
- Scans the effects chain for the **first** supported effect.
- Outputs plotting script code (to stdout).
- **Immediately exits** without reading any input audio files, applying effects, or writing output audio.

If your command doesn't fit this narrow purpose, `--plot` either wastes time, produces no useful output, or interferes with normal operation.

### Main Cases Where `--plot` Is Inappropriate / Should Be Avoided

1. **When no supported transfer-function effect is present in the chain**  
   - SoX will find no plottable effect → outputs nothing useful (empty or minimal script) and exits.  
   - Examples of unsupported / non-plottable effects (common ones that do **not** support `--plot`):
     - `reverb`, `echo`, `chorus`, `phaser`, `flanger` (some versions may partially support, but not reliably)
     - `tempo`, `speed`, `pitch`, `rate` (resampling), `remix`, `channels`
     - `vol`, `fade`, `splice`, `trim`, `silence`, `vad`
     - `stat`, `spectrogram`, `noisered`, `synth`, `overdrive`, `compand` (wait — `compand` actually does support it in recent versions for its compression curve!)
     - Most non-linear or time-domain effects  
   - Command like: `sox --plot gnuplot input.wav output.wav reverb 50` → exits immediately with useless/empty plot output, no reverb applied.

2. **When you actually want to process audio (convert, apply effects, output a file)**  
   - `--plot` **never** processes audio — it always short-circuits and exits early.  
   - Inappropriate for: batch conversion, playback (`play`), recording (`rec`), normalization, FIR application to real audio, etc.  
   - Example: `sox --plot octave input.wav -n fir coeffs.txt norm -1` → no output file created, FIR not applied — just plot script (if `fir` supports it).

3. **When using multiple plottable effects in chain**  
   - Only the **first** supported effect gets plotted — subsequent ones (e.g., `equalizer` after `highpass`) are ignored.  
   - If you need combined response, `--plot` alone won't help (some users script multiple runs or use external tools).  
   - Inappropriate if your goal is to visualize the full chain.

4. **In production scripts, automation, or piped workflows**  
   - It exits early → breaks pipes (`|`) or multi-command scripts expecting audio output.  
   - Can confuse tools parsing stdout (plot script instead of audio data).  
   - Wastes CPU if accidentally left in long-running batch jobs.

5. **When `--plot off` is explicitly needed (rare)**  
   - Default is off anyway, but if some wrapper/script forces `--plot`, you override with `--plot off` to restore normal behavior.

### Quick Rule of Thumb
Use `--plot` **only** when:
- Your **sole goal** is to tune/visualize parameters of a supported filter effect (e.g., `highpass`, `lowpass`, `bandpass`, `equalizer`, `sinc`, `fir`, `biquad`, sometimes `compand`).
- You're okay with **no audio processing** happening.
- You're redirecting output to a .gp / .m file for gnuplot/Octave.

Otherwise → **omit `--plot`** entirely (or use `--plot off` if it's somehow forced on you).

Examples of good use:
```bash
sox --plot gnuplot -n -n equalizer 1000 1.5q +6 > eq.gp     # Tune EQ curve
sox --plot octave -n -n fir coeffs.txt > fir.m               # Check FIR response
```

Bad/inappropriate use:
```bash
sox --plot gnuplot song.mp3 song-normalized.mp3 gain -n      # Useless — exits early, no normalization
play --plot octave track.wav                                 # play won't play anything
```

If you're running into a specific command where `--plot` behaves oddly (e.g., empty output, unexpected exit), share it — it's likely one of these mismatch cases!
