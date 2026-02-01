---
apply: always
---

# AI Rules Programmer SoX

## Before taking action...
- Make sure that you understand key SoX and torchaudio.sox_effects data structures.
- Make sure that you understand key SoX and torchaudio.sox_effects programming paradigms.
- Search or look up any data structures or programming best practices in SoX and torchaudio.sox_effects docs if you have a doubt regarding something.
- When creating input widgets for an interface regarding audio effects parameters
  - verify the limits and valid values of the parameters and apply appropriate limits to the input widget
  - if actual docs are not available for an effects parameters, use documented "useful" ranges if available
    - or determine appropriate ranges based on your understanding of the effect or pipeline of effects.

### SoX Resources

#### Official / Primary Sources
- **Official SoX Homepage** (SourceForge)  
  - features list, basic docs, and history https://sox.sourceforge.net  
  - Includes the full list of supported formats and effects: https://sox.sourceforge.net/Docs/Features  
  - Documentation overview: https://sox.sourceforge.net/Docs/Documentation

- **Man Pages (Detailed Reference)**  
  - `sox(1)` main man page (effects, syntax, examples): https://linux.die.net/man/1/sox or https://man.archlinux.org/man/sox.1.en  
  - `soxformat(7)` (file formats & devices): https://manpages.ubuntu.com/manpages/jammy/man7/soxformat.7.html  

- **libSoX Reference** 
  - command line: man libsox

#### Tutorials & Guides
- **Opensource.com Article: "Convert audio files with this versatile Linux command"** (2021, still relevant)  
  https://opensource.com/article/20/2/linux-sox

- **Hexmos Cmd Cheatsheet: SoX Command Examples**  
  Quick, copy-paste examples for common tasks (merge, trim, normalize, reverse, sample rate change, generate tones, volume adjust).  
  https://hexmos.com/freedevtools/c/cmd/sox

- **A Basic Introduction to SoX** (Aiyumi blog, 2011)  
  Solid older intro covering basics, formats, effects — timeless fundamentals.  
  https://aiyumi.github.io/en/blog/sox-basic-intro

#### Community & Advanced
- **GitHub Forks & Variants** (for modern patches/builds):  
  - Original-ish repo mirrors: https://github.com/mansr/sox (or similar).  
  - DSD/high-res patches: https://github.com/turbulentie/sox-dsd-win (Windows binaries with extras).  
  - DAW plugin ports: https://github.com/prof-spock/SoX-Plugins (Reimplementation as VST3/AU/JSFX for Reaper/etc.).

- **Mailing List/Support** (for questions): sox-users on SourceForge — still active for niche issues.  
  https://sourceforge.net/projects/sox/support

- **Reddit & Forums**: 
  - Search /r/linuxaudio, /r/audioengineering, or /r/commandline for "SoX" — tons of real-world scripts (e.g., batch processing voice effects, resampling debates, degradation for retro sounds).

#### Official Documentation (Best Starting Point)
The PyTorch Audio docs have the most authoritative reference. Version-specific pages vary slightly due to deprecations:

- **Latest / Main Branch** (shows deprecation notice):  
  https://pytorch.org/audio/main/sox_effects.html  
  Covers `apply_effects_tensor`, `apply_effects_file`, `effect_names`, etc.

- **Stable / Recent Versions** (e.g., 2.5.0 or 2.7.0 — still usable without heavy deprecation warnings):  
  https://pytorch.org/audio/2.5.0/sox_effects.html  
  https://pytorch.org/audio/stable/sox_effects.html (redirects to current stable)

- **Older but Detailed** (great for legacy code/examples):  
  https://pytorch.org/audio/0.12.0/sox_effects.html  
  https://pytorch.org/audio/0.7.0/sox_effects.html (shows deprecated classes like `SoxEffectsChain`)

Key functions documented:
- `torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects)` — Apply chain to tensor (e.g., `[["pitch", "150"], ["tremolo", "5", "30"]]`).
- `torchaudio.sox_effects.apply_effects_file(path, effects)` — Load + apply from file.
- `torchaudio.sox_effects.effect_names()` — List available effects (matches SoX).

#### Tutorials & Examples
- **Official TorchAudio Tutorials** (use `apply_effects_tensor` for data augmentation):  
  https://pytorch.org/audio/stable/tutorials/audio_data_augmentation_tutorial.html  

- **Deepgram Blog Intro** (2024):  
  https://deepgram.com/learn/pytorch-intro-with-torchaudio

- **Scaler Topics Guide**:  
  https://www.scaler.com/topics/pytorch/torchaudio-in-pytorch

- **GitHub Issues** (real-world troubleshooting/examples):  
  - MP3 loading fails in newer versions (use FFmpeg backend or load first): https://github.com/pytorch/audio/issues/2691  
  - Effect option mismatches with SoX CLI: https://github.com/pytorch/audio/issues/1666  
  - Multiprocessing crashes with certain effects: https://github.com/pytorch/audio/issues/1021  

