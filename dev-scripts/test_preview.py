import torch
import torchaudio
import tempfile
import os
import subprocess
import shutil
import numpy as np

print("Testing torchaudio.save/load + SoX CLI")

waveform = torch.randn(1, 2, 44100, dtype=torch.float32)  # batch=1, stereo
sr = 44100
tmpdir = tempfile.mkdtemp(prefix='test_')
tmp_in = os.path.join(tmpdir, "input.wav")
tmp_out = os.path.join(tmpdir, "output.wav")

try:
    print("1. Saving input...")
    torchaudio.save(tmp_in, waveform, sr)
    print(f"Save OK: exists={os.path.exists(tmp_in)}, size={os.path.getsize(tmp_in)}")

    print("2. Running DeepOldMan SoX chain...")
    effect_params = ["pitch", "+1.8", "tremolo", "4.8", "35.0", "equalizer", "2500", "1000h", "-4.0", "lowpass", "2800", "gain", "-2.5"]
    cmd = ["sox", tmp_in, tmp_out] + effect_params
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"SoX rc={result.returncode}")
    if result.returncode != 0:
        print("SoX stderr:", repr(result.stderr))
    else:
        print("SoX stdout:", result.stdout.strip())

    print("3. Loading output...")
    loaded_w, loaded_sr = torchaudio.load(tmp_out)
    print(f"Load OK: shape={loaded_w.shape}, sr={loaded_sr}, norm_orig={torch.norm(waveform):.2f}, norm_loaded={torch.norm(loaded_w):.2f}")

    print("Test passed: preview works!")
except Exception as e:
    print("Error:", repr(str(e)))
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
print("Cleanup done.")