import sys
sys.path.insert(0, '.')
exec(open('sox_voices.py').read())
import torch
import tempfile
import subprocess
import shutil
import os

audio = {"waveform": torch.randn(1, 2, 44100, dtype=torch.float32), "sample_rate": 44100}
node = SoxVeDeepOldManNode()
print("Node instantiated OK")

orig_norm = torch.norm(audio["waveform"])
print(f"Orig norm: {orig_norm:.3f}")

res = node.process(audio=audio, enable_voice_deep_old_man=True, intensity=1.0)
proc_audio, sox_params, dbg_text = res[1], res[2], res[3]
proc_norm = torch.norm(proc_audio["waveform"])
ratio = proc_norm / orig_norm

print(f"Proc norm: {proc_norm:.3f}")
print(f"Ratio: {ratio:.3f} (expect ~0.6-0.8 due to gain -2.5dB etc.)")
print(f"SoX params len: {len(sox_params['sox_params'])}")
print("Dbg:")
print(dbg_text)
print("SUCCESS: Preview modified!" if ratio < 0.95 else "FAIL: No change")