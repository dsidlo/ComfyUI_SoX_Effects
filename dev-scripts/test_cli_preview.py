import torch
import torchaudio
import tempfile
import subprocess
import os

waveform = torch.randn(1, 1, 16000*5).float()  # 5s mono
sr = 16000

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    input_path = f.name

torchaudio.save(input_path, waveform[0], sr)

output_path = os.path.splitext(input_path)[0] + '_preview.wav'
effect_params = ['bass', '+12', '100', '1q']
cmd = ['sox', input_path, output_path] + effect_params
result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
print('returncode:', result.returncode)
print('stdout:', repr(result.stdout))
print('stderr:', repr(result.stderr))
if result.returncode == 0:
    processed, psr = torchaudio.load(output_path)
    print('loaded shape:', processed.shape, 'psr:', psr)
    # check if different from orig
    orig = waveform[0]
    print('orig mean:', orig.mean().item(), 'processed mean:', processed.mean().item())
else:
    print('sox failed')
os.unlink(input_path)
if os.path.exists(output_path):
    os.unlink(output_path)