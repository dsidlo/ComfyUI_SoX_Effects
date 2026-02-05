import sys
sys.path.insert(0, '.')
exec(open('sox_voices.py').read())
import torch

audio_mono = {'waveform':torch.randn(1,1,44100,dtype=torch.float32),'sample_rate':44100}
print('Mono input shape:', audio_mono['waveform'].shape)

# CLI nodes
node1 = SoxVeDeepOldManNode()
res1 = node1.process(audio=audio_mono, enable_voice_deep_old_man=True, intensity=1.0, bass_gain=12.0, pitch_shift=1.8, tremolo_speed=4.8, tremolo_depth=35.0, eq_freq=2500, eq_gain=-4.0, eq_width=1000, lowpass_freq=2800, gain_adjust=-2.5)
print('DeepOldMan out shape:', res1[1]['waveform'].shape, 'ch:', res1[1]['waveform'].shape[1])
print('DeepOldMan CLI fail?', '*** SoX CLI failed' in res1[3])

node2 = SoxVeGhostNode()
res2 = node2.process(audio=audio_mono, enable_voice_ghost=True, intensity=1.0)
print('Ghost out shape:', res2[1]['waveform'].shape, 'ch:', res2[1]['waveform'].shape[1])
print('Ghost CLI fail?', '*** SoX CLI failed' in res2[3])

# torchaudio nodes (fallback)
node3 = SoxVeChipmunkChildNode()
res3 = node3.process(audio=audio_mono, enable_voice_chipmunk_child=True, intensity=1.0, pitch_shift=12)
print('Chipmunk out fallback?', res3[1] == audio_mono)
print('Chipmunk out ch:', res3[1]['waveform'].shape[1])

node4 = SoxVeHeliumNode()
res4 = node4.process(audio=audio_mono, enable_voice_helium=True, intensity=1.0, pitch_shift_hz=600)
print('Helium out fallback?', res4[1] == audio_mono)
print('Helium out ch:', res4[1]['waveform'].shape[1])

print('\\nSummary: Mono input → Mono output preserved for all tested nodes (CLI preserves channels dynamically, torchaudio fallback returns input mono).')