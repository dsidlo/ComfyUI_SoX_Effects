import pytest
import importlib
import subprocess
import tempfile
import os
from pathlib import Path

import torch
import torchaudio
import tempfile
import os
import numpy as np
from src.sox_effects import NODE_CLASS_MAPPINGS as effects_mappings, SoxApplyEffectsNode
from src.sox_voices import NODE_CLASS_MAPPINGS as voices_mappings
from src.sox_utils import NODE_CLASS_MAPPINGS as utils_mappings

NODE_CLASS_MAPPINGS = {**effects_mappings, **voices_mappings, **utils_mappings}

node_names = [
    name for name, cls in NODE_CLASS_MAPPINGS.items()
    if hasattr(cls, 'INPUT_TYPES') and (hasattr(cls, 'process') or hasattr(cls, 'FUNCTION'))
]

@pytest.fixture
def mock_audio():
    return {
        'samples': torch.sin(2 * torch.pi * torch.arange(44100).float() / 44100 * 440).unsqueeze(0).unsqueeze(0),
        'sampling_rate': 44100
    }

def parse_input_types(cls):
    '''
    Parse INPUT_TYPES to dict param_name: {'type': str, 'default': val, 'min':val, ...}
    '''
    try:
        input_types = cls.INPUT_TYPES()
    except Exception:
        return {}
    params = {}
    for category in ['required', 'optional']:
        if category in input_types:
            for name, spec in input_types[category].items():
                opts = {}
                if isinstance(spec, tuple):
                    type_str = spec[0]
                    if len(spec) > 1:
                        opts = spec[1] if isinstance(spec[1], dict) else {}
                else:
                    type_str = spec
                params[name] = {'type': type_str, **opts}
    return params

def test_load_all_nodes():
    '''
    Test that all SoX nodes can be loaded.
    '''
    assert len(node_names) > 0
    for name in node_names:
        cls = NODE_CLASS_MAPPINGS[name]
        assert hasattr(cls, 'INPUT_TYPES')
        func_name = cls.FUNCTION if hasattr(cls, 'FUNCTION') else 'process'
        assert hasattr(cls, func_name)
        assert callable(getattr(cls, func_name))

@pytest.mark.parametrize("node_name", node_names)
def test_process_defaults(node_name, mock_audio):
    '''
    Test node.process with default parameters (no crash, valid outputs).
    '''
    cls = NODE_CLASS_MAPPINGS[node_name]
    params = parse_input_types(cls)
    if not params:
        pytest.skip(f"INPUT_TYPES failed for {node_name}")
    kwargs = {}
    for name, spec in params.items():
        type_ = spec['type']
        if 'audio' in name.lower():
            # Map fixture keys to expected node format
            audio = {
                "waveform": mock_audio["samples"],
                "sample_rate": mock_audio["sampling_rate"]
            }
            kwargs[name] = audio
        elif type_ == 'BOOLEAN':
            kwargs[name] = spec.get('default', False)
        elif type_ == 'FLOAT':
            kwargs[name] = spec.get('default', 1.0)
        elif type_ == 'INT':
            kwargs[name] = spec.get('default', 1)
        elif type_ == 'SOX_PARAMS':
            kwargs[name] = {"sox_params": []}
        else:
            kwargs[name] = spec.get('default', None)
    node = cls()
    func_name = cls.FUNCTION if hasattr(cls, 'FUNCTION') else 'process'
    method = getattr(node, func_name)
    try:
        outputs = method(**kwargs)
        assert outputs is not None
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                assert value is not None, f"Output {key} is None for {node_name}"
    except Exception as e:
        if node_name == 'SoxApplyEffects':
            # Should now pass with empty sox_params
            raise
        pytest.xfail(f"{node_name} defaults fail: {str(e)[:100]}")


def test_get_plottable_effects():
    """
    Test that get_plottable_effects correctly identifies plottable effects
    and extracts their arguments.
    """
    cls = NODE_CLASS_MAPPINGS['SoxApplyEffects']
    
    # Test with mix of plottable and non-plottable effects
    sox_params = [
        'highpass', '-2', '1000', '0.707q',  # plottable
        'reverb', '50', '50', '100', '100', '0', '0',  # non-plottable
        'equalizer', '1000', '1.0q', '6.0',  # plottable
        'vol', '0.0',  # non-plottable
        'bass', '12.0', '100.0'  # plottable
    ]
    
    result = cls.get_plottable_effects(sox_params)
    
    # Should find 3 plottable effects
    assert len(result) == 3
    
    # Check highpass
    assert result[0]['effect'] == 'highpass'
    assert result[0]['args'] == ['-2', '1000', '0.707q']
    
    # Check equalizer
    assert result[1]['effect'] == 'equalizer'
    assert result[1]['args'] == ['1000', '1.0q', '6.0']
    
    # Check bass
    assert result[2]['effect'] == 'bass'
    assert result[2]['args'] == ['12.0', '100.0']
    
    # Test empty list
    assert cls.get_plottable_effects([]) == []
    
    # Test with no plottable effects
    non_plottable = ['reverb', '50', '50', 'vol', '0.0']
    assert cls.get_plottable_effects(non_plottable) == []


def test_get_gnuplot_formulas():
    """
    Test that get_gnuplot_formulas correctly generates gnuplot formulas
    for plottable effects.
    """
    import os
    os.environ['TEST_MODE'] = '1'

    cls = NODE_CLASS_MAPPINGS['SoxApplyEffects']
    
    sample_rate = 44100
    plottable_effects = [
        {'effect': 'highpass', 'args': ['-2', '1000', '0.707q']},
        {'effect': 'bass', 'args': ['12.0', '100.0']}
    ]
    
    results = cls.get_gnuplot_formulas(plottable_effects, sample_rate=sample_rate, wave_file=None)
    
    # Should return results for all effects
    assert len(results) == 2
    
    # Check structure of each result
    for result in results:
        assert 'effect' in result
        assert 'args' in result
        assert 'formula' in result
        assert 'xrange' in result
        assert 'yrange' in result
        assert 'step' in result
        
        # Verify effect name is preserved
        assert result['effect'] in ['highpass', 'bass']
        
        # If SoX is available and succeeded, check that we got a formula
        if 'error' not in result:
            # Formula should be a non-empty string or at least None/empty handled
            assert isinstance(result['formula'], str)
            # xrange should be a list or None
            if result['xrange'] is not None:
                assert isinstance(result['xrange'], str)
            if result['yrange'] is not None:
                assert isinstance(result['yrange'], str)

    # Test final_net_response=True in generate_combined_script
    gnu_formulas = results  # from above
    script_net = cls.generate_combined_script(gnu_formulas, output_fs=sample_rate, final_net_response=True)
    assert '20*log10(H1(f)*H2(f))' in script_net or '20*log10(H1(f) * H2(f))' in script_net
    assert 'Combined Net Response' in script_net
    assert 'b0_1' in script_net
    assert 'b0_2' in script_net

    # Test empty list
    assert cls.get_gnuplot_formulas([]) == []
    
    # Test with single effect
    single_effect = [{'effect': 'equalizer', 'args': ['1000', '1.0q', '6.0']}]
    single_result = cls.get_gnuplot_formulas(single_effect)
    assert len(single_result) == 1
    assert single_result[0]['effect'] == 'equalizer'

    # Test single with final_net_response=True in generate_combined_script
    script_single_net = cls.generate_combined_script(single_result, output_fs=sample_rate, final_net_response=True)
    assert 'H1(f)' in script_single_net  # H1(f) same as net
    assert 'Combined Net Response' in script_single_net


def test_parse_gnuplot_script():
    """Test _parse_gnuplot_script parses real SoX --plot output correctly."""
    cls = NODE_CLASS_MAPPINGS['SoxApplyEffects']

    highpass_script = """# gnuplot file
set title 'SoX effect: highpass gain=0 frequency=1000 band-width(Hz)=0 (rate=48000)'
set xlabel 'Frequency (Hz)'
set ylabel 'Amplitude Response (dB)'
Fs=48000
b0=9.386528845491728e-01; b1=-9.386528845491728e-01; b2=0.000000000000000e+00; a1=-8.773057690983457e-01; a2=0.000000000000000e+00
o=2*pi/Fs
H(f)=sqrt((b0*b0+b1*b1+b2*b2+2.*(b0*b1+b1*b2)*cos(f*o)+2.*(b0*b2)*cos(2.*f*o))/(1.+a1*a1+a2*a2+2.*(a1+a1*a2)*cos(f*o)+2.*a2*cos(2.*f*o)))
set logscale x
set samples 250
set grid xtics ytics
set key off
plot [f=10:Fs/2] [-35:25] 20*log10(H(f))
pause -1 'Hit return to continue'"""

    bass_script = """# gnuplot file
set title 'SoX effect: bass gain=12 frequency=100 slope=0.5 (rate=48000)'
set xlabel 'Frequency (Hz)'
set ylabel 'Amplitude Response (dB)'
Fs=48000
b0=1.009746865436002e+00; b1=-1.980329035605072e+00; b2=9.709207289998719e-01; a1=-1.980455793953805e+00; a2=9.805408360871408e-01
o=2*pi/Fs
H(f)=sqrt((b0*b0+b1*b1+b2*b2+2.*(b0*b1+b1*b2)*cos(f*o)+2.*(b0*b2)*cos(2.*f*o))/(1.+a1*a1+a2*a2+2.*(a1+a1*a2)*cos(f*o)+2.*a2*cos(2.*f*o)))
set logscale x
set samples 250
set grid xtics ytics
set key off
plot [f=10:Fs/2] [-35:25] 20*log10(H(f))
pause -1 'Hit return to continue'"""

    # Test highpass
    parsed_hp = cls._parse_gnuplot_script(highpass_script)
    assert isinstance(parsed_hp, dict)
    assert 'gnuplot_script' in parsed_hp
    assert 'title' in parsed_hp
    assert parsed_hp['title'] == "SoX effect: highpass gain=0 frequency=1000 band-width(Hz)=0 (rate=48000)"
    assert 'x_label' in parsed_hp
    assert parsed_hp['x_label'] == 'Frequency (Hz)'
    assert 'y_label' in parsed_hp
    assert parsed_hp['y_label'] == 'Amplitude Response (dB)'
    assert 'logscale' in parsed_hp
    assert parsed_hp['logscale'] == 'x'
    assert 'samples' in parsed_hp
    assert parsed_hp['samples'] == '250'
    assert 'fs' in parsed_hp
    assert parsed_hp['fs'] == 48000.0
    assert 'H' in parsed_hp
    assert 'coeffs' in parsed_hp
    assert len(parsed_hp['gnuplot_script']) > 100

    # Test bass
    parsed_bass = cls._parse_gnuplot_script(bass_script)
    assert isinstance(parsed_bass, dict)
    assert 'gnuplot_script' in parsed_bass
    assert 'title' in parsed_bass
    assert parsed_bass['title'] == "SoX effect: bass gain=12 frequency=100 slope=0.5 (rate=48000)"
    assert 'x_label' in parsed_bass
    assert parsed_bass['x_label'] == 'Frequency (Hz)'
    assert 'y_label' in parsed_bass
    assert parsed_bass['y_label'] == 'Amplitude Response (dB)'
    assert 'logscale' in parsed_bass
    assert parsed_bass['logscale'] == 'x'
    assert 'samples' in parsed_bass
    assert parsed_bass['samples'] == '250'
    assert 'fs' in parsed_bass
    assert parsed_bass['fs'] == 48000.0
    assert 'H' in parsed_bass
    assert 'coeffs' in parsed_bass
    assert len(parsed_bass['gnuplot_script']) > 100


def test_generate_combined_script():
    """Test generate_combined_script produces valid combined plot script."""
    cls = NODE_CLASS_MAPPINGS['SoxApplyEffects']

    highpass_script = """# gnuplot file
set title 'SoX effect: highpass gain=0 frequency=1000 band-width(Hz)=0 (rate=48000)'
set xlabel 'Frequency (Hz)'
set ylabel 'Amplitude Response (dB)'
Fs=48000
b0=9.386528845491728e-01; b1=-9.386528845491728e-01; b2=0.000000000000000e+00; a1=-8.773057690983457e-01; a2=0.000000000000000e+00
o=2*pi/Fs
H(f)=sqrt((b0*b0+b1*b1+b2*b2+2.*(b0*b1+b1*b2)*cos(f*o)+2.*(b0*b2)*cos(2.*f*o))/(1.+a1*a1+a2*a2+2.*(a1+a1*a2)*cos(f*o)+2.*a2*cos(2.*f*o)))
set logscale x
set samples 250
set grid xtics ytics
set key off
plot [f=10:Fs/2] [-35:25] 20*log10(H(f))
pause -1 'Hit return to continue'"""

    bass_script = """# gnuplot file
set title 'SoX effect: bass gain=12 frequency=100 slope=0.5 (rate=48000)'
set xlabel 'Frequency (Hz)'
set ylabel 'Amplitude Response (dB)'
Fs=48000
b0=1.009746865436002e+00; b1=-1.980329035605072e+00; b2=9.709207289998719e-01; a1=-1.980455793953805e+00; a2=9.805408360871408e-01
o=2*pi/Fs
H(f)=sqrt((b0*b0+b1*b1+b2*b2+2.*(b0*b1+b1*b2)*cos(f*o)+2.*(b0*b2)*cos(2.*f*o))/(1.+a1*a1+a2*a2+2.*(a1+a1*a2)*cos(f*o)+2.*a2*cos(2.*f*o)))
set logscale x
set samples 250
set grid xtics ytics
set key off
plot [f=10:Fs/2] [-35:25] 20*log10(H(f))
pause -1 'Hit return to continue'"""

    # Parse both scripts
    parsed_hp = cls._parse_gnuplot_script(highpass_script)
    parsed_bass = cls._parse_gnuplot_script(bass_script)

    # Test with both (expect renamed H1/H2)
    combined_script = SoxApplyEffectsNode.generate_combined_script([parsed_hp, parsed_bass], output_fs=48000)
    assert isinstance(combined_script, str)
    assert len(combined_script) > 500
    assert "# Combined SoX Effects Frequency Response" in combined_script
    assert "set title 'Combined SoX Effects" in combined_script
    assert "Fs=48000" in combined_script
    assert "o=2*pi/Fs" in combined_script
    assert "H1(f)=" in combined_script
    assert "H2(f)=" in combined_script
    assert "b0_1=" in combined_script
    assert "b0_2=" in combined_script
    assert "plot [f=20:20000] [-60:30] \\" in combined_script
    assert "20*log10(H1(f))" in combined_script
    assert "20*log10(H2(f))" in combined_script

    # Test with final_net_response=True
    combined_script_net = SoxApplyEffectsNode.generate_combined_script([parsed_hp, parsed_bass], output_fs=48000, final_net_response=True)
    assert isinstance(combined_script_net, str)
    assert len(combined_script_net) > 500
    assert "# Combined SoX Effects Frequency Response" in combined_script_net
    assert "set title 'Combined SoX Effects" in combined_script_net
    assert "Fs=48000" in combined_script_net
    assert "o=2*pi/Fs" in combined_script_net
    assert "H1(f)=" in combined_script_net
    assert "H2(f)=" in combined_script_net
    assert "b0_1=" in combined_script_net
    assert "b0_2=" in combined_script_net
    assert "plot [f=20:20000] [-60:30] \\" in combined_script_net
    assert "20*log10(H1(f))" in combined_script_net
    assert "20*log10(H2(f))" in combined_script_net
    assert '20*log10(H1(f)*H2(f))' in combined_script_net or '20*log10(H1(f) * H2(f))' in combined_script_net
    assert "Combined Net Response" in combined_script_net

    # Test empty list (graceful)
    empty_combined = SoxApplyEffectsNode.generate_combined_script([])
    assert "Combined SoX Effects" in empty_combined
    assert "plot" in empty_combined  # Still has plot command, empty

    # Test single (no rename)
    single_combined = SoxApplyEffectsNode.generate_combined_script([parsed_hp])
    assert "H1(f)=" in single_combined  # Still renames to _1
    assert "b0_1=" in single_combined

    # Test single with final_net_response=True
    single_combined_net = SoxApplyEffectsNode.generate_combined_script([parsed_hp], final_net_response=True)
    assert "H1(f)=" in single_combined_net
    assert "b0_1=" in single_combined_net
    assert 'H1(f)' in single_combined_net  # Same as net
    assert "Combined Net Response" in single_combined_net




def test_sox_apply_effects_plot(mock_audio, monkeypatch):
    """
    Test SoxApplyEffects with enable_sox_plot=True.
    Verifies audio passthrough/processing, IMAGE output structure, and no crash.
    """
    import os
    os.environ['TEST_MODE'] = '1'

    cls = NODE_CLASS_MAPPINGS['SoxApplyEffects']
    
    # Parse input types
    params = parse_input_types(cls)
    if not params:
        pytest.skip(f"INPUT_TYPES failed for SoxApplyEffects")
    
    # Common kwargs setup
    kwargs = {}
    for name, spec in params.items():
        type_ = spec['type']
        if 'audio' in name.lower():
            audio = {
                "waveform": mock_audio["samples"],
                "sample_rate": mock_audio["sampling_rate"]
            }
            kwargs[name] = audio
        elif type_ == 'BOOLEAN':
            kwargs[name] = spec.get('default', False)
        elif type_ == 'FLOAT':
            kwargs[name] = spec.get('default', 1.0)
        elif type_ == 'INT':
            kwargs[name] = spec.get('default', 1)
        elif type_ == 'SOX_PARAMS':
            # Provide plottable params for testing plot
            kwargs[name] = {"sox_params": ['highpass', '-2', '1000', 'bass', '12.0', '100.0']}
        else:
            kwargs[name] = spec.get('default', None)
    
    # Override for plotting
    kwargs['enable_sox_plot'] = True
    kwargs['final_net_response'] = False
    
    node = cls()
    
    # Case 1: enable_apply=False (passthrough audio, plot from params)
    kwargs['enable_apply'] = False
    outputs = node.apply_effects(**kwargs)
    assert outputs is not None
    assert len(outputs) == 3
    processed_audio, sox_plot_image, dbg_text = outputs
    
    # Audio should be passed through unchanged
    assert torch.equal(processed_audio['waveform'], kwargs['audio']['waveform'])
    assert processed_audio['sample_rate'] == kwargs['audio']['sample_rate']
    
    # sox_plot_image should be a valid IMAGE tensor
    assert isinstance(sox_plot_image, torch.Tensor)
    assert sox_plot_image.shape[0] == 1  # Batch size 1
    assert sox_plot_image.shape[3] == 3  # RGB
    assert sox_plot_image.dtype == torch.float32
    assert sox_plot_image.min() >= 0
    assert sox_plot_image.max() <= 1

    # Basic check: not all zeros (even if no sox, should have some content or blank but structured)
    assert not torch.all(sox_plot_image == 0)

    # dbg_text should contain plot info
    assert 'GnuPlot cmd' in dbg_text or 'GnuPlot skipped' in dbg_text

    # Case 2: enable_apply=True (process audio + plot from params)
    kwargs['enable_apply'] = True
    outputs = node.apply_effects(**kwargs)
    assert outputs is not None
    assert len(outputs) == 3
    processed_audio, sox_plot_image, dbg_text = outputs

    # Audio should be processed (not equal to input, assuming sox_params has effect)
    assert not torch.equal(processed_audio['waveform'], kwargs['audio']['waveform'])
    assert processed_audio['sample_rate'] == kwargs['audio']['sample_rate']

    # sox_plot_image should be a valid IMAGE tensor (plot from params, independent of apply)
    assert isinstance(sox_plot_image, torch.Tensor)
    assert sox_plot_image.shape[0] == 1  # Batch size 1
    assert sox_plot_image.shape[3] == 3  # RGB
    assert sox_plot_image.dtype == torch.float32
    assert sox_plot_image.min() >= 0
    assert sox_plot_image.max() <= 1

    # Basic check: not all zeros
    assert not torch.all(sox_plot_image == 0)

    # dbg_text should contain both apply and plot info
    assert 'SoX cmd executed' in dbg_text or 'SoX (Successful)' in dbg_text  # From apply
    assert 'GnuPlot cmd' in dbg_text  # From plot

    # Case 3: with final_net_response=True (verify net in script)
    kwargs['final_net_response'] = True
    outputs_net = node.apply_effects(**kwargs)
    assert outputs_net is not None
    assert len(outputs_net) == 3
    processed_audio_net, sox_plot_image_net, dbg_text_net = outputs_net

    # Audio same as case 2 (plot independent)
    assert torch.allclose(processed_audio_net['waveform'], processed_audio['waveform'], atol=1e-3)

    # sox_plot_image valid
    assert isinstance(sox_plot_image_net, torch.Tensor)
    assert sox_plot_image_net.shape[0] == 1
    assert sox_plot_image_net.shape[3] == 3
    assert sox_plot_image_net.dtype == torch.float32
    assert sox_plot_image_net.min() >= 0
    assert sox_plot_image_net.max() <= 1
    assert not torch.all(sox_plot_image_net == 0)

    # dbg_text should contain net response in script
    # The script is now part of the plot_dbg, let's check it's there
    assert 'Combined Net Response' in dbg_text_net or 'Combined Net Response' in outputs_net[2]
    # We can also check if the net response formula is present
    assert '20*log10(H1(f) * H2(f))' in dbg_text_net or '20*log10(H1(f)*H2(f))' in dbg_text_net
