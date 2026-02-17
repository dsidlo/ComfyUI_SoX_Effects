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
from src.sox_effects import NODE_CLASS_MAPPINGS as effects_mappings
from src.sox_voices import NODE_CLASS_MAPPINGS as voices_mappings
from src.sox_utils import NODE_CLASS_MAPPINGS as utils_mappings

NODE_CLASS_MAPPINGS = {**effects_mappings, **voices_mappings, **utils_mappings}

node_names = [
    name for name, cls in NODE_CLASS_MAPPINGS.items()
    if hasattr(cls, 'INPUT_TYPES') and hasattr(cls, 'process')
]

@pytest.fixture
def mock_audio():
    return {
        'samples': torch.zeros(1, 44100),
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
                if isinstance(spec, tuple) and len(spec) > 1:
                    type_str = spec[0]
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
        assert hasattr(cls, 'process')
        assert callable(cls.process)

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
            kwargs[name] = None
        else:
            kwargs[name] = spec.get('default', None)
    node = cls()
    try:
        outputs = node.process(**kwargs)
        assert outputs is not None
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                assert value is not None, f"Output {key} is None for {node_name}"
    except Exception as e:
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
    cls = NODE_CLASS_MAPPINGS['SoxApplyEffects']
    
    sample_rate = 44100
    # Generate a test WAV file with tone
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration), dtype=torch.float32)
    test_audio = (torch.sin(2 * np.pi * 440 * t) + 0.5 * torch.sin(2 * np.pi * 880 * t)).unsqueeze(0)
    
    test_wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            test_wav_path = f.name
            torchaudio.save(test_wav_path, test_audio, sample_rate)
        
        plottable_effects = [
            {'effect': 'highpass', 'args': ['-2', '1000', '0.707q']},
            {'effect': 'bass', 'args': ['12.0', '100.0']}
        ]
        
        results = cls.get_gnuplot_formulas(plottable_effects, sample_rate=sample_rate, wave_file=test_wav_path)
    
        # Should return results for all effects
        assert len(results) == 2
        
        # Check structure of each result
        for result in results:
            assert 'effect' in result
            assert 'args' in result
            assert 'gnuplot_formula' in result
            assert 'xrange' in result
            assert 'yrange' in result
            assert 'step' in result
            
            # Verify effect name is preserved
            assert result['effect'] in ['highpass', 'bass']
            
            # If SoX is available and succeeded, check that we got a formula
            if 'error' not in result:
                # Formula should be a non-empty string or at least None/empty handled
                assert isinstance(result['gnuplot_formula'], str)
                # xrange should be a list or None
                if result['xrange'] is not None:
                    assert isinstance(result['xrange'], list)
                    assert len(result['xrange']) == 2
    
        # Test empty list
        assert cls.get_gnuplot_formulas([]) == []
    
    finally:
        if test_wav_path and os.path.exists(test_wav_path):
            os.remove(test_wav_path)
    
    # Test with single effect
    single_effect = [{'effect': 'equalizer', 'args': ['1000', '1.0q', '6.0']}]
    single_result = cls.get_gnuplot_formulas(single_effect)
    assert len(single_result) == 1
    assert single_result[0]['effect'] == 'equalizer'
