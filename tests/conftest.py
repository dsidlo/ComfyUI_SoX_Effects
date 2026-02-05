import pytest
import torch
import sys
import os
import importlib
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

node_list = [
    'sox_effects',
    'sox_voices',
    'sox_utils',
]

NODE_CLASS_MAPPINGS = {}
for module_name in node_list:
    imported_module = importlib.import_module(module_name)
    if hasattr(imported_module, 'NODE_CLASS_MAPPINGS'):
        NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)

@pytest.fixture(scope='session')
def all_node_classes():
    '''
    All node classes with INPUT_TYPES and process method.
    '''
    node_classes = {}
    for name, cls in NODE_CLASS_MAPPINGS.items():
        if hasattr(cls, 'INPUT_TYPES') and hasattr(cls, 'process'):
            node_classes[name] = cls
    return node_classes

@pytest.fixture(params=[1, 2])
def mock_audio(request):
    '''
    Mock AUDIO input: (waveform, sample_rate)
    '''
    channels = request.param
    sr = 44100
    duration = 1.0
    length = int(sr * duration)
    waveform = torch.randn(1, channels, length, dtype=torch.float32)
    return (waveform, sr)

def parse_input_types(cls):
    '''
    Parse INPUT_TYPES to dict param_name: {'type': str, 'default': val, 'min':val, ...}
    '''
    input_types = cls.INPUT_TYPES()
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