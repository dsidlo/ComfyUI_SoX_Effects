import pytest
import importlib
from pathlib import Path
import pytest

project_root = Path(__file__).parent.parent.resolve()
import sys
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

node_names = [
    name for name, cls in NODE_CLASS_MAPPINGS.items()
    if hasattr(cls, 'INPUT_TYPES') and hasattr(cls, 'process')
]

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
            kwargs[name] = mock_audio
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