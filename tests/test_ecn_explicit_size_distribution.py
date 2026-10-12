"""The ECN configuration must never infer its size distribution from a mean."""
from pathlib import Path
import ast
import math

import pytest
from trinity.defaults import default_config


def _size_validator():
    # Extract the pure app helper without importing the optional Dash runtime.
    root = Path(__file__).resolve().parents[1]
    module = ast.parse((root / 'app.py').read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_ecn_size_probabilities')
    namespace = {'Any': object, 'math': math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(root / 'app.py'), 'exec'), namespace)
    return namespace['_ecn_size_probabilities']


def test_default_ecn_has_explicit_size_probabilities():
    probs = default_config()['passiveEcn']['tradeSizeProbabilities']
    assert len(probs) == 5
    assert sum(probs) == pytest.approx(1.0)


def test_missing_distribution_rejected_even_with_legacy_mean():
    validate = _size_validator()
    with pytest.raises(ValueError, match='requires explicit tradeSizeProbabilities'):
        validate({'meanTradeSize': 0.5})
    with pytest.raises(ValueError, match='requires explicit tradeSizeProbabilities'):
        validate({})


def test_explicit_distribution_preserved():
    validate = _size_validator()
    probs = [0.3, 0.2, 0.15, 0.1, 0.25]
    assert validate({'tradeSizeProbabilities': probs, 'meanTradeSize': 123.0}) == probs
