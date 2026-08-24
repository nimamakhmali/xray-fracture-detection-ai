"""
Unit tests for src/data/preprocessor.py (skeleton for Phase 2).
Tests here validate the module interface contract expected by the rest of the system.
"""

import numpy as np
import pytest


class TestPreprocessorInterface:
    """
    These tests define the expected interface of preprocessor.py.
    They will be implemented fully in Phase 2 when preprocessor.py is complete.
    """

    def test_preprocessor_module_importable(self):
        """Preprocessor module must be importable without errors."""
        try:
            from src.data import preprocessor  # noqa: F401
        except ImportError as e:
            pytest.skip(f"Preprocessor not yet implemented: {e}")

    def test_output_is_numpy_array(self):
        """Preprocessor must return numpy arrays."""
        try:
            from src.data.preprocessor import Preprocessor
            p = Preprocessor()
            dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            result = p.preprocess(dummy)
            assert isinstance(result, np.ndarray)
        except (ImportError, AttributeError):
            pytest.skip("Preprocessor not yet implemented")