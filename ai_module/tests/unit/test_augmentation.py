"""
Unit tests for src/data/augmentation.py (skeleton for Phase 3).
"""

import numpy as np
import pytest


class TestAugmentationInterface:
    """
    Skeleton tests — will be expanded in Phase 3.
    """

    def test_augmentation_module_importable(self):
        try:
            from src.data import augmentation  # noqa: F401
        except ImportError:
            pytest.skip("Augmentation not yet implemented")

    def test_augmented_image_same_shape(self):
        try:
            from src.data.augmentation import AugmentationPipeline
            pipeline = AugmentationPipeline(mode="train")
            dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            result = pipeline(dummy)
            assert result.shape == dummy.shape
        except (ImportError, AttributeError):
            pytest.skip("Augmentation not yet implemented")