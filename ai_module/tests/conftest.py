"""
Shared pytest fixtures for all tests.
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_xray_image():
    """
    A synthetic grayscale image simulating an X-ray.
    Shape: (640, 640, 3) — uint8
    """
    image = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_small_image():
    """Small image to test resize logic."""
    return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_yolo_label():
    """
    A valid YOLO format label.
    class x_center y_center width height (all normalized 0-1)
    """
    return [0, 0.5, 0.5, 0.2, 0.15]


@pytest.fixture
def sample_raw_detections():
    """
    Simulated raw YOLO output before postprocessing.
    Each item: [x1, y1, x2, y2, confidence, class_id]
    """
    return [
        [100.0, 150.0, 250.0, 300.0, 0.87, 0],
        [310.0, 200.0, 420.0, 350.0, 0.45, 0],
        [100.5, 151.0, 250.5, 300.5, 0.83, 0],  # near-duplicate for NMS test
    ]


@pytest.fixture
def project_root():
    return Path(__file__).parent.parent


@pytest.fixture
def configs_dir(project_root):
    return project_root / "configs"