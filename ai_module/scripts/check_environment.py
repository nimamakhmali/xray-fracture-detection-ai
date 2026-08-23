"""
Environment verification script.
Run this script to verify that all required dependencies are correctly installed.
"""

import sys
import importlib


def check_python_version():
    version = sys.version_info
    status = version.major == 3 and version.minor >= 10
    print(f"  Python {version.major}.{version.minor}.{version.micro} — {'OK' if status else 'FAIL (need 3.10+)'}")
    return status


def check_package(package_name: str, import_name: str = None, min_version: str = None):
    name = import_name or package_name
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"  {package_name} {version} — OK")
        return True
    except ImportError:
        print(f"  {package_name} — FAIL (not installed)")
        return False


def check_cuda():
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                print(f"  GPU {i}: {name} — {memory:.1f} GB VRAM — OK")
        else:
            print("  CUDA — NOT AVAILABLE (CPU only mode)")
        return cuda_available
    except ImportError:
        print("  CUDA check — FAIL (torch not installed)")
        return False


def check_yolo():
    try:
        from ultralytics import YOLO
        import numpy as np

        model = YOLO("yolov8n.pt")
        dummy = np.zeros((640, 640, 3), dtype="uint8")
        results = model.predict(dummy, verbose=False)
        print(f"  YOLOv8 predict — OK (results: {len(results)})")
        return True
    except Exception as e:
        print(f"  YOLOv8 predict — FAIL ({e})")
        return False


def check_dicom():
    try:
        import pydicom
        import SimpleITK
        print(f"  pydicom {pydicom.__version__} — OK")
        print(f"  SimpleITK {SimpleITK.Version.VersionString()} — OK")
        return True
    except ImportError as e:
        print(f"  DICOM libraries — FAIL ({e})")
        return False


def main():
    print("\n" + "=" * 50)
    print("  Environment Check")
    print("=" * 50)

    results = {}

    print("\n[1] Python Version")
    results["python"] = check_python_version()

    print("\n[2] Core Packages")
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("ultralytics", "ultralytics"),
        ("opencv", "cv2"),
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("albumentations", "albumentations"),
        ("scikit-learn", "sklearn"),
        ("mlflow", "mlflow"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("PyYAML", "yaml"),
        ("tqdm", "tqdm"),
    ]
    for pkg, imp in packages:
        results[pkg] = check_package(pkg, imp)

    print("\n[3] CUDA / GPU")
    results["cuda"] = check_cuda()

    print("\n[4] DICOM Support")
    results["dicom"] = check_dicom()

    print("\n[5] YOLO Smoke Test")
    results["yolo"] = check_yolo()

    print("\n" + "=" * 50)
    failed = [k for k, v in results.items() if not v]
    if not failed:
        print("  All checks passed. Environment is ready.")
    else:
        print(f"  Failed checks: {', '.join(failed)}")
        print("  Fix the above issues before proceeding.")
    print("=" * 50 + "\n")

    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)