"""
Quick dataset structure inspector.
Run this to discover the actual directory structure of both datasets.

Usage:
    python scripts/inspect_datasets.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in list(current.parents)[:6]:
        has_ai = (parent / "ai_module").exists()
        has_data = (
            (parent / "fracatlas").exists() or
            (parent / "FracAtlas").exists() or
            (parent / "GRAZPEDWRI-DX").exists()
        )
        if has_ai and has_data:
            return parent
    return current.parent.parent.parent


def print_tree(root: Path, max_depth: int = 4, max_files: int = 5) -> None:
    """Print directory tree with file counts."""

    def _tree(path: Path, depth: int, prefix: str = "") -> None:
        if depth > max_depth:
            return

        try:
            items = sorted(path.iterdir())
        except PermissionError:
            return

        dirs = [i for i in items if i.is_dir()]
        files = [i for i in items if i.is_file()]

        # Print dirs
        for i, d in enumerate(dirs):
            connector = "└── " if (i == len(dirs) - 1 and not files) else "├── "
            file_count = sum(1 for _ in d.rglob("*") if _.is_file())
            print(f"{prefix}{connector}[DIR]  {d.name}/  ({file_count} files)")
            extension = "    " if (i == len(dirs) - 1 and not files) else "│   "
            _tree(d, depth + 1, prefix + extension)

        # Print sample files
        ext_counter: dict = {}
        for f in files:
            ext = f.suffix.lower()
            ext_counter[ext] = ext_counter.get(ext, 0) + 1

        if files:
            shown = 0
            for f in files[:max_files]:
                connector = "└── " if shown == len(files) - 1 else "├── "
                print(f"{prefix}{connector}[FILE] {f.name}")
                shown += 1
            if len(files) > max_files:
                print(f"{prefix}      ... and {len(files) - max_files} more files")
                print(f"{prefix}      Extensions: {ext_counter}")

    _tree(root, depth=1)


def inspect_dataset(root: Path, name: str) -> dict:
    """Inspect a dataset root and return structure info."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Root   : {root}")
    print(f"  Exists : {root.exists()}")

    if not root.exists():
        print(f"  ⚠️  Directory not found!")
        return {"exists": False}

    print(f"\n  Directory Tree (max depth 4):")
    print_tree(root, max_depth=4)

    # Count files by extension
    ext_counts: dict = {}
    total_files = 0
    for f in root.rglob("*"):
        if f.is_file():
            total_files += 1
            ext = f.suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

    print(f"\n  File Summary:")
    print(f"    Total files : {total_files}")
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {ext or '(no ext)':15s}: {count}")

    # Look for annotation directories
    annotation_dirs = []
    for pattern in ["**/Annotations*", "**/annotations*", "**/labels*",
                    "**/Labels*", "**/yolo*", "**/YOLO*",
                    "**/pascalvoc*", "**/pascal_voc*", "**/PascalVOC*",
                    "**/supervisely*"]:
        found = list(root.glob(pattern))
        annotation_dirs.extend([str(f.relative_to(root)) for f in found if f.is_dir()])

    if annotation_dirs:
        print(f"\n  Annotation-related directories detected:")
        for d in sorted(set(annotation_dirs)):
            print(f"    {d}")

    # Look for CSV files
    csvs = list(root.rglob("*.csv"))
    if csvs:
        print(f"\n  CSV files found:")
        for csv in csvs[:10]:
            print(f"    {csv.relative_to(root)}")

    # Look for XML files (sample)
    xmls = list(root.rglob("*.xml"))
    print(f"\n  XML files : {len(xmls)}")
    if xmls:
        print(f"  Sample XML locations:")
        for xml in xmls[:3]:
            print(f"    {xml.relative_to(root)}")

    # Look for JSON files (sample)
    jsons = list(root.rglob("*.json"))
    print(f"\n  JSON files: {len(jsons)}")
    if jsons:
        print(f"  Sample JSON locations:")
        for j in jsons[:3]:
            print(f"    {j.relative_to(root)}")

    # Look for image files
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    img_files = [f for f in root.rglob("*") if f.suffix.lower() in img_exts]
    print(f"\n  Image files: {len(img_files)}")
    if img_files:
        img_dirs = list({f.parent for f in img_files})
        print(f"  Image directories ({len(img_dirs)} unique):")
        for d in sorted(img_dirs)[:5]:
            count = sum(1 for f in d.iterdir() if f.suffix.lower() in img_exts)
            print(f"    {d.relative_to(root)}/  ({count} images)")

    # Look for .txt files (YOLO labels)
    txts = list(root.rglob("*.txt"))
    print(f"\n  TXT files : {len(txts)}")
    if txts:
        txt_dirs = list({f.parent for f in txts})
        print(f"  TXT directories ({len(txt_dirs)} unique):")
        for d in sorted(txt_dirs)[:5]:
            count = sum(1 for f in d.iterdir() if f.suffix.lower() == ".txt")
            print(f"    {d.relative_to(root)}/  ({count} txt files)")
        print(f"  Sample TXT content (first file):")
        try:
            sample_content = txts[0].read_text(encoding="utf-8")[:200]
            print(f"    File: {txts[0].relative_to(root)}")
            print(f"    Content preview: {repr(sample_content)}")
        except Exception as e:
            print(f"    Could not read: {e}")

    return {
        "exists": True,
        "total_files": total_files,
        "ext_counts": ext_counts,
        "annotation_dirs": annotation_dirs,
        "csv_count": len(csvs),
        "xml_count": len(xmls),
        "json_count": len(jsons),
        "image_count": len(img_files),
        "txt_count": len(txts),
    }


def main():
    project_root = find_project_root()
    print(f"Project root: {project_root}")

    # Try common dataset directory names
    fa_candidates = [
        project_root / "fracatlas",
        project_root / "FracAtlas",
        project_root / "fracAtlas",
        project_root / "FRACATLAS",
    ]
    grz_candidates = [
        project_root / "GRAZPEDWRI-DX",
        project_root / "grazpedwri-dx",
        project_root / "grazpedwri",
        project_root / "GRAZPEDWRI",
    ]

    fa_root = next((c for c in fa_candidates if c.exists()), fa_candidates[0])
    grz_root = next((c for c in grz_candidates if c.exists()), grz_candidates[0])

    fa_info = inspect_dataset(fa_root, "FracAtlas")
    grz_info = inspect_dataset(grz_root, "GRAZPEDWRI-DX")

    print(f"\n{'=' * 60}")
    print("  ACTION REQUIRED")
    print(f"{'=' * 60}")

    if fa_info.get("exists"):
        print("\nFracAtlas paths to update in prepare_dataset.py:")
        print("  Based on inspection above, update these in FracAtlasProcessor:")
        print("  self.yolo_dir = raw_root / '<actual YOLO label directory>'")
        print("  self._images_dir = raw_root / '<actual images directory>'")
    else:
        print(f"\n⚠️  FracAtlas not found at: {fa_root}")

    if grz_info.get("exists"):
        print("\nGRAZPEDWRI-DX paths to update in prepare_dataset.py:")
        print("  Based on inspection above, update these in GRAZPEDWRIProcessor:")
        print("  self.voc_dir = raw_root / '<actual Pascal VOC directory>'")
    else:
        print(f"\n⚠️  GRAZPEDWRI-DX not found at: {grz_root}")


if __name__ == "__main__":
    main()