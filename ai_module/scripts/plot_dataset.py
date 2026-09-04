#!/usr/bin/env python3
"""
scripts/plot_dataset.py
=======================
Generate ALL dataset figures for the Phase 1/2 report — READ-ONLY.

Every number is computed from actual artifacts:
    * data/processed/manifest.csv           (canonical source of truth)
    * data/processed/<split>/labels/*.txt   (YOLO labels, for box-level stats)
and then CROSS-CHECKED (not copied) against:
    * configs/dataset.yaml  -> stats block
    * reports/validation_report.json
    * reports/dataset_preparation_report.json  (optional, for the drop funnel)

Outputs: figNN_*.png/.svg + summary.json + summary.md  (all traceable).

Usage
-----
    python scripts/plot_dataset.py
    python scripts/plot_dataset.py --verify-hashes          # md5 of every image vs manifest (slow, ~24k files)
    python scripts/plot_dataset.py --hash-sync-report data/manifest_hash_sync_20260904T072559Z.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # ai_module/
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("plot_dataset")

SPLITS = ["train", "val", "test"]
SOURCE_KEYS = {"fracatlas": "FracAtlas", "grazpedwri": "GRAZPEDWRI-DX"}   # manifest value -> display
SOURCES = list(SOURCE_KEYS.values())
COLORS = {"FracAtlas": "#1f77b4", "GRAZPEDWRI-DX": "#ff7f0e",
          "positive": "#d62728", "negative": "#2ca02c",
          "train": "#4c72b0", "val": "#dd8452", "test": "#55a868", "neutral": "#7f7f7f"}

plt.rcParams.update({"savefig.bbox": "tight", "font.size": 11, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


# ----------------------------------------------------------------------------- helpers
def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_unavailable(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip().upper() in {"", "UNAVAILABLE", "NAN", "NONE"}


def annotate(ax, total=None, fs=9):
    for p in ax.patches:
        h = p.get_height()
        if not h or np.isnan(h):
            continue
        lbl = f"{int(h):,}" + (f"\n({100*h/total:.1f}%)" if total else "")
        ax.annotate(lbl, (p.get_x() + p.get_width() / 2, h), ha="center", va="bottom",
                    fontsize=fs, xytext=(0, 2), textcoords="offset points")


class FigureWriter:
    def __init__(self, out_dir: Path, dpi: int, formats: list[str]):
        self.out_dir, self.dpi, self.formats, self.saved = out_dir, dpi, formats, []
        out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, fig, name: str):
        for ext in self.formats:
            fig.savefig(self.out_dir / f"{name}.{ext}", dpi=self.dpi if ext == "png" else None)
        plt.close(fig)
        self.saved.append(name)
        log.info("saved %s", name)


# ----------------------------------------------------------------------------- loading
def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"sample_id", "dataset", "image_path", "image_hash", "patient_id", "fracture_positive",
                "num_boxes", "annotation_status", "width", "height", "split"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"manifest missing columns: {sorted(missing)}")
    unknown_src = set(df["dataset"]) - set(SOURCE_KEYS)
    if unknown_src:
        log.warning("unknown dataset values in manifest: %s", unknown_src)
    df["source"] = df["dataset"].map(lambda s: SOURCE_KEYS.get(s, s))
    df["positive"] = df["fracture_positive"].str.strip().str.lower().eq("true")
    df["num_boxes"] = pd.to_numeric(df["num_boxes"], errors="coerce").fillna(0).astype(int)
    for c in ("width", "height"):
        df[c] = pd.to_numeric(df[c].where(~df[c].map(is_unavailable)), errors="coerce")
    unknown_split = set(df["split"]) - set(SPLITS)
    if unknown_split:
        log.warning("unknown split values: %s", unknown_split)
    return df


def load_yaml(p: Path) -> dict:
    if not p.exists():
        log.warning("not found: %s", p)
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_json(p: Path | None) -> dict:
    if p is None or not p.exists():
        if p is not None:
            log.warning("not found: %s", p)
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_boxes(df: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, dict]:
    """Parse <root>/<split>/labels/<stem>.txt for every manifest row."""
    rows, missing, malformed, class_hist, per_image = [], 0, 0, {}, {}
    for r in df.itertuples(index=True):
        lp = root / r.split / "labels" / (Path(r.image_path).stem + ".txt")
        if not lp.exists():
            missing += 1
            continue
        n = 0
        for line in lp.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                malformed += 1
                continue
            try:
                cls = int(float(parts[0])); cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                malformed += 1
                continue
            class_hist[cls] = class_hist.get(cls, 0) + 1
            rows.append((r.sample_id, r.source, r.split, cls, cx, cy, w, h))
            n += 1
        per_image[r.Index] = n
    boxes = pd.DataFrame(rows, columns=["sample_id", "source", "split", "cls", "cx", "cy", "w", "h"])
    if len(boxes):
        boxes["area"] = boxes["w"] * boxes["h"]
        boxes["aspect"] = boxes["w"] / boxes["h"].replace(0, np.nan)
    counted = pd.Series(per_image, dtype=int)
    mismatch_idx = counted.index[df.loc[counted.index, "num_boxes"].values != counted.values]
    stats = {
        "skipped": False,
        "label_files_found": int(len(counted)), "label_files_missing": int(missing),
        "malformed_lines": int(malformed), "total_boxes_parsed": int(len(boxes)),
        "boxes_per_split": {sp: int((boxes["split"] == sp).sum()) for sp in SPLITS},
        "boxes_per_source": {s: int((boxes["source"] == s).sum()) for s in SOURCES},
        "class_id_histogram": {str(k): int(v) for k, v in sorted(class_hist.items())},
        "num_boxes_mismatch_vs_manifest": int(len(mismatch_idx)),
        "num_boxes_mismatch_sample_ids": df.loc[mismatch_idx, "sample_id"].tolist()[:50],
    }
    if stats["num_boxes_mismatch_vs_manifest"]:
        log.warning("%d rows: manifest.num_boxes != parsed labels (investigate, do NOT edit)", len(mismatch_idx))
    if set(class_hist) - {0}:
        log.warning("non-zero class ids present in labels: %s", sorted(class_hist))
    return boxes, stats


def verify_hashes(df: pd.DataFrame, root: Path) -> dict:
    """Recompute md5 of every processed image and compare to manifest.image_hash."""
    mism, missing = [], []
    for r in df.itertuples():
        p = root / r.split / "images" / r.image_path
        if not p.exists():
            missing.append(r.image_path); continue
        if md5_of(p) != r.image_hash:
            mism.append({"path": f"{r.split}/images/{r.image_path}", "source": r.source})
    out = {"checked": int(len(df)), "missing_files": len(missing), "hash_mismatches": len(mism),
           "mismatch_list": mism[:100], "missing_list": missing[:100]}
    if mism or missing:
        log.warning("hash verification: %d mismatches, %d missing", len(mism), len(missing))
    else:
        log.info("hash verification: all %d images match manifest", len(df))
    return out


# ----------------------------------------------------------------------------- figures
def fig01_split(df, fw):
    c = df["split"].value_counts().reindex(SPLITS).fillna(0).astype(int); t = int(c.sum())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(c.index, c.values, color=[COLORS[s] for s in c.index]); annotate(ax, total=t)
    ax.set_ylim(0, c.max() * 1.18); ax.set_ylabel("images")
    ax.set_title(f"Fig 1 – Images per split (total = {t:,}, seed = 42)")
    fw.save(fig, "fig01_split_counts")
    return {"total": t, **{k: int(v) for k, v in c.items()}, "ratio": {k: round(v / t, 4) for k, v in c.items()}}


def fig02_balance(df, fw):
    pos, neg = int(df["positive"].sum()), int((~df["positive"]).sum())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].bar(["fracture-positive", "fracture-negative"], [pos, neg], color=[COLORS["positive"], COLORS["negative"]])
    annotate(ax[0], total=pos + neg); ax[0].set_ylim(0, max(pos, neg) * 1.18); ax[0].set_ylabel("images")
    ax[0].set_title("Image-level class balance")
    ax[1].pie([pos, neg], labels=["positive", "negative"], autopct="%1.1f%%", startangle=90,
              colors=[COLORS["positive"], COLORS["negative"]], wedgeprops={"edgecolor": "white"})
    ax[1].set_title("Share")
    fig.suptitle("Fig 2 – Fracture-positive vs negative images", fontweight="bold")
    fw.save(fig, "fig02_label_balance")
    return {"positive": pos, "negative": neg, "positive_ratio": round(pos / (pos + neg), 4)}


def fig03_sources(df, fw):
    c = df["source"].value_counts().reindex(SOURCES).dropna().astype(int); t = int(c.sum())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(c.index, c.values, color=[COLORS[s] for s in c.index]); annotate(ax, total=t)
    ax.set_ylim(0, c.max() * 1.18); ax.set_ylabel("images"); ax.set_title("Fig 3 – Images per source dataset")
    fw.save(fig, "fig03_source_counts")
    return {k: int(v) for k, v in c.items()}


def fig04_source_split(df, fw):
    ct = pd.crosstab(df["split"], df["source"]).reindex(index=SPLITS, columns=SOURCES).fillna(0).astype(int)
    ratio = pd.crosstab(df["source"], df["split"], normalize="index").reindex(index=SOURCES, columns=SPLITS).fillna(0)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    bottom = np.zeros(len(SPLITS))
    for src in SOURCES:
        ax[0].bar(SPLITS, ct[src].values, bottom=bottom, label=src, color=COLORS[src])
        for i, v in enumerate(ct[src].values):
            if v:
                ax[0].text(i, bottom[i] + v / 2, f"{v:,}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        bottom += ct[src].values
    for i, tt in enumerate(bottom):
        ax[0].text(i, tt, f"{int(tt):,}", ha="center", va="bottom", fontsize=9)
    ax[0].set_title("Source composition per split"); ax[0].set_ylabel("images"); ax[0].legend()
    left = np.zeros(len(SOURCES))
    for sp in SPLITS:
        vals = ratio[sp].values * 100
        ax[1].barh(SOURCES, vals, left=left, label=sp, color=COLORS[sp])
        for i, v in enumerate(vals):
            if v > 3:
                ax[1].text(left[i] + v / 2, i, f"{v:.1f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        left += vals
    ax[1].set_xlim(0, 100); ax[1].set_xlabel("% of source images"); ax[1].legend(loc="lower right")
    ax[1].set_title("Split ratio within each source (target 70/15/15)")
    fig.suptitle("Fig 4 – Source × split (split performed independently per source)", fontweight="bold")
    fw.save(fig, "fig04_source_by_split")
    return {"counts": {sp: {s: int(ct.loc[sp, s]) for s in SOURCES} for sp in SPLITS},
            "ratio_within_source": {s: {sp: round(float(ratio.loc[s, sp]), 4) for sp in SPLITS} for s in SOURCES}}


def fig05_label_split(df, fw):
    ct = pd.crosstab(df["split"], df["positive"]).reindex(index=SPLITS).fillna(0).astype(int)
    pos, neg = ct.get(True, 0), ct.get(False, 0); tot = pos + neg
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8)); x = np.arange(3)
    ax[0].bar(x - .2, pos.values, .4, label="positive", color=COLORS["positive"])
    ax[0].bar(x + .2, neg.values, .4, label="negative", color=COLORS["negative"])
    ax[0].set_xticks(x, SPLITS); annotate(ax[0]); ax[0].legend(); ax[0].set_ylabel("images")
    ax[0].set_ylim(0, max(pos.max(), neg.max()) * 1.18); ax[0].set_title("Positive / negative per split")
    pct = (pos / tot * 100).values
    ax[1].bar(SPLITS, pct, color=[COLORS[s] for s in SPLITS])
    for i, v in enumerate(pct):
        ax[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    ax[1].axhline(df["positive"].mean() * 100, ls="--", color="k", lw=1, label=f"overall {df['positive'].mean()*100:.1f}%")
    ax[1].set_ylim(0, 100); ax[1].set_ylabel("% positive"); ax[1].legend(); ax[1].set_title("Positive rate per split (stratification check)")
    fig.suptitle("Fig 5 – Label balance across splits", fontweight="bold")
    fw.save(fig, "fig05_label_by_split")
    return {sp: {"positive": int(pos[sp]), "negative": int(neg[sp]), "positive_rate": round(float(pos[sp] / tot[sp]), 4)} for sp in SPLITS}


def fig06_label_source(df, fw):
    ct = pd.crosstab(df["source"], df["positive"]).reindex(index=SOURCES).fillna(0).astype(int)
    pos, neg = ct.get(True, 0), ct.get(False, 0)
    fig, ax = plt.subplots(figsize=(8, 4.8)); x = np.arange(len(SOURCES))
    ax.bar(x - .2, pos.values, .4, label="positive", color=COLORS["positive"])
    ax.bar(x + .2, neg.values, .4, label="negative", color=COLORS["negative"])
    ax.set_xticks(x, SOURCES); annotate(ax); ax.legend(); ax.set_ylabel("images")
    for i, s in enumerate(SOURCES):
        ax.text(i, max(pos[s], neg[s]) * 1.10, f"positive rate {100*pos[s]/(pos[s]+neg[s]):.1f}%", ha="center", fontsize=9, style="italic")
    ax.set_ylim(0, max(pos.max(), neg.max()) * 1.25); ax.set_title("Fig 6 – Class balance per source dataset")
    fw.save(fig, "fig06_label_by_source")
    return {s: {"positive": int(pos[s]), "negative": int(neg[s]), "positive_rate": round(float(pos[s] / (pos[s] + neg[s])), 4)} for s in SOURCES}


def fig07_grid(df, fw):
    g = df.groupby(["source", "split", "positive"]).size().unstack(fill_value=0)
    g = g.reindex(pd.MultiIndex.from_product([SOURCES, SPLITS]), fill_value=0)
    tbl = pd.DataFrame({"positive": g.get(True, 0), "negative": g.get(False, 0)}); tbl["total"] = tbl.sum(axis=1)
    fig, ax = plt.subplots(figsize=(9, 4.2)); ax.axis("off")
    t = ax.table(cellText=[[f"{int(v):,}" for v in row] for row in tbl.values],
                 rowLabels=[f"{s} / {sp}" for s, sp in tbl.index], colLabels=list(tbl.columns), loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.5)
    ax.set_title("Fig 7 – Source × split × label", pad=12)
    fw.save(fig, "fig07_source_split_label_table")
    return {f"{s}/{sp}": {c: int(tbl.loc[(s, sp), c]) for c in tbl.columns} for s, sp in tbl.index}


def fig08_boxes_per_image(df, fw):
    pos = df[df["positive"]]; mx = int(pos["num_boxes"].max()); bins = np.arange(.5, mx + 1.5)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for s in SOURCES:
        v = pos.loc[pos["source"] == s, "num_boxes"]
        ax.hist(v, bins=bins, alpha=.65, color=COLORS[s], label=f"{s} (n={len(v):,}, mean={v.mean():.2f})")
    ax.set_xticks(range(1, mx + 1)); ax.set_xlabel("boxes per positive image"); ax.set_ylabel("images"); ax.legend()
    ax.set_title("Fig 8 – Fracture boxes per positive image")
    fw.save(fig, "fig08_boxes_per_image")
    d = pos.groupby("source")["num_boxes"].describe()
    return {"total_boxes_manifest": int(df["num_boxes"].sum()),
            "positive_images_with_zero_boxes": int((df["positive"] & (df["num_boxes"] == 0)).sum()),
            "negative_images_with_boxes": int((~df["positive"] & (df["num_boxes"] > 0)).sum()),
            "per_source": {s: {k: round(float(v), 3) for k, v in d.loc[s].items()} for s in d.index}}


def fig09_box_area(boxes, fw):
    if boxes.empty: return {}
    fig, ax = plt.subplots(figsize=(9, 4.8)); bins = np.logspace(np.log10(max(boxes["area"].min(), 1e-6)), 0, 50)
    for s in SOURCES:
        v = boxes.loc[boxes["source"] == s, "area"]
        ax.hist(v, bins=bins, alpha=.65, color=COLORS[s], label=f"{s} (n={len(v):,}, median={v.median():.4f})")
    ax.set_xscale("log"); ax.set_xlabel("normalized box area (w×h)"); ax.set_ylabel("boxes"); ax.legend()
    ax.set_title("Fig 9 – Box area distribution (log) — small-object burden")
    fw.save(fig, "fig09_box_area_distribution")
    return {s: {"n": int(len(v)), "median_area": round(float(v.median()), 5), "mean_area": round(float(v.mean()), 5),
                "share_below_1pct": round(float((v < .01).mean()), 4), "share_below_0_1pct": round(float((v < .001).mean()), 4)}
            for s in SOURCES for v in [boxes.loc[boxes["source"] == s, "area"]] if len(v)}


def fig10_box_shape(boxes, fw):
    if boxes.empty: return {}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for s in SOURCES:
        b = boxes[boxes["source"] == s]
        ax[0].scatter(b["w"], b["h"], s=4, alpha=.25, color=COLORS[s], label=f"{s} (n={len(b):,})")
        ax[1].hist(b["aspect"].dropna(), bins=np.logspace(-1.5, 1.5, 50), alpha=.65, color=COLORS[s], label=s)
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("normalized width"); ax[0].set_ylabel("normalized height"); ax[0].set_title("w vs h"); ax[0].legend(markerscale=4)
    ax[1].set_xscale("log"); ax[1].axvline(1, color="k", ls="--", lw=.8); ax[1].set_xlabel("aspect ratio w/h (log)"); ax[1].set_ylabel("boxes")
    ax[1].set_title("Aspect ratio"); ax[1].legend()
    fig.suptitle("Fig 10 – Bounding-box geometry", fontweight="bold")
    fw.save(fig, "fig10_box_shape")
    return {s: {"median_w": round(float(b["w"].median()), 4), "median_h": round(float(b["h"].median()), 4),
                "median_aspect": round(float(b["aspect"].median()), 4)} for s in SOURCES for b in [boxes[boxes["source"] == s]] if len(b)}


def fig11_centers(boxes, fw):
    if boxes.empty: return {}
    panels = [("Combined", boxes)] + [(s, boxes[boxes["source"] == s]) for s in SOURCES]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5))
    for ax, (name, b) in zip(np.atleast_1d(axes), panels):
        hm, _, _ = np.histogram2d(b["cy"], b["cx"], bins=40, range=[[0, 1], [0, 1]])
        im = ax.imshow(hm, origin="upper", cmap="magma", extent=[0, 1, 1, 0]); ax.grid(False)
        ax.set_title(f"{name} (n={len(b):,})"); ax.set_xlabel("cx"); ax.set_ylabel("cy"); fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.suptitle("Fig 11 – Spatial distribution of box centers", fontweight="bold")
    fw.save(fig, "fig11_box_center_heatmap")
    return {n: {"mean_cx": round(float(b["cx"].mean()), 4), "mean_cy": round(float(b["cy"].mean()), 4)} for n, b in panels}


def fig12_resolution(df, fw):
    d = df.dropna(subset=["width", "height"])
    if d.empty: return {}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for s in SOURCES:
        x = d[d["source"] == s]
        ax[0].scatter(x["width"], x["height"], s=5, alpha=.3, color=COLORS[s], label=f"{s} (n={len(x):,})")
        ax[1].hist(x["width"] / x["height"], bins=40, alpha=.65, color=COLORS[s], label=s)
    ax[0].axvline(640, color="k", ls=":", lw=.8); ax[0].axhline(640, color="k", ls=":", lw=.8)
    ax[0].set_xlabel("width (px)"); ax[0].set_ylabel("height (px)"); ax[0].set_title("Resolution"); ax[0].legend(markerscale=4)
    ax[1].set_xlabel("image aspect ratio w/h"); ax[1].set_ylabel("images"); ax[1].set_title("Image aspect ratio"); ax[1].legend()
    fig.suptitle("Fig 12 – Image size statistics (model input 640, letterboxed)", fontweight="bold")
    fw.save(fig, "fig12_image_resolution")
    out = {}
    for s in SOURCES:
        x = d[d["source"] == s]
        res = x.groupby(["width", "height"]).size().sort_values(ascending=False)
        out[s] = {"n": int(len(x)), "unique_resolutions": int(len(res)),
                  "top5_resolutions": {f"{int(w)}x{int(h)}": int(c) for (w, h), c in res.head(5).items()},
                  "width_median": float(x["width"].median()), "height_median": float(x["height"].median())}
    return out


def fig13_annotation_status(df, fw):
    ct = pd.crosstab(df["source"], df["annotation_status"]).reindex(index=SOURCES).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(9, 4.8)); x = np.arange(len(SOURCES)); w = .8 / max(len(ct.columns), 1)
    for j, col in enumerate(ct.columns):
        ax.bar(x + j * w - .4 + w / 2, ct[col].values, w, label=col)
    ax.set_xticks(x, SOURCES); annotate(ax); ax.set_yscale("log"); ax.set_ylabel("images (log)"); ax.legend(title="annotation_status")
    ax.set_title("Fig 13 – annotation_status per source")
    fw.save(fig, "fig13_annotation_status")
    return {s: {c: int(ct.loc[s, c]) for c in ct.columns} for s in SOURCES}


def fig14_patient(df, fw):
    d = df.copy(); d["pid_ok"] = ~d["patient_id"].map(is_unavailable)
    ct = pd.crosstab(d["source"], d["pid_ok"]).reindex(index=SOURCES).fillna(0).astype(int)
    av, un = ct.get(True, 0), ct.get(False, 0)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8)); x = np.arange(len(SOURCES))
    ax[0].bar(x - .2, av.values, .4, label="patient_id available", color="#4c72b0")
    ax[0].bar(x + .2, un.values, .4, label="UNAVAILABLE", color="#c44e52")
    ax[0].set_xticks(x, SOURCES); annotate(ax[0]); ax[0].legend(); ax[0].set_ylabel("images")
    ax[0].set_ylim(0, max(av.max(), un.max()) * 1.18); ax[0].set_title("patient_id coverage")
    out = {s: {"available": int(av[s]), "unavailable": int(un[s]), "coverage": round(float(av[s] / (av[s] + un[s])), 4)} for s in SOURCES}
    a = d[d["pid_ok"]]
    for s in SOURCES:
        g = a[a["source"] == s]
        if g.empty: continue
        per = g.groupby("patient_id").size(); nsplit = g.groupby("patient_id")["split"].nunique()
        ax[1].hist(per.clip(upper=30), bins=np.arange(.5, 31.5), alpha=.65, color=COLORS[s], label=f"{s} ({len(per):,} patients)")
        out[s].update({"unique_patients": int(len(per)), "images_per_patient_mean": round(float(per.mean()), 3),
                       "images_per_patient_max": int(per.max()), "patients_in_more_than_one_split": int((nsplit > 1).sum())})
    ax[1].set_xlabel("images per patient (clipped at 30)"); ax[1].set_ylabel("patients"); ax[1].legend(); ax[1].set_title("Images per patient")
    fig.suptitle("Fig 14 – Patient grouping availability (leakage-control basis)", fontweight="bold")
    fw.save(fig, "fig14_patient_id_coverage")
    # independent hash-overlap recomputation across splits
    hs = {sp: set(df.loc[(df["split"] == sp) & ~df["image_hash"].map(is_unavailable), "image_hash"]) for sp in SPLITS}
    out["hash_overlap_recomputed"] = {f"{a_}_{b_}": len(hs[a_] & hs[b_]) for a_, b_ in [("train", "val"), ("train", "test"), ("val", "test")]}
    out["duplicate_hashes_within_manifest"] = int(df["image_hash"].duplicated().sum())
    return out


def fig15_class_ids(stats, fw):
    hist = stats.get("class_id_histogram")
    if not hist: return {}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"class {k}" for k in hist], list(hist.values()), color=COLORS["positive"]); annotate(ax)
    ax.set_ylabel("boxes"); ax.set_title("Fig 15 – Class ids in processed labels (expected: only 0)")
    fw.save(fig, "fig15_class_ids_in_labels")
    return hist


def fig16_funnel(prep: dict, df: pd.DataFrame, fw):
    """Per-source drop accounting from dataset_preparation_report.json: raw -> orphan -> dedup -> integrity -> manifest."""
    if not prep: return {}
    dup_by_src, drop_by_src = {}, {}
    for dd in prep.get("deduplication", {}).get("duplicate_list", []):
        dup_by_src[dd["source"]] = dup_by_src.get(dd["source"], 0) + 1
    for dd in prep.get("dropped_samples", []):
        reason = dd["reason"].split(":")[0]
        drop_by_src.setdefault(dd["source"], {}).setdefault(reason, 0)
        drop_by_src[dd["source"]][reason] += 1
    out, fig = {}, None
    fig, axes = plt.subplots(1, len(SOURCE_KEYS), figsize=(6.5 * len(SOURCE_KEYS), 4.8))
    for ax, (key, name) in zip(np.atleast_1d(axes), SOURCE_KEYS.items()):
        src = prep.get("sources", {}).get(key, {})
        raw, valid = int(src.get("total", 0)), int(src.get("valid", 0))
        dups = dup_by_src.get(key, 0); drops = drop_by_src.get(key, {})
        n_int = sum(v for k, v in drops.items() if k.startswith("image_integrity")); n_other = sum(drops.values()) - n_int
        final = int((df["dataset"] == key).sum())
        stages = [("raw annotations", raw), ("with matching image", valid), ("after dedup", valid - dups),
                  ("after integrity check", valid - dups - n_int - n_other), ("in manifest", final)]
        expected_final = valid - dups - n_int - n_other
        ax.barh([s for s, _ in stages][::-1], [v for _, v in stages][::-1], color=COLORS[name])
        for i, (s, v) in enumerate(stages[::-1]):
            ax.text(v, i, f" {v:,}", va="center", fontsize=9)
        notes = f"orphan −{raw-valid} | dup −{dups} | integrity −{n_int} | other −{n_other}"
        ax.set_title(f"{name}\n{notes}", fontsize=10); ax.set_xlim(0, raw * 1.15 if raw else 1)
        ok = expected_final == final
        if not ok:
            log.warning("FUNNEL MISMATCH %s: report arithmetic gives %d, manifest has %d", name, expected_final, final)
        out[name] = {"raw": raw, "valid": valid, "orphan": raw - valid, "duplicates": dups, "integrity_failed": n_int,
                     "other_drops": n_other, "expected_final": expected_final, "manifest_final": final, "reconciled": ok}
    fig.suptitle("Fig 16 – Exclusion accounting (from dataset_preparation_report.json)", fontweight="bold")
    fw.save(fig, "fig16_exclusion_funnel")
    return out


def fig17_summary(summary, fw):
    s1, s2, s3, s6 = summary["fig01"], summary["fig02"], summary["fig03"], summary["fig06"]
    rows = [["Total images", f"{s1['total']:,}"],
            ["Train / Val / Test", f"{s1['train']:,} / {s1['val']:,} / {s1['test']:,}"],
            ["Split ratio", " / ".join(f"{100*s1['ratio'][k]:.1f}%" for k in SPLITS)],
            ["Positive / Negative", f"{s2['positive']:,} / {s2['negative']:,}"]]
    rows += [[f"{s} (pos / neg)", f"{s3[s]:,}  ({s6[s]['positive']:,} / {s6[s]['negative']:,})"] for s in SOURCES if s in s3]
    rows.append(["Boxes (manifest / parsed labels)", f"{summary['fig08']['total_boxes_manifest']:,} / {summary['labels'].get('total_boxes_parsed', '—')}"])
    rows.append(["Random seed", "42"])
    fig, ax = plt.subplots(figsize=(9, .45 * len(rows) + 1)); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=["Item", "Value"], loc="center", cellLoc="left", colWidths=[.45, .55])
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.5)
    ax.set_title("Fig 17 – Dataset summary (computed from manifest / labels)", pad=10)
    fw.save(fig, "fig17_summary_table")
    return {r[0]: r[1] for r in rows}


# ----------------------------------------------------------------------------- cross-checks
def cross_check(summary: dict, vr: dict, ds_yaml: dict) -> dict:
    s1, s2, lb = summary["fig01"], summary["fig02"], summary["labels"]
    st = ds_yaml.get("stats", {})
    checks = {
        "validation_report": {
            "total_images": (vr.get("total_images"), s1["total"]),
            "positive_images": (vr.get("positive_images"), s2["positive"]),
            "negative_images": (vr.get("negative_images"), s2["negative"]),
            "total_boxes": (vr.get("total_boxes"), lb.get("total_boxes_parsed")),
            **{f"{sp}_images": (vr.get("split_stats", {}).get(sp, {}).get("total_images"), s1[sp]) for sp in SPLITS},
            **{f"{sp}_positive": (vr.get("split_stats", {}).get(sp, {}).get("positive_images"), summary["fig05"][sp]["positive"]) for sp in SPLITS},
            **{f"{sp}_boxes": (vr.get("split_stats", {}).get(sp, {}).get("total_boxes"), lb.get("boxes_per_split", {}).get(sp)) for sp in SPLITS},
        },
        "dataset_yaml_stats": {
            "total_images": (st.get("total_images"), s1["total"]),
            **{f"{sp}_images": (st.get(f"{sp}_images"), s1[sp]) for sp in SPLITS},
            "fracture_positive": (st.get("fracture_positive"), s2["positive"]),
            "fracture_negative": (st.get("fracture_negative"), s2["negative"]),
            **{f"by_dataset_{k}": (st.get("by_dataset", {}).get(k), summary["fig03"].get(v)) for k, v in SOURCE_KEYS.items()},
        },
    }
    out, all_ok = {}, True
    for grp, items in checks.items():
        out[grp] = {}
        for k, (ref, comp) in items.items():
            match = None if ref is None or comp is None else (int(ref) == int(comp))
            out[grp][k] = {"reference": ref, "computed_here": comp, "match": match}
            if match is False:
                all_ok = False; log.warning("MISMATCH [%s] %s: reference=%s computed=%s", grp, k, ref, comp)
    out["all_matched"] = all_ok
    out["validator_status"] = vr.get("status")
    out["validator_patient_leakage"] = vr.get("patient_leakage")
    out["validator_hash_leakage"] = vr.get("hash_leakage")
    return out


def write_md(s: dict, path: Path):
    L = ["# Dataset figures — computed summary", "",
         f"- generated: {s['meta']['generated_at']}", f"- manifest: `{s['meta']['manifest_path']}`",
         f"- manifest sha256: `{s['meta']['manifest_sha256']}`", f"- rows: {s['meta']['manifest_rows']:,}", ""]
    L += ["## Source × split × label", "", "| source / split | positive | negative | total |", "|---|---:|---:|---:|"]
    L += [f"| {k} | {v['positive']:,} | {v['negative']:,} | {v['total']:,} |" for k, v in s["fig07"].items()]
    lb = s["labels"]
    if not lb.get("skipped"):
        L += ["", "## Labels (parsed from YOLO .txt)", "", f"- label files found / missing: {lb['label_files_found']:,} / {lb['label_files_missing']}",
              f"- malformed lines: {lb['malformed_lines']}", f"- boxes parsed: {lb['total_boxes_parsed']:,}  per split: {lb['boxes_per_split']}",
              f"- class ids: {lb['class_id_histogram']}", f"- rows with manifest.num_boxes ≠ parsed: {lb['num_boxes_mismatch_vs_manifest']}"]
    if s.get("fig14"):
        L += ["", "## patient_id / leakage (recomputed from manifest)", ""]
        for src in SOURCES:
            v = s["fig14"][src]
            L.append(f"- {src}: coverage {100*v['coverage']:.1f}%, unique patients {v.get('unique_patients','—')}, "
                     f"patients in >1 split: {v.get('patients_in_more_than_one_split','—')}")
        L.append(f"- cross-split hash overlap: {s['fig14']['hash_overlap_recomputed']}; duplicate hashes in manifest: {s['fig14']['duplicate_hashes_within_manifest']}")
    if s.get("fig16"):
        f16 = s["fig16"]
        L += ["", "## Exclusion accounting", "",
              "| source | raw | orphan | dup | dup(+) | integrity | other | expected | manifest | reconciled | positives lost |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|"]
        L += [f"| {k} | {v['raw']} | {v['orphan']} | {v['duplicates']} | {v['duplicates_positive']} | {v['integrity_failed']} | "
              f"{v['other_drops']} | {v['expected_final']} | {v['manifest_final']} | {v['reconciled']} | {v['positives_lost_to_integrity']} |"
              for k, v in f16.items() if k in SOURCES]
        L.append(f"- duplicate label conflicts: {f16.get('duplicate_label_conflicts')}")
        if f16.get("split_pre_vs_post_drop"):
            L += ["", "| split | pre-drop | post-drop | Δ | pos Δ |", "|---|---:|---:|---:|---:|"]
            L += [f"| {r['split']} | {r['pre_count']} | {r['post_count']} | {r['delta']:+d} | {r['post_positive'] - r['pre_positive']:+d} |"
                  for r in f16["split_pre_vs_post_drop"]]
            L.append(f"- total Δ = {f16['split_delta_total']:+d}")
    if s.get("fig18"):
        L += ["", "## FracAtlas integrity locality", "",
              f"- dropped index range: {s['fig18']['dropped_range']}, re-synced range: {s['fig18']['synced_range']}"]
    if s.get("synced_dimension_check"):
        sd = s["synced_dimension_check"]
        L += ["", "## Re-synced images — dimension check", "",
              f"- checked {sd['checked']}: width/height mismatch vs manifest = {sd['dimension_mismatch']}"]
        
    if s.get("hash_verification"):
        hv = s["hash_verification"]
        L += ["", "## Image hash verification (md5 on disk vs manifest)", "",
              f"- checked {hv['checked']:,}: mismatches {hv['hash_mismatches']}, missing {hv['missing_files']}"]
    if s.get("hash_sync_events"):
        hs = s["hash_sync_events"]
        L += ["", "## Provenance note — manifest hash sync", "",
              f"- {hs['updated_rows']} image hashes were re-synced on {hs['synced_at']} (files changed on disk after manifest creation).",
              f"- sources: {hs['by_source']}", "- These files must be listed in the report as *modified after preparation* (e.g. Ultralytics in-place JPEG repair)."]
    L += ["", "## Figures", ""] + [f"- `{n}`" for n in s["meta"]["figures"]]
    path.write_text("\n".join(L), encoding="utf-8")





def fig16_funnel(prep: dict, df: pd.DataFrame, fw):
    """Exclusion accounting per source + positive accounting + pre/post-drop split reconciliation."""
    if not prep:
        return {}
    dup_by_src, dup_pos_by_src, drop_by_src = {}, {}, {}
    for d in prep.get("deduplication", {}).get("duplicate_list", []):
        dup_by_src[d["source"]] = dup_by_src.get(d["source"], 0) + 1
        if d["was_positive"]:
            dup_pos_by_src[d["source"]] = dup_pos_by_src.get(d["source"], 0) + 1
    label_conflicts = sum(1 for d in prep.get("deduplication", {}).get("duplicate_list", [])
                          if d["was_positive"] != d["kept_sample_was_positive"])
    for d in prep.get("dropped_samples", []):
        r = d["reason"].split(":")[0]
        drop_by_src.setdefault(d["source"], {}).setdefault(r, 0)
        drop_by_src[d["source"]][r] += 1

    out = {"duplicate_label_conflicts": label_conflicts}
    fig, axes = plt.subplots(1, len(SOURCE_KEYS), figsize=(6.5 * len(SOURCE_KEYS), 4.8))
    for ax, (key, name) in zip(np.atleast_1d(axes), SOURCE_KEYS.items()):
        src = prep.get("sources", {}).get(key, {})
        raw, valid, raw_pos = int(src.get("total", 0)), int(src.get("valid", 0)), int(src.get("positive", 0))
        dups, dup_pos = dup_by_src.get(key, 0), dup_pos_by_src.get(key, 0)
        drops = drop_by_src.get(key, {})
        n_int = sum(v for k, v in drops.items() if k.startswith("image_integrity"))
        n_other = sum(drops.values()) - n_int
        sub = df[df["dataset"] == key]
        final, final_pos = int(len(sub)), int(sub["positive"].sum())
        expected, expected_pos_ub = valid - dups - n_int - n_other, raw_pos - dup_pos
        stages = [("raw annotations", raw), ("with matching image", valid), ("after dedup", valid - dups),
                  ("after integrity check", expected), ("in manifest", final)]
        ax.barh([s for s, _ in stages][::-1], [v for _, v in stages][::-1], color=COLORS[name])
        for i, (_, v) in enumerate(stages[::-1]):
            ax.text(v, i, f" {v:,}", va="center", fontsize=9)
        ax.set_xlim(0, (raw or 1) * 1.15)
        ax.set_title(f"{name}\norphan −{raw-valid} | dup −{dups} | integrity −{n_int} | other −{n_other}", fontsize=10)
        ok = expected == final
        if not ok:
            log.warning("FUNNEL MISMATCH %s: report gives %d, manifest has %d", name, expected, final)
        out[name] = {"raw": raw, "valid": valid, "orphan": raw - valid, "duplicates": dups,
                     "duplicates_positive": dup_pos, "integrity_failed": n_int, "other_drops": n_other,
                     "expected_final": expected, "manifest_final": final, "reconciled": ok,
                     "raw_positive": raw_pos, "manifest_positive": final_pos,
                     "positives_lost_to_integrity": expected_pos_ub - final_pos}
    fig.suptitle("Fig 16 – Exclusion accounting (dataset_preparation_report.json vs manifest)", fontweight="bold")
    fw.save(fig, "fig16_exclusion_funnel")

    # pre-drop (report.splitting) vs post-drop (manifest) per split
    pre = prep.get("splitting", {})
    if pre:
        rows = []
        for sp in SPLITS:
            m = df[df["split"] == sp]
            p = pre.get(sp, {})
            rows.append({"split": sp, "pre_count": p.get("count"), "post_count": int(len(m)),
                         "delta": int(len(m)) - int(p.get("count", 0)),
                         "pre_positive": p.get("positive"), "post_positive": int(m["positive"].sum()),
                         "pre_negative": p.get("negative"), "post_negative": int((~m["positive"]).sum())})
        out["split_pre_vs_post_drop"] = rows
        out["split_delta_total"] = sum(r["delta"] for r in rows)
        fig, ax = plt.subplots(figsize=(9, 4.5)); x = np.arange(3)
        ax.bar(x - .2, [r["pre_count"] for r in rows], .4, label="pre-drop (report.splitting)", color=COLORS["neutral"])
        ax.bar(x + .2, [r["post_count"] for r in rows], .4, label="post-drop (manifest)", color=COLORS["train"])
        for i, r in enumerate(rows):
            ax.text(i, r["pre_count"], f"Δ = {r['delta']:+d}\n(pos Δ {r['post_positive'] - r['pre_positive']:+d})",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x, SPLITS); ax.set_ylabel("images"); ax.legend()
        ax.set_ylim(0, max(r["pre_count"] for r in rows) * 1.2)
        ax.set_title(f"Fig 16b – Split sizes before vs after integrity drop (total Δ = {out['split_delta_total']:+d})")
        fw.save(fig, "fig16b_split_pre_post_drop")
    return out


def fig18_integrity_locality(prep: dict, hash_sync: dict, df: pd.DataFrame, fw):
    """Where in FracAtlas do corrupt / repaired files sit? (index of IMGxxxxxxx)."""
    def idx(stem):  # IMG0004073 -> 4073
        digits = "".join(ch for ch in Path(stem).stem.replace("fa_", "") if ch.isdigit())
        return int(digits) if digits else None
    kept = [i for i in df.loc[df["dataset"] == "fracatlas", "original_filename"].map(idx) if i is not None]
    dropped = [i for i in (idx(d["stem"]) for d in prep.get("dropped_samples", []) if d["source"] == "fracatlas") if i is not None]
    synced = [i for i in (idx(u["path"]) for u in hash_sync.get("updated", []) if "fa_" in u["path"]) if i is not None]
    if not kept:
        return {}
    fig, ax = plt.subplots(figsize=(12, 4.5))
    bins = np.arange(0, max(kept + dropped + synced) + 100, 100)
    ax.hist(kept, bins=bins, color="#cccccc", label=f"kept in manifest (n={len(kept):,})")
    if dropped:
        ax.hist(dropped, bins=bins, color=COLORS["positive"], alpha=.8, label=f"dropped: truncated JPEG (n={len(dropped)})")
    if synced:
        ax.hist(synced, bins=bins, color="#9467bd", alpha=.8, label=f"hash re-synced after Ultralytics re-encode (n={len(synced)})")
    ax.set_xlabel("FracAtlas image index (IMG#######)"); ax.set_ylabel("files per 100-index bin"); ax.legend()
    ax.set_title("Fig 18 – Locality of JPEG integrity problems in FracAtlas")
    fw.save(fig, "fig18_fracatlas_integrity_locality")
    rng = lambda v: [min(v), max(v)] if v else None
    return {"kept_range": rng(kept), "dropped_range": rng(dropped), "synced_range": rng(synced),
            "dropped_all_negative": None}  # filled from fig16


def check_synced_dimensions(hash_sync: dict, df: pd.DataFrame, root: Path) -> dict:
    """Did re-encoding (exif_transpose) change width/height vs manifest? Read-only."""
    from PIL import Image
    changed = []
    for u in hash_sync.get("updated", []):
        p = root / u["path"]
        row = df[(df["split"] == Path(u["path"]).parts[0]) & (df["image_path"] == Path(u["path"]).name)]
        if row.empty or not p.exists():
            continue
        with Image.open(p) as im:
            w, h = im.size
        mw, mh = int(row.iloc[0]["width"]), int(row.iloc[0]["height"])
        if (w, h) != (mw, mh):
            changed.append({"path": u["path"], "manifest": [mw, mh], "disk": [w, h]})
    if changed:
        log.warning("%d re-synced images have width/height on disk != manifest (possible EXIF rotation)", len(changed))
    return {"checked": len(hash_sync.get("updated", [])), "dimension_mismatch": len(changed), "list": changed}








# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Generate all dataset figures (read-only).")
    ap.add_argument("--dataset-yaml", default="configs/dataset.yaml")
    ap.add_argument("--manifest", help="default: <dataset.yaml path>/manifest.csv")
    ap.add_argument("--validation-report", default="reports/validation_report.json")
    ap.add_argument("--prep-report", default="reports/dataset_preparation_report.json")
    ap.add_argument("--hash-sync-report", help="manifest_hash_sync_*.json (optional provenance note)")
    ap.add_argument("--out-dir", default="reports/figures/dataset")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--formats", default="png,svg")
    ap.add_argument("--skip-labels", action="store_true")
    ap.add_argument("--verify-hashes", action="store_true", help="md5 every processed image vs manifest (slow)")
    a = ap.parse_args()

    ds_yaml = load_yaml(PROJECT_ROOT / a.dataset_yaml)
    root = Path(ds_yaml.get("path", "data/processed"))
    root = root if root.is_absolute() else PROJECT_ROOT / root
    manifest = Path(a.manifest) if a.manifest else root / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"manifest not found: {manifest}")
    df = load_manifest(manifest)
    log.info("manifest: %s rows from %s | processed root: %s", len(df), manifest, root)

    prep = load_json(PROJECT_ROOT / a.prep_report)
    hs_path = Path(a.hash_sync_report) if a.hash_sync_report else None
    hash_sync = load_json(hs_path if (hs_path is None or hs_path.is_absolute()) else PROJECT_ROOT / hs_path) if hs_path else {}

    fw = FigureWriter(PROJECT_ROOT / a.out_dir, a.dpi, [f.strip() for f in a.formats.split(",") if f.strip()])
    boxes, lb = (pd.DataFrame(), {"skipped": True}) if a.skip_labels else load_boxes(df, root)

    s = {"meta": {"generated_at": datetime.now().isoformat(timespec="seconds"), "manifest_path": str(manifest),
                  "manifest_sha256": sha256_of(manifest), "manifest_rows": int(len(df)), "processed_root": str(root),
                  "dataset_yaml_core": {k: ds_yaml.get(k) for k in ("nc", "names", "path", "train", "val", "test", "split")}},
         "labels": lb}
    s["fig01"] = fig01_split(df, fw);          s["fig02"] = fig02_balance(df, fw)
    s["fig03"] = fig03_sources(df, fw);        s["fig04"] = fig04_source_split(df, fw)
    s["fig05"] = fig05_label_split(df, fw);    s["fig06"] = fig06_label_source(df, fw)
    s["fig07"] = fig07_grid(df, fw);           s["fig08"] = fig08_boxes_per_image(df, fw)
    s["fig09"] = fig09_box_area(boxes, fw);    s["fig10"] = fig10_box_shape(boxes, fw)
    s["fig11"] = fig11_centers(boxes, fw);     s["fig12"] = fig12_resolution(df, fw)
    s["fig13"] = fig13_annotation_status(df, fw); s["fig14"] = fig14_patient(df, fw)
    s["fig15"] = fig15_class_ids(lb, fw)
    s["fig16"] = fig16_funnel(prep, df, fw)
    s["fig17"] = fig17_summary(s, fw)
    s["fig18"] = fig18_integrity_locality(prep, hash_sync, df, fw)
   
    if hash_sync:
        s["synced_dimension_check"] = check_synced_dimensions(hash_sync, df, root)
    s["cross_check"] = cross_check(s, load_json(PROJECT_ROOT / a.validation_report), ds_yaml)
    if a.verify_hashes:
        s["hash_verification"] = verify_hashes(df, root)
    if a.hash_sync_report:
        hs = load_json(Path(a.hash_sync_report) if Path(a.hash_sync_report).is_absolute() else PROJECT_ROOT / a.hash_sync_report)
        if hs:
            by_src = {}
            for u in hs.get("updated", []):
                stem = Path(u["path"]).stem; key = "fracatlas" if stem.startswith("fa_") else "grazpedwri" if stem.startswith("grz_") else "unknown"
                by_src[key] = by_src.get(key, 0) + 1
            s["hash_sync_events"] = {"synced_at": hs.get("synced_at"), "updated_rows": hs.get("updated_rows"), "by_source": by_src,
                                     "paths": [u["path"] for u in hs.get("updated", [])]}
    s["meta"]["figures"] = fw.saved

    out = fw.out_dir
    (out / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_md(s, out / "summary.md")
    log.info("done: %d figures + summary.json/.md in %s | cross-check all_matched=%s",
             len(fw.saved), out, s["cross_check"]["all_matched"])







if __name__ == "__main__":
    main()