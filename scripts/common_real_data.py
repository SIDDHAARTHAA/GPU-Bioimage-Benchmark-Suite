#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite

ROOT_DIR = Path(__file__).resolve().parents[1]
VENDOR_DATA_DIR = ROOT_DIR / "vendor" / "cubic" / "examples" / "data"
DATA_DIR = ROOT_DIR / "data" / "real"
SEGMENTATION_DATA_DIR = DATA_DIR / "segmentation"
DECONVOLUTION_DATA_DIR = DATA_DIR / "deconvolution"
OUTPUTS_DIR = ROOT_DIR / "outputs"

SEGMENTATION_FILES = [
    "3d_monolayer_xy1_ch0.tif",
    "3d_monolayer_xy1_ch1.tif",
    "3d_monolayer_xy1_ch2.tif",
    "3d_monolayer_xy1_ch0_CellsLabels.tiff",
    "3d_monolayer_xy1_ch2_NucleiLabels.tiff",
]

DECONVOLUTION_IMAGE_NAME = "astr_vpa_hoechst.tif"
DECONVOLUTION_PSF_NAME = "astr_vpa_hoechst_psf_na095_cropped.tif"
DECONVOLUTION_INFO_NAME = "source_info.json"


def log(message: str) -> None:
    print(f"[project] {message}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_segmentation_data() -> dict[str, Path]:
    ensure_dir(SEGMENTATION_DATA_DIR)
    copied: dict[str, Path] = {}

    for name in SEGMENTATION_FILES:
        src = VENDOR_DATA_DIR / name
        dst = SEGMENTATION_DATA_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Missing source segmentation TIFF: {src}")
        if not dst.exists():
            shutil.copy2(src, dst)
            log(f"Copied segmentation TIFF to {dst}")
        copied[name] = dst

    return copied


def build_demo_psf(
    shape: tuple[int, int, int] = (17, 33, 33),
    sigma: tuple[float, float, float] = (2.0, 3.5, 3.5),
) -> np.ndarray:
    z = np.arange(shape[0], dtype=np.float32) - (shape[0] - 1) / 2.0
    y = np.arange(shape[1], dtype=np.float32) - (shape[1] - 1) / 2.0
    x = np.arange(shape[2], dtype=np.float32) - (shape[2] - 1) / 2.0
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    psf = np.exp(
        -(
            (zz**2) / (2.0 * sigma[0] ** 2)
            + (yy**2) / (2.0 * sigma[1] ** 2)
            + (xx**2) / (2.0 * sigma[2] ** 2)
        )
    )
    psf = psf.astype(np.float32)
    psf /= float(psf.sum())
    return psf


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    volume = volume - float(volume.min())
    vmax = float(volume.max())
    if vmax > 0:
        volume = volume / vmax
    return volume


def prepare_deconvolution_data() -> dict[str, Path]:
    ensure_dir(DECONVOLUTION_DATA_DIR)
    seg_files = prepare_segmentation_data()
    image_path = DECONVOLUTION_DATA_DIR / DECONVOLUTION_IMAGE_NAME
    psf_path = DECONVOLUTION_DATA_DIR / DECONVOLUTION_PSF_NAME
    info_path = DECONVOLUTION_DATA_DIR / DECONVOLUTION_INFO_NAME

    if not image_path.exists():
        dna = imread(seg_files["3d_monolayer_xy1_ch2.tif"]).astype(np.float32)
        dna = normalize_volume(dna)
        imwrite(image_path, dna)
        log(f"Created local deconvolution source TIFF at {image_path}")

    if not psf_path.exists():
        psf = build_demo_psf()
        imwrite(psf_path, psf.astype(np.float32))
        log(f"Created local deconvolution PSF TIFF at {psf_path}")

    info = {
        "source_type": "local main deconvolution dataset",
        "image_source": "real DNA TIFF copied from data/real/segmentation/3d_monolayer_xy1_ch2.tif",
        "psf_source": "synthetic normalized 3D Gaussian PSF generated in scripts/common_real_data.py",
        "replaced_broken_remote_links": True,
    }
    info_path.write_text(json.dumps(info, indent=2))

    return {
        DECONVOLUTION_IMAGE_NAME: image_path,
        DECONVOLUTION_PSF_NAME: psf_path,
        DECONVOLUTION_INFO_NAME: info_path,
    }


def normalize_preview(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return clipped


def save_tiff(path: Path, array: np.ndarray) -> None:
    ensure_dir(path.parent)
    imwrite(path, np.asarray(array))


def xy_xz_views(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    return np.asarray(volume[z_mid]), np.asarray(volume[:, y_mid, :])
