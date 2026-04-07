#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import cupy as cp
import cupyx.scipy.ndimage as cpx_ndimage
import cucim.skimage.restoration as cucim_restoration
import cucim.skimage.transform as cucim_transform
import matplotlib.pyplot as plt
import numpy as np
from cubic import scipy as cubic_scipy
from cubic.cuda import asnumpy
from cubic.skimage import restoration as cubic_restoration
from cubic.skimage import transform as cubic_transform
from scipy import ndimage
from skimage.restoration import richardson_lucy as sk_richardson_lucy
from skimage.transform import rescale as sk_rescale
from tifffile import imread

from common_real_data import (
    OUTPUTS_DIR,
    ensure_dir,
    log,
    normalize_preview,
    prepare_deconvolution_data,
    prepare_segmentation_data,
    save_tiff,
    xy_xz_views,
)


ARTIFACT_ROOT = OUTPUTS_DIR / 'real_data'



def synchronize(array: object) -> None:
    if hasattr(array, 'device'):
        cp.cuda.Stream.null.synchronize()



def timed_call(label: str, func: Any, array: object) -> tuple[object, float]:
    log(f'Running {label}')
    start = perf_counter()
    out = func(array)
    synchronize(out)
    elapsed = perf_counter() - start
    log(f'Finished {label} in {elapsed:.4f} s')
    return out, elapsed



def save_panel(panel_path: Path, original: np.ndarray, outputs: dict[str, np.ndarray], title: str) -> None:
    views = {'original': original, **outputs}
    ncols = len(views)
    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6))
    for col, (name, volume) in enumerate(views.items()):
        xy, xz = xy_xz_views(volume)
        axes[0, col].imshow(normalize_preview(xy), cmap='gray')
        axes[0, col].set_title(f'{name} XY')
        axes[1, col].imshow(normalize_preview(xz), cmap='gray', aspect='equal')
        axes[1, col].set_title(f'{name} XZ')
        axes[0, col].axis('off')
        axes[1, col].axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(panel_path, dpi=160)
    plt.close(fig)



def summarize_pair(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    diff = np.abs(np.asarray(reference, dtype=np.float32) - np.asarray(candidate, dtype=np.float32))
    return {
        'max_abs_diff': float(diff.max()),
        'mean_abs_diff': float(diff.mean()),
    }


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    volume = volume - float(volume.min())
    vmax = float(volume.max())
    if vmax > 0:
        volume = volume / vmax
    return volume


def build_demo_psf(shape: tuple[int, int, int] = (17, 33, 33), sigma: tuple[float, float, float] = (2.0, 3.5, 3.5)) -> np.ndarray:
    z = np.arange(shape[0], dtype=np.float32) - (shape[0] - 1) / 2.0
    y = np.arange(shape[1], dtype=np.float32) - (shape[1] - 1) / 2.0
    x = np.arange(shape[2], dtype=np.float32) - (shape[2] - 1) / 2.0
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
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



def run_operation_suite(operation_name: str, input_volume: np.ndarray, cases: dict[str, Any], output_dir: Path) -> None:
    ensure_dir(output_dir)
    cpu_input = np.asarray(input_volume, dtype=np.float32)
    gpu_input = cp.asarray(cpu_input)

    results_np: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    for case_name, func in cases.items():
        working_input = gpu_input if 'gpu' in case_name else cpu_input
        result, elapsed = timed_call(f'{operation_name} [{case_name}]', func, working_input)
        timings[case_name] = elapsed
        result_np = asnumpy(result).astype(np.float32)
        results_np[case_name] = result_np
        save_tiff(output_dir / f'{case_name}.tif', result_np)

    save_tiff(output_dir / 'original.tif', cpu_input)
    save_panel(output_dir / 'comparison.png', cpu_input, results_np, operation_name)

    summary = {
        'operation': operation_name,
        'input_shape': list(cpu_input.shape),
        'timings_seconds': timings,
        'cpu_wrapper_vs_raw': summarize_pair(results_np['raw_cpu'], results_np['cubic_cpu']),
        'gpu_wrapper_vs_raw': summarize_pair(results_np['raw_gpu'], results_np['cubic_gpu']),
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    log(f'Saved {operation_name} artifacts into {output_dir}')



def build_dna_operation_suites(volume: np.ndarray) -> dict[str, dict[str, Any]]:
    return {
        'rescale_down_order3': {
            'raw_cpu': lambda arr: sk_rescale(arr, scale=(1.0, 0.5, 0.5), order=3, preserve_range=True, anti_aliasing=True),
            'cubic_cpu': lambda arr: cubic_transform.rescale(arr, scale=(1.0, 0.5, 0.5), order=3, preserve_range=True, anti_aliasing=True),
            'raw_gpu': lambda arr: cucim_transform.rescale(arr, scale=(1.0, 0.5, 0.5), order=3, preserve_range=True, anti_aliasing=True),
            'cubic_gpu': lambda arr: cubic_transform.rescale(arr, scale=(1.0, 0.5, 0.5), order=3, preserve_range=True, anti_aliasing=True),
        },
        'rescale_up_order3': {
            'raw_cpu': lambda arr: sk_rescale(arr, scale=(1.0, 2.0, 2.0), order=3, preserve_range=True, anti_aliasing=False),
            'cubic_cpu': lambda arr: cubic_transform.rescale(arr, scale=(1.0, 2.0, 2.0), order=3, preserve_range=True, anti_aliasing=False),
            'raw_gpu': lambda arr: cucim_transform.rescale(arr, scale=(1.0, 2.0, 2.0), order=3, preserve_range=True, anti_aliasing=False),
            'cubic_gpu': lambda arr: cubic_transform.rescale(arr, scale=(1.0, 2.0, 2.0), order=3, preserve_range=True, anti_aliasing=False),
        },
        'gaussian_filter_sigma_1.2': {
            'raw_cpu': lambda arr: ndimage.gaussian_filter(arr, sigma=(0.0, 1.2, 1.2)),
            'cubic_cpu': lambda arr: cubic_scipy.ndimage.gaussian_filter(arr, sigma=(0.0, 1.2, 1.2)),
            'raw_gpu': lambda arr: cpx_ndimage.gaussian_filter(arr, sigma=(0.0, 1.2, 1.2)),
            'cubic_gpu': lambda arr: cubic_scipy.ndimage.gaussian_filter(arr, sigma=(0.0, 1.2, 1.2)),
        },
        'median_filter_size5': {
            'raw_cpu': lambda arr: ndimage.median_filter(arr, size=5, mode='nearest'),
            'cubic_cpu': lambda arr: cubic_scipy.ndimage.median_filter(arr, size=5, mode='nearest'),
            'raw_gpu': lambda arr: cpx_ndimage.median_filter(arr, size=5, mode='nearest'),
            'cubic_gpu': lambda arr: cubic_scipy.ndimage.median_filter(arr, size=5, mode='nearest'),
        },
    }



def run_dna_operations() -> None:
    files = prepare_segmentation_data()
    dna = imread(files['3d_monolayer_xy1_ch2.tif']).astype(np.float32)
    output_root = ensure_dir(ARTIFACT_ROOT / 'dna_channel_ops')
    save_tiff(output_root / 'original_dna.tif', dna)
    log(f'Loaded real DNA TIFF with shape {dna.shape}')

    for operation_name, cases in build_dna_operation_suites(dna).items():
        run_operation_suite(operation_name, dna, cases, output_root / operation_name)



def crop_deconvolution_image(image: np.ndarray, crop_xy: int = 512) -> np.ndarray:
    y0, x0 = 1000, 1300
    return image[:, y0:y0 + crop_xy, x0:x0 + crop_xy]



def run_deconvolution_suite(iterations: int = 10) -> None:
    output_name = f'richardson_lucy_iter{iterations}'
    files = prepare_deconvolution_data()
    image = imread(files['astr_vpa_hoechst.tif']).astype(np.float32)
    psf = imread(files['astr_vpa_hoechst_psf_na095_cropped.tif']).astype(np.float32)
    source_description = 'local main deconvolution dataset: real DNA TIFF + synthetic Gaussian PSF'

    output_dir = ensure_dir(ARTIFACT_ROOT / 'deconvolution' / output_name)
    save_tiff(output_dir / 'original_crop.tif', image)
    save_tiff(output_dir / 'psf.tif', psf)
    log(f'Loaded deconvolution image crop with shape {image.shape} and PSF shape {psf.shape}')

    cpu_image = image
    cpu_psf = psf
    gpu_image = cp.asarray(cpu_image)
    gpu_psf = cp.asarray(cpu_psf)

    cases = {
        'raw_cpu': lambda arr: sk_richardson_lucy(arr, cpu_psf, num_iter=iterations, clip=False),
        'cubic_cpu': lambda arr: cubic_restoration.richardson_lucy(arr, cpu_psf, num_iter=iterations, clip=False),
        'raw_gpu': lambda arr: cucim_restoration.richardson_lucy(arr, gpu_psf, num_iter=iterations, clip=False),
        'cubic_gpu': lambda arr: cubic_restoration.richardson_lucy(arr, gpu_psf, num_iter=iterations, clip=False),
    }

    results_np: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    for case_name, func in cases.items():
        working_input = gpu_image if 'gpu' in case_name else cpu_image
        result, elapsed = timed_call(f'deconvolution [{case_name}]', func, working_input)
        timings[case_name] = elapsed
        result_np = asnumpy(result).astype(np.float32)
        results_np[case_name] = result_np
        save_tiff(output_dir / f'{case_name}.tif', result_np)

    save_panel(output_dir / 'comparison.png', cpu_image, results_np, f'richardson_lucy_iter{iterations}')
    summary = {
        'operation': 'richardson_lucy',
        'iterations': iterations,
        'source': source_description,
        'input_shape': list(cpu_image.shape),
        'psf_shape': list(cpu_psf.shape),
        'timings_seconds': timings,
        'cpu_wrapper_vs_raw': summarize_pair(results_np['raw_cpu'], results_np['cubic_cpu']),
        'gpu_wrapper_vs_raw': summarize_pair(results_np['raw_gpu'], results_np['cubic_gpu']),
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    log(f'Saved deconvolution artifacts into {output_dir}')



def main() -> None:
    ensure_dir(ARTIFACT_ROOT)
    run_dna_operations()
    run_deconvolution_suite(iterations=10)
    log(f'All real-data artifacts saved under {ARTIFACT_ROOT}')


if __name__ == '__main__':
    main()
