#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread

warnings.filterwarnings('ignore', message='Cellpose is not available.*')

from cubic.cuda import CUDAManager, asnumpy, to_device, to_same_device
from cubic.image_utils import (
    distance_transform_edt,
    normalize_min_max,
    rescale_xy,
)
from cubic.metrics.average_precision import average_precision
from cubic.segmentation.segment_utils import (
    cleanup_segmentation,
    downscale_and_filter,
    segment_watershed,
)
from cubic.skimage import filters, morphology, transform, util

from common_real_data import OUTPUTS_DIR, ensure_dir, log, prepare_segmentation_data, save_tiff



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the real-data 3D segmentation demo with saved outputs.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='gpu')
    return parser.parse_args()



def label_cmap(label_image: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap('tab20')
    labels = asnumpy(label_image)
    rgb = np.zeros(labels.shape + (3,), dtype=np.float32)
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        rgb[labels == lbl] = cmap(((int(lbl) - 1) % 20 + 0.5) / 20)[:3]
    return rgb



def save_single_panel(path: Path, images: list[np.ndarray], titles: list[str], cmaps: list[str | None], figsize: tuple[float, float]) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:
        axes = [axes]
    for ax, image, title, cmap in zip(axes, images, titles, cmaps, strict=True):
        ax.imshow(image, cmap=cmap, aspect='equal')
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)



def main() -> None:
    args = parse_args()
    files = prepare_segmentation_data()

    use_gpu = CUDAManager().num_gpus > 0
    if args.device == 'cpu':
        device = 'CPU'
    elif args.device == 'gpu':
        if not use_gpu:
            raise RuntimeError('GPU was requested but is not available.')
        device = 'GPU'
    else:
        device = 'GPU' if use_gpu else 'CPU'

    output_dir = ensure_dir(OUTPUTS_DIR / 'segmentation' / device.lower())
    log(f'Running segmentation demo on {device}')

    memb = util.img_as_float(imread(files['3d_monolayer_xy1_ch0.tif']))
    mito = util.img_as_float(imread(files['3d_monolayer_xy1_ch1.tif']))
    dna = util.img_as_float(imread(files['3d_monolayer_xy1_ch2.tif']))
    cp_nuclei = imread(files['3d_monolayer_xy1_ch2_NucleiLabels.tiff']).astype(np.uint16)
    cp_cells = imread(files['3d_monolayer_xy1_ch0_CellsLabels.tiff']).astype(np.uint16)

    memb = to_device(memb, device)
    mito = to_device(mito, device)
    dna = to_device(dna, device)
    cp_nuclei = to_device(cp_nuclei, device)
    cp_cells = to_device(cp_cells, device)

    z_mid = dna.shape[0] // 2
    y_mid = dna.shape[1] // 2
    height_ratio = dna.shape[0] / dna.shape[1]

    log('Saving input channel panel')
    fig, axes = plt.subplots(2, 3, figsize=(12, 5), gridspec_kw={'height_ratios': [1, 0.24]})
    for i, (name, img) in enumerate({'Membrane': memb, 'Mitochondria': mito, 'DNA': dna}.items()):
        axes[0, i].imshow(asnumpy(img[z_mid]), cmap='gray')
        axes[0, i].set_title(f'{name} XY')
        axes[0, i].axis('off')
        axes[1, i].imshow(asnumpy(img[:, y_mid, :]), cmap='gray', aspect='equal')
        axes[1, i].set_title(f'{name} XZ')
        axes[1, i].axis('off')
    fig.tight_layout()
    fig.savefig(output_dir / 'channels.png', dpi=160)
    plt.close(fig)

    log('Starting nuclei segmentation pipeline')
    t0 = perf_counter()
    dna_norm = normalize_min_max(dna, q=(0.0, 100.0))
    dna_ds = downscale_and_filter(dna_norm, filter_size=5, filter_mode='constant')
    thresh = filters.threshold_otsu(dna_ds)
    nuc_binary = dna_ds > thresh
    nuc_binary = morphology.remove_small_holes(nuc_binary, area_threshold=20)
    nuclei_ds = segment_watershed(nuc_binary, ball_size=10, dilate_seeds=True)
    nuclei = rescale_xy(
        nuclei_ds,
        1.0 / 0.5,
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    ).astype(np.uint16)
    nuclei = cleanup_segmentation(nuclei, min_obj_size=50, max_hole_size=500)
    t_nuclei = perf_counter() - t0
    log(f'Nuclei segmentation finished in {t_nuclei:.2f} s with {int(nuclei.max())} objects')

    save_single_panel(
        output_dir / 'nuclei_steps.png',
        [
            asnumpy(dna_norm[z_mid]),
            asnumpy(nuc_binary[z_mid]),
            label_cmap(nuclei_ds[z_mid]),
            label_cmap(nuclei[z_mid]),
        ],
        ['DNA normalized', 'Binary', 'Watershed downscaled', f'Final nuclei ({int(nuclei.max())})'],
        ['gray', 'gray', None, None],
        (16, 4),
    )

    log('Starting cell segmentation pipeline')
    t0 = perf_counter()
    thresh = filters.threshold_multiotsu(memb, classes=3, nbins=128)
    memb_bin = memb > thresh[0]
    memb_bin = morphology.remove_small_holes(~memb_bin, area_threshold=4189)

    mono = np.clip(dna + memb + mito, 0, 1)
    mono_ds = rescale_xy(mono, 0.25)
    disk_fp = to_same_device(morphology.disk(17), mono_ds)
    mono_closed = np.zeros_like(asnumpy(mono_ds))
    mono_closed = to_device(mono_closed, device)
    for z in range(mono_ds.shape[0]):
        mono_closed[z] = morphology.closing(mono_ds[z], disk_fp)
    mono_mask = rescale_xy(
        mono_closed,
        1.0 / 0.25,
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    )
    mono_mask = mono_mask > filters.threshold_li(mono_mask)

    cell_mask = memb_bin & mono_mask
    cell_mask = morphology.binary_erosion(
        cell_mask, to_same_device(morphology.ball(1), cell_mask)
    )

    nuc_ds_shape = (nuclei.shape[0], nuclei.shape[1] // 2, nuclei.shape[2] // 2)
    nuc_ds = transform.resize(
        asnumpy(nuclei),
        nuc_ds_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.uint16)
    seeds_ds = morphology.erosion(nuc_ds, morphology.ball(5))
    orig_labels = set(np.unique(nuc_ds).tolist()) - {0}
    surv_labels = set(np.unique(seeds_ds).tolist()) - {0}
    for lost in orig_labels - surv_labels:
        binary = nuc_ds == lost
        dt = distance_transform_edt(binary)
        seeds_ds[dt == dt.max()] = lost
    seeds = transform.resize(
        seeds_ds,
        nuclei.shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.uint16)

    cells_raw = segment_watershed(cell_mask, markers=seeds, mask=cell_mask)
    cells_raw = morphology.remove_small_objects(cells_raw, min_size=100)
    old_labels = np.unique(cells_raw)
    old_labels = old_labels[old_labels > 0]
    cells = np.zeros_like(cells_raw, dtype=np.uint16)
    for new_id, old_id in enumerate(asnumpy(old_labels), 1):
        cells[cells_raw == int(old_id)] = new_id

    t_cells = perf_counter() - t0
    log(f'Cell segmentation finished in {t_cells:.2f} s with {int(cells.max())} objects')

    save_single_panel(
        output_dir / 'cells_steps.png',
        [
            asnumpy(memb[z_mid]),
            asnumpy(memb_bin[z_mid]),
            asnumpy(cell_mask[z_mid]),
            label_cmap(cells[z_mid]),
        ],
        ['Membrane', 'Membrane binary', 'Cell mask', f'Final cells ({int(cells.max())})'],
        ['gray', 'gray', 'gray', None],
        (16, 4),
    )

    thresholds = np.arange(0.5, 1.0 + 0.05, 0.1)
    nuclei_ap, *_ = average_precision(cp_nuclei.astype(np.uint16), nuclei.astype(np.uint16), thresholds)
    cells_ap, *_ = average_precision(cp_cells.astype(np.uint16), cells.astype(np.uint16), thresholds)
    nuclei_ap_np = asnumpy(nuclei_ap)
    cells_ap_np = asnumpy(cells_ap)
    log(f'Nuclei mAP: {nuclei_ap_np.mean():.3f}')
    log(f'Cells mAP: {cells_ap_np.mean():.3f}')

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(16, 2 * (4 + 4 * height_ratio)),
        gridspec_kw={'height_ratios': [1, height_ratio, 1, height_ratio], 'width_ratios': [1, 1, 1, 1], 'wspace': 0.08},
    )
    for row, sl, view in [(0, np.s_[z_mid], 'XY'), (1, np.s_[:, y_mid, :], 'XZ')]:
        axes[row, 0].imshow(asnumpy(dna[sl]), cmap='gray', aspect='equal')
        axes[row, 0].set_title(f'DNA ({view})')
        axes[row, 1].imshow(label_cmap(cp_nuclei[sl]), aspect='equal')
        axes[row, 1].set_title(f'Ref nuclei ({view})')
        axes[row, 2].imshow(label_cmap(nuclei[sl]), aspect='equal')
        axes[row, 2].set_title(f'Pred nuclei ({view})')
        axes[row, 0].axis('off')
        axes[row, 1].axis('off')
        axes[row, 2].axis('off')
    ax_nuc = axes[0, 3]
    ax_nuc.plot(thresholds, nuclei_ap_np, 'o-', color='C0', markersize=5)
    ax_nuc.set_title(f'Nuclei mAP={nuclei_ap_np.mean():.3f}')
    ax_nuc.set_xlabel('IoU')
    ax_nuc.set_ylabel('AP')
    ax_nuc.set_xlim(0.45, 1.05)
    ax_nuc.set_ylim(0, 1.05)
    ax_nuc.grid(True, alpha=0.3)
    axes[1, 3].axis('off')

    for row, sl, view in [(2, np.s_[z_mid], 'XY'), (3, np.s_[:, y_mid, :], 'XZ')]:
        axes[row, 0].imshow(asnumpy(memb[sl]), cmap='gray', aspect='equal')
        axes[row, 0].set_title(f'Membrane ({view})')
        axes[row, 1].imshow(label_cmap(cp_cells[sl]), aspect='equal')
        axes[row, 1].set_title(f'Ref cells ({view})')
        axes[row, 2].imshow(label_cmap(cells[sl]), aspect='equal')
        axes[row, 2].set_title(f'Pred cells ({view})')
        axes[row, 0].axis('off')
        axes[row, 1].axis('off')
        axes[row, 2].axis('off')
    ax_cell = axes[2, 3]
    ax_cell.plot(thresholds, cells_ap_np, 's-', color='C1', markersize=5)
    ax_cell.set_title(f'Cells mAP={cells_ap_np.mean():.3f}')
    ax_cell.set_xlabel('IoU')
    ax_cell.set_ylabel('AP')
    ax_cell.set_xlim(0.45, 1.05)
    ax_cell.set_ylim(0, 1.05)
    ax_cell.grid(True, alpha=0.3)
    axes[3, 3].axis('off')
    fig.savefig(output_dir / 'evaluation.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    save_tiff(output_dir / 'nuclei_labels.tif', asnumpy(nuclei).astype(np.uint16))
    save_tiff(output_dir / 'cell_labels.tif', asnumpy(cells).astype(np.uint16))

    summary = {
        'device': device,
        'nuclei_seconds': t_nuclei,
        'cells_seconds': t_cells,
        'nuclei_count': int(nuclei.max()),
        'cells_count': int(cells.max()),
        'nuclei_map': float(nuclei_ap_np.mean()),
        'cells_map': float(cells_ap_np.mean()),
        'thresholds': thresholds.tolist(),
        'nuclei_ap': nuclei_ap_np.tolist(),
        'cells_ap': cells_ap_np.tolist(),
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    log(f'Saved segmentation outputs into {output_dir}')


if __name__ == '__main__':
    main()
