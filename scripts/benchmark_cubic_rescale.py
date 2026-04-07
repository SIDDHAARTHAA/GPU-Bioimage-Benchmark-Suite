#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from skimage import data as skdata
from skimage.transform import resize

from cubic.cuda import CUDAManager, ascupy
from cubic.skimage import transform
from common_real_data import log


RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'



def synchronize_if_needed() -> None:
    manager = CUDAManager()
    if manager.cp is not None and manager.num_gpus > 0:
        manager.cp.cuda.Stream.null.synchronize()



def uses_downscale(scale: Any) -> bool:
    values = scale if isinstance(scale, tuple) else (scale,)
    return any(float(v) < 1.0 for v in values)



def benchmark_rescale(image: np.ndarray, device: str, scale: Any, order: int, repeats: int = 1) -> float:
    arr = ascupy(image) if device == 'gpu' else image

    if device == 'gpu':
        _ = transform.rescale(
            arr,
            scale=scale,
            order=order,
            preserve_range=True,
            anti_aliasing=uses_downscale(scale),
        )
        synchronize_if_needed()

    timings: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        _ = transform.rescale(
            arr,
            scale=scale,
            order=order,
            preserve_range=True,
            anti_aliasing=uses_downscale(scale),
        )
        synchronize_if_needed()
        timings.append(perf_counter() - start)

    return median(timings)



def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    log('Loading real microscopy benchmark volume from skimage.data.cells3d()')

    cells = skdata.cells3d()[:, 1]
    cells_large = resize(
        cells,
        output_shape=(cells.shape[0], cells.shape[1] * 2, cells.shape[2] * 2),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    datasets = {
        '60x256x256': cells.astype(np.float32),
        '60x512x512': cells_large,
    }

    records: list[dict[str, float | int | str]] = []

    for shape_name, image in datasets.items():
        log(f'Prepared cubic-only rescale benchmark dataset {shape_name}')
        for order in range(6):
            log(f'Starting order={order} rescale timings on {shape_name}')
            cpu_up = benchmark_rescale(image, 'cpu', (1.0, 2.0, 2.0), order)
            log(f'  cubic CPU upscale: {cpu_up:.4f} s')
            gpu_up = benchmark_rescale(image, 'gpu', (1.0, 2.0, 2.0), order)
            log(f'  cubic GPU upscale: {gpu_up:.4f} s')

            cpu_down = benchmark_rescale(image, 'cpu', (1.0, 0.5, 0.5), order)
            log(f'  cubic CPU downscale: {cpu_down:.4f} s')
            gpu_down = benchmark_rescale(image, 'gpu', (1.0, 0.5, 0.5), order)
            log(f'  cubic GPU downscale: {gpu_down:.4f} s')

            records.extend(
                [
                    {
                        'shape': shape_name,
                        'order': order,
                        'operation': 'upscale',
                        'cpu_backend': 'cubic_cpu',
                        'gpu_backend': 'cubic_gpu',
                        'cpu_seconds': cpu_up,
                        'gpu_seconds': gpu_up,
                        'speedup': cpu_up / gpu_up,
                    },
                    {
                        'shape': shape_name,
                        'order': order,
                        'operation': 'downscale',
                        'cpu_backend': 'cubic_cpu',
                        'gpu_backend': 'cubic_gpu',
                        'cpu_seconds': cpu_down,
                        'gpu_seconds': gpu_down,
                        'speedup': cpu_down / gpu_down,
                    },
                ]
            )

    json_path = RESULTS_DIR / 'rescale_benchmark.json'
    json_path.write_text(json.dumps(records, indent=2))

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    colors = {
        '60x256x256': '#4C78A8',
        '60x512x512': '#F58518',
    }

    for ax, operation in zip(axes, ['upscale', 'downscale'], strict=True):
        for shape_name in datasets:
            subset = [r for r in records if r['shape'] == shape_name and r['operation'] == operation]
            subset = sorted(subset, key=lambda r: int(r['order']))
            ax.bar(
                [int(r['order']) + (-0.18 if shape_name == '60x256x256' else 0.18) for r in subset],
                [float(r['speedup']) for r in subset],
                width=0.36,
                label=shape_name,
                color=colors[shape_name],
            )
        ax.set_title(operation)
        ax.set_ylabel('CPU / GPU speedup')
        ax.grid(axis='y', alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel('interpolation order')
    plot_path = RESULTS_DIR / 'rescale_benchmark.png'
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(json.dumps({'json': str(json_path), 'plot': str(plot_path)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
