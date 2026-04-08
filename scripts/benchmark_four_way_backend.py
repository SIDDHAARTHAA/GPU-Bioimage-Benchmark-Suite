#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import tempfile
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import cupy as cp
import cupyx.scipy.ndimage as cpx_ndimage
import cucim.skimage.transform as cucim_transform
import matplotlib.pyplot as plt
import numpy as np
from cubic import scipy as cubic_scipy
from cubic.skimage import transform as cubic_transform
from scipy import ndimage
from skimage import data as skdata
from skimage.transform import rescale as sk_rescale
from skimage.transform import resize

from common_real_data import log


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / 'results'


@dataclass(frozen=True)
class Operation:
    name: str
    category: str
    func_raw_cpu: Any
    func_cubic: Any
    func_raw_gpu: Any


FIELDNAMES = [
    'shape',
    'operation',
    'backend_family',
    'raw_cpu_seconds',
    'raw_cpu_total_seconds',
    'cubic_cpu_seconds',
    'cubic_cpu_total_seconds',
    'raw_gpu_seconds',
    'raw_gpu_total_seconds',
    'cubic_gpu_seconds',
    'cubic_gpu_total_seconds',
    'cpu_ratio',
    'gpu_ratio',
    'cpu_overhead_percent',
    'gpu_overhead_percent',
]



def synchronize_if_needed(array: object) -> None:
    if hasattr(array, 'device'):
        cp.cuda.Stream.null.synchronize()



def benchmark_call(
    func: Any, array: object, repeats: int = 5, warmups: int = 1
) -> tuple[float, float]:
    warmup_total = 0.0
    for _ in range(warmups):
        warmup_start = perf_counter()
        out = func(array)
        synchronize_if_needed(out)
        warmup_total += perf_counter() - warmup_start

    timings: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        out = func(array)
        synchronize_if_needed(out)
        timings.append(perf_counter() - start)
    timed_total = sum(timings)
    return median(timings), warmup_total + timed_total



def format_seconds(value: float) -> str:
    return f'{value:.4f}'



def build_operations() -> list[Operation]:
    return [
        Operation(
            name='rescale_down_order3',
            category='skimage/cuCIM',
            func_raw_cpu=lambda arr: sk_rescale(
                arr,
                scale=(1.0, 0.5, 0.5),
                order=3,
                preserve_range=True,
                anti_aliasing=True,
            ),
            func_cubic=lambda arr: cubic_transform.rescale(
                arr,
                scale=(1.0, 0.5, 0.5),
                order=3,
                preserve_range=True,
                anti_aliasing=True,
            ),
            func_raw_gpu=lambda arr: cucim_transform.rescale(
                arr,
                scale=(1.0, 0.5, 0.5),
                order=3,
                preserve_range=True,
                anti_aliasing=True,
            ),
        ),
        Operation(
            name='rescale_up_order3',
            category='skimage/cuCIM',
            func_raw_cpu=lambda arr: sk_rescale(
                arr,
                scale=(1.0, 2.0, 2.0),
                order=3,
                preserve_range=True,
                anti_aliasing=False,
            ),
            func_cubic=lambda arr: cubic_transform.rescale(
                arr,
                scale=(1.0, 2.0, 2.0),
                order=3,
                preserve_range=True,
                anti_aliasing=False,
            ),
            func_raw_gpu=lambda arr: cucim_transform.rescale(
                arr,
                scale=(1.0, 2.0, 2.0),
                order=3,
                preserve_range=True,
                anti_aliasing=False,
            ),
        ),
        Operation(
            name='gaussian_filter_sigma_1.2',
            category='scipy/cupyx.scipy',
            func_raw_cpu=lambda arr: ndimage.gaussian_filter(arr, sigma=(0.0, 1.2, 1.2)),
            func_cubic=lambda arr: cubic_scipy.ndimage.gaussian_filter(
                arr, sigma=(0.0, 1.2, 1.2)
            ),
            func_raw_gpu=lambda arr: cpx_ndimage.gaussian_filter(
                arr, sigma=(0.0, 1.2, 1.2)
            ),
        ),
        Operation(
            name='median_filter_size5',
            category='scipy/cupyx.scipy',
            func_raw_cpu=lambda arr: ndimage.median_filter(arr, size=5, mode='nearest'),
            func_cubic=lambda arr: cubic_scipy.ndimage.median_filter(
                arr, size=5, mode='nearest'
            ),
            func_raw_gpu=lambda arr: cpx_ndimage.median_filter(arr, size=5, mode='nearest'),
        ),
    ]



def write_table(rows: list[dict[str, str | float]], output_dir: Path) -> tuple[Path, Path]:
    csv_path = output_dir / 'four_way_benchmark.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / 'four_way_benchmark.json'
    json_path.write_text(json.dumps(rows, indent=2))
    return csv_path, json_path



def write_plots(
    rows: list[dict[str, str | float]], operations: list[Operation], output_dir: Path
) -> list[Path]:
    bar_names = ['raw_cpu_seconds', 'cubic_cpu_seconds', 'raw_gpu_seconds', 'cubic_gpu_seconds']
    bar_labels = ['raw CPU', 'cubic CPU', 'raw GPU', 'cubic GPU']
    colors = ['#4C78A8', '#72B7B2', '#F58518', '#E45756']

    plot_paths: list[Path] = []
    for operation in operations:
        subset = [row for row in rows if row['operation'] == operation.name]
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(subset))
        width = 0.18

        for idx, (bar_name, bar_label, color) in enumerate(
            zip(bar_names, bar_labels, colors, strict=True)
        ):
            ax.bar(
                x + (idx - 1.5) * width,
                [float(row[bar_name]) for row in subset],
                width=width,
                label=bar_label,
                color=color,
            )

        ax.set_title(operation.name)
        ax.set_xticks(x)
        ax.set_xticklabels([str(row['shape']) for row in subset])
        ax.set_ylabel('seconds')
        ax.set_xlabel('data shape')
        ax.grid(axis='y', alpha=0.25)
        ax.legend(ncols=2)
        fig.tight_layout()

        plot_path = output_dir / f'four_way_{operation.name}.png'
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_paths.append(plot_path)

    return plot_paths



def clean_previous_four_way_artifacts() -> None:
    if not RESULTS_DIR.exists():
        return

    patterns = [
        'four_way_benchmark.csv',
        'four_way_benchmark.json',
        'four_way_benchmark_meta.json',
        'four_way_*.png',
    ]
    for pattern in patterns:
        for path in RESULTS_DIR.glob(pattern):
            if path.is_file():
                path.unlink()



def publish_results(staging_dir: Path) -> tuple[Path, Path, list[Path]]:
    RESULTS_DIR.mkdir(exist_ok=True)

    published_csv = RESULTS_DIR / 'four_way_benchmark.csv'
    published_json = RESULTS_DIR / 'four_way_benchmark.json'
    published_plots: list[Path] = []

    for staged_path in staging_dir.iterdir():
        final_path = RESULTS_DIR / staged_path.name
        if final_path.exists():
            final_path.unlink()
        shutil.move(str(staged_path), str(final_path))
        if final_path.suffix == '.png':
            published_plots.append(final_path)

    published_plots.sort()
    return published_csv, published_json, published_plots



def main() -> None:
    run_started = datetime.now().astimezone()
    wall_start = perf_counter()
    clean_previous_four_way_artifacts()
    log('Loading real microscopy benchmark volume from skimage.data.cells3d()')

    cells = skdata.cells3d()[:, 1].astype(np.float32)
    cells_large = resize(
        cells,
        output_shape=(cells.shape[0], cells.shape[1] * 2, cells.shape[2] * 2),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    datasets = {
        '60x256x256': cells,
        '60x512x512': cells_large,
    }
    operations = build_operations()
    rows: list[dict[str, str | float]] = []

    for shape_name, cpu_array in datasets.items():
        log(f'Prepared dataset {shape_name} on CPU')
        gpu_array = cp.asarray(cpu_array)
        log(f'Prepared dataset {shape_name} on GPU')

        for operation in operations:
            log(f'Starting {operation.name} on {shape_name}')
            raw_cpu_s, raw_cpu_total_s = benchmark_call(operation.func_raw_cpu, cpu_array)
            log(f'  raw CPU finished in {format_seconds(raw_cpu_s)} s')
            cubic_cpu_s, cubic_cpu_total_s = benchmark_call(operation.func_cubic, cpu_array)
            log(f'  cubic CPU finished in {format_seconds(cubic_cpu_s)} s')
            raw_gpu_s, raw_gpu_total_s = benchmark_call(operation.func_raw_gpu, gpu_array)
            log(f'  raw GPU finished in {format_seconds(raw_gpu_s)} s')
            cubic_gpu_s, cubic_gpu_total_s = benchmark_call(operation.func_cubic, gpu_array)
            log(f'  cubic GPU finished in {format_seconds(cubic_gpu_s)} s')

            row = {
                'shape': shape_name,
                'operation': operation.name,
                'backend_family': operation.category,
                'raw_cpu_seconds': raw_cpu_s,
                'raw_cpu_total_seconds': raw_cpu_total_s,
                'cubic_cpu_seconds': cubic_cpu_s,
                'cubic_cpu_total_seconds': cubic_cpu_total_s,
                'raw_gpu_seconds': raw_gpu_s,
                'raw_gpu_total_seconds': raw_gpu_total_s,
                'cubic_gpu_seconds': cubic_gpu_s,
                'cubic_gpu_total_seconds': cubic_gpu_total_s,
                'cpu_ratio': cubic_cpu_s / raw_cpu_s,
                'gpu_ratio': cubic_gpu_s / raw_gpu_s,
                'cpu_overhead_percent': (cubic_cpu_s / raw_cpu_s - 1.0) * 100.0,
                'gpu_overhead_percent': (cubic_gpu_s / raw_gpu_s - 1.0) * 100.0,
            }
            rows.append(row)

    staging_dir = Path(tempfile.mkdtemp(prefix='four_way_results_', dir=ROOT_DIR))
    try:
        write_table(rows, staging_dir)
        write_plots(rows, operations, staging_dir)
        run_finished = datetime.now().astimezone()
        wall_elapsed = perf_counter() - wall_start
        sum_of_medians = sum(
            float(row['raw_cpu_seconds'])
            + float(row['cubic_cpu_seconds'])
            + float(row['raw_gpu_seconds'])
            + float(row['cubic_gpu_seconds'])
            for row in rows
        )
        sum_of_measured_totals = sum(
            float(row['raw_cpu_total_seconds'])
            + float(row['cubic_cpu_total_seconds'])
            + float(row['raw_gpu_total_seconds'])
            + float(row['cubic_gpu_total_seconds'])
            for row in rows
        )
        meta = {
            'started_at': run_started.isoformat(),
            'finished_at': run_finished.isoformat(),
            'elapsed_wall_seconds': wall_elapsed,
            'sum_median_seconds': sum_of_medians,
            'sum_measured_total_seconds': sum_of_measured_totals,
            'non_operation_seconds': wall_elapsed - sum_of_measured_totals,
            'benchmark_call_repeats': 5,
            'benchmark_call_warmups': 1,
        }
        (staging_dir / 'four_way_benchmark_meta.json').write_text(json.dumps(meta, indent=2))
        csv_path, json_path, plot_paths = publish_results(staging_dir)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    print('Four-way benchmark results', flush=True)
    print(
        'shape | operation | raw_cpu | cubic_cpu | raw_gpu | cubic_gpu | cpu_overhead | gpu_overhead',
        flush=True,
    )
    for row in rows:
        print(
            ' | '.join(
                [
                    str(row['shape']),
                    str(row['operation']),
                    format_seconds(float(row['raw_cpu_seconds'])),
                    format_seconds(float(row['cubic_cpu_seconds'])),
                    format_seconds(float(row['raw_gpu_seconds'])),
                    format_seconds(float(row['cubic_gpu_seconds'])),
                    f"{float(row['cpu_overhead_percent']):.2f}%",
                    f"{float(row['gpu_overhead_percent']):.2f}%",
                ]
            ),
            flush=True,
        )
    print(f'\nCSV: {csv_path}', flush=True)
    print(f'JSON: {json_path}', flush=True)
    print(f'META: {RESULTS_DIR / "four_way_benchmark_meta.json"}', flush=True)
    print(
        f'Wall elapsed: {wall_elapsed:.2f} s | Sum medians: {sum_of_medians:.2f} s | '
        f'Sum measured totals: {sum_of_measured_totals:.2f} s | '
        f'Non-operation overhead: {wall_elapsed - sum_of_measured_totals:.2f} s',
        flush=True,
    )
    for plot_path in plot_paths:
        print(f'Plot: {plot_path}', flush=True)


if __name__ == '__main__':
    main()
