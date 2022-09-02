from datetime import datetime
from pathlib import Path
from typing import Callable
import numpy as np
import scipy.stats
import pandas as pd


COLORS = {
    'docker': 'royalblue',
    'kvm-vhost-net': 'seagreen',
    'kvm-virtio-net': 'mediumseagreen',
    'kvm': 'seagreen',
}

NF_COLORS = {
    'amf': 'red',
    'ausf': 'blue',
    'bsf': 'g',
    'db': 'k',
    'mongo': 'k',
    'nrf': 'c',
    'nssf': 'm',
    'pcf': 'coral',
    'smf': 'navy',
    'udm': 'brown',
    'udr': 'teal',
    'upf': 'goldenrod'
}

NF_NAMES = NF_COLORS.keys()

def get_color(nf_name):
    return next(
        color for name, color in NF_COLORS.items() if name in nf_name
    )

def get_nf_name(nf_name):
    return next(
        name for name in NF_NAMES if name in nf_name
    )


def find_in_iterdir(dir_path: Path, fn_part: str) -> Path:
    dir_path = Path(dir_path)
    return next(
        (path for path in dir_path.iterdir()
        if fn_part in str(path)), None
    )


def _strip_seconds(line: str):
    return line.strip()[:-1] if line.strip().endswith('s') else line.strip()


def read_general_file(path, n_ues_line = 'N_UES', n_iterations_line = 'N_ITERATIONS'):
    timestamp_line = 'started'
    timestamp = ''
    n_ues = '-1'
    n_iterations = '-1'
    with open(path, 'r') as fd:
        for line in fd.readlines():
            if timestamp_line in line:
                timestamp = line[:8].replace('-', ':')
                timestamp = datetime.strptime(timestamp, '%H:%M:%S')
            elif n_ues_line in line:
                n_ues = int(line.split(':')[1].strip())
            elif n_iterations_line in line:
                n_iterations = int(_strip_seconds(line.split(':')[1]))
    return timestamp, n_ues, n_iterations


def calc_conf_int(df, col, n_samples, confidence=0.95):
    # print(df[col, 'std'])
    h = (df[col, 'std']/np.sqrt(n_samples)) * scipy.stats.t.ppf((1 + confidence) / 2., n_samples-1)
    m = df[col, 'mean']
    return m - h, m + h, h


def concat_multiple_logs(test_dir: Path, read_sample_func: Callable, n_samples=20, **kwargs):
    stat_df = pd.DataFrame()
    n_reads = 0
    for sample_dir in test_dir.iterdir():
        # garbage
        if not sample_dir.name.startswith('test'):
            print(f'[DEBUG] skip {sample_dir}...')
            continue

        print(f'[DEBUG] read {sample_dir}...')
        df = read_sample_func(sample_dir, **kwargs)
        stat_df = pd.concat((stat_df, df))

        n_reads += 1
        if n_reads >= n_samples:
            break

    return stat_df


def _to_ms_numeric_timedelta(delta):
    return delta.astype('timedelta64[us]') / 1e6


def add_stat_err(df, stat, n_samples):
    df[stat, 'ci_lower'], df[stat, 'ci_upper'], df[stat, 'err'] = calc_conf_int(df, stat, n_samples)
