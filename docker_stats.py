from pathlib import Path
import re

from datetime import datetime, timedelta
import sys
import matplotlib.pyplot as plt
import pandas as pd

import utils

HEADER_LINE = 'NAME'
TIMESTAMP_LINE = ':'
N_THREADS = 12

DROP_NFS = ['nr_gnb']  # do not plot these nfs
COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k',
          'lime', 'dodgerblue', 'peru', 'indigo',
          'crimson', 'slategray']

# TEST_DIR = Path('..', 'containers', 'test-connect-ues', 'test-08-50-48-472238')# old
# TEST_DIR = Path('..', 'containers', 'test-connect-ues', 'test-16-51-50-334107')  # 20 30
TEST_DIR = Path('..', 'containers', 'test-connect-ues', 'test-15-46-45-451074')  # 10 30
docker_stats_file = utils.find_in_iterdir(TEST_DIR, 'docker_stats')
general_file = utils.find_in_iterdir(TEST_DIR, 'general')

def read_docker_stats(start_time, end_time):
    df_dict = {
        'timestamp': [],
        'nf_name': [],
        'cpu': [],
        'mem': [],
        'net_io_tx': [],
        'net_io_rx': [],
        'block_io_tx': [],
        'block_io_rx': [],
    }

    NUMERIC_RES_REGEX = '^\d+\.?\d*'
    pattern = re.compile('([\d.]+)\s*(\w+)')

    def _append_row(timestamp, nf_name, cpu, mem, net_io, block_io):
        def _parse_value_with_unit(raw_val: str):
            raw_val = raw_val.strip()
            val, unit = pattern.match(raw_val).groups()
            val = float(val)
            if unit == 'kB':
                val *= 1024
            elif unit == 'MB':
                val *= 1024*1024
            return val

        cpu = float(cpu[:-1]) / N_THREADS  # docker stats reports %/1cpu
        mem = re.findall(NUMERIC_RES_REGEX, mem)[0]
        mem = float(mem.split('MiB')[0])
        net_io_tx, net_io_rx = net_io.split('/')
        net_io_tx = _parse_value_with_unit(net_io_tx)
        net_io_rx = _parse_value_with_unit(net_io_rx)
        block_io_tx, block_io_rx = block_io.split('/')
        block_io_tx = _parse_value_with_unit(block_io_tx)
        block_io_rx = _parse_value_with_unit(block_io_rx)
        df_dict['timestamp'].append(timestamp)
        df_dict['nf_name'].append(nf_name)
        df_dict['cpu'].append(cpu)
        df_dict['mem'].append(mem)  # MiB
        df_dict['net_io_tx'].append(net_io_tx)
        df_dict['net_io_rx'].append(net_io_rx)
        df_dict['block_io_tx'].append(block_io_tx)
        df_dict['block_io_rx'].append(block_io_rx)


    last_timestamp = datetime(1000, 1, 1, 12, 30)
    TIME_FORMAT = '%H:%M:%S.%f'
    # BREAK_TIMESTAMP =  TODO (read start timestamp and add)

    with open(docker_stats_file, 'r') as ds_fd:
        for line in ds_fd:
            line = line.strip()

            # stop processing if test ended
            if last_timestamp > end_time:
                break

            # parse line
            if HEADER_LINE in line:
                continue
            elif TIMESTAMP_LINE in line:
                last_timestamp = datetime.strptime(line, TIME_FORMAT) + timedelta(0, 1)
                continue
            elif not line:
                continue
            elif last_timestamp < start_time:
                continue

            nf_name, cpu, mem, net_io, block_io, _ = tuple(
                map(lambda s: s.strip(), line.split(',')))
            _append_row(last_timestamp, nf_name, cpu, mem, net_io, block_io)

    df = pd.DataFrame.from_dict(df_dict)
    return df

start_time, n_ues, duration = utils.read_general_file(general_file)
end_time = start_time + timedelta(0, duration+15)

df = read_docker_stats(start_time, end_time)

def _to_ms_numeric_timedelta(delta):
    return delta.astype('timedelta64[us]') / 1e6

FIRST_TIMESTAMP = df['timestamp'].iloc[0]
df['delta'] = _to_ms_numeric_timedelta(df['timestamp'] - FIRST_TIMESTAMP)
df['net_io_txs'] = df['net_io_tx'] / df['delta']
nf_names = df['nf_name'].unique()

def plot(df, nf_names, y, title, ylabel, save=False):
    _, ax = plt.subplots(1)
    for idx, nf_name in enumerate(nf_names):
        if nf_name in DROP_NFS:
            continue

        nf_stats= df[df['nf_name'] == nf_name]
        nf_stats.plot(
            x='delta',
            y=y,
            label=nf_name,
            ax=ax,
            color=COLORS[idx],
            style='--',
            title=title,
            xlabel='Czas [s]',
            ylabel=ylabel,
        )
    if save:
        fn = title.replace(' ', '_')
        plt.savefig(f'{fn}.png')
    plt.show()

# plot(df, nf_names, 'cpu', f'Porównanie obciążenia procesora [docker,{n_ues}ue]', 'Obciążenie systemu [%]', save=True)
# plot(df, nf_names, 'net_io_rx', f'Porównanie obciążenia interfejsów sieciowych [docker_rx {n_ues}ue]', 'Rx [B]', save=True)
# plot(df, nf_names, 'net_io_tx', f'Porównanie obciążenia interfejsów sieciowych [docker_tx {n_ues}ue]', 'Tx [B]', save=True)
# plt.savefig('container_cpu.png')

TEST_DIR = Path('..', 'containers', 'test-uplane', 'test-21-03-32-800792')  # 10 30
docker_stats_file = utils.find_in_iterdir(TEST_DIR, 'docker_stats')
general_file = utils.find_in_iterdir(TEST_DIR, 'general')

start_time, n_ues, duration = utils.read_general_file(general_file, 'VUS', 'STRESS_DURATION')
print(start_time, n_ues, duration)
end_time = start_time + timedelta(0, duration+15)

df = read_docker_stats(start_time, end_time)
print(df)
FIRST_TIMESTAMP = df['timestamp'].iloc[0]
df['delta'] = _to_ms_numeric_timedelta(df['timestamp'] - FIRST_TIMESTAMP)
nf_names = df['nf_name'].unique()
plot(df, nf_names, 'cpu', f'Porównanie obciążenia procesora [docker,{n_ues}ue,test_uplane]', 'Obciążenie systemu [%]', save=True)
plot(df, nf_names, 'mem', f'Porównanie użycia pamięci [docker,{n_ues}ue,test_uplane]', 'Zużycie pamięci [MB]', save=True)
