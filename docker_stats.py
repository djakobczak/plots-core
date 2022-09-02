from pathlib import Path
import re

from datetime import datetime, timedelta
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas as pd

import utils

matplotlib.rcParams.update({'font.size': 28})


HEADER_LINE = 'NAME'
TIMESTAMP_LINE = ':'
N_THREADS = 12

DROP_NFS = ['nr_gnb']  # do not plot these nfs
COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k',
          'lime', 'dodgerblue', 'peru', 'indigo',
          'crimson', 'slategray']

BEFORE_TEST_TIME = -3
AFTER_TEST_TIME = -5

def read_docker_stats(start_time, end_time, docker_stats_file):
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
                val *= 1000
            elif unit == 'kiB':
                val *= 1024
            elif unit == 'MB':
                val *= 1000*1000
            elif unit == 'MiB':
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
    df['net_io_tx_per_s'] = df.groupby('nf_name')['net_io_tx'].diff().fillna(0)
    df['net_io_rx_per_s'] = df.groupby('nf_name')['net_io_rx'].diff().fillna(0)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df[df['nf_name'] == 'amf'][['timestamp', 'net_io_tx_per_s', 'net_io_tx']])
    return df


def plot(df, nf_names, y, title, ylabel, ymax, tick=0.5, save=False, yerr='cpu_err'):
    _, ax = plt.subplots(1, figsize=(14, 10))
    for idx, nf_name in enumerate(nf_names):
        if nf_name in DROP_NFS:
            continue
        nf_label = nf_name
        if nf_name == 'mongo':
            nf_label = 'db'

        nf_stats= df[df['nf_name_'] == nf_name]
        nf_stats.plot(
            x='delta_mean',
            y=y,
            label=nf_label,
            ax=ax,
            color=utils.get_color(nf_label),
            fmt='--',
            # title=title,
            xlabel='Czas [s]',
            ylabel=ylabel,
            yerr=yerr,
            capsize=5,
        )
    ax = plt.gca()
    ax.set_ylim([0, ymax+0.01])
    ax.set_yticks(numpy.arange(0, ymax+0.01, tick))
    plt.legend(loc = "upper right")
    if save:
        fn = title.replace(' ', '_')
        plt.savefig(f'{fn}_v3.png')
    plt.show()


def _to_ms_numeric_timedelta(delta):
    return delta.astype('timedelta64[us]') / 1e6


def read_sample(sample_dir: Path, end_time_extra_time: int = 5):
    docker_stats_file = utils.find_in_iterdir(sample_dir, 'docker_stats')
    general_file = utils.find_in_iterdir(sample_dir, 'general')
    start_time, n_ues, duration = utils.read_general_file(general_file, n_iterations_line='DURATION')  #N_ITERATIONS DURATION
    start_time = start_time - timedelta(0, BEFORE_TEST_TIME)
    end_time = start_time + timedelta(0, duration + AFTER_TEST_TIME)

    df = read_docker_stats(start_time, end_time, docker_stats_file)
    first_timestamp = df['timestamp'].iloc[0]
    df['delta'] = _to_ms_numeric_timedelta(df['timestamp'] - first_timestamp)
    df = df.drop('timestamp', axis=1)
    df.index.name = 'index'
    return df


def prepare_df(test_dir: Path):

    def _get_number_of_samples(df, uniq_name):
        df = df[df['nf_name'] == uniq_name]
        df = df[df['delta'] == 0]
        return len(df)

    stat_df = utils.concat_multiple_logs(test_dir, read_sample).fillna(0)  # long format
    pd.set_option('display.max_rows', 500)
    # stat_df.loc[(stat_df.nf_name == 'upf') & (stat_df.cpu < 0.1), 'cpu'] = 1.6  # ONLY UPLANE MEASUREMENT
    # print('[DEBUG] stat_df:', stat_df.loc[stat_df.nf_name == 'upf', 'cpu' < 0.1])
    # print('[DEBUG] stat_df:', stat_df[stat_df.nf_name == 'upf']['cpu'])
    # print('[DEBUG] stat_df:', stat_df[stat_df.nf_name == 'upf']['cpu'].replace(0.0, 1.5))
    # print('[DEBUG] stat_df:', stat_df.loc[stat_df['nf_name'] == 'upf'])
    # print('[DEBUG] stat_df:', stat_df.loc[stat_df.nf_name == 'upf', 'cpu'].replace(0.0, 1.5))
    tidy_df = stat_df.groupby(['index', 'nf_name']).agg(['mean', 'std']).reset_index()
    n_samples = _get_number_of_samples(stat_df, 'amf')
    tidy_df['cpu', 'ci_lower'], tidy_df['cpu', 'ci_upper'], tidy_df['cpu', 'err'] = utils.calc_conf_int(tidy_df, 'cpu', n_samples)
    tidy_df['mem', 'ci_lower'], tidy_df['mem', 'ci_upper'], tidy_df['mem', 'err'] = utils.calc_conf_int(tidy_df, 'mem', n_samples)
    utils.add_stat_err(tidy_df, 'net_io_tx_per_s', n_samples)
    utils.add_stat_err(tidy_df, 'net_io_rx_per_s', n_samples)
    return tidy_df


test_dir_30_30 = Path('..', 'containers', 'test-connect-ues', '30_30')
test_dir_20_30 = Path('..', 'containers', 'test-connect-ues', '20_30')
test_dir_10_30 = Path('..', 'containers', 'test-connect-ues', '10_30')
test_dir_uplane = Path('..', 'containers', 'test-uplane')

IDLE = False
DO_10_30 = False
DO_20_30 = False
DO_30_30 = False
TEST_UPLANE = True

# df_uplane = prepare_df(test_dir_uplane)

def prepare_and_plot(df, n_ues, ymax, tick, save=True):
    nf_names = df['nf_name'].unique()
    df = df[~df['nf_name'].str.contains('ue')]
    df = df[~df['nf_name'].str.contains('gnb')]
    df.columns = df.columns.map('_'.join)
    plot(df, nf_names, 'cpu_mean',
         f'Porównanie obciążenia procesora [docker,{n_ues}ue]', 'Użycie CPU [%]',
         ymax=ymax,
         tick=tick,
         save=save)

# TEST_IDLE_DIR =  Path('..', 'containers', 'test-idle')
# df_idle = prepare_df(TEST_IDLE_DIR)
if DO_10_30:
    df_10_30 = prepare_df(test_dir_10_30)
    prepare_and_plot(df_10_30, 10, 2.0, 0.25)

if DO_20_30:
    df_20_30 = prepare_df(test_dir_20_30)
    prepare_and_plot(df_20_30, 20, 4.0, 0.5)

if DO_30_30:
    df_30_30 = prepare_df(test_dir_30_30)
    prepare_and_plot(df_30_30, 30, 6.0, 0.75)

if TEST_UPLANE:
    df_uplane = prepare_df(test_dir_uplane)
    print(df_uplane)
    prepare_and_plot(df_uplane, 'uplane', 2.0, 0.25)

# df_20_30 = prepare_df(test_dir_20_30)
# df_30_30 = prepare_df(test_dir_30_30)


# nf_names = df_10_30['nf_name'].unique()
# df_10_30 = df_10_30[~df_10_30['nf_name'].str.contains('ue')]
# df_10_30 = df_10_30[~df_10_30['nf_name'].str.contains('gnb')]
# mean_df = df_idle.groupby('nf_name').mean()
# mean_df.columns = mean_df.columns.map('_'.join)
# df_10_30.columns = df_10_30.columns.map('_'.join)
# mean_df.to_csv('mean_idle_docker.csv')
# print(df)
# print(df.columns)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df[df['nf_name_'] == 'ausf'])
# plot(df, nf_names, 'cpu_mean', f'Porównanie obciążenia procesora [docker,uplane]', 'Użycie CPU [%]', save=save)
# plot(df, nf_names, 'cpu_mean', f'Porównanie obciążenia procesora [docker,20ue]', 'Użycie CPU [%]', save=save)
# plot(df_10_30, nf_names, 'net_io_tx_per_s_mean', f'Porównanie obciążenia procesora [docker,10ue,tx]', 'Tx [B/s]', yerr='net_io_tx_per_s_err', save=save)
# plot(df_10_30, nf_names, 'net_io_rx_per_s_mean', f'Porównanie obciążenia procesora [docker,10ue,rx]', 'Rx [B/s]', yerr='net_io_rx_per_s_err', save=save)
# plot(df, nf_names, 'net_io_tx_per_s_mean', f'Porównanie obciążenia interfejsów docker 30ue tx', 'Tx [B/s]', yerr='net_io_tx_per_s_err', save=save)
# plot(df, nf_names, 'net_io_rx_per_s_mean', f'Porównanie obciążenia interfejsów docker 30ue rx', 'Rx [B/s]', yerr='net_io_rx_per_s_err', save=save)

