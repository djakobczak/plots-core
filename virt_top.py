from pathlib import Path
import subprocess

from datetime import datetime, timedelta
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utils

# TEST_DIR = Path('..', 'vms-split', 'test-connect-ues', 'test-19-57-12-727809')  # 50 30
# TEST_DIR = Path('..', 'vms-split', 'test-connect-ues', 'test-19-50-19-982548')  # 40 30
# TEST_DIR = Path('..', 'vms-split', 'test-connect-ues', 'test-18-56-26-071935')  # 30 30
# TEST_DIR = Path('..', 'vms-split', 'test-connect-ues', 'test-18-34-04-266416')  # 20 30
# TEST_DIR = Path('..', 'vms-split', 'test-connect-ues', 'test-20-07-59-122517')  # 10 30
TEST_DIR = Path('..', 'vms-split', 'test-uplane', 'test-22-03-03-128862')  # 10 30
# TEST_DIR = Path('..', 'vms-split', 'test-connect-ues', 'test-22-43-15-073106') # old
# virt_top_file = next(path for path in TEST_DIR.iterdir() if 'host_virt' in str(path))

DROP_NFS = ['ue', 'gnb01']  # do not plot these nfs
COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k',
          'lime', 'dodgerblue', 'peru', 'indigo',
          'crimson', 'slategray']
TIMESTAMP_LINE = 'timestamp'


def get_cols_start_with(df, expr):
    return [col for col in df if col[0].startswith(expr)]


def prepare_virt_top(virt_top_path):
    PREPARE_VIRT_TOP_SH = 'prepare_virt_top_data.sh'
    # virt_top_fn = virt_top_file.name
    cmd = ['bash', PREPARE_VIRT_TOP_SH, str(virt_top_path)]
    subprocess.run(cmd, capture_output=True)


def read_dommemstat(virsh_domemm_file, start_time, end_time) -> pd.DataFrame:
    df_dict = {
        'timestamp': [],
        'nf_name': [],
        'mem': [],
    }
    last_timestamp = datetime(1000, 1, 1, 12, 30)
    TIME_FORMAT = '%H:%M:%S.%f'
    with open(virsh_domemm_file, 'r') as stat_file:
        for line in stat_file:
            key, val = line.split(',')
            val = val.strip()

            # stop processing if test ended
            if last_timestamp > end_time:
                break
            # parse line
            elif TIMESTAMP_LINE in key:
                last_timestamp = datetime.strptime(val, TIME_FORMAT) + timedelta(0, 1)  # - (?)
                continue
            elif not line:
                continue
            elif last_timestamp < start_time:
                continue

            df_dict['timestamp'].append(last_timestamp)
            df_dict['nf_name'].append(key)
            df_dict['mem'].append(float(val) / 1024)  # to MiB
    df = pd.DataFrame.from_dict(df_dict)
    return df


# !TODO to decorator
def read_mem_sample(sample_dir: Path):
    virsh_domemm_file = utils.find_in_iterdir(sample_dir, 'dommenstat')
    general_file = utils.find_in_iterdir(sample_dir, 'general')
    stime, n_ues, duration = utils.read_general_file(general_file, n_iterations_line='DURATION')
    etime = stime + timedelta(0, duration + 1)
    df = read_dommemstat(virsh_domemm_file, stime, etime)
    first_timestamp = df['timestamp'].iloc[0]
    df['delta'] = utils._to_ms_numeric_timedelta(df['timestamp'] - first_timestamp)
    df = df.drop('timestamp', axis=1)
    df.index.name = 'index'
    return df


def read_virt_top(sample_dir):
    pd.options.mode.chained_assignment = None  # default='warn' fixme
    virt_top_file = utils.find_in_iterdir(sample_dir, 'host_virt')
    general_file = utils.find_in_iterdir(sample_dir, 'general')
    prepare_virt_top(virt_top_file)

    # should be moved
    start_time, n_ues, duration = utils.read_general_file(general_file, n_iterations_line='DURATION')
    # start_time += timedelta(0, 20)
    end_time = start_time + timedelta(0, duration + 1)

    dateparse = lambda x: datetime.strptime(x, '%H:%M:%S')
    df_virt = pd.read_csv(virt_top_file,
                        parse_dates=[1],
                        date_parser=dateparse)
    mask = df_virt['Time'].between(start_time , end_time)
    df_tidy = df_virt[mask]
    # print(df_tidy['Time'].iloc[0])

    nf_cpu_perc_cols = [col for col in df_tidy if col.startswith('%CPU')][1:]  # cut off host cpu perc
    df_tidy[nf_cpu_perc_cols] = \
        df_tidy[nf_cpu_perc_cols].apply(
        lambda val: (val.str.replace(',','.')).astype('float32')
    )
    nf_net_tx_cols = [col for col in df_tidy if col.startswith('Net TXBY')]
    nf_net_rx_cols = [col for col in df_tidy if col.startswith('Net RXBY')]
    df_tidy['delta'] = (df_tidy['Time'] - df_tidy['Time'].iloc[0]).astype('timedelta64[us]') / 1e6
    for idx, col in enumerate(nf_net_tx_cols):
        df_tidy[f'nf_net_tx_cum.{idx}'] = df_tidy[col].cumsum()
    for idx, col in enumerate(nf_net_rx_cols):
        df_tidy[f'nf_net_rx_cum.{idx}'] = df_tidy[col].cumsum()

    df_tidy.index.name = 'index'
    return df_tidy


def prepare_virt_df(test_dir):
    stat_df = utils.concat_multiple_logs(test_dir, read_virt_top)  # wide format
    # drop summary host stats
    stat_df = stat_df.drop(['%CPU', 'Hostname', 'Arch', 'Physical CPUs', 'Count',
                            'Running', 'Blocked', 'Paused', 'Shutdown', 'Shutoff', 'Active',
                            'Crashed', 'Inactive', 'Total hardware memory (KB)', 'Total memory (KB)',
                            'Total guest memory (KB)', 'Total CPU time (ns)'], axis=1)
    base_col_names = ['delta', '%CPU', 'Domain ID', 'Domain name', '%Mem', 'Block RDRQ',
                      'Block WRRQ', 'Net RXBY', 'nf_net_rx_cum']
    first_cols_names = ['delta', '%CPU.1', 'Domain ID', 'Domain name', '%Mem', 'Block RDRQ',
                        'Block WRRQ', 'Net RXBY', 'nf_net_rx_cum.0']
    df_concat = pd.DataFrame()

    # first case is special
    nf_stats_df = stat_df[first_cols_names]
    nf_stats_df.columns = base_col_names
    df_concat = pd.concat((df_concat, nf_stats_df))
    number_of_nfs = len(stat_df.filter(regex="Domain name.*").columns)

    # wide to long
    for nf_num in range(1, number_of_nfs):
        idx_spec = 1 + nf_num
        rest_cols_names = ['delta', f'%CPU.{idx_spec}', f'Domain ID.{nf_num}', f'Domain name.{nf_num}',
                           f'%Mem.{nf_num}', f'Block RDRQ.{nf_num}', f'Block WRRQ.{nf_num}', f'Net RXBY.{nf_num}',
                           f'nf_net_rx_cum.{nf_num}']
        nf_stats_df = stat_df[rest_cols_names]
        nf_stats_df.columns = base_col_names
        df_concat = pd.concat((df_concat, nf_stats_df))

    # drop
    df_concat = df_concat[df_concat['Domain name'] != 'ue']
    df_concat = df_concat[~df_concat['Domain name'].str.contains('gnb')]

    # we got long multiple stacked (based on number of samples) df
    # !TODO how to calculate grouped conf int
    df_gp = df_concat.groupby(['delta', 'Domain name']).mean().reset_index()
    # !TODO extract to separate func
    n_samples = len(df_gp[df_gp['Domain name'] == 'cp-amf'])
    df_gp = df_gp.groupby('Domain name').agg(['mean', 'std']).reset_index()
    df_gp['%CPU', 'ci_lower'], df_gp['%CPU', 'ci_upper'], df_gp['%CPU', 'err'] = utils.calc_conf_int(df_gp, '%CPU', n_samples)
    return df_gp


def prepare_df_mem(test_dir, **kwargs):
    stat_df = utils.concat_multiple_logs(test_dir, read_mem_sample, **kwargs)
    tidy_df = stat_df.groupby('nf_name').agg(['mean', 'std']).reset_index()
    n_samples = len(stat_df[stat_df['nf_name'] == 'cp-amf'])
    tidy_df['mem', 'ci_lower'], tidy_df['mem', 'ci_upper'], tidy_df['mem', 'err'] = utils.calc_conf_int(tidy_df, 'mem', n_samples)
    return tidy_df


def plot(df, stat, nf_names_cols, ylabel, title, save=False):
    _, ax = plt.subplots(1)
    for idx in range(len(stat)):
        nf_name = df[nf_names_cols[idx]].iloc[0]
        if nf_name in DROP_NFS:
            continue
        df.plot(
            x='delta',
            y=stat[idx],
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


# df = read_virt_top(TEST_DIR)
# test_dir = Path('..', 'vms-split', 'test-idle')
# df = prepare_virt_df(test_dir)

# # dfu = df.reset_index()
# df_vms = df.copy()
# df_vms.columns = df_vms.columns.map('_'.join)
# df_vms.to_csv('mean_idle_vms.csv')

virsh_domemm_file = utils.find_in_iterdir(TEST_DIR, 'dommenstat')
general_file = utils.find_in_iterdir(TEST_DIR, 'general')
stime, n_ues, duration = utils.read_general_file(general_file, n_iterations_line='DURATION')
etime = stime + timedelta(0, duration + 1)

test_dir = Path('..', 'vms-split', 'test-idle')
print(read_dommemstat(virsh_domemm_file, stime, etime))
df_mem = prepare_df_mem(test_dir)
df_mem = df_mem[~df_mem['nf_name'].str.contains('ue')]
df_mem = df_mem[~df_mem['nf_name'].str.contains('gnb')]
df_mem.columns = df_mem.columns.map('_'.join)
df_mem.to_csv('mem_vms_idle.csv')


# nf_names_cols = [col for col in df if col.startswith('Domain name')]
# nf_cpu_perc_cols = [col for col in df if col.startswith('%CPU')][1:]
# nf_net_tx_cum_cols = [col for col in df if col.startswith('nf_net_tx_cum')]
# nf_net_rx_cum_cols = [col for col in df if col.startswith('nf_net_rx_cum')]
# # cpu plot
# plot(
#     df,
#     nf_cpu_perc_cols,
#     nf_names_cols,
#     'Obciążenie systemu [%]',
#     f'Porównanie obciążenia procesora przez funkcje sieciowe [vms uplane_test]',
#     save=True
# )
# plot(
#     df_tidy,
#     nf_net_rx_cum_cols,
#     nf_names_cols,
#     'Rx [B]',
#     f'Porównanie obciążenia interfejsów sieciowych [VM Rx {n_ues}ue uplane_test]',
#     save=True
# )
# plot(
#     df_tidy,
#     nf_net_tx_cum_cols,
#     nf_names_cols,
#     'Tx [B]',
#     f'Porównanie obciążenia interfejsów sieciowych [VM Tx {n_ues}ue uplane_test]',
#     save=True
# )
