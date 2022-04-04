from pathlib import Path
import subprocess

from datetime import datetime, timedelta
import sys
import matplotlib.pyplot as plt
import pandas as pd

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

# PREPARE_VIRT_TOP_SH = 'prepare_virt_top_data.sh'
# virt_top_fn = virt_top_file.name
# cmd = ['bash', PREPARE_VIRT_TOP_SH, f'{TEST_DIR}/{virt_top_fn}']
# cmd_res = subprocess.run(cmd, capture_output=True)
# print(cmd_res)


def _first_el_of_col(df, col_name):
    return df[col_name].iloc[0]


def prepare_virt_top(virt_top_path):
    PREPARE_VIRT_TOP_SH = 'prepare_virt_top_data.sh'
    # virt_top_fn = virt_top_file.name
    cmd = ['bash', PREPARE_VIRT_TOP_SH, str(virt_top_path)]
    subprocess.run(cmd, capture_output=True)


def read_virt_top(sample_dir):
    pd.options.mode.chained_assignment = None  # default='warn' fixme
    virt_top_file = utils.find_in_iterdir(sample_dir, 'host_virt')
    general_file = utils.find_in_iterdir(sample_dir, 'general')
    prepare_virt_top(virt_top_file)

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


def prepare_df(test_dir):
    stat_df = utils.concat_multiple_logs(test_dir, read_virt_top)  # long format
    tidy_df = stat_df.groupby(['index']).agg(['mean', 'std']).reset_index()
    nf_cpu_perc_cols = [col for col in tidy_df if col[0].startswith('%CPU')][1:]
    # !TODO add confidance intervals to df
    print(tidy_df[nf_cpu_perc_cols])


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
# print(df)
test_dir = Path('..', 'vms-split', 'test-idle')
df = prepare_df(test_dir)
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
