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
TEST_DIR = Path('..', 'vms-split', 'test-connect-ues', 'test-20-07-59-122517')  # 10 30
# TEST_DIR = Path('..', 'vms-split', 'test-connect-ues', 'test-22-43-15-073106') # old
# TEST_DIR = Path('..', 'vms-split', 'test-connect-ues', '')  # 10 30
virt_top_file = next(path for path in TEST_DIR.iterdir() if 'host_virt' in str(path))

PREPARE_VIRT_TOP_SH = 'prepare_virt_top_data.sh'
virt_top_fn = virt_top_file.name
cmd = ['bash', PREPARE_VIRT_TOP_SH, f'{TEST_DIR}/{virt_top_fn}']
cmd_res = subprocess.run(cmd, capture_output=True)
print(cmd_res)

dateparse = lambda x: datetime.strptime(x, '%H:%M:%S')
df_virt = pd.read_csv(virt_top_file,
                      parse_dates=[1],
                      date_parser=dateparse)

def _first_el_of_col(df, col_name):
    return df[col_name].iloc[0]

# print(df_virt.info())
# print(df_virt)

# drop_nfs_colnames = list(filter(
#     lambda nf_col_name: _first_el_of_col(df_virt, nf_col_name) in DROP_NFS,
#     nf_names_cols)
# )
# df_virt.drop(columns=drop_nfs_colnames)
# nf_names_cols = list(filter(lambda el: el not in drop_nfs_colnames, nf_names_cols))
# print(nf_names_cols)
# print(df_virt)
# sys.exit()

start_time, n_ues, duration = utils.read_general_file(f'{TEST_DIR}/general.log')

# START_TIME_SCRIPT="get_test_start.sh"
# cmd = ['bash', START_TIME_SCRIPT, f'{TEST_DIR}/general.log', 's']
# start_time = datetime.strptime(subprocess.run(cmd, capture_output=True)\
#     .stdout.decode("utf-8").strip(), '%H:%M:%S')
# test_time = 20 # TODO grep
end_time = start_time + timedelta(0, duration)

mask = df_virt['Time'].between(start_time, end_time)
df_tidy = df_virt[mask]
# print(df_tidy['Time'].iloc[0])

nf_names_cols = [col for col in df_tidy if col.startswith('Domain name')]
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
nf_net_tx_cum_cols = [col for col in df_tidy if col.startswith('nf_net_tx_cum')]
nf_net_rx_cum_cols = [col for col in df_tidy if col.startswith('nf_net_rx_cum')]

DROP_NFS = ['ue', 'gnb01']  # do not plot these nfs
COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k',
          'lime', 'dodgerblue', 'peru', 'indigo',
          'crimson', 'slategray']

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
    # !TODO deploy mongo on dedicated vm
    if save:
        fn = title.replace(' ', '_')
        plt.savefig(f'{fn}.png')
    plt.show()

# cpu plot
plot(
    df_tidy,
    nf_cpu_perc_cols,
    nf_names_cols,
    'Obciążenie systemu [%]',
    f'Porównanie obciążenia procesora przez funkcje sieciowe [VM {n_ues}ue]',
    save=True
)
plot(
    df_tidy,
    nf_net_rx_cum_cols,
    nf_names_cols,
    'Rx [B]',
    f'Porównanie obciążenia interfejsów sieciowych [VM Rx {n_ues}ue]',
    save=True
)
plot(
    df_tidy,
    nf_net_tx_cum_cols,
    nf_names_cols,
    'Tx [B]',
    f'Porównanie obciążenia interfejsów sieciowych [VM Tx {n_ues}ue]',
    save=True
)
