from pathlib import Path
import utils
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


test_dir_vms = Path('..', 'vms-split', 'test-rtt')
test_dir_containers = Path('..', 'containers', 'test-rtt')

SUMMARY_REGEX = r'^rtt'
TIME_RGX = r'(?<=time=)[\w\.]+'

def read_ping_log(ping_file, test_type):
    with open(ping_file, 'r') as ping_fd:
        stats = next(
            (line
            for line in ping_fd
            if re.search(SUMMARY_REGEX, line)), None
        )
        # def _get_time(line):
        #     match = re.search(TIME_RGX, str(line))
        #     return match.group() if match else None

        # stats = [float(_get_time(line)) for line in ping_fd if _get_time(line)]

    if not stats:
        return pd.DataFrame()

    min_rtt, avg_rtt, max_rtt, mdev_rtt = \
        tuple(map(float, stats.split()[3].split('/')))  # retrieve min/avg/max/mdev
    stat_df = pd.DataFrame({
        'min_rtt': [min_rtt],
        'avg_rtt': [avg_rtt],
        'max_rtt': [max_rtt],
        'mdev_rtt': [mdev_rtt],
        'type': test_type
    })

    return stat_df


def read_sample(sample_dir, test_type):
    ping_log_file = utils.find_in_iterdir(sample_dir, 'ping')
    if not ping_log_file:  # needed for working during test run
        return pd.DataFrame()

    return read_ping_log(ping_log_file, test_type)


def prepare_df(test_dir_vms, test_dir_containers):
    stat_df_vms = utils.concat_multiple_logs(test_dir_vms, read_sample, test_type='kvm')  #https://stats.stackexchange.com/questions/55999/is-it-possible-to-find-the-combined-standard-deviation
    stat_df_containers = utils.concat_multiple_logs(test_dir_containers, read_sample, test_type='docker')
    df_concat = pd.concat((stat_df_vms, stat_df_containers))
    return df_concat


df = prepare_df(test_dir_vms, test_dir_containers)
df_mean = df.groupby(['type']).agg(['mean', 'std']).reset_index()
n_samples = len(df) // 2
utils.add_stat_err(df_mean, 'avg_rtt', n_samples)
df_mean.columns = df_mean.columns.map('_'.join)
df_mean.plot.bar(
    x='type_',
    y='avg_rtt_mean',
    rot=0,
    color=utils.COLORS.values(),
    xlabel='Typ wirtualizacji',
    ylabel='RTT [ms]',
    title='Porównanie opóźnienia płaszczyzny użytkownika',
    yerr='avg_rtt_err',
    capsize=2,
    legend=False
)
plt.savefig('user_plane_rtt.png')
plt.show()
