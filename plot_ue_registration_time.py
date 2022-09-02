from pathlib import Path
import utils
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib

matplotlib.rcParams.update({'font.size': 24})

PDU_SESSION_REQ = 'Sending PDU Session Establishment Request'
PDU_SESSION_RESP = 'PDU Session establishment is successful'
REGISTRATION_REQ = 'Sending Initial Registration'
REGISTRATION_RESP = 'Registration accept received'

TIME_RGX = r'(?<=\[).+?(?=\])'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

def read_ue_log(ue_log_file, test_type):

    def get_time(line):
        date = re.search(TIME_RGX, line).group()
        return datetime.strptime(date, TIME_FORMAT)

    with open(ue_log_file, 'r') as log_fd:
        for line in log_fd:
            if line.startswith('+') or line.startswith('UERANSIM'):
                continue

            if REGISTRATION_REQ in line:
                reg_st = get_time(line)
            if REGISTRATION_RESP in line:
                reg_et = get_time(line)
            elif PDU_SESSION_REQ in line:
                pdu_st = get_time(line)
            elif PDU_SESSION_RESP in line:
                pdu_et = get_time(line)
                break

    reg_delta = (reg_et - reg_st).microseconds / 10**3
    pdu_delta = (pdu_et - pdu_st).microseconds / 10**3
    stat_df = pd.DataFrame({
        'msr_type': ['pdu', 'registration'],
        'delta': [pdu_delta, reg_delta],
        'type': test_type
    })
    return stat_df


def read_sample(sample_dir, test_type):
    ue_log_file = utils.find_in_iterdir(sample_dir, 'ue')
    if not ue_log_file:  # needed for working during test run
        return pd.DataFrame()

    return read_ue_log(ue_log_file, test_type)

def prepare_df(test_dir_vms, test_dir_containers):
    stat_df_vms = utils.concat_multiple_logs(test_dir_vms, read_sample, test_type='kvm')  #https://stats.stackexchange.com/questions/55999/is-it-possible-to-find-the-combined-standard-deviation
    stat_df_containers = utils.concat_multiple_logs(test_dir_containers, read_sample, test_type='docker')  #https://stats.stackexchange.com/questions/55999/is-it-possible-to-find-the-combined-standard-deviation
    df_concat = pd.concat((stat_df_vms, stat_df_containers))
    return df_concat

BAR_WIDTH = 0.2       # the width of the bars

test_dir_vms = Path('..', 'vms-split', 'test-session')
test_dir_containers = Path('..', 'containers', 'test-session-v2')
df = prepare_df(test_dir_vms, test_dir_containers)
df_mean = df.groupby(['type', 'msr_type']).agg(['mean', 'std']).reset_index()
n_samples = len(df) // 2
utils.add_stat_err(df_mean, 'delta', n_samples)
df_mean.columns = df_mean.columns.map('_'.join)

# plot
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111)
ind = np.arange(2)
delta_names = ['Rejestracja', 'Zestawienie sesji PDU']
dn = ['a', 'b', 'c','d']
docker_df = df_mean[df_mean['type_'] == 'docker'].sort_values(by='delta_mean')
kvm_df = df_mean[df_mean['type_'] == 'kvm'].sort_values(by='delta_mean')
docker_plot = ax.bar(ind, docker_df['delta_mean'], BAR_WIDTH, color='royalblue', yerr=docker_df['delta_err'], capsize=5)
vm_plot = ax.bar(ind+BAR_WIDTH, kvm_df['delta_mean'], BAR_WIDTH, color='seagreen', yerr=kvm_df['delta_err'], capsize=5)
ax.legend((docker_plot[0], vm_plot[0]), ('docker', 'kvm'))
ax.set_ylabel('Czas [ms]')
ax.set_xlabel('Rodzaj pomiaru')
# ax.set_title('Porównanie czasu rejestracji użytkownika')
ax.set_xticks(ind + BAR_WIDTH / 2)
ax.set_xticklabels(delta_names)
plt.savefig('registration_times_v2.png')
plt.show()
