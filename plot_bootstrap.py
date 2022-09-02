from datetime import datetime
from pathlib import Path
import re
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import utils

RES_FILE_CONTAINERS="test-bootstrap-container-results.csv"
RES_FILE_VMS="test-bootstrap-vm-results.csv"

matplotlib.rcParams.update({'font.size': 18})


def _to_ms_numeric_timedelta(delta):
    return delta.astype('timedelta64[us]') / 1e6


def bootstrap_logs_to_row(base_path, vms=False):
    type_ = 'vm' if vms else 'container'
    general_logs = base_path / 'general.log'
    gnb_logs = base_path / 'gnb.log' if not vms else base_path / 'gnb-err.log'
    open5gs_logs = base_path / 'open5gs.log' if not vms else base_path / 'open5gs/upf.log'
    cplane_logs = base_path / 'cplane_alive.log'
    uplane_logs = base_path / 'uplane_alive.log'

    with open(general_logs, 'r') as fd:
        general_fc = fd.read()

    start_test = general_fc.split()[0][:15]  # cut ns
    if start_test == '20:53:40.879868':
        raise Exception()
    start_test = datetime.strptime(start_test, '%H:%M:%S.%f')
    with open(gnb_logs, 'r') as fd:
        sctp_est_line = next(
            l for l in fd.readlines()
            if l.find('SCTP connection established') > -1)
    sctp_est_time = re.findall(r'\d+:\d+:\d+\.\d+', sctp_est_line)[0]
    print(start_test, sctp_est_time)
    sctp_est_time = datetime.strptime(sctp_est_time, '%H:%M:%S.%f')


    with open(open5gs_logs) as fd:
        pfcp_est_line = next(
            l for l in fd.readlines()[::-1]
            if l.find('PFCP associated') > -1)
    pfcp_est_time = re.findall(r'\d+:\d+:\d+\.\d+', pfcp_est_line)[0]
    pfcp_est_time = datetime.strptime(pfcp_est_time, '%H:%M:%S.%f')

    with open(cplane_logs, 'r') as fd:
        cplane_alive_line = next(
            l for l in fd.readlines()
            if l.find('ping successful') > -1)
    cplane_alive_time = re.findall(r'\d+:\d+:\d+\.\d+', cplane_alive_line)[0]
    cplane_alive_time = datetime.strptime(cplane_alive_time, '%H:%M:%S.%f')

    with open(uplane_logs, 'r') as fd:
        uplane_alive_line = next(
            l for l in fd.readlines()
            if l.find('ping successful') > -1)
    uplane_alive_time = re.findall(r'\d+:\d+:\d+\.\d+', uplane_alive_line)[0]
    uplane_alive_time = datetime.strptime(uplane_alive_time, '%H:%M:%S.%f')

    return {
        'type': type_,
        'start_test': start_test,
        'sctp_est_time': sctp_est_time,
        'pfcp_est_time': pfcp_est_time,
        'cplane_alive_time': cplane_alive_time,
        'uplane_alive_time': uplane_alive_time
    }


# y = ['gnb_session_delta', 'boot_time_cplane', 'boot_time_uplane', 'pfcp_sesstion_delta']
# # y.reverse()
# df_combined.plot.bar(
#     x='test_type',
#     y=y,
#     rot="0",
#     title='Porównanie czasów uruchomienia sieci szkieletowej',
#     ylabel='Czas [s]',
#     xlabel='Typ wirtualizacji')
# plt.savefig('bootstrap_time.png')
# plt.show()


df_dict = {
    'type': [],
    'start_test': [],
    'sctp_est_time': [],
    'pfcp_est_time': [],
    'cplane_alive_time': [],
    'uplane_alive_time': []
}

container_tests_path = Path('../containers/test-bootstrap/')
vms_tests_path = Path('../vms/test-bootstrap-no-shared/')
tests = 20
rows = []

for tidx, test_path in enumerate(container_tests_path.iterdir()):
    if tidx > tests:
        break
    if not test_path.name.startswith('test'):
        continue
    print(f'Reading {test_path}...')
    row = bootstrap_logs_to_row(test_path)
    rows.append(row)
    print(row)

for tidx, test_path in enumerate(vms_tests_path.iterdir()):
    if tidx > tests:
        break
    if not test_path.name.startswith('test'):
        continue

    print(f'Reading {test_path}...')
    rows.append(bootstrap_logs_to_row(test_path, vms=True))

for row in rows:
    for key in df_dict.keys():
        df_dict[key].append(row[key])
df = pd.DataFrame(df_dict)
print(df)

df['sctp_delta'] = _to_ms_numeric_timedelta(df['sctp_est_time'] - df['start_test'])
df['boot_time_cplane'] = _to_ms_numeric_timedelta(df['cplane_alive_time'] - df['start_test'])
df['boot_time_uplane'] = _to_ms_numeric_timedelta(df['uplane_alive_time'] - df['start_test'])
df['pfcp_delta'] = _to_ms_numeric_timedelta(df['pfcp_est_time'] - df['start_test'])

print(df)
df_stats = df.groupby(['type']).agg(['mean', 'std']).reset_index()
print(df_stats)
utils.add_stat_err(df_stats, 'sctp_delta', tests)
utils.add_stat_err(df_stats, 'pfcp_delta', tests)
utils.add_stat_err(df_stats, 'boot_time_uplane', tests)
utils.add_stat_err(df_stats, 'boot_time_cplane', tests)
# print(df_stats.columns)
df_stats.columns = df_stats.columns.get_level_values(0) + '_' + df_stats.columns.get_level_values(1)
# print(df_stats.columns)
df_means = df_stats[['boot_time_cplane_mean', 'boot_time_uplane_mean', 'pfcp_delta_mean', 'sctp_delta_mean']]
yerrs = df_stats[['boot_time_cplane_err', 'boot_time_uplane_err', 'pfcp_delta_err', 'sctp_delta_err']]
labels = ['uruchomienie AMF', 'uruchomienie UPF', 'nawiązanie\nsesji PFCP', 'nawiązanie\nsesji SCTP']

width = 0.4
idxs = list(range(len(df_means.iloc[0])))
idxs_docker = [el - width/2 for el in idxs]
idxs_kvm = [el + width/2 for el in idxs]
fig, ax = plt.subplots(figsize=(14,8))
ax.bar(idxs_docker, df_means.iloc[0], width=width, color=utils.COLORS['docker'], yerr=yerrs.iloc[0], capsize=5)
ax.bar(idxs_kvm, df_means.iloc[1], width=width, color=utils.COLORS['kvm'], yerr=yerrs.iloc[1], capsize=5)
ax.set_xticks(idxs)
ax.set_xticklabels(labels)
ax.set_ylabel('Czas [s]')
ax.set_xlabel('Rodzaj pomiaru')
# plt.xticks(rotation=15)
ax.legend(['docker', 'kvm'])
plt.savefig('bootstrap_time_v2.png', bbox_inches="tight")
plt.show()
