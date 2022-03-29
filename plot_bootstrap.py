from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


RES_FILE_CONTAINERS="test-bootstrap-container-results.csv"
RES_FILE_VMS="test-bootstrap-vm-results.csv"

def _to_ms_numeric_timedelta(delta):
    return delta.astype('timedelta64[us]') / 1e6

dateparse = lambda x: datetime.strptime(x, '%H:%M:%S.%f')
df_vms = pd.read_csv(RES_FILE_VMS, parse_dates=[1, 2, 3, 4, 5], date_parser=dateparse)
df_containers = pd.read_csv(RES_FILE_CONTAINERS, parse_dates=[1, 2, 3, 4, 5], date_parser=dateparse)
df_combined = pd.concat([df_vms, df_containers])

# nanoseconds not supported by plot
df_combined['gnb_session_delta'] = \
    _to_ms_numeric_timedelta(df_combined['gnb_session'] - df_combined['start_test'])
df_combined['boot_time_cplane'] = \
    _to_ms_numeric_timedelta(df_combined['cplane_ping'] - df_combined['start_test'])
df_combined['boot_time_uplane'] = \
    _to_ms_numeric_timedelta(df_combined['uplane_ping'] - df_combined['start_test'])
df_combined['pfcp_sesstion_delta'] = \
    _to_ms_numeric_timedelta(df_combined['pfcp_associated'] - df_combined['start_test'])

df_combined.plot.bar(
    x='test_type',
    y=['gnb_session_delta', 'boot_time_cplane', 'boot_time_uplane', 'pfcp_sesstion_delta'],
    rot="0",
    title='Porównanie czasów uruchomienia sieci szkieletowej',
    ylabel='Czas [s]',
    xlabel='Typ wirtualizacji')
plt.savefig('bootstrap_time.png')
plt.show()
