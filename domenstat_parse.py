from datetime import datetime, timedelta
import pandas as pd
import numpy as np

TIMESTAMP_LINE = 'timestamp'

def read_dommemstat(stat_path, end_time, start_time):
    df_dict = {
        'timestamp': [],
        'nf_name': [],
        'mem': [],
    }
    last_timestamp = datetime(1000, 1, 1, 12, 30)
    TIME_FORMAT = '%H:%M:%S.%f'
    with open(stat_path, 'r') as stat_file:
        for line in stat_file:
            key, val = line.split()
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

            df_dict['timestamp'] = last_timestamp
            df_dict['nf_name'] = key
            df_dict['mem'] = val / 1024
    print(df_dict)
