from datetime import datetime
from pathlib import Path


def find_in_iterdir(dir_path: Path, fn_part: str) -> Path:
    dir_path = Path(dir_path)
    return next(
        path for path in dir_path.iterdir()
        if fn_part in str(path)
    )


def read_general_file(path):
    timestamp_line = 'started'
    n_ues_line = 'N_UES'
    n_iterations_line = 'N_ITERATIONS'
    timestamp = ''
    n_ues = '-1'
    n_iterations = '-1'
    with open(path, 'r') as fd:
        for line in fd.readlines():
            if timestamp_line in line:
                timestamp = line[:8].replace('-', ':')
                timestamp = datetime.strptime(timestamp, '%H:%M:%S')
            elif n_ues_line in line:
                n_ues = int(line.split(':')[1].strip())
            elif n_iterations_line in line:
                n_iterations = int(line.split(':')[1].strip())
    return timestamp, n_ues, n_iterations