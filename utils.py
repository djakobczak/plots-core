from datetime import datetime
from pathlib import Path


def find_in_iterdir(dir_path: Path, fn_part: str) -> Path:
    dir_path = Path(dir_path)
    return next(
        path for path in dir_path.iterdir()
        if fn_part in str(path)
    )


def _strip_seconds(line: str):
    return line.strip()[:-1] if line.strip().endswith('s') else line.strip()


def read_general_file(path, n_ues_line = 'N_UES', n_iterations_line = 'N_ITERATIONS'):
    timestamp_line = 'started'
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
                n_iterations = int(_strip_seconds(line.split(':')[1]))
    return timestamp, n_ues, n_iterations
