import os
import platform
import socket
import subprocess
import sys
import uuid
import warnings
from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt

REQUIRED_PYTHON_VERSION = (3, 8)

DEBUG = os.environ.get('DEBUG', '').lower() in ('1', 'y', 'yes', 't', 'true')

if sys.version_info < REQUIRED_PYTHON_VERSION:
    exit(f'Python 3.8 or higher version is required')

host_name: str = socket.gethostname()
system_name: str = platform.system().lower()

if system_name != 'darwin':
    plt.switch_backend('agg')

app_dir = Path(__file__).resolve().parent
project_dir = app_dir.parent
app_name = app_dir.name

system_data_dir = Path.home() / 'data'
system_data_dir.mkdir(parents=True, exist_ok=True)

project_data_dir = project_dir / 'data'
project_data_dir.mkdir(parents=True, exist_ok=True)


def git_describe(num: int = 10) -> str:
    try:
        version = subprocess.check_output(f'git describe --tags --dirty --abbrev={num} --always', shell=True)
        return str(version, encoding='utf-8').strip()
    except subprocess.CalledProcessError:
        warnings.warn('git is not initialized in this project')
        return 'untracked'


def time_date(time_format: str = r'%y%m%d-%H%M%S') -> str:
    return datetime.strftime(datetime.now(), time_format).strip()


def get_out_dir(study: str) -> Path:
    name = f'{time_date()}-{git_describe()}-{str(uuid.uuid4())[:8]}'
    out_dir = project_dir / 'out' / study / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
