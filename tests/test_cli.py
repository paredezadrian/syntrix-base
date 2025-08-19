import subprocess, sys


def test_cli_train_help():
    out = subprocess.check_output([sys.executable, '-m', 'syntrix.cli_train', '--help']).decode()
    assert 'syntrix.train' in out


def test_cli_sample_help():
    out = subprocess.check_output([sys.executable, '-m', 'syntrix.cli_sample', '--help']).decode()
    assert 'syntrix.sample' in out


