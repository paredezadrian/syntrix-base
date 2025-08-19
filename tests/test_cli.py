import subprocess
import sys


def test_cli_train_help():
    out = subprocess.check_output(
        [sys.executable, "-m", "syntrix.cli_train", "--help"]
    ).decode()
    assert "syntrix.train" in out


def test_cli_sample_help():
    out = subprocess.check_output(
        [sys.executable, "-m", "syntrix.cli_sample", "--help"]
    ).decode()
    assert "syntrix.sample" in out


def test_cli_train_help_verbose():
    out = subprocess.check_output(
        [sys.executable, "-m", "syntrix.cli_train", "--help"]
    ).decode()
    assert "-v" in out and "--verbose" in out


def test_cli_eval_help():
    out = subprocess.check_output(
        [sys.executable, "-m", "syntrix.cli_eval", "--help"]
    ).decode()
    assert "syntrix.eval" in out


def test_cli_config_help():
    out = subprocess.check_output(
        [sys.executable, "-m", "syntrix.cli_config", "--help"]
    ).decode()
    assert "syntrix.config" in out
