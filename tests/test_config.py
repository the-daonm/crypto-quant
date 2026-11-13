import io
import sys
from pathlib import Path

import pytest
import yaml

from cq import pipeline
from cq.pipeline import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_falls_back_to_packaged_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no config.yaml here
    cfg = load_config()
    assert cfg["symbols"] and cfg["timeframes"]
    assert {"fee", "threshold", "band", "horizon", "holdout_start", "walk_forward"} <= cfg.keys()


def test_config_in_working_directory_overrides_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("fee: 0.5\n")
    assert load_config() == {"fee": 0.5}


def test_explicit_path_wins_over_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("fee: 0.5\n")
    other = tmp_path / "other.yaml"
    other.write_text("fee: 0.25\n")
    assert load_config(other) == {"fee": 0.25}


def test_repository_config_matches_packaged_default():
    """`config.yaml` at the repository root and the packaged default must not drift apart."""
    repo_cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert repo_cfg == load_config(REPO_ROOT / "src" / "cq" / "default_config.yaml")


def test_cli_help_works_without_a_config_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        pipeline.main(["--help"])
    assert exit_info.value.code == 0
    assert "--config" in capsys.readouterr().out


def test_data_and_results_default_to_working_directory(tmp_path, monkeypatch):
    from cq import data

    monkeypatch.chdir(tmp_path)
    assert data._cache_path("binance", "BTC/USDT", "1d").resolve().parent == tmp_path / "data"
    assert pipeline.RESULTS_DIR.resolve() == tmp_path / "results"


def test_cli_help_survives_a_legacy_output_encoding(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    buffer = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(buffer, encoding="cp1252"))
    with pytest.raises(SystemExit) as exit_info:
        pipeline.main(["--help"])
    sys.stdout.flush()
    assert exit_info.value.code == 0
    assert b"--config" in buffer.getvalue()


def test_cli_reports_a_missing_config_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        pipeline.main(["--config", "missing.yaml"])
    assert exit_info.value.code == 2
    assert "config file not found: missing.yaml" in capsys.readouterr().err
