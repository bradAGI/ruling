"""The command line's own behavior, apart from what each command does."""

import sys

import pytest

from ruling.cli import main


def test_progress_reaches_a_log_file_as_it_happens(tmp_path, capsys):
    """A training run redirected to a file showed nothing until it ended: block buffering."""
    main(["export-decisions", "--ledger", str(tmp_path / "ledger.sqlite"), "--output", str(tmp_path / "out.jsonl")])
    assert sys.stdout.line_buffering


def test_train_with_wandb_refuses_before_loading_anything_when_the_package_is_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ImportError, match=r"ruling\[wandb\]"):
        main(["train", "--data", str(tmp_path / "never-read.jsonl"), "--output", str(tmp_path / "out"), "--holdout",
              "--model", "not-a-model-because-it-must-never-load", "--wandb"])
