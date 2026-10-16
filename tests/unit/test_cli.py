"""The command line's own behavior, apart from what each command does."""

import sys

from ruling.cli import main


def test_progress_reaches_a_log_file_as_it_happens(tmp_path, capsys):
    """A training run redirected to a file showed nothing until it ended: block buffering."""
    main(["export-decisions", "--ledger", str(tmp_path / "ledger.sqlite"), "--output", str(tmp_path / "out.jsonl")])
    assert sys.stdout.line_buffering
