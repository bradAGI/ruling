"""Model metadata the API reports, for weights that never came from Hugging Face."""

from datetime import date

from ruling.engine import model_released


def test_local_weights_report_the_date_they_landed(tmp_path):
    """A trained checkpoint is a directory, not a repository id; asking the Hub about it is an error."""
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    assert model_released(str(checkpoint)) == date.today().isoformat()
