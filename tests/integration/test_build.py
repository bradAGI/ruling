import pytest

from ruling.build import SST5_LEVELS, build_sst5, build_wanli
from ruling.evals import Record

pytestmark = pytest.mark.integration


def test_wanli_rebuilds_the_published_256_rows():
    records = build_wanli()
    assert len(records) == 256
    parsed = [Record.model_validate(r) for r in records]
    assert {r.labels["verdict"] for r in parsed} == {"supported", "insufficient", "contradicted"}
    assert all(r.labels["supported"] == (r.labels["verdict"] == "supported") for r in parsed)


def test_sst5_sample_is_deterministic_and_ordinal():
    a = build_sst5(20)
    b = build_sst5(20)
    assert a == b and len(a) == 20
    parsed = [Record.model_validate(r) for r in a]
    assert all(0 <= r.labels["sentiment"] < len(SST5_LEVELS) for r in parsed)


def test_typesafe_rows_carry_jev_and_reference_answers():
    from ruling.build import build_typesafe
    records = [Record.model_validate(r) for r in build_typesafe()]
    assert len(records) == 102
    for r in records:
        (qid, refs), = r.references.items()
        assert set(refs) == {"jev", "reference"}
        assert sum(refs["jev"].values()) == pytest.approx(1.0, abs=0.02)
        assert qid in r.labels
    families = {r.family for r in records}
    assert families == {"typesafe_security_incidents", "typesafe_agent_trace_observability",
                        "typesafe_invoice_processing", "typesafe_customer_service"}


def test_every_rows_carry_jev_answers_and_author_labels_where_they_exist():
    from ruling.build import build_every
    records = [Record.model_validate(r) for r in build_every()]
    labeled = [r for r in records if r.labels]
    assert sum(len(r.labels) for r in labeled) == 154
    assert sum(len(r.references) for r in records) >= 700
    assert all(all("jev" in refs for refs in r.references.values()) for r in records)
    assert {r.family for r in records} >= {"every_judge-grid", "every_code-rag", "every_inbox-urgency", "every_persona-panel"}


def test_public_corpus_builds_from_every_pinned_source():
    from ruling.corpus import build

    records = build(per_source=12, seed=0)
    assert {r.family for r in records} == {"wanli", "snli", "clinc", "boolq", "emotions", "civil"}
    dev = build(per_source=12, seed=0, split="dev")
    assert {r.family for r in dev} == {"snli", "clinc", "boolq", "civil"}
    for r in records + dev:
        assert set(r.labels) == set(r.questions)
    from ruling.train import _state_key
    assert not {_state_key(r.state) for r in records} & {_state_key(r.state) for r in dev}


def test_dev_set_command_writes_loadable_records(tmp_path):
    from ruling.cli import main
    from ruling.evals import load_dataset

    output = tmp_path / "dev.jsonl"
    main(["build-dataset", "dev", "--output", str(output), "--limit", "4"])
    records = load_dataset(output)
    assert {r.family for r in records} >= {"snli", "clinc", "boolq", "civil", "state_orders"}
