from ruling.distill import parse_questions, writer_messages


def test_parses_a_fenced_json_object_and_keeps_only_valid_questions():
    text = """<think>
plan
</think>
Here you go:
```json
{"questions": {
  "wants_refund": {"type": "noul", "instructions": "The writer asks for money back."},
  "tone": {"type": "score", "instructions": "How upset is the writer?", "criteria": ["calm", "annoyed", "furious"]},
  "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Charges", "shipping": "Deliveries"}},
  "broken": {"type": "choice", "instructions": "Only one option", "criteria": {"x": null}},
  "unknown": {"type": "rank", "instructions": "Rank them"}
}}
```"""
    questions = parse_questions(text, max_options=12)
    assert set(questions) == {"wants_refund", "tone", "team"}
    assert questions["team"].criteria == {"billing": "Charges", "shipping": "Deliveries"}


def test_rejects_choices_with_too_many_options_and_bad_json():
    many = {f"o{i}": None for i in range(30)}
    import json
    text = json.dumps({"questions": {"big": {"type": "choice", "instructions": "q", "criteria": many},
                                     "ok": {"type": "noul", "instructions": "q"}}})
    assert set(parse_questions(text, max_options=12)) == {"ok"}
    assert parse_questions("not json at all", max_options=12) == {}
    assert parse_questions('{"questions": []}', max_options=12) == {}


def test_writer_prompt_carries_the_state_and_count():
    messages = writer_messages({"order": 42, "status": "late"}, 6)
    assert '"order": 42' in messages[-1]["content"] and "6" in messages[-1]["content"]


def test_a_resumed_run_answers_only_the_records_not_yet_written(tmp_path):
    from ruling.distill import unanswered
    from ruling.evals import Record

    question = {"q": {"type": "noul", "instructions": "It is about money."}}
    too_long, paid, shipped = (Record(state=s, questions=question) for s in ("word " * 9, "Paid twice.", "Lost parcel."))
    written = tmp_path / "answers.jsonl"
    # A record too long to answer leaves no line, so the first line written belongs to the second record.
    answered = paid.model_copy(update={"references": {"q": {"target": {"no": 0.2, "yes": 0.8}}}})
    written.write_text(answered.model_dump_json() + "\n")
    assert unanswered([too_long, paid, shipped], written) == [too_long, shipped]
    assert unanswered([paid], tmp_path / "missing.jsonl") == [paid]
