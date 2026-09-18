"""Command line: serve the API, ask one question, evaluate, calibrate, or build datasets."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
import uvicorn

from ruling import corpus, states
from ruling.against import collect_live
from ruling.build import build_every, build_sst5, build_typesafe, build_wanli, write
from ruling.calibration import Calibration
from ruling.config import Settings, configure_memory
from ruling.engine import Engine
from ruling.evals import collect, fit, load_dataset, report, write_json, write_predictions
from ruling.questions import Choice, Noul, Score, SystemOneRequest
from ruling.hosted import OpenRouterEngine
from ruling.server import app_from_env, build_engine


def bundled_evaluations() -> list[Path]:
    """The labeled evaluation datasets shipped in `datasets/`, excluding row-selection manifests."""
    folder = Path(__file__).resolve().parents[1] / "datasets"
    return sorted(p for p in folder.glob("*.jsonl") if not p.name.endswith("-selection.jsonl"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ruling")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("serve", help="run the HTTP API")

    ask = commands.add_parser("ask", help="answer questions once, without a server")
    ask.add_argument("state", help="state text, or @path to read a file")
    ask.add_argument("--choice", action="append", default=[], metavar="ID=Q|a,b,c",
                     help="choice question: id=instructions|key1,key2,...")
    ask.add_argument("--score", action="append", default=[], metavar="ID=Q|lo,mid,hi",
                     help="score question: id=instructions|level1,level2,...")
    ask.add_argument("--noul", action="append", default=[], metavar="ID=STATEMENT")
    _engine_overrides(ask)

    for name, help_text in (("eval", "score a labeled dataset and report metrics"),
                            ("calibrate", "fit per-type temperatures on a labeled dataset")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("dataset", type=Path)
        sub.add_argument("--output", type=Path, help="write JSON here instead of stdout")
        _engine_overrides(sub)
        sub.add_argument("--think-tokens", type=int,
                         help="let a decoder model reason up to this many tokens before each answer (slow; for teachers)")
        if name == "eval":
            sub.add_argument("--predictions", type=Path, help="also write each question's distribution as JSONL")
            sub.add_argument("--against", choices=["typesafe"],
                             help="also send every record to the hosted TypeSafe API (TYPESAFE_API_KEY) and report agreement")
            sub.add_argument("--typesafe-model", help="model alias for --against typesafe; the service default when omitted")

    build = commands.add_parser("build-dataset", help="download a pinned public dataset into ruling format")
    build.add_argument("name", choices=["wanli", "sst5", "typesafe", "every", "dev"])
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--limit", type=int, default=300, help="rows to sample (sst5), or rows per source (dev)")

    data = commands.add_parser("build-corpus", help="build training records from open data, program state, or synthetic worlds")
    data.add_argument("kind", choices=["public", "states", "worlds", "questions", "answers"])
    data.add_argument("--output", type=Path, required=True)
    data.add_argument("--count", type=int, help="records per public source, or total states, worlds, or question sets")
    data.add_argument("--seed", type=int, default=0)
    data.add_argument("--catalog", type=Path, default=Path(__file__).resolve().parents[1] / "datasets" / "worlds.json")
    data.add_argument("--writer", help="writer and judge model for worlds; defaults to RULING_MODEL")
    data.add_argument("--batch-size", type=int, default=32)
    data.add_argument("--min-probability", type=float, default=0.05,
                      help="drop a world record when the judge gives any labeled answer less than this")
    data.add_argument("--temperature", type=float, default=0.9)
    data.add_argument("--states", type=Path, nargs="*", default=[], help="record files whose states get questions")
    data.add_argument("--questions", type=Path, help="question records to answer (answers)")
    data.add_argument("--teacher", choices=["model", "typesafe"], default="model",
                      help="answer with the --writer model (local, or openrouter:<model>) or the hosted TypeSafe API")
    data.add_argument("--questions-per-state", type=int, default=6)
    data.add_argument("--max-options", type=int, default=8)
    data.add_argument("--rotations", type=int, default=3, help="option orderings the teacher averages")
    data.add_argument("--think-tokens", type=int, help="let the teacher reason up to this many tokens before each answer")
    data.add_argument("--holdout", type=Path, nargs="*",
                      default=bundled_evaluations(),
                      help="evaluation datasets whose states are never distilled")

    train = commands.add_parser("train", help="LoRA fine-tune the answer readout on training records")
    train.add_argument("--data", type=Path, nargs="+", required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--holdout", type=Path, nargs="*",
                       default=bundled_evaluations(),
                       help="evaluation datasets whose states must not appear in training data")
    train.add_argument("--model", help="base model; defaults to RULING_MODEL")
    for name, kind in (("valid-fraction", float), ("steps", int), ("batch-size", int), ("learning-rate", float), ("max-grad-norm", float),
                       ("warmup", int), ("lora-layers", int), ("lora-rank", int), ("lora-scale", float),
                       ("lora-dropout", float), ("max-tokens", int), ("eval-every", int), ("valid-examples", int), ("seed", int)):
        train.add_argument(f"--{name}", type=kind)
    train.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false",
                       help="keep all activations; faster but needs far more memory")

    encoder = commands.add_parser("train-encoder", help="fully fine-tune a decision encoder on training records")
    encoder.add_argument("--data", type=Path, nargs="+", required=True)
    encoder.add_argument("--output", type=Path, required=True)
    encoder.add_argument("--base", required=True, help="a ModernBERT-architecture encoder repository")
    encoder.add_argument("--revision", required=True, help="the base repository commit to pin")
    encoder.add_argument("--holdout", type=Path, nargs="*",
                         default=bundled_evaluations(),
                         help="evaluation datasets whose states must not appear in training data")
    for name, kind in (("valid-fraction", float), ("steps", int), ("batch-tokens", int), ("learning-rate", float),
                       ("head-learning-rate", float), ("weight-decay", float), ("warmup", int), ("max-tokens", int),
                       ("eval-every", int), ("valid-records", int), ("seed", int)):
        encoder.add_argument(f"--{name}", type=kind)
    encoder.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false",
                         help="store all activations; faster but uses far more memory")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    configure_memory(Settings.from_env())

    if args.command == "serve":
        settings = Settings.from_env()
        uvicorn.run(app_from_env(), host=settings.host, port=settings.port)
        return
    if args.command == "build-corpus":
        if args.kind != "answers" and args.count is None:
            parser.error(f"build-corpus {args.kind} needs --count")
        if args.kind == "answers" and args.questions is None:
            parser.error("build-corpus answers needs --questions")
        if args.kind == "questions":
            from ruling.distill import write_questions
            from ruling.train import _state_key
            held = {_state_key(r.state) for path in args.holdout for r in load_dataset(path)}
            pool = {}
            for path in args.states:
                for record in load_dataset(path):
                    key = _state_key(record.state)
                    if key not in held:
                        pool.setdefault(key, (record.family or path.stem, record.state))
            chosen = [list(pool.values())[i] for i in np.random.default_rng(args.seed).permutation(len(pool))[:args.count]]
            settings = dataclasses.replace(Settings.from_env(), model=args.writer or Settings.from_env().model, rotations=1,
                                           prior_debias=False)
            writer = build_engine(settings)
            if isinstance(writer, OpenRouterEngine):
                generate = lambda prompts: writer.write(prompts, max_tokens=1200)
            else:
                from ruling.distill import local_generator
                generate = local_generator(writer, args.seed, args.temperature)
            _stream(args.output, write_questions(chosen, generate, args.questions_per_state, args.batch_size, args.max_options),
                    append=False)
            _report_spend(writer)
            return
        if args.kind == "answers":
            records = load_dataset(args.questions)
            done = sum(1 for line in args.output.open() if line.strip()) if args.output.exists() else 0
            pending = records[done:]
            print(f"{done} already answered, {len(pending)} to go")
            if args.teacher == "typesafe":
                from ruling.against import live_answers
                answered = (pending[i].model_copy(update={"references": {qid: {"target": t} for qid, t in answers.items()}})
                            for i, answers in live_answers(pending))
            else:
                from ruling.distill import answer_locally
                teacher = build_engine(dataclasses.replace(Settings.from_env(), model=args.writer or Settings.from_env().model,
                                                           rotations=args.rotations, prior_debias=False))
                if args.think_tokens:
                    from ruling.thinking import ThinkingEngine
                    teacher = ThinkingEngine(teacher, args.think_tokens)
                answered = answer_locally(pending, teacher)
            _stream(args.output, ([r] for r in answered), append=True)
            if args.teacher == "model":
                _report_spend(teacher)
            return
        if args.kind == "public":
            records = corpus.build(args.count, args.seed)
        elif args.kind == "states":
            records = states.generate(args.count, args.seed)
        else:
            from ruling.worlds import Catalog, synthesize
            model = args.writer or Settings.from_env().model
            engine = Engine(model, Calibration(), Settings.from_env().max_input_tokens, rotations=1, prior_debias=False)
            written = 0
            with args.output.open("w") as out:
                for accepted in synthesize(Catalog.load(args.catalog), engine, args.count, args.seed, args.batch_size,
                                           args.min_probability, args.temperature):
                    out.writelines(r.model_dump_json() + "\n" for r in accepted)
                    out.flush()
                    written += len(accepted)
            print(f"wrote {written} records to {args.output}")
            return
        write([r.model_dump(mode="json") for r in records], args.output)
        print(f"wrote {len(records)} records to {args.output}")
        return
    if args.command == "train":
        from ruling.train import Trainer, TrainConfig
        options = {name.replace("-", "_"): getattr(args, name.replace("-", "_"))
                   for name in ("valid-fraction", "steps", "batch-size", "learning-rate", "max-grad-norm", "warmup", "lora-layers", "lora-rank",
                                "lora-scale", "lora-dropout", "max-tokens", "eval-every", "valid-examples", "seed")}
        config = TrainConfig(model=args.model or Settings.from_env().model, data=args.data, output=args.output,
                             holdout=args.holdout, grad_checkpoint=args.grad_checkpoint,
                             **{k: v for k, v in options.items() if v is not None})
        Trainer(config).run()
        return
    if args.command == "train-encoder":
        from ruling.decision_train import DecisionTrainConfig, DecisionTrainer
        names = ("valid_fraction", "steps", "batch_tokens", "learning_rate", "head_learning_rate", "weight_decay", "warmup",
                 "max_tokens", "eval_every", "valid_records", "seed")
        config = DecisionTrainConfig(base=args.base, revision=args.revision, data=args.data, holdout=args.holdout,
                                     output=args.output, grad_checkpoint=args.grad_checkpoint,
                                     **{n: getattr(args, n) for n in names if getattr(args, n) is not None})
        DecisionTrainer(config).run()
        return
    if args.command == "build-dataset":
        builders = {"wanli": build_wanli, "sst5": lambda: build_sst5(args.limit),
                    "typesafe": build_typesafe, "every": build_every,
                    "dev": lambda: [r.model_dump(mode="json") for r in
                                    corpus.build(args.limit, seed=1, split="dev") + states.generate(args.limit, seed=10_001)]}
        records = builders[args.name]()
        write(records, args.output)
        print(f"wrote {len(records)} records to {args.output}")
        return

    engine = build_engine(_settings(args))
    if getattr(args, "think_tokens", None):
        from ruling.thinking import ThinkingEngine
        engine = ThinkingEngine(engine, args.think_tokens)
    if args.command == "ask":
        state = Path(args.state[1:]).read_text() if args.state.startswith("@") else args.state
        try:
            state = json.loads(state)
        except json.JSONDecodeError:
            pass
        questions = {}
        for spec in args.choice:
            qid, instructions, items = _parse_spec(spec)
            questions[qid] = Choice(instructions=instructions, criteria={k: None for k in items})
        for spec in args.score:
            qid, instructions, items = _parse_spec(spec)
            questions[qid] = Score(instructions=instructions, criteria=items)
        for spec in args.noul:
            qid, _, statement = spec.partition("=")
            questions[qid] = Noul(instructions=statement)
        if not questions:
            parser.error("ask needs at least one --choice, --score, or --noul")
        response = engine.evaluate(SystemOneRequest(state=state, questions=questions))
        print(response.model_dump_json(indent=2))
        return

    records = load_dataset(args.dataset)
    live = collect_live(records, model=args.typesafe_model) if getattr(args, "against", None) else None
    observations = collect(engine, records, live)
    header = {"model": engine.model_id, "revision": engine.revision, "adapter": engine.adapter, "rotations": engine.rotations,
              "prior_debias": engine.prior_debias, "dataset": str(args.dataset)}
    _report_spend(engine)
    if args.command == "eval":
        data = {**header, **report(observations, engine.calibration)}
        if args.predictions:
            write_predictions(observations, engine.calibration, args.predictions)
    else:
        calibration = fit(observations, engine.provenance)
        data = {**header, "temperatures": calibration.temperatures,
                "before": report(observations, engine.calibration)["question_types"],
                "after": report(observations, calibration)["question_types"]}
        if args.output:
            calibration.save(args.output)
            print(f"wrote {args.output}")
            print(json.dumps({k: data[k] for k in ("temperatures", "before", "after")}, indent=2))
            return
    if args.output:
        write_json(args.output, data)
        print(f"wrote {args.output}")
    else:
        print(json.dumps(data, indent=2))


def _report_spend(engine) -> None:
    if isinstance(engine, OpenRouterEngine):
        print(f"OpenRouter spend this run: ${engine.budget.spent_usd:.4f}")


def _stream(output: Path, batches, append: bool) -> None:
    """Write record batches as they arrive, so an interrupted run keeps everything finished so far."""
    written = 0
    with output.open("a" if append else "w") as out:
        for records in batches:
            out.writelines(r.model_dump_json() + "\n" for r in records)
            out.flush()
            written += len(records)
    print(f"wrote {written} records to {output}")


def _engine_overrides(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--model", help="override RULING_MODEL")
    sub.add_argument("--rotations", type=int, help="override RULING_ROTATIONS")
    sub.add_argument("--prior-debias", choices=["on", "off"], help="override RULING_PRIOR_DEBIAS")
    sub.add_argument("--calibration", type=Path, help="override RULING_CALIBRATION")
    sub.add_argument("--adapter", type=Path, help="override RULING_ADAPTER, a LoRA directory from `ruling train`")


def _settings(args: argparse.Namespace) -> Settings:
    overrides = {
        "model": args.model,
        "rotations": args.rotations,
        "prior_debias": None if args.prior_debias is None else args.prior_debias == "on",
        "calibration_path": args.calibration,
        "adapter_path": args.adapter,
    }
    return dataclasses.replace(Settings.from_env(), **{k: v for k, v in overrides.items() if v is not None})


def _parse_spec(spec: str) -> tuple[str, str, list[str]]:
    qid, _, rest = spec.partition("=")
    instructions, _, items = rest.partition("|")
    if not (qid and instructions and items):
        raise SystemExit(f"bad question spec {spec!r}; expected id=instructions|item1,item2,...")
    return qid, instructions, [item.strip() for item in items.split(",") if item.strip()]
