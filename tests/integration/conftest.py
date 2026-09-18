import os
import socket
import threading
import time

import pytest
import uvicorn

from ruling.calibration import Calibration
from ruling.config import Settings, configure_memory
from ruling.engine import Engine
from ruling.server import create_app

configure_memory(Settings.from_env())

TEST_MODEL = os.environ.get("RULING_TEST_MODEL", "mlx-community/Qwen3.5-0.8B-4bit")


@pytest.fixture(scope="session")
def engine() -> Engine:
    """The shipped defaults for rotations and debiasing, on the small test model."""
    defaults = Settings()
    return Engine(TEST_MODEL, Calibration(), max_input_tokens=8192,
                  rotations=defaults.rotations, prior_debias=defaults.prior_debias)


def _variant(engine: Engine, rotations: int) -> Engine:
    variant = Engine.__new__(Engine)
    variant.__dict__.update(engine.__dict__)
    variant.rotations = rotations
    return variant


@pytest.fixture(scope="session")
def single_engine(engine) -> Engine:
    return _variant(engine, 1)


@pytest.fixture(scope="session")
def rotating_engine(engine) -> Engine:
    return _variant(engine, 4)


@pytest.fixture(scope="session")
def debiased_engine(engine) -> Engine:
    variant = _variant(engine, 1)
    variant.prior_debias = True
    return variant


def _serve(app):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.05)
    return server, thread, f"http://127.0.0.1:{port}"


@pytest.fixture(scope="session")
def live_server(engine):
    server, thread, url = _serve(create_app(engine))
    yield url
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="session")
def live_server_with_key(engine):
    server, thread, url = _serve(create_app(engine, api_key="s3cret"))
    yield url
    server.should_exit = True
    thread.join(timeout=10)
