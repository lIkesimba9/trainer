"""Keep all test artifacts; never use or clean a system temporary directory."""

import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class LocalTempFactory:
    def mktemp(self, basename, numbered=True):
        path = ROOT / ".tmp" / "tests" / f"{basename}-{uuid.uuid4().hex}"
        path.mkdir(parents=True)
        return path


@pytest.fixture(scope="session")
def tmp_path_factory():
    return LocalTempFactory()


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("case")
