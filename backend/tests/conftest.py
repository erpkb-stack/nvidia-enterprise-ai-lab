import os
import tempfile

# Configure an isolated, offline environment BEFORE the app is imported.
_tmp = tempfile.mkdtemp(prefix="nvlab-test-")
os.environ.update({
    "APP_ENV": "test",
    "DATABASE_URL": f"sqlite:///{_tmp}/test.db",
    "QDRANT_URL": "",  # in-process Qdrant
    "AUTO_CREATE_TABLES": "true",  # override any local backend/.env
    "NVIDIA_NIM_API_KEY": "",
    "NVIDIA_NIM_BASE_URL": "",
    "LOG_LEVEL": "WARNING",
})

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
