import pytest
from fastapi.testclient import TestClient

from ..main import app  # Leave as relative for use in template: ssec-jhu/base-template.


@pytest.fixture(scope="class")
def app_client():
    return TestClient(app)
