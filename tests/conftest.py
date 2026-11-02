from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from support_agent.app import create_app
from support_agent.config import PROJECT_ROOT, Settings
from support_agent.db import SupportDB
from support_agent.kb import KnowledgeBase
from support_agent.tools import ToolExecutor


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    return KnowledgeBase.from_directory(PROJECT_ROOT / "knowledge_base")


@pytest.fixture
def db(tmp_path: Path) -> SupportDB:
    return SupportDB(tmp_path / "test.db")


@pytest.fixture
def tools(kb: KnowledgeBase, db: SupportDB) -> ToolExecutor:
    return ToolExecutor(kb, db)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = Settings(llm_provider="mock", embeddings="none", db_path=tmp_path / "api.db")
    return TestClient(create_app(settings))
