"""
Pytest configuration and shared fixtures

This module provides reusable fixtures and configuration for all tests.
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator
from pathlib import Path
from datetime import datetime, UTC

from fastapi.testclient import TestClient

from backend.app.models.domain import DecisionState, ProcessInfo
from backend.app.services.redis_repository import InMemoryProcessRepository
from backend.app.services.process_manager import ProcessManager
from backend.app.services.decision_service import DecisionService


# ===== Pytest Configuration =====

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ===== Test Data Fixtures =====

@pytest.fixture
def sample_decision_query() -> str:
    """Sample decision query for testing."""
    return "Should I invest in renewable energy stocks?"


@pytest.fixture
def sample_decision_state() -> DecisionState:
    """Sample decision state with populated fields."""
    return DecisionState(
        decision_requested="Should I invest in renewable energy stocks?",
        trigger="Rising demand for clean energy solutions",
        root_cause="Global shift towards sustainability and climate action",
        scope_definition="Investment decision in renewable energy sector with $50K capital",
        decision_drafted="Initial recommendation to invest in diversified renewable portfolio",
        goals="Maximize returns (15% target) while supporting environmental sustainability",
        stakeholders="Personal investor, financial advisor, environmental impact",
        complementary_info="Solar market growing 20% annually, wind energy costs down 40%",
        complementary_info_num=2,
        decision_draft_updated="Revised recommendation with specific allocation strategy",
        generated_alternatives="1. Solar ETF, 2. Wind energy stocks, 3. Mixed renewable fund",
        alternatives="Alternative 1: Solar ETF (ICLN)\nAlternative 2: Wind stocks (VWND)\nAlternative 3: Mixed fund (RENE)",
        result="Invest $30K in Solar ETF (ICLN) and $20K in mixed fund (RENE)",
        result_comment="Balanced approach providing growth potential and diversification",
        best_alternative_result="Invest $50K in Wind energy stocks (VWND)",
        best_alternative_result_comment="Higher growth potential but concentrated risk"
    )


@pytest.fixture
def sample_process_info() -> ProcessInfo:
    """Sample process info for testing."""
    return ProcessInfo(
        process_id="test-process-123",
        query="Should I invest in renewable energy stocks?",
        status="pending",
        current_step=None,
        completed_steps=[],
        result=None,
        error=None,
        created_at=datetime.now(UTC),
        completed_at=None
    )


# ===== Repository Fixtures =====

@pytest.fixture
def in_memory_repository() -> InMemoryProcessRepository:
    """Create a fresh in-memory repository for each test."""
    return InMemoryProcessRepository()


@pytest.fixture
async def populated_repository(
    in_memory_repository: InMemoryProcessRepository,
    sample_process_info: ProcessInfo
) -> InMemoryProcessRepository:
    """Create a repository with sample data."""
    await in_memory_repository.save(sample_process_info)
    return in_memory_repository


# ===== Service Fixtures =====

@pytest.fixture
def decision_service() -> DecisionService:
    """Create a DecisionService instance."""
    return DecisionService()


@pytest.fixture
def process_manager(in_memory_repository: InMemoryProcessRepository) -> ProcessManager:
    """Create a ProcessManager with in-memory repository."""
    return ProcessManager(repository=in_memory_repository)


# ===== API Client Fixtures =====

@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    from backend.app.main import app
    with TestClient(app) as client:
        yield client


# ===== Temporary File Fixtures =====

@pytest.fixture
def temp_persistence_file(tmp_path: Path) -> Path:
    """Create a temporary file for persistence testing."""
    return tmp_path / "test_decision_graph.json"


# ===== Mock Data Generators =====

@pytest.fixture
def create_process_info():
    """Factory fixture for creating ProcessInfo instances."""
    def _create_process_info(
        process_id: str = "test-process",
        query: str = "Test decision?",
        status: str = "pending",
        **kwargs
    ) -> ProcessInfo:
        return ProcessInfo(
            process_id=process_id,
            query=query,
            status=status,
            current_step=kwargs.get("current_step"),
            completed_steps=kwargs.get("completed_steps", []),
            result=kwargs.get("result"),
            error=kwargs.get("error"),
            created_at=kwargs.get("created_at", datetime.now(UTC)),
            completed_at=kwargs.get("completed_at")
        )
    return _create_process_info


@pytest.fixture
def create_decision_state():
    """Factory fixture for creating DecisionState instances."""
    def _create_decision_state(
        decision_requested: str = "Test decision?",
        **kwargs
    ) -> DecisionState:
        return DecisionState(
            decision_requested=decision_requested,
            trigger=kwargs.get("trigger", ""),
            root_cause=kwargs.get("root_cause", ""),
            scope_definition=kwargs.get("scope_definition", ""),
            decision_drafted=kwargs.get("decision_drafted", ""),
            goals=kwargs.get("goals", ""),
            stakeholders=kwargs.get("stakeholders", ""),
            complementary_info=kwargs.get("complementary_info", ""),
            complementary_info_num=kwargs.get("complementary_info_num", 0),
            decision_draft_updated=kwargs.get("decision_draft_updated", ""),
            generated_alternatives=kwargs.get("generated_alternatives", ""),
            alternatives=kwargs.get("alternatives", ""),
            result=kwargs.get("result", ""),
            result_comment=kwargs.get("result_comment", ""),
            best_alternative_result=kwargs.get("best_alternative_result", ""),
            best_alternative_result_comment=kwargs.get("best_alternative_result_comment", "")
        )
    return _create_decision_state
