# Test Suite for MAS-CDM

This directory contains comprehensive tests for the Multi-Agent System for Collaborative Decision Making.

## Test Structure

```
backend/tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest fixtures and configuration
├── test_models.py           # Tests for domain models
├── test_repository.py       # Tests for repository pattern
├── test_decision_service.py # Tests for decision service
├── test_process_manager.py  # Tests for process manager
├── test_api.py              # Tests for API endpoints
└── test_integration.py      # Integration and E2E tests
```

# Run All Tests (coverage on by default via pytest.ini)

```bash
# Using pytest directly
pytest backend/tests/

# Using uv
uv run pytest backend/tests/

# With verbose output
pytest backend/tests/ -v

# With HTML coverage
pytest backend/tests/ --cov=backend --cov-report=html
```
pytest backend/tests/ --cov=backend --cov-report=html
```

### Run Specific Test Files

```bash
# Unit tests only
pytest backend/tests/test_models.py
pytest backend/tests/test_repository.py
pytest backend/tests/test_decision_service.py
pytest backend/tests/test_process_manager.py

# API tests only
pytest backend/tests/test_api.py

# Integration tests only
pytest backend/tests/test_integration.py
```

### Run Tests by Marker

```bash
# Run only unit tests (fast)
pytest backend/tests/ -m unit

# Run only integration tests
pytest backend/tests/ -m integration

# Run only API tests
pytest backend/tests/ -m api

# Skip slow tests
pytest backend/tests/ -m "not slow"
```

### Run Specific Test Classes or Methods

```bash
# Run a specific test class
pytest backend/tests/test_models.py::TestDecisionState

# Run a specific test method
pytest backend/tests/test_models.py::TestDecisionState::test_empty_decision_state_creation

# Run tests matching a pattern
pytest backend/tests/ -k "test_decision"
```

## Test Categories

### Unit Tests

**Fast, isolated tests** that test individual components without external dependencies:

- `test_models.py` - Domain model validation and serialization
- `test_repository.py` - Repository operations (in-memory)
- `test_decision_service.py` - DecisionService methods
- `test_process_manager.py` - ProcessManager operations

Run with:
```bash
pytest backend/tests/ -m unit
```

### API Tests

**Tests for FastAPI endpoints**:

- `test_api.py` - HTTP endpoint tests, request/response validation

Run with:
```bash
pytest backend/tests/test_api.py
```

### Integration Tests

**End-to-end tests** that verify multiple components working together:

- `test_integration.py` - Full workflow tests, concurrent operations, data consistency

Run with:
```bash
pytest backend/tests/ -m integration
```

## Test Fixtures

Common fixtures are defined in `conftest.py`:

### Data Fixtures
- `sample_decision_query` - Sample decision query string
- `sample_decision_state` - Populated DecisionState instance
- `sample_process_info` - Sample ProcessInfo instance

### Repository Fixtures
- `in_memory_repository` - Fresh InMemoryProcessRepository
- `populated_repository` - Repository with sample data

### Service Fixtures
- `decision_service` - DecisionService instance
- `process_manager` - ProcessManager with in-memory repository

### API Fixtures
- `test_client` - FastAPI TestClient

### Factory Fixtures
- `create_process_info()` - Factory for creating ProcessInfo instances
- `create_decision_state()` - Factory for creating DecisionState instances

## Writing New Tests

### Example Unit Test

```python
import pytest
from backend.app.models.domain import DecisionState

class TestMyFeature:
    """Tests for my new feature."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        state = DecisionState(decision_requested="Test")
        assert state.decision_requested == "Test"
    
    @pytest.mark.asyncio
    async def test_async_operation(self, process_manager):
        """Test async operation."""
        process = await process_manager.create_process("Test")
        assert process is not None
```

### Example Integration Test

```python
import pytest

class TestFeatureIntegration:
    """Integration tests for feature."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end(self, process_manager):
        """Test complete workflow."""
        # Create
        process = await process_manager.create_process("Query")
        
        # Verify
        status = await process_manager.get_process_status(process.process_id)
        assert status.status == "pending"
```

### Example API Test

```python
def test_api_endpoint(test_client):
    """Test API endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

## Coverage

Generate coverage report:

```bash
# Generate HTML coverage report
pytest backend/tests/ --cov=backend --cov-report=html

# View report
open htmlcov/index.html

# Generate terminal report (already default)
pytest backend/tests/ --cov=backend --cov-report=term-missing
```

## Continuous Integration

Tests should be run in CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    uv run pytest backend/tests/ -v
    
- name: Run tests with coverage
  run: |
    uv run pytest backend/tests/ --cov=backend --cov-report=xml
```

## Troubleshooting

### Tests Fail with Import Errors

Make sure you're in the project root directory and have installed dependencies:

```bash
cd /path/to/MAS-CDM
uv sync
```

### Async Tests Fail

Ensure pytest-asyncio is installed:

```bash
uv add --dev pytest-asyncio
```

### API Tests Fail

Make sure the FastAPI app is properly configured and all dependencies are installed.

## Best Practices

1. **Keep tests independent** - Each test should work in isolation
2. **Use fixtures** - Reuse common setup code via fixtures
3. **Test edge cases** - Don't just test the happy path
4. **Use descriptive names** - Test names should describe what they test
5. **Keep tests fast** - Mock external dependencies when possible
6. **Mark slow tests** - Use `@pytest.mark.slow` for tests that take >1 second
7. **Clean up resources** - Use fixtures with cleanup or context managers

## Test Coverage Goals

- **Models**: 100% coverage (easy to achieve)
- **Repositories**: 90%+ coverage
- **Services**: 80%+ coverage
- **API Endpoints**: 80%+ coverage
- **Integration**: Key workflows covered

## Dependencies

Required test dependencies:

```toml
[project.optional-dependencies]
test = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "httpx>=0.27.0",
]
```

Install with:

```bash
uv add --dev pytest pytest-asyncio pytest-cov httpx
```
