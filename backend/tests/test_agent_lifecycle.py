import pytest

from backend.app.config import get_settings
from backend.app.core.agents.lifecycle import AgentStatus, lifecycle_manager
from backend.app.core.exceptions import ConcurrencyError


@pytest.fixture(autouse=True)
def reset_lifecycle_state():
    settings = get_settings()
    config_backup = settings.agent_config.model_copy(deep=True)
    lifecycle_manager.reset()
    yield
    settings.agent_config = config_backup
    lifecycle_manager.reset()


@pytest.mark.asyncio
async def test_initialize_sets_status_active():
    agent = await lifecycle_manager.initialize_agent("identify_trigger_agent")
    assert agent is not None
    assert lifecycle_manager.get_agent_status("identify_trigger_agent") == AgentStatus.ACTIVE


@pytest.mark.asyncio
async def test_pause_and_resume_agent():
    await lifecycle_manager.initialize_agent("identify_trigger_agent")
    paused_status = await lifecycle_manager.pause_agent("identify_trigger_agent")
    assert paused_status == AgentStatus.PAUSED
    assert lifecycle_manager.get_agent_status("identify_trigger_agent") == AgentStatus.PAUSED

    resumed_status = await lifecycle_manager.resume_agent("identify_trigger_agent")
    assert resumed_status == AgentStatus.ACTIVE
    assert lifecycle_manager.get_agent_status("identify_trigger_agent") == AgentStatus.ACTIVE


@pytest.mark.asyncio
async def test_concurrent_limit_enforced():
    settings = get_settings()
    settings.agent_config.max_concurrent_agents = 1

    await lifecycle_manager.initialize_agent("identify_trigger_agent")
    with pytest.raises(ConcurrencyError):
        await lifecycle_manager.initialize_agent("root_cause_analyzer_agent")


@pytest.mark.asyncio
async def test_timeout_enforcement_marks_agent():
    settings = get_settings()
    settings.agent_config.agent_timeout = 1

    await lifecycle_manager.initialize_agent("identify_trigger_agent")
    lifecycle_manager._start_times["identify_trigger_agent"] -= 2  # force expiry

    status = lifecycle_manager.get_agent_status("identify_trigger_agent")
    assert status == AgentStatus.TIMEOUT


def test_update_agent_config_overrides_settings():
    settings = get_settings()
    lifecycle_manager.update_agent_config(
        "identify_trigger_agent", model="gpt-test", temperature=0.6
    )

    assert settings.agent_model_mapping["identify_trigger_agent"] == "gpt-test"
    assert settings.agent_temperature_mapping["identify_trigger_agent"] == 0.6
