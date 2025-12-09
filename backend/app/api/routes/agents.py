"""Agent management endpoints."""

from fastapi import APIRouter, HTTPException

from backend.app.core.agents.lifecycle import AgentStatus, lifecycle_manager
from backend.app.core.agents.registry import agent_registry
from backend.app.core.exceptions import ConcurrencyError
from backend.app.models.requests import AgentConfigUpdateRequest

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("/")
async def list_agents():
    agents = []
    for metadata in agent_registry.list_metadata():
        status = lifecycle_manager.get_agent_status(metadata.name)
        agents.append({
            "agent_id": metadata.name,
            "status": status,
            "metadata": metadata,
        })
    return {"agents": agents}


@router.get("/{agent_id}")
async def get_agent(agent_id: str):
    metadata = agent_registry.get_metadata(agent_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    status = lifecycle_manager.get_agent_status(agent_id)
    return {"agent_id": agent_id, "status": status, "metadata": metadata}


@router.post("/{agent_id}/pause")
async def pause_agent(agent_id: str):
    try:
        status = await lifecycle_manager.pause_agent(agent_id)
    except ValueError as exc:  # agent missing or terminated
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"agent_id": agent_id, "status": status}


@router.post("/{agent_id}/resume")
async def resume_agent(agent_id: str):
    try:
        status = await lifecycle_manager.resume_agent(agent_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConcurrencyError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return {"agent_id": agent_id, "status": status}


@router.put("/{agent_id}/config")
async def update_agent_config(agent_id: str, payload: AgentConfigUpdateRequest):
    if agent_registry.get(agent_id) is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    lifecycle_manager.update_agent_config(
        agent_id=agent_id,
        model=payload.model,
        temperature=payload.temperature,
    )
    status = lifecycle_manager.get_agent_status(agent_id)
    metadata = agent_registry.get_metadata(agent_id)
    return {
        "agent_id": agent_id,
        "status": status,
        "metadata": metadata,
        "model": payload.model or metadata.model,
        "temperature": payload.temperature,
    }
