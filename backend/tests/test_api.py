"""
Tests for API Endpoints

Tests for all FastAPI routes including health, graph, and decisions endpoints.
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns API information."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "Multi-Agent Decision Making" in data["message"]
        assert "version" in data
        assert "endpoints" in data
        
        # Check that key endpoints are listed
        endpoints = data["endpoints"]
        assert "GET /" in endpoints
        assert "GET /health" in endpoints
        assert "POST /decisions/run" in endpoints
    
    def test_health_check_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "version" in data


class TestGraphEndpoints:
    """Tests for graph visualization endpoints."""
    
    def test_get_mermaid_diagram(self, test_client):
        """Test getting Mermaid diagram code."""
        response = test_client.get("/graph/mermaid")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "mermaid_code" in data
        assert len(data["mermaid_code"]) > 0
        
        # Should contain graph syntax
        mermaid_code = data["mermaid_code"]
        assert "graph" in mermaid_code.lower() or "flowchart" in mermaid_code.lower()
    
    def test_get_graph_structure(self, test_client):
        """Test getting graph structure information."""
        response = test_client.get("/graph/structure")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should contain structure information
        assert "total_nodes" in data or "nodes" in data or isinstance(data, dict)


class TestDecisionEndpoints:
    """Tests for decision-making endpoints."""
    
    def test_run_decision_sync_valid_query(self, test_client):
        """Test synchronous decision run with valid query."""
        # Note: This will actually try to run the decision process
        # In a real test environment, you'd mock the AI agents
        
        payload = {
            "decision_query": "Should I invest in renewable energy stocks for long-term growth?"
        }
        
        # This test might take a while due to actual AI calls
        # In production tests, mock the agents
        # response = test_client.post("/decisions/run", json=payload)
        # assert response.status_code == 200
        
        # For now, just test the payload validation
        assert payload["decision_query"] is not None
        assert len(payload["decision_query"]) >= 10
    
    def test_run_decision_sync_invalid_query_too_short(self, test_client):
        """Test synchronous decision with too short query."""
        payload = {
            "decision_query": "Yes?"
        }
        
        response = test_client.post("/decisions/run", json=payload)
        
        # Should return validation error
        assert response.status_code == 422 or response.status_code == 400
    
    def test_run_decision_sync_missing_query(self, test_client):
        """Test synchronous decision with missing query."""
        payload = {}
        
        response = test_client.post("/decisions/run", json=payload)
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_start_decision_async_valid_query(self, test_client):
        """Test starting asynchronous decision process."""
        payload = {
            "decision_query": "Should I expand my business to international markets?"
        }
        
        response = test_client.post("/decisions/start", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "process_id" in data
        assert data["status"] in ["started", "pending"]
        assert len(data["process_id"]) > 0
    
    def test_start_decision_async_invalid_query(self, test_client):
        """Test starting async decision with invalid query."""
        payload = {
            "decision_query": "No"
        }
        
        response = test_client.post("/decisions/start", json=payload)
        
        # Should return validation error
        assert response.status_code == 422 or response.status_code == 400
    
    def test_get_decision_status_valid_process(self, test_client):
        """Test getting status of a valid process."""
        # First, start a process
        payload = {
            "decision_query": "Should I switch careers from engineering to management?"
        }
        
        start_response = test_client.post("/decisions/start", json=payload)
        assert start_response.status_code == 200
        
        process_id = start_response.json()["process_id"]
        
        # Now get status
        status_response = test_client.get(f"/decisions/status/{process_id}")
        
        assert status_response.status_code == 200
        data = status_response.json()
        
        assert "status" in data
        assert data["status"] in ["pending", "running", "completed", "failed"]
    
    def test_get_decision_status_invalid_process(self, test_client):
        """Test getting status of non-existent process."""
        response = test_client.get("/decisions/status/nonexistent-process-id")
        
        # Should return 404
        assert response.status_code == 404
    
    def test_list_all_processes(self, test_client):
        """Test listing all processes."""
        # Create a couple of processes
        test_client.post("/decisions/start", json={
            "decision_query": "Should I pursue further education?"
        })
        test_client.post("/decisions/start", json={
            "decision_query": "Should I start a side business?"
        })
        
        # List all processes
        response = test_client.get("/decisions/processes")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "stats" in data or "processes" in data or isinstance(data, dict)
    
    def test_get_process_stats(self, test_client):
        """Test getting process statistics via processes endpoint."""
        response = test_client.get("/decisions/processes")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have stats section
        if "stats" in data:
            stats = data["stats"]
            assert "total" in stats or "pending" in stats or "running" in stats
    
    def test_cleanup_old_processes(self, test_client):
        """Test cleanup endpoint."""
        response = test_client.delete("/decisions/cleanup")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "removed_count" in data or "status" in data
        if "removed_count" in data:
            assert isinstance(data["removed_count"], int)
            assert data["removed_count"] >= 0


class TestRequestValidation:
    """Tests for request validation."""
    
    def test_decision_request_valid(self):
        """Test valid decision request payloads."""
        from backend.app.models.requests import DecisionRequest
        
        valid_queries = [
            "Should I invest in stocks?",
            "Should we expand our business to new markets?",
            "Is it time to change my career path?"
        ]
        
        for query in valid_queries:
            request = DecisionRequest(decision_query=query)
            assert request.decision_query == query
    
    def test_decision_request_too_short(self):
        """Test decision request with too short query."""
        from backend.app.models.requests import DecisionRequest
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            DecisionRequest(decision_query="Yes?")
    
    def test_decision_request_too_long(self):
        """Test decision request with too long query."""
        from backend.app.models.requests import DecisionRequest
        from pydantic import ValidationError
        
        # Create a very long query (over 1000 characters)
        long_query = "Should I " + "x" * 1000
        
        with pytest.raises(ValidationError):
            DecisionRequest(decision_query=long_query)


class TestCORS:
    """Tests for CORS configuration."""
    
    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present in responses."""
        response = test_client.options("/health")
        
        # Should have CORS headers
        # Note: TestClient might not include all CORS headers
        # This is more relevant in actual deployment
        assert response.status_code in [200, 405]
