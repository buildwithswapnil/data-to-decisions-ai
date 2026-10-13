from agents.healthcare_agent import run_agentic_analysis


def test_agent_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_agentic_analysis("Show Medicare high risk readmission patients")

    assert result["analysis_domain"] == "readmissions"
    assert result["agent_enabled"] is False
    assert result["agent_trace"]
