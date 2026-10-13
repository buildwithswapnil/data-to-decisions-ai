from typing import Any, Dict

from tools.hybrid_rag_lookup import hybrid_rag_lookup


def run_sql_analytics_agent(query: str) -> Dict[str, Any]:
    """
    Compatibility wrapper for older code that imported agents.sql_agent.

    The project no longer asks an LLM to generate arbitrary SQL. Instead, it routes
    natural-language healthcare questions through the governed analytics tool,
    which builds parameterized read-only SQLite queries and returns traceable
    evidence.
    """
    return hybrid_rag_lookup(query)


def explain_sql_trace(query: str) -> str:
    result = run_sql_analytics_agent(query)
    filters = ", ".join(result.get("applied_filters", [])) or "none"
    return (
        f"Domain: {result.get('analysis_domain')}\n"
        f"Intent: {result.get('query_intent')}\n"
        f"Filters: {filters}\n"
        f"Count SQL:\n{result.get('count_sql') or 'No SQL executed'}"
    )
