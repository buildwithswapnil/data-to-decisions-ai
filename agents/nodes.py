from typing import Any, Dict

from typing_extensions import TypedDict

from tools.hybrid_rag_lookup import hybrid_rag_lookup


class AgentState(TypedDict):
    query: str
    plan: str
    context: Any
    draft: str
    critique: str
    approved: bool
    insights: Any
    revision_count: int


def planner_node(state: AgentState) -> Dict[str, Any]:
    context = state.get("context") or {}
    intent = context.get("query_intent", "Unclassified")
    filters = context.get("applied_filters", [])
    plan = [
        f"Classify query intent as {intent}.",
        "Execute read-only analytics against the local healthcare SQLite database.",
        "Use detected filters: " + (", ".join(filters) if filters else "none"),
        "Return evidence-backed summary, limitations, and recommendations.",
    ]
    return {"plan": "\n".join(f"{idx}. {step}" for idx, step in enumerate(plan, start=1))}


def generator_node(state: AgentState) -> Dict[str, Any]:
    context = state.get("context") or hybrid_rag_lookup(state["query"])
    summary = context.get("executive_summary") or "No executive summary was generated."
    recommendations = context.get("recommendations", [])
    recommendation_text = "\n".join(f"- {item}" for item in recommendations)
    return {
        "draft": (
            f"{summary}\n\n"
            f"Recommendations:\n{recommendation_text or '- Review the filtered cohort and trace output.'}"
        ),
        "context": context,
    }


def critic_node(state: AgentState) -> Dict[str, Any]:
    context = state.get("context") or {}
    has_trace = bool(context.get("reasoning_trace"))
    has_sql_or_rejection = bool(context.get("generated_sql")) or context.get("analysis_domain") == "unsupported"
    approved = has_trace and has_sql_or_rejection
    return {
        "approved": approved,
        "critique": (
            "Approved: analytics trace and SQL/rejection evidence are present."
            if approved
            else "Rejected: missing reasoning trace or execution evidence."
        ),
        "revision_count": state.get("revision_count", 0) + (0 if approved else 1),
    }


def insight_node(state: AgentState) -> Dict[str, Any]:
    context = state.get("context") or {}
    return {
        "insights": {
            "answer": state.get("draft", ""),
            "intent": context.get("query_intent"),
            "domain": context.get("analysis_domain"),
            "records": context.get("total_records"),
            "filters": context.get("applied_filters", []),
            "limitations": context.get("limitations", []),
        }
    }
