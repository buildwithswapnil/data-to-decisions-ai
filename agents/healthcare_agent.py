import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

from openai import OpenAI

from tools.hybrid_rag_lookup import hybrid_rag_lookup
from utils.config import BASE_DIR


load_dotenv(os.path.join(BASE_DIR, ".env"))

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

SYSTEM_INSTRUCTIONS = """
You are Data-to-Decisions AI, a healthcare operations analytics agent.

Your job:
- Turn the user's business question into a concrete analytics action.
- Use the healthcare_analytics_lookup tool whenever the user asks about patient cohorts,
  readmissions, payer mix, diagnosis drivers, discharge patterns, risk, claims, denials,
  operational KPIs, or anything inside the local healthcare dataset.
- Do not invent patient facts. Ground the final answer in the tool output.
- If the tool says the request is unsupported, explain that the app is scoped to the local
  healthcare operations dataset and suggest a valid query.
- Keep the answer executive-friendly: headline finding, key numbers, and next actions.
- Mention limitations when they materially affect interpretation.
"""

TOOLS = [
    {
        "type": "function",
        "name": "healthcare_analytics_lookup",
        "description": (
            "Run safe read-only healthcare operations analytics over the local SQLite "
            "database. Returns cohort counts, filters, SQL trace, charts, rows, summaries, "
            "limitations, and recommendations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's healthcare analytics question.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }
]


def _compact_tool_output(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "analysis_domain": metrics.get("analysis_domain"),
        "query_intent": metrics.get("query_intent"),
        "total_records": metrics.get("total_records"),
        "pct_share": metrics.get("pct_share"),
        "risk_index": metrics.get("risk_index"),
        "applied_filters": metrics.get("applied_filters", []),
        "aggregate_summary": metrics.get("aggregate_summary", {}),
        "summary_data": metrics.get("summary_data", [])[:10],
        "executive_summary": metrics.get("executive_summary"),
        "recommendations": metrics.get("recommendations", []),
        "limitations": metrics.get("limitations", []),
        "generated_sql": metrics.get("generated_sql"),
        "count_sql": metrics.get("count_sql"),
        "sql_params": metrics.get("sql_params", []),
    }


def _fallback_metrics(query: str, reason: str) -> Dict[str, Any]:
    metrics = hybrid_rag_lookup(query)
    metrics.update(
        {
            "agent_enabled": False,
            "agent_model": None,
            "agent_response": metrics.get("executive_summary", ""),
            "agent_trace": [
                {
                    "step": "Agent fallback",
                    "detail": reason,
                },
                {
                    "step": "Tool executed",
                    "detail": "Used deterministic healthcare_analytics_lookup directly.",
                },
            ],
        }
    )
    return metrics


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text

    chunks: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            content_text = getattr(content, "text", None)
            if content_text:
                chunks.append(content_text)
    return "\n".join(chunks).strip()


def run_agentic_analysis(query: str) -> Dict[str, Any]:
    query = str(query or "").strip()
    if not query:
        return _fallback_metrics(query, "Empty query; no model call attempted.")

    if not os.getenv("OPENAI_API_KEY"):
        return _fallback_metrics(
            query,
            "OPENAI_API_KEY is not configured. Add it to .env to enable the tool-calling agent.",
        )

    client = OpenAI()
    input_list: List[Dict[str, Any]] = [{"role": "user", "content": query}]

    try:
        first_response = client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            tools=TOOLS,
            input=input_list,
        )

        input_list.extend(first_response.output)
        tool_metrics: Dict[str, Any] | None = None
        tool_calls = []

        for item in first_response.output:
            if item.type != "function_call":
                continue

            tool_calls.append(item.name)
            if item.name != "healthcare_analytics_lookup":
                continue

            arguments = json.loads(item.arguments or "{}")
            tool_query = arguments.get("query") or query
            tool_metrics = hybrid_rag_lookup(tool_query)
            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(_compact_tool_output(tool_metrics), default=str),
                }
            )

        if tool_metrics is None:
            tool_metrics = hybrid_rag_lookup(query)
            tool_metrics["agent_trace"] = [
                {
                    "step": "Tool selection",
                    "detail": "Model did not call the analytics tool, so the app executed it directly.",
                }
            ]
        else:
            tool_metrics["agent_trace"] = [
                {
                    "step": "Tool selection",
                    "detail": f"Model requested tools: {', '.join(tool_calls)}.",
                },
                {
                    "step": "Tool execution",
                    "detail": "Executed healthcare_analytics_lookup against local SQLite.",
                },
            ]

        final_response = client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            tools=TOOLS,
            input=input_list,
        )

        tool_metrics.update(
            {
                "agent_enabled": True,
                "agent_model": DEFAULT_MODEL,
                "agent_response": _response_text(final_response),
                "agent_trace": tool_metrics.get("agent_trace", [])
                + [
                    {
                        "step": "Final synthesis",
                        "detail": "Model summarized the tool output into an executive answer.",
                    }
                ],
            }
        )
        return tool_metrics
    except Exception as exc:
        return _fallback_metrics(query, f"Agent call failed: {exc}")
