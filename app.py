
import time

import pandas as pd
import streamlit as st

from agents.healthcare_agent import run_agentic_analysis


st.set_page_config(
    page_title="Healthcare Operational Analytics Engine",
    page_icon="🏥",
    layout="wide",
)


def _records_to_df(records):
    return pd.DataFrame(records or [])


def _render_bar_chart(records, index_col="segment", value_cols=None):
    df = _records_to_df(records)
    if df.empty or index_col not in df.columns:
        st.info("No chartable records available for this view.")
        return

    if value_cols is None:
        value_cols = ["records"]
    available_cols = [col for col in value_cols if col in df.columns]
    if not available_cols:
        st.info("No numeric measure available for this chart.")
        return

    chart_df = df[[index_col] + available_cols].set_index(index_col)
    st.bar_chart(chart_df, use_container_width=True)


def _render_overview(metrics):
    intent_col, confidence_col, rows_col, share_col = st.columns(4)
    with intent_col:
        st.metric("Query Intent", metrics.get("query_intent", "Unclassified"))
    with confidence_col:
        st.metric("Confidence", metrics.get("confidence", "Low"))
    with rows_col:
        st.metric(
            metrics.get("metric_label", "Matching Records"),
            f"{metrics.get('total_records', 0):,}",
        )
    with share_col:
        st.metric(
            metrics.get("tertiary_metric_label", "Share of Source Data"),
            str(metrics.get("tertiary_metric_value", metrics.get("pct_share", "0.0%"))),
        )

    st.metric(
        metrics.get("secondary_metric_label", "Mean Risk Score"),
        str(metrics.get("secondary_metric_value", metrics.get("risk_index", 0))),
    )


def _render_visuals(metrics):
    chart_data = metrics.get("chart_data", {})
    domain = metrics.get("analysis_domain")

    if domain == "unsupported":
        st.info("No visualizations were generated because the request was outside the healthcare operations dataset.")
        return

    if domain == "claims":
        st.markdown("#### Claim Status")
        _render_bar_chart(
            chart_data.get("status_bar", []),
            value_cols=["records", "total_claim_amount"],
        )

        st.markdown("#### Denial Reason Mix")
        _render_bar_chart(
            chart_data.get("denial_reasons", []),
            value_cols=["records", "total_claim_amount"],
        )
        return

    st.markdown("#### Segment Summary")
    segment_records = chart_data.get("segment_bar") or metrics.get("summary_data", [])
    _render_bar_chart(
        segment_records,
        value_cols=["records", "mean_risk_score"],
    )

    st.markdown("#### Risk Score Distribution")
    _render_bar_chart(
        chart_data.get("risk_distribution", []),
        value_cols=["records"],
    )

    st.markdown("#### Insurance Type vs Readmission Label")
    _render_bar_chart(
        chart_data.get("insurance_readmission_stack", []),
        value_cols=["readmitted", "not_readmitted"],
    )

    st.markdown("#### Top Diagnosis Drivers")
    _render_bar_chart(
        chart_data.get("top_drivers", []),
        value_cols=["records", "mean_risk_score", "mean_length_of_stay"],
    )


def _render_trace(metrics):
    trace_df = _records_to_df(metrics.get("reasoning_trace", []))
    if not trace_df.empty:
        st.dataframe(trace_df, use_container_width=True, hide_index=True)

    generated_sql = metrics.get("generated_sql")
    if generated_sql:
        st.markdown("#### Row Query")
        st.code(generated_sql, language="sql")

    count_sql = metrics.get("count_sql")
    if count_sql:
        st.markdown("#### Count Query")
        st.code(count_sql, language="sql")

    st.markdown("#### SQL Parameters")
    st.write(metrics.get("sql_params", []))

    limitations = metrics.get("limitations", [])
    if limitations:
        st.markdown("#### Limitations")
        for limitation in limitations:
            st.write(f"- {limitation}")


st.title("🏥 Healthcare Operational Analytics Dashboard")
st.markdown("---")

if "cached_telemetry" not in st.session_state:
    st.session_state["cached_telemetry"] = None

user_query = st.text_input(
    "🔍 Enter Operational Query or Analysis Objective:",
    value="",
    placeholder="Ask about readmission risk, payer mix, diagnosis drivers, claim denials, or cohorts...",
)

if st.button("Execute Autonomous Analysis", type="primary"):
    if not user_query.strip():
        st.warning("Please enter an analytics objective first.")
    else:
        with st.status("🚀 Computing database filters, aggregates, and trace...", expanded=True) as status:
            time.sleep(0.3)
            computed_metrics = run_agentic_analysis(user_query)
            st.session_state["cached_telemetry"] = computed_metrics
            if computed_metrics.get("analysis_domain") == "unsupported":
                status.update(label="⚠️ Request rejected by domain guardrail.", state="complete")
            else:
                status.update(label="✅ Analysis complete.", state="complete")

if st.session_state["cached_telemetry"] is not None:
    metrics = st.session_state["cached_telemetry"]

    st.markdown(f"### 📋 {metrics.get('display_title', 'Filtered Patient Cohort')}")
    if metrics.get("analysis_domain") == "unsupported":
        st.caption(
            f"🟡 Domain guardrail response | Source: {metrics.get('source_table', 'none')} | "
            f"Snapshot: {metrics.get('snapshot_utc', 'unknown')}"
        )
    else:
        st.caption(
            f"🟢 Live SQLite analytical snapshot | Source: {metrics.get('source_table', 'database')} | "
            f"Snapshot: {metrics.get('snapshot_utc', 'unknown')}"
        )

    if metrics.get("error"):
        st.error(metrics["error"])

    if metrics.get("agent_enabled"):
        st.caption(f"🤖 Agentic synthesis enabled | Model: {metrics.get('agent_model')}")
    else:
        st.caption("🧭 Deterministic analytics fallback active. Add OPENAI_API_KEY to enable model tool-calling.")

    _render_overview(metrics)

    applied_filters = metrics.get("applied_filters", [])
    if metrics.get("analysis_domain") == "unsupported":
        st.warning("No healthcare operations domain match was found. SQL execution was skipped.")
    elif applied_filters:
        st.info("Active filters: " + " | ".join(applied_filters))
    else:
        st.info("Active filters: none. Showing the broadest matching operational view.")

    executive_summary = metrics.get("executive_summary")
    if executive_summary:
        if metrics.get("analysis_domain") == "unsupported":
            st.warning(executive_summary)
        else:
            st.success(executive_summary)

    suggestions = metrics.get("no_result_suggestions", [])
    if suggestions:
        st.warning("No-result guidance:")
        for suggestion in suggestions:
            st.write(f"- {suggestion}")

    answer_tab, visual_tab, rows_tab, trace_tab = st.tabs(
        ["Answer", "Visuals", "Rows", "Trace"]
    )

    with answer_tab:
        agent_response = metrics.get("agent_response")
        if agent_response:
            st.markdown("#### Agent Answer")
            st.write(agent_response)

        summary_rows = metrics.get("summary_data", [])
        aggregate_summary = metrics.get("aggregate_summary", {})

        if aggregate_summary:
            st.markdown("#### Aggregate Summary")
            st.dataframe(pd.DataFrame([aggregate_summary]), use_container_width=True, hide_index=True)

        if summary_rows:
            st.markdown("#### Dynamic Segment Summary")
            st.dataframe(_records_to_df(summary_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Analyst Strategic Recommendations")
        for recommendation in metrics.get("recommendations", []):
            st.write(f"- {recommendation}")

    with visual_tab:
        _render_visuals(metrics)

    with rows_tab:
        raw_rows = metrics.get("df_data", None)
        if raw_rows:
            st.dataframe(_records_to_df(raw_rows), use_container_width=True)
            st.success("Rows synchronized from live SQLite query output.")
        elif metrics.get("analysis_domain") == "unsupported":
            st.info("No database rows were queried for this out-of-scope request.")
        else:
            st.info("No matching records found for the specified criteria.")

    with trace_tab:
        agent_trace = metrics.get("agent_trace", [])
        if agent_trace:
            st.markdown("#### Agent Trace")
            st.dataframe(_records_to_df(agent_trace), use_container_width=True, hide_index=True)

        _render_trace(metrics)
