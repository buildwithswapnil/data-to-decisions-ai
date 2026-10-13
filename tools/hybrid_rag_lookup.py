import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "db" / "healthcare_data.db"

CLAIMS_TERMS = {
    "claim",
    "claims",
    "denial",
    "denied",
    "billing",
    "revenue",
    "leakage",
    "icd",
    "code",
    "coding",
}

HEALTHCARE_DOMAIN_TERMS = {
    "admission",
    "admissions",
    "appendicitis",
    "billing",
    "claim",
    "claims",
    "clinical",
    "coding",
    "cohort",
    "comorbidities",
    "comorbidity",
    "conservative",
    "copd",
    "denial",
    "denied",
    "diabetes",
    "diagnosis",
    "diagnoses",
    "discharge",
    "facility",
    "followup",
    "follow-up",
    "fracture",
    "gender",
    "healthcare",
    "heart failure",
    "hospital",
    "hypertension",
    "icd",
    "influenza",
    "insurance",
    "interventional",
    "kidney disease",
    "length of stay",
    "los",
    "medicaid",
    "medical",
    "medicare",
    "medication",
    "medications",
    "patient",
    "patients",
    "payer",
    "pneumonia",
    "private",
    "readmission",
    "readmissions",
    "region",
    "revenue",
    "risk",
    "sepsis",
    "stroke",
    "surgical",
    "treatment",
    "uninsured",
}

READMISSION_DIMENSIONS = {
    "diagnosis": "primary_diagnosis",
    "diagnoses": "primary_diagnosis",
    "condition": "primary_diagnosis",
    "conditions": "primary_diagnosis",
    "disease": "primary_diagnosis",
    "treatment": "treatment_type",
    "treatments": "treatment_type",
    "therapy": "treatment_type",
    "insurance": "insurance_type",
    "payer": "insurance_type",
    "plan": "insurance_type",
    "region": "region",
    "regions": "region",
    "gender": "gender",
    "sex": "gender",
    "season": "season",
    "discharge": "discharge_disposition",
    "disposition": "discharge_disposition",
}

READMISSION_CATEGORICAL_COLUMNS = [
    "season",
    "gender",
    "region",
    "primary_diagnosis",
    "treatment_type",
    "insurance_type",
    "discharge_disposition",
]

READMISSION_NUMERIC_SPECS = [
    ("age", ["age", "aged", "years old", "year old", "yo"], "Age"),
    (
        "readmission_risk_score",
        ["readmission risk score", "risk score", "readmission risk", "risk"],
        "Readmission risk score",
    ),
    ("length_of_stay", ["length of stay", "los", "stay"], "Length of stay"),
    ("comorbidities_count", ["comorbidities", "comorbidity"], "Comorbidities"),
    ("medications_count", ["medications", "medication"], "Medications"),
    (
        "followup_visits_last_year",
        ["follow up visits", "follow-up visits", "followup visits", "visits"],
        "Follow-up visits",
    ),
    (
        "prev_readmissions",
        ["previous readmissions", "prior readmissions", "prev readmissions"],
        "Previous readmissions",
    ),
]

DEFAULT_READMISSION_COLUMNS = [
    "patient_id",
    "admission_date",
    "age",
    "gender",
    "region",
    "primary_diagnosis",
    "treatment_type",
    "insurance_type",
    "discharge_disposition",
    "length_of_stay",
    "prev_readmissions",
    "readmission_risk_score",
    "label",
]


def hybrid_rag_lookup(query_text: str) -> dict:
    """
    Dynamic schema-aware retrieval layer for the healthcare operations app.

    The previous implementation queried a non-existent patient_records table and
    returned hard-coded demo counts after the SQL failed. This version inspects the
    live SQLite schema, builds parameterized filters from the user's wording, and
    returns the actual cohort or claims records that match the ask.
    """
    query = str(query_text or "").strip()
    metrics = _empty_metrics(query)

    if not DB_PATH.exists():
        metrics["error"] = f"Database not found at {DB_PATH}"
        metrics["recommendations"] = [
            "Verify that db/healthcare_data.db exists before running analysis."
        ]
        return metrics

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            available_tables = _list_tables(conn)

            if not _is_supported_healthcare_question(conn, query, available_tables):
                return _unsupported_query_metrics(metrics)

            if _is_claims_question(query) and "claims" in available_tables:
                return _lookup_claims(conn, query, metrics)

            if "readmissions" in available_tables:
                return _lookup_readmissions(conn, query, metrics)

            metrics["error"] = "No supported operational table was found."
            metrics["recommendations"] = [
                "Expected a readmissions table or claims table in the SQLite database."
            ]
            return metrics
    except Exception as exc:
        metrics["error"] = str(exc)
        metrics["recommendations"] = [
            "The database query failed. Review the generated filters and table schema before trusting the result."
        ]
        return metrics


def _empty_metrics(query: str) -> Dict[str, Any]:
    return {
        "query": query,
        "analysis_domain": "unknown",
        "is_list": True,
        "total_records": 0,
        "risk_index": 0.0,
        "pct_share": "0.0%",
        "df_data": [],
        "summary_data": [],
        "aggregate_summary": {},
        "chart_data": {},
        "applied_filters": [],
        "query_intent": "Unclassified",
        "executive_summary": "",
        "reasoning_trace": [],
        "confidence": "Low",
        "limitations": [],
        "no_result_suggestions": [],
        "display_title": "Dynamic Healthcare Operations Analysis",
        "generated_sql": "",
        "count_sql": "",
        "sql_params": [],
        "source_table": "",
        "metric_label": "Matching Records",
        "secondary_metric_label": "Mean Risk Score",
        "secondary_metric_value": "0.000",
        "tertiary_metric_label": "Share of Source Data",
        "recommendations": [],
        "snapshot_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "error": None,
    }


def _unsupported_query_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    metrics.update(
        {
            "analysis_domain": "unsupported",
            "query_intent": "Out-of-Scope Request",
            "display_title": "Unsupported Analytics Request",
            "source_table": "none",
            "metric_label": "Rows Queried",
            "secondary_metric_label": "Database Query Executed",
            "secondary_metric_value": "No",
            "tertiary_metric_label": "Domain Match",
            "tertiary_metric_value": "Rejected",
            "total_records": 0,
            "pct_share": "0.0%",
            "confidence": "Rejected",
            "executive_summary": (
                "This request is outside the healthcare operations dataset, so no SQL was executed. "
                "Ask about readmissions, patient cohorts, diagnoses, treatment types, payer mix, risk, "
                "admissions, discharge, or claims."
            ),
            "recommendations": [
                "Rephrase the question using healthcare operations terms such as readmission, diagnosis, treatment, insurance, payer, risk, admission, discharge, or claims.",
                "For non-healthcare research tasks, use a separate research workflow instead of this database analytics agent.",
            ],
            "limitations": [
                "The app is intentionally scoped to the local healthcare operations database.",
                "Out-of-domain questions are blocked to prevent misleading broad-table fallback results.",
            ],
            "reasoning_trace": [
                {
                    "step": "Domain Guardrail",
                    "detail": "No healthcare operations terms, table names, columns, or known categorical values were detected.",
                },
                {"step": "Action", "detail": "Skipped SQL execution and returned an unsupported-query response."},
            ],
        }
    )
    return metrics


def _list_tables(conn: sqlite3.Connection) -> set:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
    return [row[1] for row in rows]


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[_/\\-]+", " ", text)
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_phrase(normalized_query: str, phrase: Any) -> bool:
    normalized_phrase = _normalize_text(phrase)
    if not normalized_phrase:
        return False
    return bool(
        re.search(rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])", normalized_query)
    )


def _is_claims_question(query: str) -> bool:
    query_norm = _normalize_text(query)
    return any(_contains_phrase(query_norm, term) for term in CLAIMS_TERMS)


def _is_supported_healthcare_question(
    conn: sqlite3.Connection, query: str, available_tables: set
) -> bool:
    query_norm = _normalize_text(query)
    if not query_norm:
        return False

    if any(_contains_phrase(query_norm, term) for term in HEALTHCARE_DOMAIN_TERMS):
        return True

    ignored_columns = {
        "adm_id",
        "claim_id",
        "diag_id",
        "id",
        "label",
        "name",
        "patient_id",
        "status",
    }
    for table in available_tables:
        if _contains_phrase(query_norm, table):
            return True
        for column in _table_columns(conn, table):
            if column in ignored_columns:
                continue
            if _contains_phrase(query_norm, column):
                return True

    value_checks = {
        "readmissions": ["primary_diagnosis", "treatment_type", "insurance_type", "discharge_disposition"],
        "claims": ["denial_reason"],
        "diagnoses": ["icd_code", "description"],
    }
    for table, columns in value_checks.items():
        if table not in available_tables:
            continue
        table_columns = set(_table_columns(conn, table))
        for column in columns:
            if column not in table_columns:
                continue
            for value in _distinct_values(conn, table, column):
                if _contains_phrase(query_norm, value):
                    return True

    return False


def _lookup_readmissions(
    conn: sqlite3.Connection, query: str, metrics: Dict[str, Any]
) -> Dict[str, Any]:
    table = "readmissions"
    columns = _table_columns(conn, table)
    query_norm = _normalize_text(query)
    filters: List[Dict[str, Any]] = []

    _add_categorical_filters(conn, table, columns, query_norm, filters)
    _add_readmission_label_filter(query_norm, columns, filters)
    _add_year_filters(query, columns, filters)
    _add_numeric_filters(query, query_norm, columns, filters)

    where_sql, params = _compile_filters(filters)
    total_source = _scalar(conn, f"SELECT COUNT(*) FROM {_quote_identifier(table)}")
    count_sql = f"SELECT COUNT(*) FROM {_quote_identifier(table)}{where_sql}"
    total_records = _scalar(conn, count_sql, params)

    summary_sql = f"""
        SELECT
            COUNT(*) AS records,
            ROUND(AVG(readmission_risk_score), 3) AS mean_risk_score,
            ROUND(AVG(length_of_stay), 1) AS mean_length_of_stay,
            ROUND(AVG(prev_readmissions), 2) AS mean_previous_readmissions,
            ROUND(AVG(label) * 100, 1) AS readmission_rate_pct
        FROM {_quote_identifier(table)}
        {where_sql}
    """
    summary = pd.read_sql_query(summary_sql, conn, params=params).fillna(0)

    group_col = _choose_readmission_group_column(query_norm)
    summary_data: List[Dict[str, Any]] = []
    if group_col in columns:
        group_sql = f"""
            SELECT
                {_quote_identifier(group_col)} AS segment,
                COUNT(*) AS records,
                ROUND(AVG(readmission_risk_score), 3) AS mean_risk_score,
                ROUND(AVG(length_of_stay), 1) AS mean_length_of_stay,
                ROUND(AVG(prev_readmissions), 2) AS mean_previous_readmissions,
                ROUND(AVG(label) * 100, 1) AS readmission_rate_pct
            FROM {_quote_identifier(table)}
            {where_sql}
            GROUP BY {_quote_identifier(group_col)}
            ORDER BY records DESC, mean_risk_score DESC
            LIMIT 10
        """
        summary_data = pd.read_sql_query(group_sql, conn, params=params).to_dict(
            orient="records"
    )

    select_columns = [col for col in DEFAULT_READMISSION_COLUMNS if col in columns]
    select_sql = ",\n            ".join(_quote_identifier(col) for col in select_columns) or "*"
    row_sql = f"""
        SELECT
            {select_sql}
        FROM {_quote_identifier(table)}
        {where_sql}
        ORDER BY readmission_risk_score DESC, admission_date DESC
        LIMIT 50
    """
    rows = pd.read_sql_query(row_sql, conn, params=params)

    summary_row = summary.iloc[0].to_dict() if not summary.empty else {}
    risk_index = float(summary_row.get("mean_risk_score") or 0)
    pct_share = _format_pct(total_records, total_source)
    query_intent = _classify_readmission_intent(query_norm, filters, group_col)
    chart_data = _readmission_chart_data(
        conn, table, where_sql, params, columns, rows, summary_data, group_col
    )
    no_result_suggestions = (
        _readmission_no_result_suggestions(conn, table, columns) if total_records == 0 else []
    )

    metrics.update(
        {
            "analysis_domain": "readmissions",
            "source_table": table,
            "query_intent": query_intent,
            "metric_label": "Matching Patient Records",
            "secondary_metric_label": "Mean Risk Score",
            "secondary_metric_value": f"{risk_index:.3f}",
            "tertiary_metric_label": "Share of Readmission Data",
            "total_records": int(total_records),
            "risk_index": round(risk_index, 3),
            "pct_share": pct_share,
            "df_data": rows.to_dict(orient="records"),
            "summary_data": summary_data,
            "aggregate_summary": summary_row,
            "chart_data": chart_data,
            "applied_filters": [item["label"] for item in filters],
            "display_title": _build_readmission_title(filters, group_col),
            "executive_summary": _readmission_executive_summary(
                total_records, pct_share, risk_index, summary_row, summary_data, group_col, filters
            ),
            "reasoning_trace": _reasoning_trace(
                query_intent=query_intent,
                table=table,
                filters=filters,
                count_sql=count_sql,
                row_sql=row_sql,
                params=params,
                total_records=total_records,
                total_source=total_source,
                group_col=group_col,
            ),
            "confidence": _confidence_label(total_records, filters),
            "limitations": _readmission_limitations(filters, total_records),
            "no_result_suggestions": no_result_suggestions,
            "generated_sql": _format_sql(row_sql),
            "count_sql": _format_sql(count_sql),
            "sql_params": list(params),
            "recommendations": _readmission_recommendations(
                total_records, summary_row, summary_data, group_col
            ),
        }
    )
    return metrics


def _lookup_claims(
    conn: sqlite3.Connection, query: str, metrics: Dict[str, Any]
) -> Dict[str, Any]:
    query_norm = _normalize_text(query)
    filters: List[Dict[str, Any]] = []

    claim_columns = _table_columns(conn, "claims")
    diagnosis_columns = _table_columns(conn, "diagnoses") if "diagnoses" in _list_tables(conn) else []

    if "status" in claim_columns:
        statuses = _distinct_values(conn, "claims", "status")
        for status in statuses:
            if _contains_phrase(query_norm, status):
                _append_filter(filters, "c.status = ?", [status], f"Claim status = {status}")
                break
        if not any("Claim status" in item["label"] for item in filters):
            if _contains_phrase(query_norm, "denied") or _contains_phrase(query_norm, "denial"):
                _append_filter(filters, "LOWER(c.status) = ?", ["denied"], "Claim status = Denied")

    if "denial_reason" in claim_columns:
        for reason in _distinct_values(conn, "claims", "denial_reason"):
            if _contains_phrase(query_norm, reason):
                _append_filter(
                    filters, "c.denial_reason = ?", [reason], f"Denial reason = {reason}"
                )
                break

    if "description" in diagnosis_columns:
        for description in _distinct_values(conn, "diagnoses", "description"):
            if _contains_phrase(query_norm, description):
                _append_filter(
                    filters,
                    "LOWER(d.description) = LOWER(?)",
                    [description],
                    f"Diagnosis = {description}",
                )
                break

    icd_match = re.search(r"\b([a-z]\d{1,2}(?:\.\d+)?)\b", query, flags=re.IGNORECASE)
    if icd_match and "icd_code" in diagnosis_columns:
        icd_code = icd_match.group(1).upper()
        _append_filter(filters, "UPPER(d.icd_code) = ?", [icd_code], f"ICD code = {icd_code}")

    amount_filter = _extract_comparison_filter(
        query,
        ["amount", "charge", "claim amount", "revenue"],
        "c.amount",
        "Claim amount",
    )
    if amount_filter:
        filters.append(amount_filter)

    age_filter = _extract_comparison_filter(
        query,
        ["age", "aged", "years old", "year old"],
        "p.age",
        "Patient age",
    )
    if age_filter:
        filters.append(age_filter)

    facility_match = re.search(r"\bfacility\s*(?:id)?\s*(\d+)\b", query_norm)
    if facility_match:
        facility_id = int(facility_match.group(1))
        _append_filter(filters, "a.facility_id = ?", [facility_id], f"Facility ID = {facility_id}")

    where_sql, params = _compile_filters(filters)
    base_from = """
        FROM claims c
        LEFT JOIN admissions a ON c.adm_id = a.adm_id
        LEFT JOIN patients p ON a.patient_id = p.patient_id
        LEFT JOIN diagnoses d ON a.adm_id = d.adm_id
    """
    total_source = _scalar(conn, f"SELECT COUNT(*) {base_from}")
    count_sql = f"SELECT COUNT(*) {base_from} {where_sql}"
    total_records = _scalar(conn, count_sql, params)

    summary_sql = f"""
        SELECT
            c.status AS segment,
            COUNT(*) AS records,
            ROUND(AVG(c.amount), 2) AS mean_claim_amount,
            ROUND(SUM(c.amount), 2) AS total_claim_amount
        {base_from}
        {where_sql}
        GROUP BY c.status
        ORDER BY records DESC
    """
    summary_data = pd.read_sql_query(summary_sql, conn, params=params).to_dict(orient="records")

    row_sql = f"""
        SELECT
            c.claim_id,
            c.status,
            c.amount,
            c.denial_reason,
            a.adm_id,
            a.adm_date,
            a.discharge_date,
            a.facility_id,
            p.patient_id,
            p.name,
            p.age,
            p.gender,
            d.icd_code,
            d.description AS diagnosis
        {base_from}
        {where_sql}
        ORDER BY c.amount DESC, c.claim_id
        LIMIT 50
    """
    rows = pd.read_sql_query(row_sql, conn, params=params)

    avg_amount = _scalar(conn, f"SELECT ROUND(AVG(c.amount), 2) {base_from} {where_sql}", params)
    query_intent = _classify_claims_intent(query_norm, filters)
    chart_data = _claims_chart_data(conn, base_from, where_sql, params, rows, summary_data)
    no_result_suggestions = _claims_no_result_suggestions(conn) if total_records == 0 else []

    metrics.update(
        {
            "analysis_domain": "claims",
            "source_table": "claims",
            "query_intent": query_intent,
            "metric_label": "Matching Claims",
            "secondary_metric_label": "Average Claim Amount",
            "secondary_metric_value": _format_currency(avg_amount),
            "tertiary_metric_label": "Share of Claims Data",
            "total_records": int(total_records),
            "risk_index": 0.0,
            "pct_share": _format_pct(total_records, total_source),
            "df_data": rows.to_dict(orient="records"),
            "summary_data": summary_data,
            "aggregate_summary": {
                "average_claim_amount": avg_amount or 0,
                "records": total_records,
            },
            "chart_data": chart_data,
            "applied_filters": [item["label"] for item in filters],
            "display_title": _build_claims_title(filters),
            "executive_summary": _claims_executive_summary(
                total_records, avg_amount, summary_data, filters
            ),
            "reasoning_trace": _reasoning_trace(
                query_intent=query_intent,
                table="claims + admissions + patients + diagnoses",
                filters=filters,
                count_sql=count_sql,
                row_sql=row_sql,
                params=params,
                total_records=total_records,
                total_source=total_source,
                group_col="status",
            ),
            "confidence": _confidence_label(total_records, filters),
            "limitations": _claims_limitations(filters, total_records),
            "no_result_suggestions": no_result_suggestions,
            "generated_sql": _format_sql(row_sql),
            "count_sql": _format_sql(count_sql),
            "sql_params": list(params),
            "recommendations": _claims_recommendations(total_records, summary_data),
        }
    )
    return metrics


def _add_categorical_filters(
    conn: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
    query_norm: str,
    filters: List[Dict[str, Any]],
) -> None:
    for column in READMISSION_CATEGORICAL_COLUMNS:
        if column not in columns:
            continue

        values = sorted(
            _distinct_values(conn, table, column),
            key=lambda item: len(_normalize_text(item)),
            reverse=True,
        )
        for value in values:
            if _contains_phrase(query_norm, value):
                label = f"{_humanize(column)} = {value}"
                _append_filter(
                    filters,
                    f"{_quote_identifier(column)} = ?",
                    [value],
                    label,
                    column=column,
                )
                break


def _add_readmission_label_filter(
    query_norm: str, columns: Sequence[str], filters: List[Dict[str, Any]]
) -> None:
    if "label" not in columns:
        return

    if (
        _contains_phrase(query_norm, "not readmitted")
        or _contains_phrase(query_norm, "no readmission")
        or _contains_phrase(query_norm, "without readmission")
    ):
        _append_filter(filters, "label = ?", [0], "Readmission label = 0", column="label")
        return

    positive_patterns = [
        "readmitted patients",
        "readmitted",
        "patients readmitted",
        "actual readmissions",
        "readmission cases",
        "readmission rates",
        "readmission surges",
        "readmission surge",
        "causing readmission",
        "readmission root cause",
    ]
    if any(_contains_phrase(query_norm, pattern) for pattern in positive_patterns):
        _append_filter(filters, "label = ?", [1], "Readmission label = 1", column="label")


def _add_year_filters(
    query: str, columns: Sequence[str], filters: List[Dict[str, Any]]
) -> None:
    if "admission_date" not in columns:
        return

    years = sorted({year for year in re.findall(r"\b(19\d{2}|20\d{2})\b", query)})
    if not years:
        return

    placeholders = ", ".join("?" for _ in years)
    _append_filter(
        filters,
        f"strftime('%Y', admission_date) IN ({placeholders})",
        years,
        "Admission year in " + ", ".join(years),
        column="admission_date",
    )


def _add_numeric_filters(
    query: str,
    query_norm: str,
    columns: Sequence[str],
    filters: List[Dict[str, Any]],
) -> None:
    used_columns = {item.get("column") for item in filters}

    for column, aliases, label in READMISSION_NUMERIC_SPECS:
        if column not in columns or column in used_columns:
            continue

        if column == "readmission_risk_score":
            risk_band_filter = _risk_band_filter(query_norm)
            if risk_band_filter:
                filters.append(risk_band_filter)
                used_columns.add(column)
                continue

        numeric_filter = _extract_comparison_filter(
            query, aliases, _quote_identifier(column), label, column=column
        )
        if numeric_filter:
            filters.append(numeric_filter)
            used_columns.add(column)

    if "age" in columns and "age" not in used_columns:
        inferred_age = _infer_age_range_filter(query)
        if inferred_age:
            filters.append(inferred_age)


def _risk_band_filter(query_norm: str) -> Optional[Dict[str, Any]]:
    if _contains_phrase(query_norm, "high risk") or _contains_phrase(query_norm, "high-risk"):
        return {
            "sql": "readmission_risk_score >= ?",
            "params": [0.8],
            "label": "Readmission risk score >= 0.8",
            "column": "readmission_risk_score",
        }
    if _contains_phrase(query_norm, "low risk") or _contains_phrase(query_norm, "low-risk"):
        return {
            "sql": "readmission_risk_score < ?",
            "params": [0.4],
            "label": "Readmission risk score < 0.4",
            "column": "readmission_risk_score",
        }
    if _contains_phrase(query_norm, "moderate risk") or _contains_phrase(query_norm, "medium risk"):
        return {
            "sql": "readmission_risk_score BETWEEN ? AND ?",
            "params": [0.4, 0.8],
            "label": "Readmission risk score between 0.4 and 0.8",
            "column": "readmission_risk_score",
        }
    return None


def _extract_comparison_filter(
    query: str,
    aliases: Sequence[str],
    sql_column: str,
    label: str,
    column: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    cleaned_query = re.sub(r"[–—]", "-", query.lower())
    cleaned_query = re.sub(r"\s+", " ", cleaned_query)
    alias_pattern = "|".join(re.escape(alias.lower()) for alias in sorted(aliases, key=len, reverse=True))
    number = r"(\d+(?:\.\d+)?)"

    range_patterns = [
        rf"(?:{alias_pattern})\D{{0,25}}(?:between|from)?\s*{number}\s*(?:and|to|-)\s*{number}",
    ]
    for pattern in range_patterns:
        match = re.search(pattern, cleaned_query)
        if match:
            low, high = sorted([_coerce_number(match.group(1)), _coerce_number(match.group(2))])
            if not _numeric_filter_is_plausible(column, [low, high], match.group(0)):
                continue
            return {
                "sql": f"{sql_column} BETWEEN ? AND ?",
                "params": [low, high],
                "label": f"{label} between {low:g} and {high:g}",
                "column": column,
            }

    comparison_alias_first = [
        ("<=", ["at most", "no more than", "less than or equal to", "under", "below", "less than"]),
        (">=", ["at least", "no less than", "greater than or equal to", "over", "above", "more than", "older than"]),
    ]
    for operator, words in comparison_alias_first:
        words_pattern = "|".join(re.escape(word) for word in words)
        patterns = [
            rf"(?:{alias_pattern})\D{{0,25}}(?:{words_pattern}|{re.escape(operator)})\s*{number}",
            rf"(?:{words_pattern}|{re.escape(operator)})\s*{number}\D{{0,25}}(?:{alias_pattern})",
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned_query)
            if match:
                value = _coerce_number(match.group(1))
                if not _numeric_filter_is_plausible(column, [value], match.group(0)):
                    continue
                return {
                    "sql": f"{sql_column} {operator} ?",
                    "params": [value],
                    "label": f"{label} {operator} {value:g}",
                    "column": column,
                }

    exact_patterns = [
        rf"(?:{alias_pattern})\s*(?:=|is|of)?\s*{number}",
        rf"{number}\s*(?:year old|years old|yo)",
    ]
    for pattern in exact_patterns:
        match = re.search(pattern, cleaned_query)
        if match:
            value = _coerce_number(match.group(1))
            if not _numeric_filter_is_plausible(column, [value], match.group(0)):
                continue
            return {
                "sql": f"{sql_column} = ?",
                "params": [value],
                "label": f"{label} = {value:g}",
                "column": column,
            }

    return None


def _numeric_filter_is_plausible(
    column: Optional[str], values: Sequence[float], matched_text: str
) -> bool:
    if not column:
        return True

    # Prevent metric requests such as "mean risk scores for individuals aged
    # between 65 and 85" from borrowing the age range as a risk-score filter.
    if column != "age" and re.search(r"\b(age|aged|year old|years old|yo)\b", matched_text):
        return False

    bounds = {
        "age": (0, 120),
        "readmission_risk_score": (0, 1),
        "length_of_stay": (0, 60),
        "comorbidities_count": (0, 20),
        "medications_count": (0, 50),
        "followup_visits_last_year": (0, 50),
        "prev_readmissions": (0, 20),
    }
    if column not in bounds:
        return True

    low, high = bounds[column]
    return all(low <= float(value) <= high for value in values)


def _infer_age_range_filter(query: str) -> Optional[Dict[str, Any]]:
    cleaned_query = re.sub(r"[–—]", "-", query.lower())
    cleaned_query = re.sub(r"\s+", " ", cleaned_query)
    match = re.search(r"\b(?:between|from)?\s*(\d{1,3})\s*(?:and|to|-)\s*(\d{1,3})\b", cleaned_query)
    if not match:
        return None

    low, high = sorted([int(match.group(1)), int(match.group(2))])
    if not (0 < low < 120 and 0 < high < 120):
        return None

    prefix = cleaned_query[max(0, match.start() - 8) : match.start()]
    if re.search(r"\b(top|last|past)\s*$", prefix):
        return None

    return {
        "sql": "age BETWEEN ? AND ?",
        "params": [low, high],
        "label": f"Age between {low} and {high}",
        "column": "age",
    }


def _choose_readmission_group_column(query_norm: str) -> Optional[str]:
    for alias, column in sorted(READMISSION_DIMENSIONS.items(), key=lambda item: len(item[0]), reverse=True):
        if (
            _contains_phrase(query_norm, f"by {alias}")
            or _contains_phrase(query_norm, f"per {alias}")
            or _contains_phrase(query_norm, f"across {alias}")
            or _contains_phrase(query_norm, f"top {alias}")
            or _contains_phrase(query_norm, f"which {alias}")
        ):
            return column

    driver_terms = ["cause", "causing", "driver", "drivers", "root cause", "surge", "surges"]
    if any(_contains_phrase(query_norm, term) for term in driver_terms):
        return "primary_diagnosis"

    breakdown_terms = ["breakdown", "distribution", "segment", "segments", "compare"]
    if any(_contains_phrase(query_norm, term) for term in breakdown_terms):
        return "primary_diagnosis"

    return None


def _classify_readmission_intent(
    query_norm: str, filters: Sequence[Dict[str, Any]], group_col: Optional[str]
) -> str:
    if any(_contains_phrase(query_norm, term) for term in ["cause", "driver", "surge", "root cause"]):
        return "Root Cause Analysis"
    if any(_contains_phrase(query_norm, term) for term in ["high risk", "risk score", "risk"]):
        return "Risk Stratification"
    if group_col:
        return "Segment Breakdown"
    if any(item.get("column") == "label" for item in filters):
        return "Readmission Cohort Review"
    return "Cohort Lookup"


def _classify_claims_intent(
    query_norm: str, filters: Sequence[Dict[str, Any]]
) -> str:
    if any(_contains_phrase(query_norm, term) for term in ["denial", "denied", "coding"]):
        return "Claims Denial Audit"
    if any(_contains_phrase(query_norm, term) for term in ["revenue", "leakage", "billing"]):
        return "Revenue Leakage Review"
    if filters:
        return "Claims Cohort Lookup"
    return "Claims Overview"


def _readmission_chart_data(
    conn: sqlite3.Connection,
    table: str,
    where_sql: str,
    params: Sequence[Any],
    columns: Sequence[str],
    rows: pd.DataFrame,
    summary_data: Sequence[Dict[str, Any]],
    group_col: Optional[str],
) -> Dict[str, List[Dict[str, Any]]]:
    chart_data: Dict[str, List[Dict[str, Any]]] = {
        "segment_bar": list(summary_data),
        "risk_distribution": [],
        "insurance_readmission_stack": [],
        "top_drivers": [],
        "risk_score_rows": [],
    }

    if "readmission_risk_score" in columns:
        risk_sql = f"""
            SELECT risk_bucket AS segment, COUNT(*) AS records
            FROM (
                SELECT
                    CASE
                        WHEN readmission_risk_score < 0.2 THEN '0.00-0.19'
                        WHEN readmission_risk_score < 0.4 THEN '0.20-0.39'
                        WHEN readmission_risk_score < 0.6 THEN '0.40-0.59'
                        WHEN readmission_risk_score < 0.8 THEN '0.60-0.79'
                        ELSE '0.80-1.00'
                    END AS risk_bucket
                FROM {_quote_identifier(table)}
                {where_sql}
            )
            GROUP BY risk_bucket
            ORDER BY risk_bucket
        """
        chart_data["risk_distribution"] = pd.read_sql_query(
            risk_sql, conn, params=params
        ).to_dict(orient="records")

    if {"insurance_type", "label"}.issubset(set(columns)):
        stack_sql = f"""
            SELECT
                insurance_type AS segment,
                SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS readmitted,
                SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS not_readmitted,
                COUNT(*) AS records
            FROM {_quote_identifier(table)}
            {where_sql}
            GROUP BY insurance_type
            ORDER BY records DESC
            LIMIT 10
        """
        chart_data["insurance_readmission_stack"] = pd.read_sql_query(
            stack_sql, conn, params=params
        ).to_dict(orient="records")

    if "primary_diagnosis" in columns:
        drivers_sql = f"""
            SELECT
                primary_diagnosis AS segment,
                COUNT(*) AS records,
                ROUND(AVG(readmission_risk_score), 3) AS mean_risk_score,
                ROUND(AVG(length_of_stay), 1) AS mean_length_of_stay
            FROM {_quote_identifier(table)}
            {where_sql}
            GROUP BY primary_diagnosis
            ORDER BY records DESC, mean_risk_score DESC
            LIMIT 10
        """
        chart_data["top_drivers"] = pd.read_sql_query(
            drivers_sql, conn, params=params
        ).to_dict(orient="records")

    if not chart_data["segment_bar"] and group_col == "primary_diagnosis":
        chart_data["segment_bar"] = chart_data["top_drivers"]

    if not rows.empty and "readmission_risk_score" in rows.columns:
        chart_columns = [
            col for col in ["patient_id", "primary_diagnosis", "insurance_type", "readmission_risk_score"]
            if col in rows.columns
        ]
        chart_data["risk_score_rows"] = rows[chart_columns].head(25).to_dict(
            orient="records"
        )

    return chart_data


def _claims_chart_data(
    conn: sqlite3.Connection,
    base_from: str,
    where_sql: str,
    params: Sequence[Any],
    rows: pd.DataFrame,
    summary_data: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    chart_data: Dict[str, List[Dict[str, Any]]] = {
        "status_bar": list(summary_data),
        "denial_reasons": [],
        "amount_rows": [],
    }

    denial_sql = f"""
        SELECT
            COALESCE(c.denial_reason, 'Not specified') AS segment,
            COUNT(*) AS records,
            ROUND(SUM(c.amount), 2) AS total_claim_amount
        {base_from}
        {where_sql}
        GROUP BY COALESCE(c.denial_reason, 'Not specified')
        ORDER BY records DESC, total_claim_amount DESC
        LIMIT 10
    """
    chart_data["denial_reasons"] = pd.read_sql_query(
        denial_sql, conn, params=params
    ).to_dict(orient="records")

    if not rows.empty and "amount" in rows.columns:
        chart_columns = [
            col for col in ["claim_id", "status", "amount", "denial_reason", "diagnosis"]
            if col in rows.columns
        ]
        chart_data["amount_rows"] = rows[chart_columns].head(25).to_dict(orient="records")

    return chart_data


def _readmission_executive_summary(
    total_records: int,
    pct_share: str,
    risk_index: float,
    summary_row: Dict[str, Any],
    summary_data: Sequence[Dict[str, Any]],
    group_col: Optional[str],
    filters: Sequence[Dict[str, Any]],
) -> str:
    if total_records == 0:
        return "No matching patient records were found for the detected filters."

    filter_text = ", ".join(item["label"] for item in filters) if filters else "no explicit filters"
    summary = (
        f"Matched {total_records:,} patient records ({pct_share} of readmission data) "
        f"using {filter_text}. Mean readmission risk is {risk_index:.3f}, "
        f"average length of stay is {float(summary_row.get('mean_length_of_stay') or 0):.1f} days, "
        f"and observed readmission-label rate is {float(summary_row.get('readmission_rate_pct') or 0):.1f}%."
    )

    if summary_data and group_col:
        top_segment = summary_data[0]
        summary += (
            f" The largest {_humanize(group_col).lower()} segment is "
            f"{top_segment.get('segment')} with {top_segment.get('records')} records."
        )
    return summary


def _claims_executive_summary(
    total_records: int,
    avg_amount: Any,
    summary_data: Sequence[Dict[str, Any]],
    filters: Sequence[Dict[str, Any]],
) -> str:
    if total_records == 0:
        return "No matching claims were found for the detected filters."

    filter_text = ", ".join(item["label"] for item in filters) if filters else "no explicit filters"
    summary = (
        f"Matched {total_records:,} claims using {filter_text}. "
        f"Average claim amount is {_format_currency(avg_amount)}."
    )
    if summary_data:
        top_status = summary_data[0]
        summary += (
            f" The leading claim status is {top_status.get('segment')} "
            f"with {top_status.get('records')} records."
        )
    return summary


def _reasoning_trace(
    query_intent: str,
    table: str,
    filters: Sequence[Dict[str, Any]],
    count_sql: str,
    row_sql: str,
    params: Sequence[Any],
    total_records: int,
    total_source: int,
    group_col: Optional[str],
) -> List[Dict[str, str]]:
    filter_text = "; ".join(item["label"] for item in filters) or "No explicit filters"
    group_text = _humanize(group_col) if group_col else "No grouping requested"
    return [
        {"step": "Intent", "detail": query_intent},
        {"step": "Source", "detail": table},
        {"step": "Filters", "detail": filter_text},
        {"step": "Grouping", "detail": group_text},
        {"step": "Count SQL", "detail": _format_sql(count_sql)},
        {"step": "Row SQL", "detail": _format_sql(row_sql)},
        {"step": "Parameters", "detail": str(list(params))},
        {
            "step": "Cardinality",
            "detail": f"{int(total_records):,} matches from {int(total_source):,} source rows",
        },
    ]


def _confidence_label(total_records: int, filters: Sequence[Dict[str, Any]]) -> str:
    if total_records == 0:
        return "Low"
    if filters:
        return "High"
    return "Medium"


def _readmission_limitations(
    filters: Sequence[Dict[str, Any]], total_records: int
) -> List[str]:
    limitations = [
        "This local prototype uses a SQLite analytical snapshot, not a live hospital transactional database.",
        "The parser maps common operational phrasing to known schema fields; unsupported clinical synonyms may need explicit mapping.",
    ]
    if not filters:
        limitations.append("No filters were detected, so the result is a broad overview.")
    if total_records == 0:
        limitations.append("Zero-row outputs may mean the criteria are too narrow or the requested value is absent from the dataset.")
    return limitations


def _claims_limitations(filters: Sequence[Dict[str, Any]], total_records: int) -> List[str]:
    limitations = [
        "Claims output is limited to the normalized demo claims, admissions, patients, and diagnoses tables.",
        "The tool joins claims to clinical context read-only; it does not write to billing systems.",
    ]
    if not filters:
        limitations.append("No filters were detected, so all claim rows are included.")
    if total_records == 0:
        limitations.append("Zero-row outputs may mean the requested denial reason, status, ICD code, or amount threshold is absent.")
    return limitations


def _readmission_no_result_suggestions(
    conn: sqlite3.Connection, table: str, columns: Sequence[str]
) -> List[str]:
    suggestions: List[str] = []
    for column in [
        "primary_diagnosis",
        "treatment_type",
        "insurance_type",
        "region",
        "season",
        "discharge_disposition",
    ]:
        if column not in columns:
            continue
        values = _distinct_values(conn, table, column)[:8]
        if values:
            suggestions.append(f"Try {_humanize(column).lower()} values such as: {', '.join(values)}.")
    return suggestions[:4]


def _claims_no_result_suggestions(conn: sqlite3.Connection) -> List[str]:
    suggestions: List[str] = []
    if "claims" in _list_tables(conn):
        statuses = _distinct_values(conn, "claims", "status")[:6]
        reasons = _distinct_values(conn, "claims", "denial_reason")[:6]
        if statuses:
            suggestions.append(f"Try claim statuses such as: {', '.join(statuses)}.")
        if reasons:
            suggestions.append(f"Try denial reasons such as: {', '.join(reasons)}.")
    if "diagnoses" in _list_tables(conn):
        codes = _distinct_values(conn, "diagnoses", "icd_code")[:6]
        if codes:
            suggestions.append(f"Try ICD codes such as: {', '.join(codes)}.")
    return suggestions


def _distinct_values(conn: sqlite3.Connection, table: str, column: str) -> List[str]:
    sql = f"""
        SELECT DISTINCT {_quote_identifier(column)} AS value
        FROM {_quote_identifier(table)}
        WHERE {_quote_identifier(column)} IS NOT NULL
    """
    rows = conn.execute(sql).fetchall()
    return [str(row[0]) for row in rows if str(row[0]).strip()]


def _append_filter(
    filters: List[Dict[str, Any]],
    sql: str,
    params: Sequence[Any],
    label: str,
    column: Optional[str] = None,
) -> None:
    if any(item["sql"] == sql and item["params"] == list(params) for item in filters):
        return
    filters.append({"sql": sql, "params": list(params), "label": label, "column": column})


def _compile_filters(filters: Sequence[Dict[str, Any]]) -> Tuple[str, List[Any]]:
    if not filters:
        return "", []
    where_sql = " WHERE " + " AND ".join(f"({item['sql']})" for item in filters)
    params: List[Any] = []
    for item in filters:
        params.extend(item["params"])
    return where_sql, params


def _scalar(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else 0


def _coerce_number(value: str) -> float:
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _format_pct(part: Any, whole: Any) -> str:
    if not whole:
        return "0.0%"
    return f"{round((float(part or 0) / float(whole)) * 100, 1)}%"


def _format_currency(value: Any) -> str:
    if value is None:
        return "$0.00"
    return f"${float(value):,.2f}"


def _format_sql(sql: str) -> str:
    lines = [line.rstrip() for line in str(sql).strip().splitlines()]
    return "\n".join(line for line in lines if line.strip())


def _humanize(column: str) -> str:
    return column.replace("_", " ").title()


def _build_readmission_title(
    filters: Sequence[Dict[str, Any]], group_col: Optional[str]
) -> str:
    if group_col:
        return f"Readmission Analysis by {_humanize(group_col)}"
    if filters:
        return "Readmission Cohort: " + "; ".join(item["label"] for item in filters[:3])
    return "Readmission Operations Overview"


def _build_claims_title(filters: Sequence[Dict[str, Any]]) -> str:
    if filters:
        return "Claims Analysis: " + "; ".join(item["label"] for item in filters[:3])
    return "Claims Operations Overview"


def _readmission_recommendations(
    total_records: int,
    summary_row: Dict[str, Any],
    summary_data: Sequence[Dict[str, Any]],
    group_col: Optional[str],
) -> List[str]:
    if total_records == 0:
        return [
            "No rows matched the requested filters. Relax one condition or check whether the requested segment exists in the database."
        ]

    recommendations: List[str] = []
    mean_risk = float(summary_row.get("mean_risk_score") or 0)
    mean_los = float(summary_row.get("mean_length_of_stay") or 0)
    readmission_rate = float(summary_row.get("readmission_rate_pct") or 0)

    if summary_data and group_col:
        top_segment = summary_data[0]
        recommendations.append(
            f"Largest {_humanize(group_col).lower()} segment: {top_segment.get('segment')} "
            f"({top_segment.get('records')} records). Start the operational review there."
        )

    if mean_risk >= 0.8:
        recommendations.append(
            "Mean risk is high. Prioritize discharge planning, follow-up scheduling, and medication reconciliation for this cohort."
        )
    elif mean_risk >= 0.6:
        recommendations.append(
            "Mean risk is elevated. Compare this cohort against lower-risk segments to isolate preventable utilization drivers."
        )

    if mean_los >= 7:
        recommendations.append(
            "Length of stay is above a week on average. Review placement delays, discharge disposition, and complex-care pathways."
        )

    if readmission_rate >= 75:
        recommendations.append(
            "Observed readmission label rate is concentrated in this cohort. Track follow-up completion and post-discharge support gaps."
        )

    if not recommendations:
        recommendations.append(
            "Cohort returned successfully. Use the matched rows and segment summary to compare care coordination performance."
        )

    return recommendations


def _claims_recommendations(
    total_records: int, summary_data: Sequence[Dict[str, Any]]
) -> List[str]:
    if total_records == 0:
        return [
            "No claims matched the requested filters. Check claim status, ICD code, denial reason, or amount thresholds."
        ]

    recommendations: List[str] = []
    denied_rows = [
        row for row in summary_data if str(row.get("segment", "")).lower() == "denied"
    ]
    if denied_rows:
        denied_amount = denied_rows[0].get("total_claim_amount", 0)
        recommendations.append(
            f"Denied claims total {_format_currency(denied_amount)} in the matched set. Review coding and documentation patterns first."
        )

    recommendations.append(
        "Use the claim-level rows to trace denials back to admission, diagnosis, facility, and patient demographics."
    )
    return recommendations
