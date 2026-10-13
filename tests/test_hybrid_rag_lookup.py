from tools.hybrid_rag_lookup import hybrid_rag_lookup


def test_readmission_query_applies_filters_and_returns_trace():
    result = hybrid_rag_lookup(
        "Compare readmission risk by diagnosis for Medicare high risk patients in 2023"
    )

    assert result["analysis_domain"] == "readmissions"
    assert result["total_records"] > 0
    assert "Insurance Type = Medicare" in result["applied_filters"]
    assert "Admission year in 2023" in result["applied_filters"]
    assert "Readmission risk score >= 0.8" in result["applied_filters"]
    assert result["generated_sql"].startswith("SELECT")
    assert result["sql_params"] == ["Medicare", "2023", 0.8]


def test_unsupported_query_does_not_execute_sql():
    result = hybrid_rag_lookup("Who won the cricket match yesterday?")

    assert result["analysis_domain"] == "unsupported"
    assert result["confidence"] == "Rejected"
    assert result["generated_sql"] == ""
    assert result["total_records"] == 0


def test_zero_result_query_returns_suggestions():
    result = hybrid_rag_lookup("Show high risk patients with pneumonia in 2099")

    assert result["analysis_domain"] == "readmissions"
    assert result["total_records"] == 0
    assert result["no_result_suggestions"]


def test_mean_risk_score_request_does_not_treat_age_range_as_risk_filter():
    result = hybrid_rag_lookup(
        "Show me the total patient cohort volume and mean risk scores for individuals aged between 65 and 85 tracking Kidney Disease"
    )

    assert result["analysis_domain"] == "readmissions"
    assert result["total_records"] == 234
    assert result["applied_filters"] == [
        "Primary Diagnosis = Kidney Disease",
        "Age between 65 and 85",
    ]
    assert result["risk_index"] == 0.965
