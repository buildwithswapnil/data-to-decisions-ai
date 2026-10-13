# Data-to-Decisions AI

Agentic healthcare operations analytics dashboard built with Streamlit, SQLite, pandas, and OpenAI tool calling.

The app converts a natural-language business question into a governed analytics workflow:

1. Classify whether the query belongs to the healthcare operations domain.
2. Let an AI agent decide when to call the local healthcare analytics tool.
3. Execute safe read-only SQLite analytics against the local dataset.
4. Return cohort KPIs, SQL trace, chart data, row-level evidence, limitations, and recommendations.
5. Synthesize the result into an executive answer.

## Why This Project Exists

Healthcare leaders often ask questions in business language:

- Which diagnosis segments are driving readmission risk?
- How does Medicare risk compare across regions?
- Which discharge channels show higher risk?
- What cohort should operations review first?

This project demonstrates how an AI agent can sit between business users and structured healthcare data without blindly trusting generated SQL.

## Data

The main analytics table contains 8,000 synthetic/de-identified readmission records with:

- patient ID
- admission date and season
- age, gender, and region
- primary diagnosis
- comorbidities, medications, prior readmissions, and length of stay
- treatment type
- payer/insurance type
- discharge disposition
- readmission risk score
- readmission label

Raw hospital patient records should not be used in a public portfolio project. See [docs/DATA_STRATEGY.md](docs/DATA_STRATEGY.md) for the safe data strategy.

## Architecture

```text
Streamlit UI
    |
    v
Agentic wrapper
    |
    |-- OpenAI Responses API tool call, when OPENAI_API_KEY is configured
    |-- Deterministic fallback, when no API key is configured
    v
healthcare_analytics_lookup tool
    |
    v
Schema-aware filter extraction + parameterized SQLite queries
    |
    v
KPIs, SQL trace, chart data, rows, limitations, recommendations
```

## Key Files

- `app.py`: Streamlit application.
- `agents/healthcare_agent.py`: real tool-calling agent wrapper.
- `tools/hybrid_rag_lookup.py`: schema-aware analytics tool.
- `db/setup_db.py`: CSV-to-SQLite ingestion.
- `tests/`: regression tests for analytics and agent fallback.
- `evals/agent_eval_questions.json`: repeatable agent evaluation prompts.
- `docs/DATA_STRATEGY.md`: healthcare data sourcing and privacy strategy.
- `docs/DEPLOYMENT.md`: live deployment guide.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional agent mode:

```bash
cp .env.example .env
```

Then add:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5-mini
```

If `OPENAI_API_KEY` is missing, the app still works in deterministic analytics mode.

## Run

```bash
streamlit run app.py
```

Local upgraded preview currently uses:

```bash
streamlit run app.py --server.port 8765
```

## Test

```bash
pytest
```

## Example Questions

- Compare readmission risk by diagnosis for Medicare high risk patients in 2023.
- Break down operational metrics by insurance type for diabetes patients.
- Show patients aged 60 to 80 with COPD and high readmission risk.
- What is causing our operational readmission surges?
- Show denied claims due to coding error.

## Resume Positioning

Built an agentic healthcare operations analytics assistant using Streamlit, SQLite, pandas, and OpenAI tool calling to convert natural-language business questions into safe structured-data analysis, SQL traceability, cohort KPIs, chart-ready summaries, and executive recommendations over 8,000 readmission records.

## Limitations

- The project uses a local analytical snapshot, not a live hospital EHR.
- The current claims demo table is intentionally small; readmission analytics is the main project surface.
- Patient-level data in production would require privacy controls, governance, audit logging, role-based access, and legal review.
