# Deployment Guide

The simplest public deployment path is Streamlit Community Cloud or a small VM/container service.

## Streamlit Community Cloud

1. Push this project to GitHub.
2. Confirm `.env` is not committed.
3. In Streamlit Cloud, create a new app from the repository.
4. Set the main file path to `app.py`.
5. Add secrets:

```toml
OPENAI_API_KEY = "your_api_key_here"
OPENAI_MODEL = "gpt-5.4-mini"
```

6. Deploy.

If no API key is configured, the app still runs in deterministic analytics mode.

## Production Notes

- Do not deploy raw patient-level hospital data to a public app.
- Use synthetic, de-identified, or public-use healthcare data.
- For private deployment, add authentication, role-based access, request logging, PHI controls, and audit review.
- Keep the agent read-only unless a future workflow has human approval and strong governance.

## Demo Script

Use these questions during a live walkthrough:

1. `Compare readmission risk by diagnosis for Medicare high risk patients in 2023`
2. `Break down operational metrics by insurance type for diabetes patients`
3. `Show patients aged 60 to 80 with COPD and high readmission risk`
4. `Who won the cricket match yesterday?`

The fourth query demonstrates domain guardrails.
