# Healthcare Data Strategy

This project should not use raw hospital patient records. Real hospital data is confidential and may contain protected health information. For a public portfolio project, the safe approach is to use one of these paths:

1. Synthetic healthcare data generated to resemble operational workflows.
2. Public de-identified healthcare datasets.
3. Credentialed research datasets with a data-use agreement.
4. Aggregated public healthcare datasets where patient-level identifiers are removed.

## Recommended Sources

- Synthea: open-source synthetic patient and electronic health record data from MITRE.
- MIMIC-IV: de-identified critical-care EHR data available through PhysioNet credentialing and a data-use agreement.
- CMS public-use data: de-identified public data products, dashboards, public-use files, and claims-like synthetic files.
- HHS HIPAA de-identification guidance: explains Safe Harbor and Expert Determination approaches for de-identifying protected health information.

## Portfolio Positioning

For resumes and interviews, describe this app as using a local analytical healthcare snapshot for safe demonstration. Avoid saying it uses live hospital records unless you truly have permission, governance approval, and a compliant data pipeline.

Recommended wording:

> Built an agentic healthcare operations analytics prototype over synthetic/de-identified readmission data to demonstrate natural-language cohort analysis, safe SQL execution, traceability, and executive decision support.

## Future Production Path

In a real hospital deployment, the data layer would connect to governed systems such as an enterprise data warehouse, FHIR server, or claims warehouse. The agent would need role-based access control, audit logs, PHI redaction, data-loss prevention, approval workflows, and legal review before any patient-level data could be used.
