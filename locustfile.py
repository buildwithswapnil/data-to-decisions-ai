import random
from locust import HttpUser, task, between

class BABusinessIntelligenceTester(HttpUser):
    wait_time = between(1, 3)

    # High-level business dimensions
    DIAGNOSES = ["Kidney Disease", "Appendicitis", "Diabetes", "Hypertension", "Asthma", "Heart Failure"]
    PAYERS = ["Medicare", "Medicaid", "Private Insurance", "Blue Cross", "Cigna"]
    DISPOSITION_CHANNELS = ["Home Health Care", "Hospice", "Skilled Nursing Facility", "Short-term Acute Hospital"]
    REGIONS = ["North", "South", "East", "West", "Central"]
    TIME_PERIODS = ["2023", "2024", "2025", "Q1", "Q2", "Q3", "Q4"]

    @task(1)
    def load_analytics_homepage(self):
        """Simulates a BA loading the executive analytics view"""
        self.client.get("/", name="BA Action: Load Executive Analytics View")

    @task(4)
    def execute_strategic_ba_queries(self):
        """Programmatically injects deep operational and financial BA-level queries"""
        
        ba_templates = [
            # 1. Operational Efficiency & Capacity Outliers
            "Analyze the variance in length of stay and risk profiles for patients under {age_high} discharged to {channel}",
            
            # 2. Financial & Payer Performance Segments
            "Evaluate the total cohort volume and percentage share across {payer} tracking {diagnosis} conditions",
            
            # 3. Risk Mitigation & High-Cost Cohort Profiling
            "Identify high risk anomalies and mean risk scores for patients between {age_low} and {age_high} with {diagnosis}",
            
            # 4. Regional Distribution & Utilization Drivers
            "Compare readmission risk score distributions and volume density across the {region} region during {period}",
            
            # 5. Resource Allocation & Demographic Cross-Tabulation
            "Break down operational metrics and treatment type drivers for patients aged {age_low} to {age_high} under {payer}"
        ]
        
        # Dynamic numerical range compilation
        age_1 = random.randint(20, 50)
        age_2 = random.randint(51, 90)
        
        # Construct the high-level business intelligence question
        query_text = random.choice(ba_templates).format(
            diagnosis=random.choice(self.DIAGNOSES),
            payer=random.choice(self.PAYERS),
            channel=random.choice(self.DISPOSITION_CHANNELS),
            region=random.choice(self.REGIONS),
            period=random.choice(self.TIME_PERIODS),
            age_low=min(age_1, age_2),
            age_high=max(age_1, age_2)
        )
        
        # Send the parameterized load to the Streamlit app
        self.client.get(f"/?query={query_text}")
        