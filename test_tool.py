# test_tool.py
from tools.hybrid_rag_lookup import hybrid_rag_lookup

print("🚀 Testing Hybrid RAG Lookup Pipeline...")
print("==========================================")

# Simulate a Business Analyst query passing through the tool
sample_query = "What is causing our operational readmission surges?"
result_context = hybrid_rag_lookup(sample_query)

print(result_context)
print("==========================================")