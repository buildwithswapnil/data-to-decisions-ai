
import os
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, START, END

# Import shared master state and node workers
from agents.nodes import AgentState, planner_node, generator_node, critic_node, insight_node

# Import our functional database retrieval tool
from tools.hybrid_rag_lookup import hybrid_rag_lookup

# --- 1. The Retrieval Node (Bridging Live SQLite Context into the Graph) ---
def retrieval_node(state: AgentState) -> Dict[str, Any]:
    """
    Extracts the user's query from the state, passes it to our 
    production SQL DB lookup tool, and injects the live results.
    """
    print("🔍 [Retrieval Node]: Pulling quantitative metrics from local database...")
    user_query = state["query"]
    retrieved_text = hybrid_rag_lookup(user_query)
    return {"context": retrieved_text}


# --- 2. Conditional Routing Logic ---
def route_after_critic(state: AgentState) -> Literal["generator", "insight"]:
    """
    Evaluates the gate flag set by the Critic Node.
    """
    if state.get("approved", False):
        print("✅ [Routing]: Draft approved by auditor. Proceeding to Insight Extraction.")
        return "insight"
    
    if state.get("revision_count", 0) >= 3:
        print("⚠️ [Routing]: Maximum audit cycles reached. Forcing fallback completion path.")
        return "insight"
        
    print("❌ [Routing]: Draft rejected by auditor. Re-routing back to Generator Node.")
    return "generator"


# --- 3. Compile the State Graph Workstream ---
workflow = StateGraph(AgentState)

# Define all structural nodes inside the state graph
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("planner", planner_node)
workflow.add_node("generator", generator_node)
workflow.add_node("critic", critic_node)
workflow.add_node("insight", insight_node)

# Construct functional edge paths
workflow.add_edge(START, "retrieval")
workflow.add_edge("retrieval", "planner")
workflow.add_edge("planner", "generator")
workflow.add_edge("generator", "critic")

# Add conditional execution loop based on Critic audit results
workflow.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "generator": "generator",
        "insight": "insight"
    }
)

workflow.add_edge("insight", END)

# Export compiled runnable graph application
app = workflow.compile()