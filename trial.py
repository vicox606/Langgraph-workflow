import os
from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama         
from langchain_core.messages import HumanMessage, SystemMessage

MODEL_NAME   = "llama3.2"        
OLLAMA_URL   = "http://localhost:11434"  
TEMPERATURE  = 0.7               


class BlogState(TypedDict):
    topic: str           
    outline: str         
    draft: str           
    feedback: str        
    score: int           
    final_post: str      
    revision_count: int  

llm = ChatOllama(                    
    model       = MODEL_NAME,        
    base_url    = OLLAMA_URL,        
    temperature = TEMPERATURE,
    num_predict = 1024,              
)

def planner_node(state: BlogState) -> dict:
    """Creates a structured outline for the blog post."""
    print(f"\n  PLANNER  — Creating outline for: '{state['topic']}'")
    print(f"   (using {MODEL_NAME} via Ollama)")

    messages = [
        SystemMessage(content="You are a content strategist. Create concise blog outlines."),
        HumanMessage(content=(
            f"Create a 4-section outline for a blog post about: {state['topic']}\n"
            "Format: Section title — one sentence description. Keep it brief."
        ))
    ]
    response = llm.invoke(messages)
    outline = response.content
    print(f"   → Outline ready ({len(outline.split())} words)")
    return {"outline": outline}


def writer_node(state: BlogState) -> dict:
    """Writes the first draft based on the outline."""
    print(f"\n   WRITER  — Writing draft (revision #{state['revision_count'] + 1})")

    messages = [
        SystemMessage(content="You are a skilled blog writer. Write engaging, clear content."),
        HumanMessage(content=(
            f"Write a blog post about '{state['topic']}' following this outline:\n\n"
            f"{state['outline']}\n\n"
            "Write ~250 words. Use a friendly, educational tone."
            + (f"\n\nPrevious feedback to address:\n{state['feedback']}" if state.get("feedback") else "")
        ))
    ]
    response = llm.invoke(messages)
    draft = response.content
    print(f"   → Draft ready ({len(draft.split())} words)")
    return {"draft": draft, "revision_count": state["revision_count"] + 1}


def critic_node(state: BlogState) -> dict:
    """Reviews the draft and gives a score + actionable feedback."""
    print(f"\n  CRITIC   — Reviewing draft...")

    messages = [
        SystemMessage(content="You are a critical editor. Be honest but constructive."),
        HumanMessage(content=(
            f"Review this blog post draft about '{state['topic']}':\n\n"
            f"{state['draft']}\n\n"
            "Respond in this EXACT format (two lines only):\n"
            "SCORE: [number 1-10]\n"
            "FEEDBACK: [2-3 sentences of specific improvements]\n\n"
            "Be strict — only give 8+ if it's genuinely excellent. "
            "Output ONLY those two lines, nothing else."
        ))
    ]
    response = llm.invoke(messages)
    raw = response.content
    score = 5       
    feedback = raw 

    for line in raw.split("\n"):
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                num_str = line.split(":", 1)[1].strip().split()[0].rstrip(".,")
                score = max(1, min(10, int(num_str)))
            except (ValueError, IndexError):
                pass
        if line.upper().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    print(f"   → Score: {score}/10")
    return {"score": score, "feedback": feedback}


def polisher_node(state: BlogState) -> dict:
    """Applies final polish — formatting, intro hook, conclusion."""
    print(f"\n POLISHER — Applying final polish...")

    messages = [
        SystemMessage(content="You are a copy editor. Enhance clarity and structure."),
        HumanMessage(content=(
            "Polish this blog post. Add a compelling title, improve the intro hook, "
            f"and ensure a strong conclusion. Keep the core content:\n\n{state['draft']}"
        ))
    ]
    response = llm.invoke(messages)
    final = response.content
    print(f"   → Final post ready ({len(final.split())} words)")
    return {"final_post": final}



def route_after_critic(state: BlogState) -> str:
    """Decides: loop back to writer, or proceed to polisher."""
    if state["score"] < 7 and state["revision_count"] < 2:
        print(f"\n Score {state['score']}/10 — sending back for revision...")
        return "writer"
    else:
        print(f"\n Score {state['score']}/10 — proceeding to polish...")
        return "polisher"


def build_workflow():
    graph = StateGraph(BlogState)

    graph.add_node("planner",  planner_node)
    graph.add_node("writer",   writer_node)
    graph.add_node("critic",   critic_node)
    graph.add_node("polisher", polisher_node)

    graph.add_edge(START,      "planner")
    graph.add_edge("planner",  "writer")
    graph.add_edge("writer",   "critic")
    graph.add_edge("polisher", END)

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"writer": "writer", "polisher": "polisher"},
    )

    return graph.compile()

def run_workflow(topic: str):
    print("\n" + "═" * 60)
    print(f" LangGraph Blog Workflow  (Ollama / {MODEL_NAME})")
    print(f"  Topic: {topic}")
    print("═" * 60)

    workflow = build_workflow()

    initial_state: BlogState = {
        "topic": topic,
        "outline": "",
        "draft": "",
        "feedback": "",
        "score": 0,
        "final_post": "",
        "revision_count": 0,
    }

    final_state = workflow.invoke(initial_state)

    print("\n" + "═" * 60)
    print("FINAL BLOG POST")
    print("═" * 60)
    print(final_state["final_post"])
    print("\n" + "═" * 60)
    print(f"  Model    : {MODEL_NAME}")
    print(f"  Revisions: {final_state['revision_count']}  |  Final score: {final_state['score']}/10")
    print("═" * 60)

    return final_state


if __name__ == "__main__":
    run_workflow("Why every developer should learn Machine Learning")