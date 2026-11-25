from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
import config
from agent.tool import retriever_tool
from langchain_openai import ChatOpenAI
from utils import image_vers_data_url
from pathlib import Path

tools = [retriever_tool]
llm = ChatOpenAI(
    model=config.MODELE_GENERATION,            # e.g. "openai/gpt-oss-120b"
    base_url=config.BASE_URL,
    api_key=config.API_KEY,
)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful assistant answering questions about a PDF document. "
    "Use retriever_tool to find information before answering.\n"
    "Rules:\n"
    "- Break the user's question into small, independent sub-questions.\n"
    "- Call retriever_tool once per sub-question, with a short keyword-style query.\n"
    "  Example: 'what is attention and how was it discovered?' -> "
    "'attention mechanism definition' and 'origin of attention mechanism history'.\n"
    "- The tool searches both text passages and images/figures. "
    "If the user asks to see a figure or diagram, say so in the query "
    "(e.g. 'Transformer architecture diagram').\n"
    "- If the user provides an image path, pass it in the image_path argument.\n"
    "- If results are insufficient, search again with a different query.\n"
    "- Do not answer from memory. Cite sources as [pdf p.X] and mention "
    "image files when you use them."
))

def agent_node(state: MessagesState):          # MIND
    return {"messages": [llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])]}

def vision_node(state: MessagesState):
    """Attach the images retrieved by the last tool calls as a visible message."""
    paths = []
    for m in reversed(state["messages"]):          # only the latest ToolMessages
        if m.type != "tool":
            break
        paths += m.artifact or []

    paths = list(dict.fromkeys(paths))[:config.K_IMAGES]   # dedupe, cap
    if not paths:
        return {}

    content = [{"type": "text",
                "text": "Here are the images retrieved by the search. "
                        "Describe only what is actually visible in them."}]
    for i, p in enumerate(paths, 1):
        try:
            url = image_vers_data_url(p)
        except Exception as e:
            content.append({"type": "text", "text": f"Image {i} could not be loaded: {e}"})
            continue
        content.append({"type": "text", "text": f"Image {i}: {Path(p).name}"})
        content.append({"type": "image_url", "image_url": {"url": url}})

    return {"messages": [HumanMessage(content=content)]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))     # BODY
builder.add_node("vision", vision_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # tool call? -> "tools", else END
builder.add_edge("tools", "vision")      # was: tools -> agent
builder.add_edge("vision", "agent")             # Observation goes back to the brain

graph = builder.compile()

def ask(question, max_steps = config.MAX_STEPS):
    final = None
    for state in graph.stream({"messages": [HumanMessage(content=question)]},
                              config={"recursion_limit": max_steps},
                              stream_mode="values"):
        m = state["messages"][-1]
        if isinstance(m.content, list):          # the vision message
            n = sum(1 for b in m.content if b.get("type") == "image_url")
            print(f"[Vision message: {n} image(s) sent to the LLM]")
        else:
            m.pretty_print()
        final = state
    return final["messages"][-1].content