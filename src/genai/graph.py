from langgraph.graph import StateGraph, START, END

from src.genai.state import RetailAssistantState
from src.genai.nodes import resolve_followup_node, intent_detection_node, general_response_node
from src.genai.routing import route_request

from src.genai.tools.sql_tool import run_sql_tool
from src.genai.tools.rag_tool import run_rag_tool
from src.genai.tools.forecast_tool import run_forecast_tool

from langchain_core.messages import AIMessage

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)


def sql_node(state: RetailAssistantState) -> dict:
    """Answer historical sales questions using the MySQL SQL tool."""
    messages = state.get("messages", [])

    if not messages:
        return {"tool_result": "No user question was provided."}

    question = state.get("resolved_question") or messages[-1].content

    try:
        result = run_sql_tool(question)

        output = (
            f"Question: {result['question']}\n\n"
            f"SQL:\n{result['sql']}\n\n"
            f"Columns: {result['columns']}\n"
            f"Results:\n"
        )

        for row in result["rows"]:
            output += f"{row}\n"

        return {"tool_result": output}

    except Exception as exc:
        return {"tool_result": f"Unable to answer the SQL question: {exc}"}




def forecast_node(state: RetailAssistantState) -> dict:
    """Generate forecasts using the deployed forecasting API."""

    messages = state.get("messages", [])

    if not messages:
        return {"tool_result": "No forecasting question was provided."}

    question = state.get("resolved_question") or messages[-1].content

    try:
        result = run_forecast_tool(question)

        # Handle errors returned by the forecasting tool.
        if "error" in result:
            return {
                "tool_result": f"Forecast unavailable: {result['error']}"
            }

        output = (
            f"Product: {result['item_id']}\n"
            f"Historical data through: {result['historical_data_through']}\n"
            f"Forecast period: {result['forecast_start_date']} "
            f"to {result['forecast_end_date']}\n"
            f"Horizon: {result['horizon_days']} days\n\n"
            "Daily predicted sales (units):\n"
        )

        for forecast in result["forecasts"]:
            output += (
                f"- {forecast['date']}: "
                f"{forecast['predicted_sales']:.2f} units\n"
            )

        output += "\nAssumptions:\n"
        for assumption in result.get("assumptions", []):
            output += f"- {assumption}\n"

        return {"tool_result": output}

    except Exception as exc:
        return {
            "tool_result": (
                f"Unable to generate the forecast: "
                f"{type(exc).__name__}: {exc}"
            )
        }



def rag_node(state: RetailAssistantState) -> dict:
    """Answer knowledge questions using the RAG tool."""

    messages = state.get("messages", [])

    if not messages:
        return {"tool_result": "No user question was provided."}

    question = state.get("resolved_question") or messages[-1].content

    try:
        result = run_rag_tool(question)

        output = (
            f"Question: {result['question']}\n\n"
            f"Answer: {result['answer']}\n\n"
            f"Sources: {', '.join(result['sources'])}"
        )

        return {"tool_result": output}

    except Exception as exc:
        return {"tool_result": f"Unable to answer the RAG question: {exc}"}


def tool_response_node(state: RetailAssistantState) -> dict:
    """Convert tool output into a concise, grounded answer."""

    from langchain_core.messages import AIMessage, HumanMessage

    result = state.get("tool_result", "No result available.")
    intent = state.get("intent", "general")
    messages = state.get("messages", [])

    question = state.get("resolved_question") or (
        messages[-1].content
        if messages and isinstance(messages[-1], HumanMessage)
        else ""
    )

    prompt = f"""
    You are a retail sales analytics assistant.

    Answer the user's question using only the tool output.
    Never contradict the tool output or invent facts.

    Tool type: {intent}
    User question: {question}

    Tool output:
    {result}

    Instructions:
    - Give a direct, concise answer.
    - Preserve the meaning of the tool output.
    - If the tool output explicitly answers the question, use that answer.
    - If the tool output does not answer the question, say so clearly.
    - Do not present unrelated SQL rows as an answer.
    - Do not claim data is missing when the tool output provides the answer.
    - Preserve product IDs exactly.
    - Format prices to 2 decimal places and large counts with commas.
    - Mention document sources when useful.
    - Forecasted sales are unit counts, never currency.
    - Never add currency symbols to predicted sales or unit counts.
    - Preserve the numeric forecast values exactly as supplied by the tool.

    Answer:
    """

    response = llm.invoke(prompt)

    return {"messages": [AIMessage(content=response.content)]}


builder = StateGraph(RetailAssistantState)

# Register nodes
builder.add_node("resolve_followup", resolve_followup_node)
builder.add_node("detect_intent", intent_detection_node)
builder.add_node("sql", sql_node)
builder.add_node("forecast", forecast_node)
builder.add_node("rag", rag_node)
builder.add_node("general", general_response_node)
builder.add_node("tool_response", tool_response_node)

# Define workflow
builder.add_edge(START, "resolve_followup")
builder.add_edge("resolve_followup", "detect_intent")

builder.add_conditional_edges(
    "detect_intent",
    route_request,
    {
        "sql": "sql",
        "forecast": "forecast",
        "rag": "rag",
        "general": "general",
    },
)

# Specialized tool paths
builder.add_edge("sql", "tool_response")
builder.add_edge("forecast", "tool_response")
builder.add_edge("rag", "tool_response")

# End workflow
builder.add_edge("general", END)
builder.add_edge("tool_response", END)

# Compile graph
graph = builder.compile()
