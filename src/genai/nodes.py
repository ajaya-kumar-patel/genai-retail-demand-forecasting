import os
import re
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from src.genai.state import RetailAssistantState

load_dotenv()

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)



def resolve_followup_node(state: RetailAssistantState) -> dict:
    """Resolve product references from the previous conversation."""

    import re
    from langchain_core.messages import HumanMessage, AIMessage

    messages = state.get("messages", [])

    if not messages:
        return {"resolved_question": ""}

    # Find the latest user question.
    last_user_index = next(
        (
            i for i in range(len(messages) - 1, -1, -1)
            if isinstance(messages[i], HumanMessage)
        ),
        None,
    )

    if last_user_index is None:
        return {"resolved_question": ""}

    question = messages[last_user_index].content.strip()

    # Find the most recent assistant response before this question.
    previous_answer = ""
    for msg in reversed(messages[:last_user_index]):
        if isinstance(msg, AIMessage):
            previous_answer = msg.content
            break

    # Resolve ordinal references against product IDs in that answer.
    product_ids = re.findall(
        r"\b[A-Z]+_[A-Z0-9]+_[A-Z0-9]+\b",
        previous_answer,
    )

    ordinal_patterns = [
        (r"\b(first|1st)\b", 0),
        (r"\b(second|2nd)\b", 1),
        (r"\b(third|3rd)\b", 2),
        (r"\b(fourth|4th)\b", 3),
        (r"\b(fifth|5th)\b", 4),
    ]

    for pattern, index in ordinal_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            if index < len(product_ids):
                product_id = product_ids[index]

                # Replace the ordinal reference with the exact product ID.
                resolved = re.sub(
                    r"\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th)"
                    r"(?:\s+(?:one|product|item))?\b",
                    product_id,
                    question,
                    count=1,
                    flags=re.IGNORECASE,
                )

                return {"resolved_question": resolved}

    # For other follow-ups, let the existing question pass through unchanged.
    return {"resolved_question": question}


def intent_detection_node(state: RetailAssistantState) -> dict:
    """Classify the user's request using rules and the LLM."""

    messages = state.get("messages", [])

    if not messages:
        return {"intent": "general", "tool_result": ""}

    question = state.get(
        "resolved_question",
        messages[-1].content,
    ).strip()

    q = question.lower()

    # Clear documentation and business-definition questions.
    rag_patterns = [
        r"\bwhat does\b.*\b(mean|represent)\b",
        r"\bwhat is\b.*\b(column|field|sell_price|item_id|dept_id|cat_id)\b",
        r"\bcan (we|i|you) calculate profit\b",
        r"\bwhy can't\b.*\b(profit|revenue)\b",
        r"\bwithout cost data\b",
        r"\bwhat data\b.*\b(profit|cost)\b",
    ]

    if any(re.search(pattern, q) for pattern in rag_patterns):
        return {"intent": "rag"}

    # Keep the LLM for other questions.
    prompt = f"""
    Classify the question into exactly one category:

    - sql: historical sales calculations, rankings, or comparisons
    - forecast: predictions of future sales
    - rag: documentation, definitions, or business concepts
    - general: ordinary conversation or unrelated questions

    Return only one word: sql, forecast, rag, or general.

    Question: {question}
    """

    response = llm.invoke(prompt)
    intent = response.content.strip().lower()

    if intent not in {"sql", "forecast", "rag", "general"}:
        intent = "general"

    return {"intent": intent}


def general_response_node(state: RetailAssistantState) -> dict:
    """Answer a general question."""
    messages = state.get("messages", [])

    response = llm.invoke(messages)

    return {"messages": [response]}
