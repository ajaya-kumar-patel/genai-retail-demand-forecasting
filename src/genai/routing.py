from typing import Literal
from src.genai.state import RetailAssistantState

def route_request(state: RetailAssistantState) -> Literal["sql", "forecast", "rag", "general"]:
    """Choose the next workflow path based on the detected intent."""

    intent = state.get("intent", "general")

    if intent in {"sql", "forecast", "rag", "general"}:
        return intent

    return "general"

