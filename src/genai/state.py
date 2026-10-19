from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class RetailAssistantState(TypedDict, total=False):
    # Conversation history
    messages: Annotated[list, add_messages]

    # Intent detected from the user's request
    intent: Literal["sql", "forecast", "rag", "general"]

    # Result returned by the selected tool
    tool_result: str

    resolved_question: str



