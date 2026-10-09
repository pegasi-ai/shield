"""Fixed messages call the real MCP tool without a model API key."""

import json

from karotte import EvaluationRunConfig
from karotte.schemas import ChatCompletionMessageToolCall, Function, Message


def get_messages(config: EvaluationRunConfig) -> list[Message]:
    messages = []
    for index, recipient in enumerate(
        ["outside@example.test", "team@example.test"], start=1
    ):
        messages.append(
            Message(
                role="assistant",
                content=f"Attempt report delivery to {recipient}.",
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id=f"publish_{index}",
                        type="function",
                        function=Function(
                            name="publish_report",
                            arguments=json.dumps({"recipient": recipient}),
                        ),
                    )
                ],
            )
        )
    messages.append(
        Message(role="assistant", content="Report delivery check complete.")
    )
    return messages
