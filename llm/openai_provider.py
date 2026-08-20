import json

import openai

from llm.base import LLMProvider, LLMResponse, ToolCall, ToolDefinition

DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = openai.OpenAI()

    def complete(self, system, messages, tools, max_tokens=4096) -> LLMResponse:
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
        # OpenAI takes the system message as part of the messages list
        full_messages = [{"role": "system", "content": system}] + messages

        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            tools=tool_schemas,
            messages=full_messages,
        )

        choice = resp.choices[0]

        if choice.finish_reason == "stop":
            return LLMResponse(
                stop_reason="end_turn",
                text=choice.message.content or "",
                raw=choice.message,
            )

        if choice.finish_reason == "tool_calls":
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                )
                for tc in choice.message.tool_calls
            ]
            return LLMResponse(
                stop_reason="tool_use",
                tool_calls=tool_calls,
                raw=choice.message,
            )

        return LLMResponse(stop_reason="end_turn", text="", raw=choice.message)

    def format_assistant_tool_calls(self, response: LLMResponse) -> dict:
        msg = response.raw
        return {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        }

    def format_tool_results(self, results: list[tuple[ToolCall, str]]) -> list[dict]:
        # OpenAI requires one message per tool result
        return [
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": content,
            }
            for tool_call, content in results
        ]
