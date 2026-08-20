import anthropic

from llm.base import LLMProvider, LLMResponse, ToolCall, ToolDefinition

DEFAULT_MODEL = "claude-sonnet-4-5"


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = anthropic.Anthropic()

    def complete(self, system, messages, tools, max_tokens=4096) -> LLMResponse:
        tool_schemas = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=tool_schemas,
            messages=messages,
        )

        if resp.stop_reason == "end_turn":
            text = next((b.text for b in resp.content if hasattr(b, "text")), "")
            return LLMResponse(stop_reason="end_turn", text=text, raw=resp.content)

        if resp.stop_reason == "tool_use":
            tool_calls = [
                ToolCall(id=b.id, name=b.name, input=b.input)
                for b in resp.content
                if b.type == "tool_use"
            ]
            return LLMResponse(stop_reason="tool_use", tool_calls=tool_calls, raw=resp.content)

        return LLMResponse(stop_reason="end_turn", text="", raw=resp.content)

    def format_assistant_tool_calls(self, response: LLMResponse) -> dict:
        return {
            "role": "assistant",
            "content": [b.model_dump() for b in response.raw],
        }

    def format_tool_results(self, results: list[tuple[ToolCall, str]]) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": content,
                    }
                    for tool_call, content in results
                ],
            }
        ]
