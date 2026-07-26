"""Tests for DeepSeekChatOpenAI thinking-mode behaviour.

Two pieces verified:

1. ``reasoning_content`` is captured on receive into the AIMessage's
   ``additional_kwargs`` and re-attached on send so DeepSeek's API
   sees the same value across turns.
2. ``with_structured_output`` consults the capability table and uses JSON
   mode for V4/reasoner models that cannot be forced to call a schema tool.
"""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue
from pydantic import BaseModel

from tradingagents.llm_clients.openai_client import (
    DeepSeekChatOpenAI,
    NormalizedChatOpenAI,
    _input_to_messages,
)

# ---------------------------------------------------------------------------
# _input_to_messages — the helper that handles list / ChatPromptValue / other
# (Gemini bot review note: non-list inputs must also work)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInputToMessages:
    def test_list_input_returned_as_is(self):
        msgs = [HumanMessage(content="hi")]
        assert _input_to_messages(msgs) is msgs

    def test_chat_prompt_value_unwrapped(self):
        msgs = [HumanMessage(content="hi")]
        prompt_value = ChatPromptValue(messages=msgs)
        assert _input_to_messages(prompt_value) == msgs

    def test_string_input_yields_empty_list(self):
        # A bare string isn't a message-bearing input; the caller's normal
        # langchain conversion happens upstream of _get_request_payload.
        assert _input_to_messages("hello") == []


# ---------------------------------------------------------------------------
# Reasoning content propagation across turns
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeepSeekReasoningContent:
    def _client(self):
        os.environ.setdefault("DEEPSEEK_API_KEY", "placeholder")
        return DeepSeekChatOpenAI(
            model="deepseek-v4-flash",
            api_key="placeholder",
            base_url="https://api.deepseek.com",
        )

    def test_capture_on_receive(self):
        """When the response carries reasoning_content, it lands on the
        AIMessage's additional_kwargs so the next turn can echo it back."""
        client = self._client()
        result = client._create_chat_result(
            {
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Plan: buy NVDA.",
                            "reasoning_content": "Step 1: trend is up. Step 2: ...",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )
        ai = result.generations[0].message
        assert ai.additional_kwargs["reasoning_content"] == "Step 1: trend is up. Step 2: ..."

    def test_propagate_on_send(self):
        """When an outgoing AIMessage carries reasoning_content, the request
        payload echoes it on the corresponding message dict."""
        client = self._client()
        prior = AIMessage(
            content="Plan",
            additional_kwargs={"reasoning_content": "weighed bull case"},
        )
        new_user = HumanMessage(content="Refine.")
        payload = client._get_request_payload([prior, new_user])
        # Find the assistant message in the payload
        assistant_dicts = [m for m in payload["messages"] if m.get("role") == "assistant"]
        assert assistant_dicts, "assistant message missing from outgoing payload"
        assert assistant_dicts[0]["reasoning_content"] == "weighed bull case"

    def test_propagate_through_chat_prompt_value(self):
        """Gemini bot review note: non-list inputs (ChatPromptValue) must
        also propagate reasoning_content."""
        client = self._client()
        prior = AIMessage(
            content="Plan",
            additional_kwargs={"reasoning_content": "weighed bull case"},
        )
        prompt_value = ChatPromptValue(messages=[prior, HumanMessage(content="Refine.")])
        payload = client._get_request_payload(prompt_value)
        assistant_dicts = [m for m in payload["messages"] if m.get("role") == "assistant"]
        assert assistant_dicts[0]["reasoning_content"] == "weighed bull case"


# ---------------------------------------------------------------------------
# Capability-driven structured output: JSON mode for V4 + reasoner
# ---------------------------------------------------------------------------


def _bound_kwargs(runnable):
    """Extract bind() kwargs from a with_structured_output result."""
    first = runnable.steps[0] if hasattr(runnable, "steps") else runnable
    return getattr(first, "kwargs", {})


@pytest.mark.unit
class TestStructuredOutputCapabilityDispatch:
    """Thinking models use JSON mode; legacy chat keeps function calling."""

    class _Sample(BaseModel):
        answer: str

    def _client(self, model):
        return DeepSeekChatOpenAI(
            model=model, api_key="placeholder", base_url="https://api.deepseek.com",
        )

    def test_chat_sends_tool_choice(self):
        bound = self._client("deepseek-chat").with_structured_output(self._Sample)
        assert _bound_kwargs(bound).get("tool_choice") is not None

    def test_reasoner_uses_json_mode(self):
        bound = self._client("deepseek-reasoner").with_structured_output(self._Sample)
        assert _bound_kwargs(bound)["response_format"] == {"type": "json_object"}
        assert "tools" not in _bound_kwargs(bound)

    def test_v4_flash_uses_json_mode(self):
        bound = self._client("deepseek-v4-flash").with_structured_output(self._Sample)
        assert _bound_kwargs(bound)["response_format"] == {"type": "json_object"}
        assert "tools" not in _bound_kwargs(bound)

    def test_v4_pro_uses_json_mode(self):
        bound = self._client("deepseek-v4-pro").with_structured_output(self._Sample)
        assert _bound_kwargs(bound)["response_format"] == {"type": "json_object"}
        assert "tools" not in _bound_kwargs(bound)

    def test_future_v_variant_via_regex(self):
        """Forward-compat: unknown deepseek-v\\d-* IDs inherit JSON mode."""
        bound = self._client("deepseek-v5-hypothetical").with_structured_output(self._Sample)
        assert _bound_kwargs(bound)["response_format"] == {"type": "json_object"}

    def test_thinking_model_does_not_bind_schema_as_tool(self):
        bound = self._client("deepseek-reasoner").with_structured_output(self._Sample)
        kwargs = _bound_kwargs(bound)
        assert kwargs["response_format"] == {"type": "json_object"}
        assert "tools" not in kwargs


# ---------------------------------------------------------------------------
# Live API: structured output round-trips against the real DeepSeek backend
# ---------------------------------------------------------------------------


def _has_real_deepseek_key():
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    return bool(key) and key != "placeholder"


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_real_deepseek_key(),
    reason="DEEPSEEK_API_KEY not set (or placeholder); skipping live API call",
)
class TestDeepSeekLiveStructuredOutput:
    """End-to-end: a real DeepSeek V4-flash call returns a typed instance.

    Verifies JSON mode plus the explicit schema instruction parses to a
    Pydantic instance.
    """

    class _Pick(BaseModel):
        action: str
        confidence: float

    def test_v4_flash_returns_structured_output(self):
        client = DeepSeekChatOpenAI(
            model="deepseek-v4-flash",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=60,
        )
        from tradingagents.agents.utils.structured import (
            bind_structured,
            invoke_structured_with_metadata,
        )

        bound = bind_structured(client, self._Pick, "test")
        invocation = invoke_structured_with_metadata(
            bound,
            client,
            "Pick BUY or SELL or HOLD for a tech stock with strong earnings.",
            lambda result: result.action,
            "test",
        )
        result = self._Pick.model_validate(invocation.parsed)
        assert isinstance(result, self._Pick)
        assert result.action in {"BUY", "SELL", "HOLD"}
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Base class isolation: NormalizedChatOpenAI does NOT have DeepSeek behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBaseClassIsolation:
    def test_normalized_does_not_propagate_reasoning_content(self):
        """The general-purpose NormalizedChatOpenAI must not carry
        DeepSeek-specific behaviour. Only the subclass does."""
        assert not hasattr(NormalizedChatOpenAI, "_get_request_payload") or (
            NormalizedChatOpenAI._get_request_payload
            is NormalizedChatOpenAI.__bases__[0]._get_request_payload
        )
