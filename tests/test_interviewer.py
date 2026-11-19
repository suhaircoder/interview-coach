from types import SimpleNamespace
from typing import Any

from src.interviewer import Interviewer


class FakeCompletions:
    def __init__(self) -> None:
        self.arguments: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.arguments = kwargs
        message = SimpleNamespace(content="Generated interview response")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_interviewer() -> tuple[Interviewer, FakeCompletions]:
    completions = FakeCompletions()
    interviewer = Interviewer.__new__(Interviewer)
    interviewer.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    interviewer.model = "test-model"
    return interviewer, completions


def test_guided_response_builds_personalized_prompt() -> None:
    interviewer, completions = make_interviewer()
    history = [
        {"role": "user", "content": "Previous answer"},
        {"role": "assistant", "content": "Previous question"},
        {"role": "assistant", "content": {"path": "ignored-file"}},
    ]

    result = interviewer.respond(
        "My current answer",
        history,
        "Backend Engineer",
        "Senior",
        "Technical",
        "Guided practice",
        "Built Python APIs at Example Corp.",
    )

    assert result == "Generated interview response"
    messages = completions.arguments["messages"]
    system_prompt = messages[0]["content"]
    assert "Backend Engineer" in system_prompt
    assert "Built Python APIs at Example Corp." in system_prompt
    assert "Guided practice mode" in system_prompt
    assert "Relevance (20%)" in system_prompt
    assert messages[-1] == {"role": "user", "content": "My current answer"}
    assert len(messages) == 4


def test_real_interview_uses_no_live_feedback_policy() -> None:
    interviewer, completions = make_interviewer()

    interviewer.respond(
        "Start",
        [],
        "Product Manager",
        "Mid level",
        "Mixed",
        "Real interview",
    )

    system_prompt = completions.arguments["messages"][0]["content"]
    assert "Real interview mode" in system_prompt
    assert "Save all feedback for the final report" in system_prompt


def test_blank_message_does_not_call_provider() -> None:
    interviewer, completions = make_interviewer()

    result = interviewer.respond(
        "   ",
        [],
        "Engineer",
        "Entry level",
        "Mixed",
        "Guided practice",
    )

    assert result == "Please enter a response so we can continue the interview."
    assert completions.arguments == {}
