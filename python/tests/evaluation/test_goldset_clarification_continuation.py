from __future__ import annotations

from fervis.evaluation.goldsets.runner import _continue_and_follow
from fervis.questions import AskResult, QuestionPrincipal


class _Questions:
    def __init__(self) -> None:
        self.clarification_request = None

    def respond_to_clarification(self, request, *, event_sink=None):
        del event_sink
        self.clarification_request = request
        return AskResult(
            status="COMPLETED",
            conversation_id="conversation_1",
            question_id=request.question_id,
            run_id="run_2",
            answer="31",
        )


def _waiting_result(*, options: list[dict[str, str]]) -> AskResult:
    return AskResult(
        status="WAITING_FOR_CLARIFICATION",
        conversation_id="conversation_1",
        question_id="question_1",
        run_id="run_1",
        result_data={
            "kind": "needs_clarification",
            "details": {
                "clarifications": [
                    {
                        "id": "clarification_area",
                        "subjects": [
                            {
                                "id": "area_input",
                                "options": options,
                            }
                        ],
                    }
                ]
            },
        },
    )


def test_goldset_continuation_selects_the_exact_stored_option() -> None:
    questions = _Questions()

    result = _continue_and_follow(
        "Area: Nairobi",
        previous=_waiting_result(
            options=[
                {"id": "area_nairobi", "label": "Area: Nairobi"},
                {"id": "location_nairobi", "label": "Location: Nairobi"},
            ]
        ),
        questions=questions,
        question_run_follower=None,
        principal=QuestionPrincipal(principal_id="user_1", tenant_id="tenant_1"),
        wait_seconds=0,
    )

    assert result is not None
    assert questions.clarification_request is not None
    assert questions.clarification_request.selected_option_id == "area_nairobi"
    assert questions.clarification_request.response_text == "Area: Nairobi"


def test_goldset_continuation_preserves_an_unmatched_free_text_answer() -> None:
    questions = _Questions()

    result = _continue_and_follow(
        "Use the northern region",
        previous=_waiting_result(options=[]),
        questions=questions,
        question_run_follower=None,
        principal=QuestionPrincipal(principal_id="user_1", tenant_id="tenant_1"),
        wait_seconds=0,
    )

    assert result is not None
    assert questions.clarification_request is not None
    assert questions.clarification_request.selected_option_id == ""
    assert questions.clarification_request.response_text == "Use the northern region"
