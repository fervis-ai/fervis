"""Terminal rendering delegates to the public typed-result presentation contract."""
from fervis.lookup.clarification import render_clarification_question
from fervis.lookup.outcomes.model import FactResult, NeedsClarification, ResultOutcome
from fervis.lookup.outcomes.terminal_details import fact_result_terminal_details
from fervis.questions.result_data import terminal_result_message


def terminal_message(outcome: ResultOutcome) -> str:
    if isinstance(outcome, NeedsClarification):
        questions = [render_clarification_question(item) for item in outcome.clarifications]
        return "\n".join(questions) if questions else "Can you clarify the requested value?"
    return terminal_result_message(outcome.kind.value, fact_result_terminal_details(FactResult(outcome=outcome)) or {})
