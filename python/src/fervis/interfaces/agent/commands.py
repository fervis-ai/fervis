"""Typed Fervis command references for agent-facing follow-up actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import shlex
from typing import Union


class FervisCommand(StrEnum):
    AUTH_CONFIGURE = "auth.configure"
    DOCTOR = "doctor"
    EXPLAIN_QUESTION = "explain.question"
    EXPLAIN_RUN = "explain.run"
    INIT = "init"
    MIGRATE = "migrate"
    MODEL_ALLOW = "model.allow"
    MODEL_USE = "model.use"
    RUNTIME_ASK = "runtime.ask"


@dataclass(frozen=True)
class Placeholder:
    name: str
    quoted: bool = False


@dataclass(frozen=True)
class Switch:
    name: str


@dataclass(frozen=True)
class Option:
    name: str
    value: object


@dataclass(frozen=True)
class Positional:
    value: object


CommandPart = Union[Switch, Option, Positional]


@dataclass(frozen=True)
class CommandInvocation:
    command: FervisCommand
    parts: tuple[CommandPart, ...] = ()


COMMAND_PATHS: dict[FervisCommand, tuple[str, ...]] = {
    FervisCommand.AUTH_CONFIGURE: ("auth", "configure"),
    FervisCommand.DOCTOR: ("doctor",),
    FervisCommand.EXPLAIN_QUESTION: ("explain",),
    FervisCommand.EXPLAIN_RUN: ("explain",),
    FervisCommand.INIT: ("init",),
    FervisCommand.MIGRATE: ("migrate",),
    FervisCommand.MODEL_ALLOW: ("model", "allow"),
    FervisCommand.MODEL_USE: ("model", "use"),
    FervisCommand.RUNTIME_ASK: ("runtime", "ask"),
}


class CommandBuilders:
    def auth_configure(
        self,
        *,
        framework: str | None = None,
        transport_mode: str,
        principal_dependency: object | None = None,
        principal_source: object | None = None,
        principal_id_attr: object | None = None,
        principal_resolver: object | None = None,
    ) -> CommandInvocation:
        parts: list[CommandPart] = []
        if framework:
            parts.append(Option("--framework", framework))
        if principal_dependency is not None:
            parts.append(Option("--principal-dependency", principal_dependency))
        if principal_source is not None:
            parts.append(Option("--principal-source", principal_source))
        if principal_id_attr is not None:
            parts.append(Option("--principal-id-attr", principal_id_attr))
        if principal_resolver is not None:
            parts.append(Option("--principal-resolver", principal_resolver))
        parts.append(Option("--transport-mode", transport_mode))
        return CommandInvocation(FervisCommand.AUTH_CONFIGURE, tuple(parts))

    def doctor(
        self,
        *,
        probe_read_context_key: object | None = None,
    ) -> CommandInvocation:
        parts: list[CommandPart] = []
        if probe_read_context_key is not None:
            parts.append(Option("--probe-read-context-key", probe_read_context_key))
        return CommandInvocation(FervisCommand.DOCTOR, tuple(parts))

    def explain_question(
        self,
        question_id: object,
        *,
        debug: bool = False,
    ) -> CommandInvocation:
        parts: list[CommandPart] = [Option("--question-id", question_id)]
        if debug:
            parts.append(Switch("--debug"))
        return CommandInvocation(FervisCommand.EXPLAIN_QUESTION, tuple(parts))

    def explain_run(self, run_id: object) -> CommandInvocation:
        return CommandInvocation(
            FervisCommand.EXPLAIN_RUN,
            (Option("--run-id", run_id),),
        )

    def init(
        self,
        *,
        framework: str,
        app_factory: object | None = None,
        yes: bool = True,
    ) -> CommandInvocation:
        parts: list[CommandPart] = [Option("--framework", framework)]
        if app_factory is not None:
            parts.append(Option("--app-factory", app_factory))
        if yes:
            parts.append(Switch("--yes"))
        return CommandInvocation(FervisCommand.INIT, tuple(parts))

    def migrate(self) -> CommandInvocation:
        return CommandInvocation(FervisCommand.MIGRATE)

    def model_allow(self, model_ref: object) -> CommandInvocation:
        return CommandInvocation(FervisCommand.MODEL_ALLOW, (Positional(model_ref),))

    def model_use(self, model_ref: object) -> CommandInvocation:
        return CommandInvocation(FervisCommand.MODEL_USE, (Positional(model_ref),))

    def runtime_ask(
        self,
        answer: object,
        *,
        question_id: object,
        previous_run_id: object,
        clarification_id: object,
        conversation_id: object,
        tenant_id: object,
        principal_id: object,
    ) -> CommandInvocation:
        return CommandInvocation(
            FervisCommand.RUNTIME_ASK,
            (
                Positional(answer),
                Option("--question-id", question_id),
                Option("--previous-run-id", previous_run_id),
                Option("--clarification-id", clarification_id),
                Option("--conversation-id", conversation_id),
                Option("--tenant-id", tenant_id),
                Option("--principal-id", principal_id),
            ),
        )


commands = CommandBuilders()


def render_command(invocation: CommandInvocation) -> str:
    words: list[str] = [
        "fervis",
        *COMMAND_PATHS[invocation.command],
    ]
    for part in invocation.parts:
        words.extend(_render_part(part))
    return " ".join(_quote_word(word) for word in words)


def _render_part(part: CommandPart) -> tuple[str, ...]:
    if isinstance(part, Switch):
        return (part.name,)
    if isinstance(part, Option):
        return (part.name, _render_value(part.value))
    return (_render_value(part.value),)


def _render_value(value: object) -> str:
    if isinstance(value, Placeholder):
        text = f"<{value.name}>"
        if value.quoted:
            return f'"{text}"'
        return text
    return str(value)


def _quote_word(value: str) -> str:
    if value.startswith("<") and value.endswith(">"):
        return value
    if value.startswith('"') and value.endswith('"'):
        return value
    return shlex.quote(value)
