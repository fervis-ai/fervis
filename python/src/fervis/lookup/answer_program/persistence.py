"""Canonical program invocation persistence boundary."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol

from fervis.lineage.ports import LineageRecorderPort
from fervis.lineage.enums import ProgramInvocationKind
from fervis.lineage.recorder import (
    AnswerProgramWrite,
    ProgramInvocationBundleWrite,
    ProgramInvocationWrite,
    ProgramRevisionBundleWrite,
    ProgramRevisionWrite,
)
from fervis.lookup.answer_program.codec import (
    answer_program_id,
    binding_patch_id,
    canonical_answer_program_json,
    canonical_binding_patch_json,
    canonical_binding_set_json,
    decode_answer_program,
    decode_binding_patch,
    decode_binding_set,
)
from fervis.lookup.answer_program.instantiation import VerifiedExecution
from fervis.lookup.answer_program.model import (
    ANSWER_PROGRAM_SCHEMA_REVISION,
    AnswerProgram,
)
from fervis.lookup.answer_program.revisions import (
    ProgramRevision,
    canonical_capability_application_json,
    program_revision_id,
)
from fervis.lookup.answer_program.values import BindingSet
from fervis.lookup.answer_program.values import BindingPatch


@dataclass(frozen=True)
class ProgramInvocation:
    invocation_id: str
    run_id: str
    program_id: str
    bindings: BindingSet
    kind: ProgramInvocationKind
    base_invocation_id: str | None = None
    binding_patch: BindingPatch | None = None
    revision_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind is ProgramInvocationKind.COMPILED_QUESTION:
            if self.base_invocation_id is not None:
                raise ValueError("compiled question cannot have a base invocation")
            return
        if self.base_invocation_id is None:
            raise ValueError(
                "continued and rerun invocations require a base invocation"
            )

    @property
    def patch_id(self) -> str | None:
        return (
            binding_patch_id(self.binding_patch)
            if self.binding_patch is not None
            else None
        )


@dataclass(frozen=True)
class StoredProgramInvocation:
    invocation: ProgramInvocation
    program: AnswerProgram

    @property
    def bindings(self) -> BindingSet:
        return self.invocation.bindings

    @property
    def binding_patch(self) -> BindingPatch | None:
        return self.invocation.binding_patch


class ProgramInvocationBinding(Protocol):
    def bind(
        self,
        execution: VerifiedExecution,
        *,
        kind: ProgramInvocationKind,
        base_invocation_id: str | None,
    ) -> ProgramInvocation: ...


class PriorProgramInvocationReader(Protocol):
    def load_prior_answered_invocation(
        self,
        *,
        run_id: str,
        conversation_id: str,
        tenant_id: str,
    ) -> StoredProgramInvocation | None: ...


@dataclass(frozen=True)
class LineageProgramInvocationBinding:
    run_id: str
    recorder: LineageRecorderPort

    def bind(
        self,
        execution: VerifiedExecution,
        *,
        kind: ProgramInvocationKind,
        base_invocation_id: str | None,
    ) -> ProgramInvocation:
        program_id = answer_program_id(execution.answer)
        invocation = program_invocation(
            run_id=self.run_id,
            program_id=program_id,
            bindings=execution.bindings,
            kind=kind,
            base_invocation_id=base_invocation_id,
        )
        self.recorder.record_program_invocation(
            program_invocation_bundle(
                program=execution.answer,
                invocation=invocation,
            )
        )
        return invocation


@dataclass(frozen=True)
class StoredProgramInvocationBinding:
    stored: StoredProgramInvocation

    def bind(
        self,
        execution: VerifiedExecution,
        *,
        kind: ProgramInvocationKind,
        base_invocation_id: str | None,
    ) -> ProgramInvocation:
        if answer_program_id(execution.answer) != self.stored.invocation.program_id:
            raise ValueError("stored invocation program does not match execution")
        if execution.bindings != self.stored.invocation.bindings:
            raise ValueError("stored invocation bindings do not match execution")
        if self.stored.invocation.kind is not kind:
            raise ValueError("stored invocation kind does not match execution")
        if self.stored.invocation.base_invocation_id != base_invocation_id:
            raise ValueError("stored invocation base does not match execution")
        return self.stored.invocation


def program_invocation_bundle(
    *,
    program: AnswerProgram,
    invocation: ProgramInvocation,
) -> ProgramInvocationBundleWrite:
    program_id = answer_program_id(program)
    if invocation.program_id != program_id:
        raise ValueError("program invocation does not reference its canonical program")
    return ProgramInvocationBundleWrite(
        program=AnswerProgramWrite(
            program_id=program_id,
            schema_revision=ANSWER_PROGRAM_SCHEMA_REVISION,
            canonical_json=canonical_answer_program_json(program),
        ),
        invocation=ProgramInvocationWrite(
            invocation_id=invocation.invocation_id,
            run_id=invocation.run_id,
            program_id=invocation.program_id,
            bindings_json=canonical_binding_set_json(invocation.bindings),
            kind=invocation.kind,
            base_invocation_id=invocation.base_invocation_id,
            patch_id=invocation.patch_id,
            binding_patch_json=(
                canonical_binding_patch_json(invocation.binding_patch)
                if invocation.binding_patch is not None
                else None
            ),
            revision_id=invocation.revision_id,
        ),
    )


def program_revision_bundle(
    *,
    revision: ProgramRevision,
) -> ProgramRevisionBundleWrite:
    revised_program_id = answer_program_id(revision.program)
    if revised_program_id != revision.revised_program_id:
        raise ValueError("program revision does not reference its canonical program")
    if revision.revision_id != program_revision_id(
        base_program_id=revision.base_program_id,
        revised_program_id=revision.revised_program_id,
        application=revision.application,
    ):
        raise ValueError("program revision id does not match canonical content")
    return ProgramRevisionBundleWrite(
        program=AnswerProgramWrite(
            program_id=revised_program_id,
            schema_revision=ANSWER_PROGRAM_SCHEMA_REVISION,
            canonical_json=canonical_answer_program_json(revision.program),
        ),
        revision=ProgramRevisionWrite(
            revision_id=revision.revision_id,
            base_program_id=revision.base_program_id,
            revised_program_id=revision.revised_program_id,
            capability_id=revision.application.capability_id,
            application_json=canonical_capability_application_json(
                revision.application
            ),
        ),
    )


def program_invocation(
    *,
    run_id: str,
    program_id: str,
    bindings: BindingSet,
    kind: ProgramInvocationKind,
    base_invocation_id: str | None = None,
    patch: BindingPatch | None = None,
    revision_id: str | None = None,
) -> ProgramInvocation:
    return ProgramInvocation(
        invocation_id=program_invocation_id(
            run_id=run_id,
            program_id=program_id,
            bindings=bindings,
        ),
        run_id=run_id,
        program_id=program_id,
        bindings=bindings,
        kind=kind,
        base_invocation_id=base_invocation_id,
        binding_patch=patch,
        revision_id=revision_id,
    )


def program_invocation_id(*, run_id: str, program_id: str, bindings: BindingSet) -> str:
    payload = "\0".join(
        (run_id, program_id, canonical_binding_set_json(bindings))
    ).encode()
    return "pi_" + hashlib.sha256(payload).hexdigest()


def parse_stored_program_invocation(
    *,
    invocation_id: str,
    run_id: str,
    program_id: str,
    canonical_json: str,
    bindings_json: str,
    kind: str,
    base_invocation_id: str | None = None,
    patch_id: str | None = None,
    binding_patch_json: str | None = None,
    revision_id: str | None = None,
) -> StoredProgramInvocation:
    program = decode_answer_program(canonical_json)
    if answer_program_id(program) != program_id:
        raise ValueError("stored answer program id does not match canonical content")
    bindings = decode_binding_set(bindings_json)
    expected_invocation_id = program_invocation_id(
        run_id=run_id,
        program_id=program_id,
        bindings=bindings,
    )
    if invocation_id != expected_invocation_id:
        raise ValueError("stored invocation id does not match canonical content")
    if (patch_id is None) != (binding_patch_json is None):
        raise ValueError("stored binding patch id and payload must be present together")
    patch = (
        decode_binding_patch(binding_patch_json)
        if binding_patch_json is not None
        else None
    )
    if patch is not None and binding_patch_id(patch) != patch_id:
        raise ValueError("stored binding patch id does not match canonical content")
    return StoredProgramInvocation(
        invocation=ProgramInvocation(
            invocation_id=invocation_id,
            run_id=run_id,
            program_id=program_id,
            bindings=bindings,
            kind=ProgramInvocationKind(kind),
            base_invocation_id=base_invocation_id,
            binding_patch=patch,
            revision_id=revision_id,
        ),
        program=program,
    )
