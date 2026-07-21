"""Apply declared structural capabilities to immutable answer programs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

from fervis.lookup.answer_program.capability_contracts import (
    CapabilityApplication,
    NarrowPopulationCapability,
)
from fervis.lookup.answer_program.codec import (
    _decode_as,
    _encode,
    answer_program_id,
    canonicalize_answer_program,
)
from fervis.lookup.answer_program.compatibility import (
    FUNCTION_SEMANTIC_VERSION,
    compatibility_requirements,
)
from fervis.lookup.answer_program.contracts import AnswerProgramContractError
from fervis.lookup.answer_program.inputs import compile_program_inputs
from fervis.lookup.answer_program.model import AnswerProgram, FunctionSemanticVersion
from fervis.lookup.answer_program.relations import (
    FieldBindingRole,
)
from fervis.lookup.answer_program.expressions import FieldRef
from fervis.lookup.answer_program.operations import (
    FilterSpec,
    Operation,
    Predicate,
)
from fervis.lookup.answer_program.values import (
    BindingSet,
    ParameterRef,
    ProgramInputs,
)


@dataclass(frozen=True)
class ProgramRevision:
    revision_id: str
    base_program_id: str
    revised_program_id: str
    application: CapabilityApplication
    program: AnswerProgram
    bindings: BindingSet

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.revision_id,
                self.base_program_id,
                self.revised_program_id,
            )
        ):
            raise ValueError("program revision requires content identities")
        if not isinstance(self.application, CapabilityApplication):
            raise TypeError("program revision application must be typed")
        if not isinstance(self.program, AnswerProgram):
            raise TypeError("program revision program must be AnswerProgram")
        if not isinstance(self.bindings, BindingSet):
            raise TypeError("program revision bindings must be BindingSet")


def apply_capability(
    *,
    program: AnswerProgram,
    bindings: BindingSet,
    application: CapabilityApplication,
) -> ProgramRevision:
    capability = next(
        (item for item in program.capabilities if item.id == application.capability_id),
        None,
    )
    if capability is None:
        raise AnswerProgramContractError(
            "undeclared_capability",
            f"program does not declare capability {application.capability_id}",
        )
    verify_capability_declarations(program)
    _verify_application_preconditions(
        program,
        capability=capability,
        application=application,
    )
    effective_application = replace(
        application,
        binding=replace(
            application.binding,
            provenance=replace(
                application.binding.provenance,
                refs=tuple(
                    dict.fromkeys(
                        (
                            *application.binding.provenance.refs,
                            *capability.proof_refs,
                        )
                    )
                ),
            ),
        ),
    )
    revised_bindings = BindingSet.from_bindings(
        (*bindings.bindings, effective_application.binding)
    )
    compile_program_inputs(
        ProgramInputs(
            parameters=(*program.parameters, capability.parameter),
            bindings=revised_bindings,
        )
    )
    revised_graph = replace(
        program,
        parameters=(*program.parameters, capability.parameter),
        relations=tuple(
            replace(relation, id=f"{relation.id}__capability_source")
            if relation.id == capability.relation_id
            else relation
            for relation in program.relations
        ),
        operations=(
            Operation(
                id=f"capability.{capability.id}",
                spec=FilterSpec(
                    input_relation=f"{capability.relation_id}__capability_source",
                    predicate=Predicate(
                        left=FieldRef(capability.field_id),
                        operator=capability.operator,
                        right=ParameterRef(parameter_id=capability.parameter.id),
                    ),
                    proof_refs=capability.proof_refs,
                ),
                output_relation=capability.relation_id,
            ),
            *program.operations,
        ),
        capabilities=tuple(
            item for item in program.capabilities if item.id != capability.id
        ),
    )
    revised = canonicalize_answer_program(
        replace(
            revised_graph,
            compatibility=replace(
                revised_graph.compatibility,
                function_semantics=tuple(
                    FunctionSemanticVersion(
                        function_key=key,
                        version=FUNCTION_SEMANTIC_VERSION,
                    )
                    for key in compatibility_requirements(revised_graph).function_keys
                ),
            ),
        )
    )
    base_id = answer_program_id(program)
    revised_id = answer_program_id(revised)
    revision_id = program_revision_id(
        base_program_id=base_id,
        revised_program_id=revised_id,
        application=effective_application,
    )
    return ProgramRevision(
        revision_id=revision_id,
        base_program_id=base_id,
        revised_program_id=revised_id,
        application=effective_application,
        program=revised,
        bindings=revised_bindings,
    )


def verify_capability_declarations(program: AnswerProgram) -> None:
    capability_ids = tuple(item.id for item in program.capabilities)
    if len(set(capability_ids)) != len(capability_ids):
        raise AnswerProgramContractError(
            "invalid_capability_declaration",
            "program contains duplicate capability ids",
        )
    declared_parameter_ids = tuple(item.parameter.id for item in program.capabilities)
    if len(set(declared_parameter_ids)) != len(declared_parameter_ids):
        raise AnswerProgramContractError(
            "invalid_capability_declaration",
            "capabilities cannot declare the same parameter",
        )
    existing_parameter_ids = {item.id for item in program.parameters}
    known_fact_ids = {fact.id for fact in program.fact_template}
    for capability in program.capabilities:
        if capability.function_semantics_version != FUNCTION_SEMANTIC_VERSION:
            raise AnswerProgramContractError(
                "incompatible_capability_semantics",
                "program capability function semantics version is incompatible",
            )
        if capability.parameter.id in existing_parameter_ids:
            raise AnswerProgramContractError(
                "invalid_capability_declaration",
                "capability parameter already exists",
            )
        relation = next(
            (item for item in program.relations if item.id == capability.relation_id),
            None,
        )
        if relation is None:
            raise AnswerProgramContractError(
                "invalid_capability_declaration",
                "capability relation is unavailable",
            )
        field = next(
            (item for item in relation.fields if item.field_id == capability.field_id),
            None,
        )
        if field is None or FieldBindingRole.PREDICATE not in field.roles:
            raise AnswerProgramContractError(
                "invalid_capability_declaration",
                "capability field is not a declared predicate surface",
            )
        if any(item not in known_fact_ids for item in capability.requested_fact_ids):
            raise AnswerProgramContractError(
                "invalid_capability_declaration",
                "capability semantic effect references an unknown fact",
            )


def canonical_capability_application_json(
    application: CapabilityApplication,
) -> str:
    import json

    return json.dumps(
        _encode(application),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def decode_capability_application(payload: str) -> CapabilityApplication:
    import json

    return _decode_as(json.loads(payload), CapabilityApplication)


def _verify_application_preconditions(
    program: AnswerProgram,
    *,
    capability: NarrowPopulationCapability,
    application: CapabilityApplication,
) -> None:
    if application.binding.parameter_id != capability.parameter.id:
        raise AnswerProgramContractError(
            "capability_input_mismatch",
            "capability binding does not target its declared parameter",
        )


def program_revision_id(
    *,
    base_program_id: str,
    revised_program_id: str,
    application: CapabilityApplication,
) -> str:
    payload = "\n".join(
        (
            base_program_id,
            revised_program_id,
            canonical_capability_application_json(application),
        )
    )
    return "apr_" + hashlib.sha256(payload.encode()).hexdigest()
