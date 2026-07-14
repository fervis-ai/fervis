from __future__ import annotations

from dataclasses import replace
from typing import Any
from fervis.lineage.enums import ProgramInvocationKind

from fervis.lookup.answer_program import (
    AnswerProgram,
    AnswerProgramContractError,
    FunctionSemanticVersion,
    ProgramCompatibility,
    SourceContractKind,
    SourceContractPin,
    answer_program_id,
    canonical_answer_program_payload,
    canonical_binding_patch_json,
    canonical_binding_set_json,
    canonical_contract_fingerprint,
    canonicalize_answer_program,
    decode_answer_program,
    binding_patch_id,
)
from fervis.lookup.answer_program.compilation import compile_answer_program
from fervis.lookup.answer_program.inputs import apply_binding_patch
from fervis.lookup.answer_program.instantiation import (
    ExecutionEnvironment,
    instantiate_answer_program,
)
from fervis.lookup.answer_program.invocation import RuntimePorts, invoke_answer_program
from fervis.lookup.answer_program.instantiation import _instantiate_operations
from fervis.lookup.answer_program.revisions import (
    apply_capability,
    canonical_capability_application_json,
    decode_capability_application,
)
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.answer_program.persistence import program_invocation

from tests.testkit.assertions import exact_mismatches, subset_mismatches
from tests.testkit.answer_program_contracts import (
    binding_patch_from_payload,
    binding_payload,
    binding_set_from_payload,
    capability_application_from_payload,
    parameter_declarations_from_payload,
    fact_value_from_payload,
)
from fervis.lookup.answer_program.values import ConstantRef
from fervis.lookup.answer_program.values import BindingSet, FactValue, LiteralType
from fervis.lookup.answer_program.operations import (
    Operation,
    RankSpec,
    SortDirection,
    SortKey,
    TiePolicy,
)
from fervis.lookup.plan_execution.operation_runtime import ResolvedRankSpec
from tests.testkit.catalog import catalog_from_payload
from tests.testkit.question_contract import question_contract_from_payload
from tests.testkit.serialization import portable_value


def run_answer_program_canonicalize_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    programs = tuple(decode_answer_program(item) for item in input_payload["programs"])
    canonical = tuple(canonicalize_answer_program(program) for program in programs)
    canonical_payloads = tuple(
        canonical_answer_program_payload(program) for program in canonical
    )
    program_ids = tuple(answer_program_id(program) for program in canonical)
    binding_sets = tuple(
        binding_set_from_payload(item)
        for item in input_payload.get("binding_sets") or ()
    )
    patches = tuple(
        binding_patch_from_payload(item) for item in input_payload.get("patches") or ()
    )
    binding_jsons = tuple(canonical_binding_set_json(item) for item in binding_sets)
    patch_jsons = tuple(canonical_binding_patch_json(item) for item in patches)
    constant_fingerprints = tuple(
        canonical_contract_fingerprint(
            ConstantRef(
                constant_id=str(item["constant_id"]),
                version_ref=str(item["version_ref"]),
                value=fact_value_from_payload(
                    item["value"],
                    value_id=str(item["constant_id"]),
                ),
            )
        )
        for item in input_payload.get("constants") or ()
    )
    run_id = str(input_payload.get("run_id") or "canonical_run")
    invocation_ids = tuple(
        program_invocation(
            run_id=run_id,
            program_id=program_ids[0],
            bindings=bindings,
            kind=ProgramInvocationKind.COMPILED_QUESTION,
        ).invocation_id
        for bindings in binding_sets
    )
    actual = {
        "programs": [
            {"program_id": program_id, "program": program_payload}
            for program_id, program_payload in zip(
                program_ids,
                canonical_payloads,
                strict=True,
            )
        ],
        "same_program_id": len(set(program_ids)) == 1,
        "same_program": len({str(item) for item in canonical_payloads}) == 1,
        "binding_jsons": list(binding_jsons),
        "patch_jsons": list(patch_jsons),
        "patch_ids": [binding_patch_id(item) for item in patches],
        "invocation_ids": list(invocation_ids),
        "same_binding_json": len(set(binding_jsons)) <= 1,
        "same_patch_json": len(set(patch_jsons)) <= 1,
        "same_patch_id": len({binding_patch_id(item) for item in patches}) <= 1,
        "same_invocation_id": len(set(invocation_ids)) <= 1,
        "same_constant_fingerprint": len(set(constant_fingerprints)) <= 1,
    }
    return _mismatches(payload, actual=actual)


def run_answer_program_decode_case(payload: dict[str, Any]) -> list[str]:
    try:
        program = decode_answer_program(payload["input"]["program"])
    except AnswerProgramContractError as exc:
        actual = {"status": "rejected", "code": exc.code}
    else:
        actual = {
            "status": "decoded",
            "program_id": answer_program_id(program),
        }
    return _mismatches(payload, actual=actual)


def run_answer_program_compile_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    try:
        program, bindings = compile_answer_program(
            decode_answer_program(input_payload["program"]),
            question_contract=question_contract_from_payload(
                input_payload["question_contract"]
            ),
            catalog=catalog_from_payload(input_payload["catalog"]),
            bindings=binding_set_from_payload(input_payload),
        )
    except AnswerProgramContractError as exc:
        actual = {
            "status": "rejected",
            "code": exc.code,
            "reads": 0,
            "reusable": False,
        }
    else:
        actual = {
            "status": "compiled",
            "program_id": answer_program_id(program),
            "program": canonical_answer_program_payload(program),
            "bindings": binding_payload(bindings),
            "compatibility": _compatibility_payload(program.compatibility),
        }
    return _mismatches(payload, actual=actual)


def run_answer_program_patch_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    parameters = parameter_declarations_from_payload(input_payload)
    bindings = binding_set_from_payload(input_payload)
    base_payload = binding_payload(bindings)
    if "attempts" in input_payload:
        attempts: list[dict[str, str]] = []
        for attempt in input_payload["attempts"]:
            try:
                apply_binding_patch(
                    program=AnswerProgram(parameters=parameters),
                    bindings=bindings,
                    patch=binding_patch_from_payload(attempt),
                )
            except AnswerProgramContractError as exc:
                attempts.append({"id": str(attempt["id"]), "code": exc.code})
            else:
                attempts.append({"id": str(attempt["id"]), "code": "accepted"})
        actual: dict[str, Any] = {
            "status": "rejected",
            "base": base_payload,
            "attempts": attempts,
        }
    else:
        patch = binding_patch_from_payload(input_payload["patch"])
        try:
            revised = apply_binding_patch(
                program=AnswerProgram(parameters=parameters),
                bindings=bindings,
                patch=patch,
            )
        except AnswerProgramContractError as exc:
            actual = {
                "status": "rejected",
                "code": exc.code,
                "base": base_payload,
            }
        else:
            actual = {
                "status": "patched",
                "base": base_payload,
                "revised": binding_payload(revised),
                "changed_parameter_ids": sorted(
                    operation.parameter_id for operation in patch.operations
                ),
            }
    return _mismatches(payload, actual=actual)


def run_answer_program_apply_capability_case(
    payload: dict[str, Any],
) -> list[str]:
    input_payload = payload["input"]
    program = decode_answer_program(input_payload["program"])
    bindings = binding_set_from_payload(input_payload)
    application = capability_application_from_payload(input_payload["application"])
    try:
        revision = apply_capability(
            program=program,
            bindings=bindings,
            application=application,
        )
    except AnswerProgramContractError as exc:
        actual = {
            "status": "rejected",
            "code": exc.code,
            "base_program_id": answer_program_id(program),
            "reads": 0,
            "model_calls": 0,
        }
    else:
        encoded_application = canonical_capability_application_json(
            revision.application
        )
        actual = {
            "status": "revised",
            "revision_id": revision.revision_id,
            "base_program_id": revision.base_program_id,
            "revised_program_id": revision.revised_program_id,
            "application_round_trips": (
                decode_capability_application(encoded_application)
                == revision.application
            ),
            "compatibility_unchanged": (
                revision.program.compatibility == program.compatibility
            ),
            "compatibility": _compatibility_payload(revision.program.compatibility),
            "program": canonical_answer_program_payload(revision.program),
            "bindings": binding_payload(revision.bindings),
            "reads": 0,
            "model_calls": 0,
        }
    return _mismatches(payload, actual=actual)


def run_answer_program_invoke_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    program = decode_answer_program(input_payload["program"])
    catalog = catalog_from_payload(input_payload["catalog"])
    environment = _execution_environment(
        catalog=catalog,
        payload=input_payload["environment"],
    )
    invocations: list[dict[str, Any]] = []
    for invocation in input_payload["invocations"]:
        access = _ScriptedDataAccess(tuple(invocation.get("evidence") or ()))
        try:
            execution = invoke_answer_program(
                program=program,
                bindings=binding_set_from_payload(invocation),
                environment=environment,
                ports=RuntimePorts(
                    data_access_port=access,
                    memory=LookupMemory(),
                ),
            )
        except (AnswerProgramContractError, VerificationError) as exc:
            result = {
                "id": str(invocation["id"]),
                "status": "rejected",
                "code": getattr(exc, "code", str(exc)),
                "requests": portable_value(access.requests),
            }
        else:
            result = {
                "id": str(invocation["id"]),
                "status": "completed",
                "program_id": execution.program_id,
                "fact_result": portable_value(execution.fact_result),
                "relations": [
                    {
                        "id": relation.id,
                        "rows": portable_value(relation.rows),
                        "row_count": relation.completeness.row_count,
                        "snapshot_hash": relation.evidence.snapshot_hash,
                        "scope_fingerprint": (relation.completeness.scope_fingerprint),
                        "proof_refs": list(relation.completeness.proof_refs),
                    }
                    for relation in execution.relations
                ],
                "proof_graph": portable_value(execution.proof_graph),
                "effective_requested_facts": [
                    {
                        "id": fact.id,
                        "description": fact.description,
                        "population_constraints": [
                            {
                                "id": constraint.id,
                                "included_values": list(constraint.included_values),
                                "excluded_values": list(constraint.excluded_values),
                            }
                            for constraint in fact.population_constraints
                        ],
                    }
                    for fact in getattr(
                        execution,
                        "effective_requested_facts",
                        (),
                    )
                ],
                "requests": portable_value(access.requests),
            }
        invocations.append(result)
    return _mismatches(
        payload,
        actual={
            "invocations": invocations,
            "same_program_id": len(
                {
                    invocation.get("program_id")
                    for invocation in invocations
                    if invocation.get("program_id")
                }
            )
            <= 1,
        },
    )


def run_answer_program_rank_limit_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    limit = ConstantRef(
        constant_id="rank_limit",
        version_ref="fixture-v1",
        value=FactValue.literal(
            id="rank_limit",
            literal_type=LiteralType.NUMBER,
            value=str(input_payload["value"]),
        ),
    )
    operations, _inputs = _instantiate_operations(
        AnswerProgram(
            operations=(
                Operation(
                    id="rank",
                    spec=RankSpec(
                        input_relation="rows",
                        order_by=(
                            SortKey(field="value", direction=SortDirection.DESC),
                        ),
                        tie_policy=TiePolicy.FIELD,
                        limit=limit,
                        tie_breakers=(
                            SortKey(field="id", direction=SortDirection.ASC),
                        ),
                    ),
                    output_relation="ranked",
                ),
            ),
        ),
        bindings=BindingSet(),
    )
    resolved = operations[0].spec
    return _mismatches(
        payload,
        actual={"limit": resolved.limit if isinstance(resolved, ResolvedRankSpec) else None},
    )


def run_answer_program_instantiate_case(payload: dict[str, Any]) -> list[str]:
    attempts: list[dict[str, Any]] = []
    for attempt in payload["input"]["attempts"]:
        catalog = catalog_from_payload(attempt["catalog"])
        program = decode_answer_program(attempt["program"])
        if "compatibility" in attempt:
            program = replace(
                program,
                compatibility=_compatibility_from_payload(attempt["compatibility"]),
            )
        try:
            execution = instantiate_answer_program(
                program,
                binding_set_from_payload(attempt),
                _execution_environment(
                    catalog=catalog,
                    payload=attempt["environment"],
                ),
            )
        except VerificationError as exc:
            result = {
                "id": str(attempt["id"]),
                "status": "rejected",
                "code": str(exc),
                "reads": 0,
            }
        else:
            result = {
                "id": str(attempt["id"]),
                "status": "accepted",
                "program_id": answer_program_id(execution.answer),
                "reads": 0,
            }
        attempts.append(result)
    return _mismatches(payload, actual={"attempts": attempts})


def _compatibility_from_payload(payload: dict[str, Any]) -> ProgramCompatibility:
    return ProgramCompatibility(
        schema_revision=int(payload["schema_revision"]),
        compiler_version=str(payload["compiler_version"]),
        function_semantics=tuple(
            FunctionSemanticVersion(
                function_key=str(item["function_key"]),
                version=str(item["version"]),
            )
            for item in payload.get("function_semantics") or ()
        ),
        source_contracts=tuple(
            SourceContractPin(
                kind=SourceContractKind(str(item["kind"])),
                source_id=str(item["source_id"]),
                fingerprint=str(item["fingerprint"]),
            )
            for item in payload.get("source_contracts") or ()
        ),
    )


def _compatibility_payload(
    compatibility: ProgramCompatibility,
) -> dict[str, object]:
    return {
        "function_keys": [
            item.function_key for item in compatibility.function_semantics
        ],
        "source_keys": [
            {"kind": item.kind.value, "source_id": item.source_id}
            for item in compatibility.source_contracts
        ],
    }


def _mismatches(payload: dict[str, Any], *, actual: dict[str, Any]) -> list[str]:
    expect = payload["expect"]
    if "result_equals" in expect:
        return exact_mismatches(actual=actual, expected=expect["result_equals"])
    if "result_exact_fields" in expect:
        return exact_mismatches(
            actual=actual,
            expected=expect["result_exact_fields"],
        )
    return subset_mismatches(
        actual=actual,
        expected_subset=expect["result_contains"],
    )


def _execution_environment(
    *,
    catalog: object,
    payload: dict[str, Any],
) -> ExecutionEnvironment:
    return ExecutionEnvironment(
        catalog=catalog,
        authority_ref=str(payload.get("authority_ref") or ""),
    )


class _ScriptedDataAccess:
    def __init__(self, evidence: tuple[dict[str, Any], ...]) -> None:
        self._remaining = list(evidence)
        self.requests: list[dict[str, Any]] = []

    def read(self, *, endpoint_name: str, args: dict[str, Any]) -> dict[str, Any]:
        request = {
            "endpoint_name": endpoint_name,
            "args": portable_value(args),
        }
        self.requests.append(request)
        for index, item in enumerate(self._remaining):
            if item["request"] == request:
                matched = self._remaining.pop(index)["response"]
                return {
                    "endpointName": endpoint_name,
                    "responseStatus": int(matched.get("status") or 200),
                    "responseBody": matched["body"],
                }
        raise AssertionError(f"no scripted evidence for request {request!r}")
