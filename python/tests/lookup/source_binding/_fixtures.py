"""Exercise both production binding boundaries from one composite test fixture."""

from jsonschema import validate
from copy import deepcopy
from fervis.lookup.source_binding.choice_requirements import requirement_choice_surfaces

from fervis.lookup.source_binding.parser import (
    compile_source_binding_plan,
    compile_source_realization,
)
from fervis.lookup.source_binding.schema import (
    build_semantic_source_binding_schema,
    build_semantic_source_realization_schema,
)


def _parts(payload):
    row_keys = ("set_bindings", "fact_bindings", "association_bindings")
    input_keys = (
        "resolved_input_applications",
        "finite_choice_applications",
        "subject_binding",
    )
    if set(payload) != set(row_keys + input_keys):
        raise ValueError("composite fixture requires the exact two boundary payloads")
    return (
        {key: payload[key] for key in row_keys},
        {key: payload[key] for key in input_keys},
    )


def _membership_inputs(inputs, realization, *, validate_schema=False):
    inputs = deepcopy(inputs)
    subject = inputs.pop("subject_binding")
    returned = {}
    for branch in subject["branch_realizations"]:
        branch_id = branch["branch_id"]
        returned[branch_id] = {}
        wanted = {
            s.surface_ref
            for s in requirement_choice_surfaces(realization.request, branch_id)
        }
        for surface_ref, surface in branch["finite_choice_reviews"].items():
            if surface_ref in wanted:
                returned[branch_id][surface_ref] = {
                    choice: {
                        "mapping_basis": review["decision_basis"],
                        "selected_by_requirements": review["selected_by_requirements"],
                    }
                    for choice, review in surface["choice_reviews"].items()
                }
    inputs["choice_requirement_applications"] = returned
    return realization, inputs


def compile_binding_fixture(payload, *, request):
    rows, inputs = _parts(payload)
    realization = compile_source_realization(rows, request=request)
    realization, inputs = _membership_inputs(inputs, realization)
    return compile_source_binding_plan(inputs, realization=realization)


def validate_binding_fixture(payload, *, request):
    rows, inputs = _parts(payload)
    validate(rows, build_semantic_source_realization_schema(request))
    realization = compile_source_realization(rows, request=request)
    realization, inputs = _membership_inputs(inputs, realization, validate_schema=True)
    validate(inputs, build_semantic_source_binding_schema(realization.request))
