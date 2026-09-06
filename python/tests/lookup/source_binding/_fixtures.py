"""Exercise both production binding boundaries from one composite test fixture."""

from jsonschema import validate
from copy import deepcopy
from fervis.lookup.source_binding.membership import parse_source_membership, requirement_choice_surfaces

from fervis.lookup.source_binding.parser import compile_source_binding_plan, compile_source_realization
from fervis.lookup.source_binding.schema import build_semantic_source_binding_schema, build_semantic_source_realization_schema


def _parts(payload):
    row_keys = ("set_bindings", "fact_bindings", "association_bindings")
    input_keys = ("resolved_input_applications", "finite_choice_applications", "subject_binding")
    if set(payload) != set(row_keys + input_keys):
        raise ValueError("composite fixture requires the exact two boundary payloads")
    return ({key: payload[key] for key in row_keys}, {key: payload[key] for key in input_keys})


def _membership_inputs(inputs, realization, *, validate_schema=False):
    inputs = deepcopy(inputs)
    subject = inputs.pop("subject_binding")
    baseline = {}
    returned = {}
    for branch in subject["branch_realizations"]:
        branch_id = branch["branch_id"]
        returned[branch_id] = {}
        if branch["finite_choice_reviews"]:
            baseline[branch_id] = {}
        wanted = {s.surface_ref for s in requirement_choice_surfaces(realization.request, branch_id)}
        for surface_ref, surface in branch["finite_choice_reviews"].items():
            baseline[branch_id][surface_ref] = {
                "surface_mapping_basis": surface["surface_mapping_basis"],
                "choice_reviews": {choice: {k: v for k,v in review.items() if k != "selected_by_requirements"} for choice,review in surface["choice_reviews"].items()},
            }
            if surface_ref in wanted:
                returned[branch_id][surface_ref] = {choice: {
                    "mapping_basis": review["decision_basis"],
                    "selected_by_requirements": review["selected_by_requirements"],
                } for choice,review in surface["choice_reviews"].items()}
    from fervis.lookup.source_binding.membership import membership_surfaces
    for branch in membership_surfaces(realization):
        baseline.setdefault(branch,{})
    from fervis.lookup.source_binding.membership import membership_scopes
    baseline = {branch: {scope.owner_set_ref: {surface.surface_ref: baseline[branch][surface.surface_ref] for surface in scope.surfaces}
                         for scope in scopes} for branch, scopes in membership_scopes(realization).items()}
    if validate_schema:
        from fervis.lookup.source_binding.membership import membership_schema
        validate(baseline, membership_schema(realization))
    membership = parse_source_membership(baseline, realization=realization)
    for branch_id, surfaces in returned.items():
        allowed = {s.surface_ref for s in requirement_choice_surfaces(membership.realization.request, branch_id)}
        returned[branch_id] = {ref: choices for ref,choices in surfaces.items() if ref in allowed}
    inputs["choice_requirement_applications"] = returned
    return membership, inputs


def compile_binding_fixture(payload, *, request):
    rows, inputs = _parts(payload)
    realization = compile_source_realization(rows, request=request)
    membership, inputs = _membership_inputs(inputs, realization)
    return compile_source_binding_plan(inputs, membership=membership)


def validate_binding_fixture(payload, *, request):
    rows, inputs = _parts(payload)
    validate(rows, build_semantic_source_realization_schema(request))
    realization = compile_source_realization(rows, request=request)
    membership, inputs = _membership_inputs(inputs, realization, validate_schema=True)
    validate(inputs, build_semantic_source_binding_schema(membership.realization.request))
