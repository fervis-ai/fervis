"""One physical parameter projection stays one choice across response row paths."""

from types import SimpleNamespace

from jsonschema import Draft7Validator

from fervis.lookup.source_binding.schema import _branch_resolved_input_applications_schema


def test_shared_endpoint_projection_is_a_satisfiable_choice():
    # The endpoint's data and metadata row sources share its request parameter.
    options = tuple(SimpleNamespace(
        source_ref=source, value_ref="period", target_ref="report.query.start_date",
        component_ref=None, projection=SimpleNamespace(value="TEMPORAL_START"),
    ) for source in ("report_rows", "report_metadata"))
    request = SimpleNamespace(
        invocation_application_owner_refs=("time_requirement",),
        direct_value_options_for_owner=lambda owner, branch_id: options,
        unapplied_input_value_refs_for_owner=lambda owner, branch_id: ("period",),
    )
    schema = _branch_resolved_input_applications_schema(request, branch_id="scope")
    value = [{"kind": "request_application", "mapping_basis": "The declared period bounds this report.",
              "owner_ref": "time_requirement", "value_ref": "period",
              "value_component": "TEMPORAL_START", "target_ref": "report.query.start_date"}]
    assert list(Draft7Validator(schema).iter_errors(value)) == []
    value[0]["target_ref"] = "another_report.query.start_date"
    assert list(Draft7Validator(schema).iter_errors(value))
