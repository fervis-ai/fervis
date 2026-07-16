from ._helpers import *  # noqa: F403

from fervis.lookup.fact_planning.row_set_filters import (
    filter_row_set_filters_for_requested_fact,
)
from fervis.lookup.fact_planning.grounded_params import (
    unique_grounded_param_values,
)


def test_row_set_filters_use_their_exact_fact_scoped_value() -> None:
    area_value = FactValue.identity(
        id="nairobi_area",
        known_input_id="nairobi",
        key=entity_key_value("area", "primary_key", {"area_id": "area_1"}),
        applies_to_requested_fact_ids=("area_fact",),
    )
    location_value = FactValue.identity(
        id="nairobi_location",
        known_input_id="nairobi",
        key=entity_key_value(
            "location",
            "primary_key",
            {"location_id": "location_1"},
        ),
        applies_to_requested_fact_ids=("location_fact",),
    )
    filters = (
        {"value_id": area_value.id, "known_input_id": "nairobi"},
        {"value_id": location_value.id, "known_input_id": "nairobi"},
    )

    assert filter_row_set_filters_for_requested_fact(
        filters,
        requested_fact_id="area_fact",
        available_values=(area_value, location_value),
    ) == [{"value_id": "nairobi_area", "known_input_id": "nairobi"}]


def test_grounded_parameter_values_use_the_fact_scoped_application() -> None:
    row_source_id = "read:rows"
    area_value = FactValue.identity(
        id="nairobi_area",
        known_input_id="nairobi",
        key=entity_key_value("area", "primary_key", {"area_id": "area_1"}),
    )
    location_value = FactValue.identity(
        id="nairobi_location",
        known_input_id="nairobi",
        key=entity_key_value(
            "location",
            "primary_key",
            {"location_id": "location_1"},
        ),
    )
    uses = (
        GroundedInputUse(
            id="area_use",
            value_id=area_value.id,
            row_source_id=row_source_id,
            param_id="place_id",
            requested_fact_id="area_fact",
        ),
        GroundedInputUse(
            id="location_use",
            value_id=location_value.id,
            row_source_id=row_source_id,
            param_id="place_id",
            requested_fact_id="location_fact",
        ),
    )

    grounded = unique_grounded_param_values(
        values=(area_value, location_value),
        grounded_input_uses=uses,
        requested_fact_id="area_fact",
    )

    assert grounded[(row_source_id, "place_id")].value_id == "nairobi_area"


def test_fact_plan_prompt_uses_explicit_source_binding_for_required_dates():
    request = FactPlanRequest(
        question="How much revenue on January 1 and January 2?",
        question_contract=_question_contract(),
        relation_catalog=RelationCatalog(
            reads=(
                EndpointRead(
                    id="sales",
                    endpoint_name="list_sales",
                    params=(
                        CatalogParam(
                            ref="sales.query.start_date",
                            name="start_date",
                            source=ParamSource.QUERY,
                            type="date",
                            required=True,
                        ),
                    ),
                    fields=(CatalogField(ref="field.total", type="decimal"),),
                ),
            )
        ),
        available_values=(
            FactValue.time(
                id="jan_1",
                expression="January 1",
                resolved_start="2030-01-01",
                resolved_end="2030-01-01",
                granularity="day",
            ),
            FactValue.time(
                id="jan_2",
                expression="January 2",
                resolved_start="2030-01-02",
                resolved_end="2030-01-02",
                granularity="day",
            ),
        ),
        bound_sources=(
            BoundSource(
                id="sb_sales_jan_1",
                requested_fact_id="rf_answer",
                answer_population=_answer_population(),
                source=DraftRelationSource(
                    kind=SourceKind.API_READ,
                    read_id="sales",
                    param_bindings=(
                        DraftEndpointParamBinding(
                            param_id="start_date",
                            value="2030-01-01",
                        ),
                    ),
                ),
                available_field_ids=("total",),
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="rf_answer",
                        answer_output_id="answer",
                        value_evidence_ids=("total",),
                        match_basis_explanation=(
                            "answer is fulfilled by total because total provides "
                            "the requested revenue value."
                        ),
                    ),
                ),
            ),
        ),
    )

    relation_catalog = _json_prompt_section(
        _fact_plan_prompt(request),
        label="Bound sources",
        next_label="Catalog selection",
    )

    sales_source = next(
        item
        for item in _bound_sources(relation_catalog)
        if item.get("read_id") == "sales"
    )
    assert sales_source["bound_params"] == [
        {
            "param_id": "start_date",
            "value": "2030-01-01",
        }
    ]
    assert "missing_required_inputs" not in relation_catalog


def test_fact_plan_prompt_hides_missing_inputs_when_fact_has_executable_relation():
    request = _request_with_executable_relation_and_required_detail()

    relation_catalog = _json_prompt_section(
        _fact_plan_prompt(request),
        label="Bound sources",
        next_label="Catalog selection",
    )

    available_ids = {item.get("read_id") for item in _bound_sources(relation_catalog)}
    assert "sales" in available_ids
    assert "sale_detail" not in available_ids
    assert "missing_required_inputs" not in relation_catalog


def test_fact_plan_schema_hides_clarification_when_fact_has_executable_relation():
    class ProviderAssertsNoClarificationBranch:
        def generate(self, **kwargs):
            schema_text = json.dumps(kwargs["tool_specs"][0].input_schema)
            assert "needs_clarification" not in schema_text
            return {
                "answer": json.dumps(
                    {
                        "tool": "submit_pattern_fact_plan",
                        "arguments": {
                            "outcome": {
                                "kind": "impossible",
                                "blocked_facts": [
                                    {
                                        "requested_fact_id": "rf_answer",
                                        "basis": "catalog_access",
                                        "evidence_refs": [
                                            f"row_source:{api_row_source_id('sales', 'root')}"
                                        ],
                                    }
                                ],
                            }
                        },
                    }
                ),
                "usage": {},
            }

    request = _request_with_executable_relation_and_required_detail()
    generate_pattern_fact_plan(
        request=request,
        plan_selection=_plan_selection_for_request(request),
        model_port=ProviderAssertsNoClarificationBranch(),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )
