from types import SimpleNamespace
import pytest

from fervis.lookup.orchestration import semantic_compilation as module
from fervis.lookup.relation_catalog import EndpointRead, RelationCatalog
from fervis.lookup.relation_catalog.row_sources.model import (
    RowSource,
    RowSourceKind,
    RowSourceParam,
    RowSourceValueType,
)
from fervis.lookup.answer_program.values import FactValue, LiteralType


@pytest.mark.parametrize("supplied", [False, True])
def test_required_inputs_are_checked_before_spending_on_read_eligibility(
    monkeypatch, supplied
):
    catalog = RelationCatalog(
        reads=(EndpointRead("open", "open"), EndpointRead("detail", "detail"))
    )
    sources = (
        RowSource("open_rows", RowSourceKind.API_READ, "open", read_id="open"),
        RowSource(
            "detail_rows",
            RowSourceKind.API_READ,
            "detail",
            read_id="detail",
            params=(
                RowSourceParam(
                    "number",
                    "detail.number",
                    "number",
                    RowSourceValueType.INTEGER,
                    "query",
                    required=True,
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        module,
        "build_api_row_source_catalog",
        lambda catalog: SimpleNamespace(sources=sources),
    )
    values = (
        (FactValue.literal(id="value", literal_type=LiteralType.NUMBER, value="3"),)
        if supplied
        else ()
    )
    selected = module._executable_relation_catalog(catalog, values=values)
    assert [read.id for read in selected.reads] == (
        ["open", "detail"] if supplied else ["open"]
    )


def test_integer_parameter_projection_preserves_large_values_and_rejects_fractions():
    from fervis.lookup.source_binding.param_values import (
        fact_value_parameter_projection,
    )
    from fervis.lookup.answer_program.values import ValueProjectionKind

    def projected(text):
        return fact_value_parameter_projection(
            FactValue.literal(id="n", literal_type=LiteralType.NUMBER, value=text),
            projection=ValueProjectionKind.WHOLE_VALUE,
            component_id=None,
            type_name="integer",
            choices=(),
        )

    assert projected("9007199254740993") == 9007199254740993
    with pytest.raises(ValueError, match="exact integral"):
        projected("3.5")
