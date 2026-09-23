import pytest
from fervis.host_api.adapters.django.catalog import _with_framework_param_semantics
from fervis.host_api.contracts import ParameterContract


def test_view_can_declare_response_shape_controls_without_name_heuristics():
    class View:
        fervis_parameter_semantics = {"presentation": "response_shape"}

    params = (
        ParameterContract(
            "presentation", "choice", choices=("daily", "monthly"), default="daily"
        ),
        ParameterContract("status", "choice", choices=("active", "inactive")),
    )
    result = _with_framework_param_semantics(
        params, response_fields=(), view_class=View
    )
    assert [param.semantics for param in result] == ["response_shape", ""]
    assert result[0].default == "daily"


@pytest.mark.parametrize(
    "declaration",
    [{"missing": "response_shape"}, {"presentation": "unknown"}, ["presentation"]],
)
def test_invalid_parameter_semantics_declarations_fail_explicitly(declaration):
    class View:
        fervis_parameter_semantics = declaration

    with pytest.raises(ValueError, match="parameter semantics"):
        _with_framework_param_semantics(
            (ParameterContract("presentation", "string"),),
            response_fields=(),
            view_class=View,
        )
