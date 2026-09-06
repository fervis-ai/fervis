# Django / DRF

Install Fervis:

```bash
uv add "fervis[django]"
```

Run from the Django project root:

```bash
fervis init --framework django --yes
fervis migrate
fervis doctor
```

`init` adds Fervis to Django settings/URLs when it can do so safely. If it blocks, follow its `next_actions`.
If `doctor` blocks, follow every reported `next_actions` item, then rerun `fervis doctor` until it passes.

If the host API uses header auth for reads:

```bash
fervis auth configure \
  --framework django-drf \
  --transport-mode http \
  --base-url-env FERVIS_HOST_API_BASE_URL \
  --capture-credential-header Authorization
```

Then:

```bash
fervis doctor
fervis runtime ask "How many orders happened this month?"
```

Only run `runtime ask` after `fervis doctor` passes.


Declare query controls that change the response representation rather than the
underlying business-row population on the owning view:

```python
class SalesSummaryView(APIView):
    fervis_parameter_semantics = {
        "group_by": "response_shape",
        "granularity": "response_shape",
    }
```

Fervis preserves declared defaults and source variants for these controls instead
of treating every choice as a separate population to read. Ordinary filters keep
`opaque_query_param` semantics. Declaration keys must name actual query
parameters; unsupported semantics are rejected. Framework-provided ordering
controls are recognized automatically.

For a custom `APIView` that returns a plain JSON list, declare its response
cardinality explicitly. Fervis can infer this from DRF list mixins and viewset
actions, but an arbitrary `get()` method does not declare whether its serializer
is used once or with `many=True`.

```python
class ScheduledAssignmentsView(APIView):
    serializer_class = AssignmentSerializer
    fervis_response_cardinality = "many"
```

The declaration describes the actual GET response. Use a response serializer
that describes the complete envelope when the API wraps its rows in an object.

A custom view that implements pagination itself can declare the same public
pagination contract used by the adapters:

```python
fervis_pagination = {
    "kind": "offset",
    "positionQueryParam": "offset",
    "pageSizeQueryParam": "limit",
    "resultsPath": "data",
    "pageSize": 50,
    "maxPageSize": 200,
    "totalPath": "pagination.count",
    "continuationPath": "pagination.has_more",
}
```

These values must describe the endpoint's actual parameters and response paths.
Fervis validates the declaration and uses its completeness evidence when
traversing pages. Endpoint and parameter descriptions are also passed to
identity grounding and read selection; describe distinct business resources
and categories precisely.

A plain `Serializer` can declare `Meta.model` when its projected fields come from
that model. Use nested serializers for nested objects and lists so their field
and identity contracts remain inspectable.

For computed projections without an ORM relationship, a view can declare
`fervis_relation_metadata` using the same `candidateKeys` and `entityReferences`
format as OpenAPI's `x-fervis` extension:

```python
class ObservationListView(ListAPIView):
    serializer_class = ObservationSerializer
    fervis_relation_metadata = {
        "entityReferences": [{
            "referenceId": "category_reference",
            "targetEntityKind": "category",
            "targetKeyId": "unique_code",
            "components": [{
                "targetComponentId": "code",
                "localFieldPath": "category.code",
            }],
            "contextFieldPaths": ["category.name"],
        }],
    }
```

These declarations supplement inferred ORM metadata. They must describe actual
stable keys and references; they do not create database constraints or fields.
Paths address the serializer's response fields before pagination wrapping.
Unknown field paths and malformed declarations fail catalog construction.
