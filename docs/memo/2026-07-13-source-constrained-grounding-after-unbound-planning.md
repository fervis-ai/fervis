# Source-Constrained Grounding After Unbound Planning

## Context

A conversational follow-up can name a business entity that exists in more than one host relation. In the observed sales continuation, the same supplied name could resolve to either a `Store` key or a `Location` key, while the viable sales read accepts only `Location.primary_key`. Grounding currently runs before source planning, so it must choose an identity without knowing the type required by the fact-bearing source. The following response identifies that ordering defect and the deletion-shaped architectural correction.

Yes. That is the larger architectural issue.

I focused on giving grounding more semantic context, but grounding is currently being asked to choose an identity before the compiler knows which identity the answer source can consume.

Current order:

```text
Question Contract
→ grounding across the general catalog
→ read eligibility
→ plan selection
→ source binding
```

That makes grounding choose between:

```text
Store(store_id)
Location(location_id)
```

without knowing that the viable sales source requires:

```text
list_sale_list
  filter/input: location_id
  entity target: Location.primary_key
```

The correct order is:

```text
Question Contract
→ catalog selection / read eligibility
→ unbound plan and source selection
→ derive typed input requirements
→ grounding
→ source binding
→ execution
```

For this case:

```text
1. Plan selection identifies list_sale_list as the fact-bearing source.

2. Its contract says the relevant input slot requires:
   entity_kind: location
   key_id: primary_key
   component: location_id

3. Grounding is then offered only resolvers that return that key.

4. list_location_list is eligible.
   list_store_list is mechanically excluded because it returns Store.primary_key.

5. Grounding resolves Acacia Mall to Location(location_id).

6. Source binding puts that value into list_sale_list.location_id.
```

This is not the backend acting as a semantic oracle. It is ordinary type checking:

```text
required: Location.primary_key
provided by resolver: Store.primary_key
→ incompatible
```

The model still does semantic work when several resolvers can provide the required type. The backend merely prevents it from selecting a value that cannot inhabit the chosen source’s input slot.

It also explains why simply exposing the broader membership predicate to grounding is insufficient. That might improve model behavior, but the contract would still permit Store when only Location is executable. The ambiguity can be removed structurally.

The existing plan-selection prompt already says source alignment should be judged from raw ingredients and that filters and parameters are chosen later. That means plan selection conceptually does not need grounded values. Our pipeline order contradicts that intended ownership.

The clean implementation should not introduce a second candidate system:

- Build one unbound typed candidate registry.
- Run eligibility and plan selection over it.
- Derive grounding requirements from the selected source roles and their declared entity targets.
- Ground those requirements.
- Enrich the same selected candidates with the grounded bindings.
- Continue into source binding.

No additional model call is necessary; the existing grounding call moves after plan selection. The current grounding-first candidate filtering and rediscovery should be deleted.

This would solve the Store/Location case generally—not by knowing anything about Ozana, but because candidate keys, entity references, and parameter targets become the authority for what grounding is allowed to return.
