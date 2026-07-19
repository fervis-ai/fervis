from fervis.lookup.question_contract import (
    AnswerPopulationMembershipTestKind,
    AnswerPopulationMembershipTestPolarity,
    KnownInputSource,
    LiteralInputRole,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerOutput,
    RequestedFactAnswerPopulation,
    RequestedFactAnswerPopulationMembershipTest,
    RequestedFactKnownInput,
    RequestedFactLiteralInput,
)
from fervis.lookup.relation_catalog import (
    CandidateKey,
    CandidateKeyComponent,
    CatalogField,
    CatalogParam,
    EndpointRead,
    EntityKeyComponentTarget,
    EntityReference,
    EntityReferenceComponent,
    ParamSource,
    RelationCatalog,
    RowCardinality,
    RowPath,
)


def _reference_input(
    input_id: str,
    text: str,
    *,
    value_meaning_hint: str = "",
    resolved_value_text: str | None = None,
    field_label_text: str = "",
) -> RequestedFactKnownInput:
    return RequestedFactLiteralInput(
        id=input_id,
        source=KnownInputSource.QUESTION_CONTEXT,
        text=text,
        resolved_value_text=resolved_value_text or text,
        field_label_text=field_label_text,
        value_meaning_hint=value_meaning_hint,
        role=LiteralInputRole.REFERENCE_VALUE,
    )


def _time_input(input_id: str, text: str) -> RequestedFactKnownInput:
    return RequestedFactLiteralInput(
        id=input_id,
        source=KnownInputSource.QUESTION_CONTEXT,
        text=text,
        resolved_value_text=text,
        role=LiteralInputRole.TIME_VALUE,
    )


def _endpoint_result(response_body):
    return {
        "responseStatus": 200,
        "responseBody": response_body,
    }


def _question_contract(
    text: str,
    *,
    description: str = "",
) -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="sales total",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="total_sales",
                        role="ANSWER_VALUE",
                        description="total sales",
                    ),
                ),
                known_inputs=(
                    _reference_input(
                        "input_location",
                        text,
                        value_meaning_hint=description,
                    ),
                ),
            ),
        )
    )


def _city_question_contract(text: str) -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="count of stores in city",
                answer_population=RequestedFactAnswerPopulation(
                    population_label="stores in city",
                    counted_unit="store",
                    membership_tests=(
                        RequestedFactAnswerPopulationMembershipTest(
                            id="test_1",
                            kind=AnswerPopulationMembershipTestKind.SUBJECT_IDENTITY,
                            polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
                            test_question="Is the row a store?",
                        ),
                    ),
                ),
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="answer_1",
                        role="ANSWER_VALUE",
                        description="number of stores",
                    ),
                ),
                known_inputs=(
                    _reference_input(
                        "input_city",
                        text,
                        value_meaning_hint="city",
                    ),
                ),
            ),
        )
    )


def _staff_question_contract(
    text: str,
    *,
    description: str = "",
    resolved_value_text: str | None = None,
    field_label_text: str = "",
) -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="staff sales total",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="total_sales",
                        role="ANSWER_VALUE",
                        description="total sales",
                    ),
                ),
                known_inputs=(
                    _reference_input(
                        "input_staff",
                        text,
                        value_meaning_hint=description,
                        resolved_value_text=resolved_value_text,
                        field_label_text=field_label_text,
                    ),
                ),
            ),
        )
    )


def _shared_staff_question_contract(text: str) -> QuestionContract:
    staff = _reference_input("input_staff", text)
    fact_1 = RequestedFact(
        id="fact_1",
        description="staff sales total",
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="total_sales",
                role="ANSWER_VALUE",
                description="total sales",
            ),
        ),
        known_inputs=(staff,),
        input_refs=("input_staff",),
    )
    fact_2 = RequestedFact(
        id="fact_2",
        description="store associated with staff sales",
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="store",
                role="ANSWER_VALUE",
                description="store",
            ),
        ),
        known_inputs=(staff,),
        input_refs=("input_staff",),
    )
    return QuestionContract(
        question_inputs=(staff,),
        requested_facts=(fact_1, fact_2),
    )


def _staff_question_contract_with_resolved_value_text(
    *,
    reference_text: str,
    resolved_value_text: str,
) -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="staff ID",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="staff_id",
                        role="ANSWER_VALUE",
                        description="staff ID",
                    ),
                ),
                known_inputs=(
                    _reference_input(
                        "input_staff",
                        reference_text,
                        resolved_value_text=resolved_value_text,
                    ),
                ),
            ),
        )
    )


def _time_question_contract(text: str) -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="sales total",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="total_sales",
                        role="ANSWER_VALUE",
                        description="total sales",
                    ),
                ),
                known_inputs=(_time_input("input_date", text),),
            ),
        )
    )


def _quarter_question_contract(text: str) -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="sales total",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="total_sales",
                        role="ANSWER_VALUE",
                        description="total sales",
                    ),
                ),
                known_inputs=(_time_input("input_date", text),),
            ),
        )
    )


def _candidate_key(
    entity_kind: str,
    component_id: str,
    field_ref: str,
    *,
    context_field_refs: tuple[str, ...] = (),
) -> tuple[CandidateKey, ...]:
    return (
        CandidateKey(
            id="primary_key",
            entity_kind=entity_kind,
            components=(CandidateKeyComponent(id=component_id, field_ref=field_ref),),
            primary=True,
            context_field_refs=context_field_refs,
        ),
    )


def _entity_target(entity_kind: str, component_id: str) -> EntityKeyComponentTarget:
    return EntityKeyComponentTarget(
        entity_kind=entity_kind,
        key_id="primary_key",
        component_id=component_id,
    )


def _catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            _sales_read(),
            _location_read(),
        )
    )


def _staff_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            _staff_sales_read(),
            _staff_read(),
        )
    )


def _selected_catalog() -> RelationCatalog:
    return RelationCatalog(reads=(_sales_read(),))


def _flow_read() -> EndpointRead:
    return EndpointRead(
        id="list_flows",
        endpoint_name="list_flows",
        resource_names=("flow",),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="flow.id",
                path="data.id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="flow.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="flow.tags",
                path="data.tags",
                row_path_id="data",
                type="array",
            ),
        ),
        candidate_keys=_candidate_key(
            "flow",
            "id",
            "flow.id",
            context_field_refs=("flow.name",),
        ),
    )


def _location_read() -> EndpointRead:
    return EndpointRead(
        id="list_location_list",
        endpoint_name="list_location_list",
        resource_names=("location",),
        params=(
            CatalogParam(
                ref="list_location_list.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
            CatalogParam(
                ref="list_location_list.query.type",
                name="type",
                source=ParamSource.QUERY,
                type="choice",
                choices=("STORE", "WAREHOUSE"),
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.location_id",
                path="data.location_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "location",
            "location_id",
            "field.data.location_id",
            context_field_refs=("field.data.name",),
        ),
    )


def _variant_person_read() -> EndpointRead:
    return EndpointRead(
        id="list_people",
        endpoint_name="list_people",
        resource_names=("person",),
        params=(
            CatalogParam(
                ref="list_people.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
            CatalogParam(
                ref="list_people.query.shape",
                name="shape",
                source=ParamSource.QUERY,
                type="choice",
                required=True,
                choices=("SUMMARY", "DETAIL"),
                semantics="response_shape",
            ),
        ),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="person.id",
                path="data.person_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="person.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key("person", "person_id", "person.id"),
    )


def _uuid_person_read() -> EndpointRead:
    return EndpointRead(
        id="get_person",
        endpoint_name="get_person",
        resource_names=("person",),
        params=(
            CatalogParam(
                ref="get_person.query.person_id",
                name="person_id",
                source=ParamSource.QUERY,
                type="uuid",
            ),
        ),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.ONE),),
        fields=(
            CatalogField(
                ref="person.id",
                path="data.person_id",
                row_path_id="data",
                type="uuid",
            ),
            CatalogField(
                ref="person.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key("person", "person_id", "person.id"),
    )


def _location_read_without_lookup_param() -> EndpointRead:
    return EndpointRead(
        id="list_location_list",
        endpoint_name="list_location_list",
        resource_names=("location",),
        params=(
            CatalogParam(
                ref="list_location_list.query.type",
                name="type",
                source=ParamSource.QUERY,
                type="choice",
                choices=("STORE", "WAREHOUSE"),
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.location_id",
                path="data.location_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "location",
            "location_id",
            "field.data.location_id",
            context_field_refs=("field.data.name",),
        ),
    )


def _location_alias_read() -> EndpointRead:
    return EndpointRead(
        id="list_location_alias_list",
        endpoint_name="list_location_alias_list",
        resource_names=("location",),
        params=(
            CatalogParam(
                ref="list_location_alias_list.query.display_name",
                name="display_name",
                source=ParamSource.QUERY,
                type="string",
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.location_id",
                path="data.location_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.display_name",
                path="data.display_name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "location",
            "location_id",
            "field.data.location_id",
            context_field_refs=("field.data.display_name",),
        ),
    )


def _location_with_area_read() -> EndpointRead:
    return EndpointRead(
        id="list_location_list",
        endpoint_name="list_location_list",
        resource_names=("location",),
        params=(
            CatalogParam(
                ref="list_location_list.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
            CatalogParam(
                ref="list_location_list.query.type",
                name="type",
                source=ParamSource.QUERY,
                type="choice",
                choices=("STORE", "WAREHOUSE"),
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.location_id",
                path="data.location_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.type",
                path="data.type",
                row_path_id="data",
                type="choice",
                choices=("STORE", "WAREHOUSE"),
            ),
            CatalogField(
                ref="field.data.area.area_id",
                path="data.area.area_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.area.name",
                path="data.area.name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "location",
            "location_id",
            "field.data.location_id",
        ),
        entity_references=(
            EntityReference(
                id="area_reference",
                target_entity_kind="area",
                target_key_id="primary_key",
                components=(
                    EntityReferenceComponent(
                        target_component_id="area_id",
                        local_field_ref="field.data.area.area_id",
                    ),
                ),
                context_field_refs=("field.data.area.name",),
            ),
        ),
    )


def _area_read() -> EndpointRead:
    return EndpointRead(
        id="list_area_list",
        endpoint_name="list_area_list",
        resource_names=("area",),
        params=(
            CatalogParam(
                ref="list_area_list.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.area_id",
                path="data.area_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "area",
            "area_id",
            "field.data.area_id",
            context_field_refs=("field.data.name",),
        ),
    )


def _store_read() -> EndpointRead:
    return EndpointRead(
        id="list_store_list",
        endpoint_name="list_store_list",
        params=(
            CatalogParam(
                ref="list_store_list.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.store_id",
                path="data.store_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "store",
            "store_id",
            "field.data.store_id",
            context_field_refs=("field.data.name",),
        ),
    )


def _broken_store_read() -> EndpointRead:
    return EndpointRead(
        id="list_store_list",
        endpoint_name="list_store_list",
        params=(
            CatalogParam(
                ref="list_store_list.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
        ),
        row_paths=(
            RowPath(
                id="data_deposits",
                path="data.deposits",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data_deposits.store_id",
                path="data.deposits.store_id",
                row_path_id="data_deposits",
                type="string",
            ),
            CatalogField(
                ref="field.data_deposits.name",
                path="data.deposits.name",
                row_path_id="data_deposits",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "store",
            "store_id",
            "field.data_deposits.store_id",
            context_field_refs=("field.data_deposits.name",),
        ),
    )


def _staff_read() -> EndpointRead:
    return EndpointRead(
        id="list_staff_list",
        endpoint_name="list_staff_list",
        params=(
            CatalogParam(
                ref="list_staff_list.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.full_name",
                path="data.full_name",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.first_name",
                path="data.first_name",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.last_name",
                path="data.last_name",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.phone_number",
                path="data.phone_number",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "staff",
            "staff_id",
            "field.data.staff_id",
            context_field_refs=(
                "field.data.full_name",
                "field.data.first_name",
                "field.data.last_name",
            ),
        ),
    )


def _staff_uuid_only_read() -> EndpointRead:
    return EndpointRead(
        id="list_staff_uuid_list",
        endpoint_name="list_staff_uuid_list",
        resource_names=("staff",),
        params=(
            CatalogParam(
                ref="list_staff_uuid_list.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
        ),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="field.data.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="uuid",
            ),
        ),
        candidate_keys=_candidate_key(
            "staff",
            "staff_id",
            "field.data.staff_id",
        ),
    )


def _staff_detail_read(*, param_type: str = "string") -> EndpointRead:
    return EndpointRead(
        id="get_staff_detail",
        endpoint_name="get_staff_detail",
        resource_names=("staff",),
        params=(
            CatalogParam(
                ref="get_staff_detail.path.staff_id",
                name="staff_id",
                source=ParamSource.PATH,
                type=param_type,
                required=True,
                entity_target=_entity_target("staff", "staff_id"),
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.ONE,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.full_name",
                path="data.full_name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "staff",
            "staff_id",
            "field.data.staff_id",
            context_field_refs=("field.data.full_name",),
        ),
    )


def _location_detail_read() -> EndpointRead:
    return EndpointRead(
        id="get_location_detail",
        endpoint_name="get_location_detail",
        resource_names=("location",),
        params=(
            CatalogParam(
                ref="get_location_detail.path.location_id",
                name="location_id",
                source=ParamSource.PATH,
                type="string",
                required=True,
                entity_target=_entity_target("location", "location_id"),
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.ONE,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.location_id",
                path="data.location_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "location",
            "location_id",
            "field.data.location_id",
            context_field_refs=("field.data.name",),
        ),
    )


def _staff_sales_read() -> EndpointRead:
    return EndpointRead(
        id="list_sale_list",
        endpoint_name="list_sale_list",
        params=(
            CatalogParam(
                ref="list_sale_list.query.staff_id",
                name="staff_id",
                source=ParamSource.QUERY,
                type="string",
                entity_target=_entity_target("staff", "staff_id"),
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.amount",
                path="data.amount",
                row_path_id="data",
                type="decimal",
            ),
        ),
    )


def _date_sales_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="list_sales_list",
                endpoint_name="list_sales_list",
                params=(
                    CatalogParam(
                        ref="list_sales_list.query.start_date",
                        name="start_date",
                        source=ParamSource.QUERY,
                        type="date",
                    ),
                    CatalogParam(
                        ref="list_sales_list.query.end_date",
                        name="end_date",
                        source=ParamSource.QUERY,
                        type="date",
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="field.total_sales",
                        type="decimal",
                    ),
                ),
            ),
        )
    )


def _sales_read() -> EndpointRead:
    return EndpointRead(
        id="list_sales_list",
        endpoint_name="list_sales_list",
        params=(
            CatalogParam(
                ref="list_sales_list.query.location_id",
                name="location_id",
                source=ParamSource.QUERY,
                type="string",
                entity_target=_entity_target("location", "location_id"),
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.total_sales",
                path="data.total_sales",
                row_path_id="data",
                type="decimal",
            ),
        ),
    )
