from __future__ import annotations

import sys
import types
from typing import ClassVar
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, create_model
from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    JSON,
    MetaData,
    String,
    Table,
    Index,
    UniqueConstraint,
)

from fervis.host_api.adapters.fastapi.catalog import (
    get_fastapi_endpoint_contracts,
)
from fervis.lookup.relation_catalog.from_host_api import (
    relation_catalog_from_endpoint_contracts,
)
from fervis.project.integration import FastAPIAppSource


def test_fastapi_catalog_uses_explicit_source_imports_and_path_prefixes(
    tmp_path: Path,
) -> None:
    module = types.ModuleType("commerce_api")
    setattr(module, "app", _commerce_app())
    sys.modules[module.__name__] = module
    try:
        contracts = get_fastapi_endpoint_contracts(
            sources=(
                FastAPIAppSource(
                    name="commerce",
                    import_paths=["commerce_api:app"],
                    path_prefixes=["/api/v1/"],
                ),
            ),
            project_root=tmp_path,
        )
    finally:
        sys.modules.pop(module.__name__, None)

    assert [contract.path_template for contract in contracts] == ["/api/v1/orders/"]
    contract = contracts[0]
    assert contract.endpoint_name == "list_orders"
    assert [param.name for param in contract.query_params] == ["status"]
    assert {field.path for field in contract.response_fields} == {
        "id",
        "status",
        "amount",
    }
    assert contract.candidate_keys == ()
    assert contract.entity_references == ()
    assert contract.catalog_endpoint is not None
    assert contract.catalog_endpoint.framework_kind == "fastapi"
    assert contract.catalog_endpoint.source_namespace_kind == "fastapi_app"
    assert contract.catalog_endpoint.source_namespace_path == ("commerce",)


def test_fastapi_catalog_uses_live_route_and_mapped_response_model_metadata(
    tmp_path: Path,
) -> None:
    metadata = MetaData()
    users = Table(
        "user",
        metadata,
        Column("id", Integer, primary_key=True, unique=True),
    )
    items = Table(
        "item",
        metadata,
        Column("id", Integer, primary_key=True, unique=True),
        Column("owner_id", ForeignKey(users.c.id), nullable=False),
        Column("title", String, nullable=False),
        Column("labels", JSON, nullable=False),
    )

    class ItemResponse(BaseModel):
        __table__: ClassVar[Table] = items

        id: int
        owner_id: int
        title: str
        labels: list[str]

    app = FastAPI()

    @app.get("/api/items/", response_model=list[ItemResponse], tags=["items"])
    def list_items() -> list[ItemResponse]:
        return []

    def fail_if_openapi_is_used() -> dict[str, object]:
        raise AssertionError("FastAPI catalog must inspect APIRoute directly")

    setattr(app, "openapi", fail_if_openapi_is_used)
    module = types.ModuleType("mapped_fastapi")
    setattr(module, "app", app)
    sys.modules[module.__name__] = module
    try:
        contracts = get_fastapi_endpoint_contracts(
            sources=(
                FastAPIAppSource(
                    name="inventory",
                    import_paths=["mapped_fastapi:app"],
                    path_prefixes=["/api/"],
                ),
            ),
            project_root=tmp_path,
        )
    finally:
        sys.modules.pop(module.__name__, None)

    contract = contracts[0]
    assert [(field.path, field.type) for field in contract.response_fields] == [
        ("id", "integer"),
        ("owner_id", "integer"),
        ("title", "string"),
        ("labels", "array"),
    ]
    assert contract.candidate_keys[0].entity_kind == "item"
    assert contract.candidate_keys[0].components[0].field_path == "id"
    assert tuple(key.key_id for key in contract.candidate_keys) == ("primary_key",)
    assert contract.entity_references[0].target_entity_kind == "user"
    assert contract.entity_references[0].components[0].local_field_path == "owner_id"
    authority = contract.candidate_key_authorities[0]
    assert authority.entity_kind == "user"
    assert authority.components[0].type == "integer"
    read = relation_catalog_from_endpoint_contracts(contracts).reads[0]
    assert tuple(path.path for path in read.row_paths) == ("",)
    assert next(field for field in read.fields if field.path == "labels").row_path_id == (
        "root"
    )


def test_fastapi_catalog_does_not_guess_separate_response_and_orm_model_mapping(
    tmp_path: Path,
) -> None:
    module = types.ModuleType("separate_models_fastapi")
    metadata = MetaData()
    folders = Table(
        "folder",
        metadata,
        Column("id", String, primary_key=True),
        Column("name", String, nullable=False),
    )
    folder_record = type(
        "Folder",
        (),
        {"__module__": module.__name__, "__table__": folders},
    )
    folder_response = create_model(
        "FolderModel",
        __module__=module.__name__,
        __config__=ConfigDict(from_attributes=True),
        id=(str, ...),
        name=(str, ...),
    )
    app = FastAPI()

    @app.get("/folders/{folder_id}", response_model=folder_response)
    def get_folder(folder_id: str):
        raise NotImplementedError

    setattr(module, "Folder", folder_record)
    setattr(module, "FolderModel", folder_response)
    setattr(module, "app", app)
    sys.modules[module.__name__] = module
    try:
        contracts = get_fastapi_endpoint_contracts(
            sources=(
                FastAPIAppSource(
                    name="folders",
                    import_paths=["separate_models_fastapi:app"],
                    path_prefixes=["/folders/"],
                ),
            ),
            project_root=tmp_path,
        )
    finally:
        sys.modules.pop(module.__name__, None)

    [contract] = contracts
    assert contract.candidate_keys == ()
    assert contract.path_params[0].entity_target is None


def test_fastapi_catalog_maps_response_and_table_models_with_declared_shared_base(
    tmp_path: Path,
) -> None:
    metadata = MetaData()
    flows = Table(
        "fastapi_shared_base_flow",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String, nullable=False),
        Column("tags", JSON, nullable=False),
    )

    class FlowFields(BaseModel):
        name: str
        tags: list[str]

    class Flow(FlowFields):
        __table__: ClassVar[Table] = flows

        id: int

    class FlowRead(FlowFields):
        id: int

    app = FastAPI()

    @app.get("/flows/", response_model=list[FlowRead])
    def list_flows() -> list[FlowRead]:
        return []

    module = types.ModuleType("shared_base_fastapi")
    setattr(module, "app", app)
    sys.modules[module.__name__] = module
    try:
        contracts = get_fastapi_endpoint_contracts(
            sources=(
                FastAPIAppSource(
                    name="flows",
                    import_paths=["shared_base_fastapi:app"],
                    path_prefixes=["/flows/"],
                ),
            ),
            project_root=tmp_path,
        )
    finally:
        sys.modules.pop(module.__name__, None)

    [contract] = contracts
    assert tuple(key.key_id for key in contract.candidate_keys) == ("primary_key",)
    assert contract.candidate_keys[0].entity_kind == "fastapi_shared_base_flow"
    assert contract.candidate_keys[0].components[0].field_path == "id"


def test_fastapi_detail_path_can_validate_returned_single_component_key(
    tmp_path: Path,
) -> None:
    metadata = MetaData()
    items = Table(
        "item",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String, nullable=False),
    )

    class ItemResponse(BaseModel):
        __table__: ClassVar[Table] = items

        id: int
        name: str

    app = FastAPI()

    @app.get("/api/items/{item_ref}", response_model=ItemResponse)
    def get_item(item_ref: int) -> ItemResponse:
        raise NotImplementedError

    module = types.ModuleType("detail_fastapi")
    setattr(module, "app", app)
    sys.modules[module.__name__] = module
    try:
        contracts = get_fastapi_endpoint_contracts(
            sources=(
                FastAPIAppSource(
                    name="items",
                    import_paths=["detail_fastapi:app"],
                    path_prefixes=["/api/"],
                ),
            ),
            project_root=tmp_path,
        )
    finally:
        sys.modules.pop(module.__name__, None)

    target = contracts[0].path_params[0].entity_target
    assert target is not None
    assert (
        target.entity_kind,
        target.key_id,
        target.component_id,
    ) == ("item", "primary_key", "id")


def test_fastapi_catalog_does_not_map_unrelated_same_shape_model(
    tmp_path: Path,
) -> None:
    module = types.ModuleType("unrelated_fastapi")
    metadata = MetaData()
    accounts = Table(
        "account",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String),
    )

    class AccountRecord(BaseModel):
        __table__: ClassVar[Table] = accounts

        id: int
        name: str

    class PublicSummary(BaseModel):
        model_config = ConfigDict(from_attributes=True)

        id: int
        name: str

    PublicSummary.__module__ = module.__name__
    setattr(module, "AccountRecord", AccountRecord)
    setattr(module, "PublicSummary", PublicSummary)
    app = FastAPI()
    setattr(module, "app", app)

    @app.get("/api/summaries/", response_model=list[PublicSummary])
    def list_summaries() -> list[PublicSummary]:
        return []

    sys.modules[module.__name__] = module
    try:
        contracts = get_fastapi_endpoint_contracts(
            sources=(
                FastAPIAppSource(
                    name="summaries",
                    import_paths=["unrelated_fastapi:app"],
                    path_prefixes=["/api/"],
                ),
            ),
            project_root=tmp_path,
        )
    finally:
        sys.modules.pop(module.__name__, None)

    assert contracts[0].candidate_keys == ()


def test_fastapi_catalog_unifies_collection_union_with_shared_relation(
    tmp_path: Path,
) -> None:
    metadata = MetaData()
    flows = Table(
        "flow",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String),
        Column("updated_at", String),
    )

    class FlowFields(BaseModel):
        id: int
        name: str

    class FlowRead(FlowFields):
        __table__: ClassVar[Table] = flows

        updated_at: str

    class FlowHeader(FlowFields):
        __table__: ClassVar[Table] = flows

    app = FastAPI()

    @app.get("/flows/", response_model=list[FlowRead] | list[FlowHeader])
    def list_flows() -> list[FlowRead]:
        return []

    module = types.ModuleType("union_fastapi")
    setattr(module, "app", app)
    sys.modules[module.__name__] = module
    try:
        contracts = get_fastapi_endpoint_contracts(
            sources=(
                FastAPIAppSource(
                    name="flows",
                    import_paths=["union_fastapi:app"],
                    path_prefixes=["/flows/"],
                ),
            ),
            project_root=tmp_path,
        )
    finally:
        sys.modules.pop(module.__name__, None)

    contract = contracts[0]
    assert [(field.path, field.type) for field in contract.response_fields] == [
        ("id", "integer"),
        ("name", "string"),
    ]
    assert tuple(key.key_id for key in contract.candidate_keys) == ("primary_key",)


def test_fastapi_catalog_uses_only_total_mapped_uniqueness_as_candidate_keys(
    tmp_path: Path,
) -> None:
    metadata = MetaData()
    campaigns = Table(
        "campaign",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("external_code", String, nullable=False, unique=True),
        Column("public_code", String, nullable=False),
        Column("optional_code", String, nullable=True, unique=True),
        Column("account_id", Integer, nullable=False),
        Column("code", String, nullable=False),
        UniqueConstraint("account_id", "code", name="campaign_account_code"),
        Index("campaign_public_code", "public_code", unique=True),
    )

    class CampaignResponse(BaseModel):
        __table__: ClassVar[Table] = campaigns

        id: int
        external_code: str
        public_code: str
        optional_code: str | None
        account_id: int
        code: str

    app = FastAPI()

    @app.get("/campaigns/", response_model=list[CampaignResponse])
    def list_campaigns() -> list[CampaignResponse]:
        return []

    module = types.ModuleType("campaign_fastapi")
    setattr(module, "app", app)
    sys.modules[module.__name__] = module
    try:
        contracts = get_fastapi_endpoint_contracts(
            sources=(
                FastAPIAppSource(
                    name="campaigns",
                    import_paths=["campaign_fastapi:app"],
                    path_prefixes=["/campaigns/"],
                ),
            ),
            project_root=tmp_path,
        )
    finally:
        sys.modules.pop(module.__name__, None)

    assert tuple(key.key_id for key in contracts[0].candidate_keys) == (
        "primary_key",
        "unique_external_code",
        "campaign_account_code",
        "campaign_public_code",
    )


def test_fastapi_catalog_imports_from_declared_project_source_roots(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi']\n",
        encoding="utf-8",
    )
    package_dir = tmp_path / "src" / "service"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "main.py").write_text(
        """
from fastapi import FastAPI
from pydantic import BaseModel

class Order(BaseModel):
    id: str

app = FastAPI()

@app.get('/api/v1/orders/', response_model=list[Order], operation_id='list_orders')
def list_orders():
    return []
""",
        encoding="utf-8",
    )

    contracts = get_fastapi_endpoint_contracts(
        sources=(
            FastAPIAppSource(
                name="commerce",
                import_paths=["service.main:app"],
                path_prefixes=["/api/v1/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert [contract.endpoint_name for contract in contracts] == ["list_orders"]


def test_fastapi_catalog_loads_configured_factory_app(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi']\n",
        encoding="utf-8",
    )
    package_dir = tmp_path / "app"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "main.py").write_text(
        """
from fastapi import FastAPI
from pydantic import BaseModel

class Order(BaseModel):
    id: str

def create_app():
    app = FastAPI()

    @app.get('/orders/', response_model=list[Order], operation_id='list_orders')
    def list_orders():
        return []

    return app
""",
        encoding="utf-8",
    )

    contracts = get_fastapi_endpoint_contracts(
        sources=(
            FastAPIAppSource(
                name="commerce",
                import_paths=["app.main:create_app"],
                path_prefixes=["/orders/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert [contract.endpoint_name for contract in contracts] == ["list_orders"]


def test_fastapi_catalog_suppresses_host_stdout(tmp_path: Path, capsys) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi']\n",
        encoding="utf-8",
    )
    package_dir = tmp_path / "app"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "main.py").write_text(
        """
print("host import noise")

from fastapi import FastAPI
from pydantic import BaseModel

class Order(BaseModel):
    id: str

app = FastAPI()

@app.get('/orders/', response_model=list[Order], operation_id='list_orders')
def list_orders():
    return []
""",
        encoding="utf-8",
    )

    contracts = get_fastapi_endpoint_contracts(
        sources=(
            FastAPIAppSource(
                name="commerce",
                import_paths=["app.main:app"],
                path_prefixes=["/orders/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert [contract.endpoint_name for contract in contracts] == ["list_orders"]
    assert capsys.readouterr().out == ""


def test_fastapi_catalog_path_prefixes_match_path_segments() -> None:
    module = types.ModuleType("segment_api")
    app = FastAPI()

    @app.get("/api/orders/", operation_id="list_orders")
    def list_orders() -> dict[str, str]:
        return {"id": "order_1"}

    @app.get("/apiary/orders/", operation_id="apiary")
    def apiary() -> dict[str, str]:
        return {"id": "order_2"}

    setattr(module, "app", app)
    sys.modules[module.__name__] = module
    try:
        contracts = get_fastapi_endpoint_contracts(
            sources=(
                FastAPIAppSource(
                    name="commerce",
                    import_paths=["segment_api:app"],
                    path_prefixes=["/api"],
                ),
            ),
            project_root=Path.cwd(),
        )
    finally:
        sys.modules.pop(module.__name__, None)

    assert [contract.path_template for contract in contracts] == ["/api/orders/"]


def test_fastapi_catalog_exposes_only_schema_declared_routes() -> None:
    module = types.ModuleType("public_api")
    app = FastAPI()

    @app.get("/api/orders/", operation_id="list_orders")
    def list_orders() -> dict[str, str]:
        return {"id": "order_1"}

    @app.get(
        "/api/internal-state/",
        operation_id="internal_state",
        include_in_schema=False,
    )
    def internal_state() -> dict[str, str]:
        return {"status": "private"}

    setattr(module, "app", app)
    sys.modules[module.__name__] = module
    try:
        contracts = get_fastapi_endpoint_contracts(
            sources=(
                FastAPIAppSource(
                    name="commerce",
                    import_paths=["public_api:app"],
                    path_prefixes=["/api/"],
                ),
            ),
            project_root=Path.cwd(),
        )
    finally:
        sys.modules.pop(module.__name__, None)

    assert [contract.endpoint_name for contract in contracts] == ["list_orders"]


def _commerce_app() -> FastAPI:
    class Order(BaseModel):
        id: str
        status: str
        amount: float

    app = FastAPI()

    @app.get(
        "/api/v1/orders/",
        response_model=list[Order],
        operation_id="list_orders",
        tags=["orders"],
    )
    def list_orders(status: str | None = None) -> list[Order]:
        return []

    @app.get("/internal/orders/", operation_id="internal_orders")
    def internal_orders() -> list[Order]:
        return []

    return app
