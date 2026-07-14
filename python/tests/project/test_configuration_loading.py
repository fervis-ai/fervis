from pathlib import Path

from fervis.project.auth_config.loading import LoadedAuthSchema
from fervis.project.config_io import ActiveEnvironment
from fervis.project.configuration import _fastapi_principal_dependency
from fervis.project.discovery import ProjectInspection


def test_fastapi_config_keeps_principal_dependency_import_lazy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    environment = ActiveEnvironment(name="local", source="explicit")
    project = ProjectInspection(
        framework="fastapi",
        root_path=tmp_path,
        config_path=Path("config/fervis.json"),
        expected_config_path=Path("config/fervis.json"),
        confidence="high",
    )
    loaded_auth = LoadedAuthSchema(
        schema={
            "principal": {
                "source": "fastapi_dependency",
                "dependency": "host.auth:get_user",
                "id_attr": "id",
            }
        },
        config_path=tmp_path / "config" / "fervis_auth.json",
        active_environment=environment,
    )
    monkeypatch.setattr(
        "fervis.project.auth_config.loading.load_auth_project_schema",
        lambda *args, **kwargs: loaded_auth,
    )

    def fail_if_imported(path: str):
        raise AssertionError(f"loaded configuration imported {path}")

    monkeypatch.setattr("fervis.project.importing.import_object", fail_if_imported)

    dependency = _fastapi_principal_dependency(
        project,
        active_environment=environment,
    )

    assert dependency is not None
    assert callable(dependency.factory)
