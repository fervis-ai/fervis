"""Deterministic project discovery for Fervis CLI commands."""

from __future__ import annotations

from dataclasses import dataclass
import ast
import json
from pathlib import Path
import tomllib

from .config_io import PROJECT_CONFIG_PATH


@dataclass(frozen=True)
class ProjectInspection:
    framework: str
    root_path: Path
    config_path: Path | None
    expected_config_path: Path | None
    confidence: str
    blocked_reason: str | None = None

    @property
    def is_blocked(self) -> bool:
        return self.blocked_reason is not None

    def to_payload(self) -> dict[str, object]:
        return {
            "framework": self.framework,
            "root_path": str(self.root_path),
            "config_path": str(self.config_path) if self.config_path else None,
            "expected_config_path": (
                str(self.expected_config_path) if self.expected_config_path else None
            ),
            "confidence": self.confidence,
            "blocked_reason": self.blocked_reason,
        }

    def to_envelope_project(self) -> dict[str, object]:
        return {
            "framework": self.framework,
            "config_path": str(self.config_path) if self.config_path else None,
        }


def discover_project(start_path: Path | str | None = None) -> ProjectInspection:
    start = Path.cwd() if start_path is None else Path(start_path)
    root = _project_root(start.resolve())
    if root is None:
        current = start.resolve()
        return ProjectInspection(
            framework="unknown",
            root_path=current,
            config_path=None,
            expected_config_path=None,
            confidence="low",
            blocked_reason="No Django, FastAPI, or Flask project marker was found.",
        )
    existing_config = _existing_config_path(root)
    detected_frameworks = _detected_frameworks(root)
    if not detected_frameworks and existing_config is not None:
        framework = _configured_framework(root / existing_config)
        if framework in {"django", "fastapi", "flask"}:
            return ProjectInspection(
                framework=framework,
                root_path=root,
                config_path=existing_config,
                expected_config_path=_expected_config_path(root),
                confidence="high",
                blocked_reason=None,
            )
    if len(detected_frameworks) > 1:
        configured = (
            _configured_framework(root / existing_config)
            if existing_config is not None
            else ""
        )
        if configured in detected_frameworks:
            return ProjectInspection(
                framework=configured,
                root_path=root,
                config_path=existing_config,
                expected_config_path=_expected_config_path(root),
                confidence="high",
                blocked_reason=None,
            )
        markers = ", ".join(detected_frameworks)
        return ProjectInspection(
            framework="unknown",
            root_path=root,
            config_path=existing_config,
            expected_config_path=_expected_config_path(root),
            confidence="low",
            blocked_reason=(
                f"Multiple framework markers were found: {markers}. "
                "Run Fervis init with --framework to choose one."
            ),
        )
    detected_framework = next(iter(detected_frameworks), None)
    if detected_framework == "django":
        return ProjectInspection(
            framework="django",
            root_path=root,
            config_path=existing_config,
            expected_config_path=_expected_config_path(root),
            confidence="high",
            blocked_reason=_missing_config_reason(
                root, expected_config=existing_config
            ),
        )
    if detected_framework == "flask":
        return ProjectInspection(
            framework="flask",
            root_path=root,
            config_path=existing_config,
            expected_config_path=_expected_config_path(root),
            confidence=_marker_confidence(
                root,
                detected_framework=detected_framework,
                existing_config=existing_config,
            ),
            blocked_reason=_missing_config_reason(
                root, expected_config=existing_config
            ),
        )
    if detected_framework is None:
        return ProjectInspection(
            framework="unknown",
            root_path=root,
            config_path=existing_config,
            expected_config_path=_expected_config_path(root),
            confidence="low",
            blocked_reason=(
                "No Django, FastAPI, or Flask project marker was found."
                if existing_config is None
                else None
            ),
        )
    return ProjectInspection(
        framework=detected_framework,
        root_path=root,
        config_path=existing_config,
        expected_config_path=_expected_config_path(root),
        confidence=_marker_confidence(
            root,
            detected_framework=detected_framework,
            existing_config=existing_config,
        ),
        blocked_reason=_missing_config_reason(root, expected_config=existing_config),
    )


def _project_root(start: Path) -> Path | None:
    current = start if start.is_dir() else start.parent
    for path in (current, *current.parents):
        if (
            _is_django_project(path)
            or _is_fastapi_project(path)
            or _is_flask_project(path)
            or _existing_config_path(path) is not None
        ):
            return path
    return None


def _is_django_project(path: Path) -> bool:
    manage_py = path / "manage.py"
    if not manage_py.is_file():
        return False
    try:
        from .mounting.common import parse_python_source
        from .mounting.django import django_settings_modules

        tree = parse_python_source(manage_py.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    return bool(django_settings_modules(tree))


def _detected_frameworks(path: Path) -> tuple[str, ...]:
    frameworks: list[str] = []
    if _is_django_project(path):
        frameworks.append("django")
    if _is_fastapi_project(path):
        frameworks.append("fastapi")
    if _is_flask_project(path):
        frameworks.append("flask")
    return tuple(frameworks)


def _marker_confidence(
    root: Path,
    *,
    detected_framework: str,
    existing_config: Path | None,
) -> str:
    if existing_config is None:
        return "medium"
    configured = _configured_framework(root / existing_config)
    return "high" if configured == detected_framework else "medium"


def _is_fastapi_project(path: Path) -> bool:
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return False
    return _pyproject_declares_fastapi(pyproject)


def _is_flask_project(path: Path) -> bool:
    pyproject = path / "pyproject.toml"
    if pyproject.is_file() and _pyproject_declares(pyproject, dependency_name="flask"):
        return True
    return _requirements_declares(path, dependency_name="flask")


def _pyproject_declares_fastapi(pyproject: Path) -> bool:
    return _pyproject_declares(pyproject, dependency_name="fastapi")


def _pyproject_declares(pyproject: Path, *, dependency_name: str) -> bool:
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    dependencies = list(data.get("project", {}).get("dependencies") or [])
    optional_dependencies = data.get("project", {}).get("optional-dependencies") or {}
    for values in optional_dependencies.values():
        dependencies.extend(values or [])
    poetry_dependencies = (
        data.get("tool", {}).get("poetry", {}).get("dependencies") or {}
    )
    dependencies.extend(poetry_dependencies.keys())
    return any(_dependency_name(item) == dependency_name for item in dependencies)


def _dependency_name(value: object) -> str:
    text = str(value).strip().lower()
    for separator in ("[", "<", ">", "=", "~", "!", " "):
        text = text.split(separator, 1)[0]
    return text.replace("_", "-")


def _requirements_declares(path: Path, *, dependency_name: str) -> bool:
    for requirements in path.glob("requirements*.txt"):
        if _requirements_file_declares(
            requirements,
            dependency_name=dependency_name,
            project_root=path,
            seen=frozenset(),
        ):
            return True
    return False


def _requirements_file_declares(
    requirements: Path,
    *,
    dependency_name: str,
    project_root: Path,
    seen: frozenset[Path],
) -> bool:
    path = requirements.resolve()
    if path in seen:
        return False
    try:
        path.relative_to(project_root.resolve())
    except ValueError:
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        included = _requirements_include(path.parent, line)
        if included is not None and _requirements_file_declares(
            included,
            dependency_name=dependency_name,
            project_root=project_root,
            seen=seen | {path},
        ):
            return True
        if _requirement_line_name(line) == dependency_name:
            return True
    return False


def _requirements_include(base: Path, line: str) -> Path | None:
    text = line.split("#", 1)[0].strip()
    if not text.startswith(("-r ", "--requirement ")):
        return None
    return (base / text.split(maxsplit=1)[1]).resolve()


def _requirement_line_name(line: str) -> str:
    text = line.split("#", 1)[0].strip()
    if not text or text.startswith(("-", "git+", "http://", "https://")):
        return ""
    return _dependency_name(text)


def _expected_config_path(root: Path | None = None) -> Path:
    if root is not None and _is_django_project(root):
        configured = _declared_django_config_path(root)
        if configured is not None:
            return configured
    return PROJECT_CONFIG_PATH


def _existing_config_path(root: Path) -> Path | None:
    config_path = _expected_config_path(root)
    if (root / config_path).is_file():
        return config_path
    if config_path != PROJECT_CONFIG_PATH and (root / PROJECT_CONFIG_PATH).is_file():
        return PROJECT_CONFIG_PATH
    return None


def _missing_config_reason(
    root: Path,
    *,
    expected_config: Path | None = None,
) -> str | None:
    if expected_config is not None:
        return None
    return f"Fervis config was not found at {_expected_config_path(root)}."


def _declared_django_config_path(root: Path) -> Path | None:
    try:
        from .mounting.bindings import top_level_assignment
        from .mounting.common import BlockedPatch, PythonFile
        from .mounting.django import django_settings_path
    except ImportError:
        return None
    settings_path = django_settings_path(root)
    if isinstance(settings_path, BlockedPatch):
        return None
    settings = PythonFile.load(root, settings_path)
    if isinstance(settings, BlockedPatch):
        return None
    assignment = top_level_assignment(settings.tree, "FERVIS_CONFIG_PATH")
    if assignment is None:
        return None
    value = assignment.value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return None
    if not value.value.strip():
        return None
    return Path(value.value)


def _configured_framework(config_path: Path) -> str:
    try:
        schema = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(schema, dict):
        return ""
    framework = schema.get("framework")
    return str(framework) if isinstance(framework, str) else ""
