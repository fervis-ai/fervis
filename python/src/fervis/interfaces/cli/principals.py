"""CLI principal construction for local Fervis runtime commands."""

from __future__ import annotations

from fervis.host_api.contracts.authority import ReadContextRef, ReadContextScheme
from fervis.project import ProjectInspection
from fervis.questions import QuestionPrincipal


_READ_CONTEXT_SCHEME_BY_FRAMEWORK: dict[str, ReadContextScheme] = {
    "django": "django_principal",
    "fastapi": "fastapi_principal",
    "flask": "flask_principal",
}


def cli_question_principal(
    *,
    tenant_id: str,
    principal_id: str,
    project: ProjectInspection,
) -> QuestionPrincipal:
    scheme = _READ_CONTEXT_SCHEME_BY_FRAMEWORK.get(project.framework)
    read_context_ref = (
        ReadContextRef(scheme=scheme, key=principal_id)
        if scheme
        else ReadContextRef(scheme="anonymous")
    )
    return QuestionPrincipal(
        tenant_id=tenant_id,
        principal_id=principal_id,
        read_context_ref=read_context_ref,
    )
