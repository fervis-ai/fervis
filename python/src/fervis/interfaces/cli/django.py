"""Django composition root for the Fervis CLI."""

from __future__ import annotations

import sys

from fervis.interfaces.cli.contracts import FervisCliPorts
from fervis.interfaces.cli.dispatch import run_fervis
from fervis.project import ProjectInspection, discover_project
from fervis.project.django_runtime import django_project_runtime


def main(
    argv: tuple[str, ...] | None = None,
    *,
    project: ProjectInspection | None = None,
) -> int:
    resolved_project = project or discover_project()
    with django_project_runtime(resolved_project):
        from fervis.lineage.views.django import DjangoLineageQuery
        from fervis.observability.django import DjangoObservabilityQuery
        from fervis.observability.django_prompt_captures import (
            DjangoPromptCaptureQuery,
        )
        from fervis.interfaces.django.question_run_ports import (
            django_question_service,
        )
        from fervis.interfaces.django.composition import (
            question_run_request_limits,
        )
        from fervis.interfaces.common.admission import ConfiguredModelPolicy
        from fervis.project.source_scope import configured_fervis_config

        config = configured_fervis_config()
        return run_fervis(
            tuple(sys.argv[1:] if argv is None else argv),
            ports=FervisCliPorts(
                lineage_query=DjangoLineageQuery(),
                observability_query=DjangoObservabilityQuery(),
                prompt_capture_query=DjangoPromptCaptureQuery(),
                questions=django_question_service(),
                project=resolved_project,
                question_run_limits=question_run_request_limits(),
                model_policy=ConfiguredModelPolicy.from_config(config.model),
            ),
        )


if __name__ == "__main__":
    raise SystemExit(main())
