from __future__ import annotations

import json

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from fervis.host_api.adapters.django.adapter import DjangoHostApiAdapter
from fervis.host_api.context import (
    HostApiContext,
    HostContext,
    configure_host_api_context,
)
from fervis.interfaces.django.composition import reset_runtime_for_tests
from fervis.model_io.backbone.factory import build_test_provider_backbone
from fervis.project.source_scope import configured_django_source_scopes
from fervis.run_work.queue.django.queue import reset_question_run_queue_for_tests
from tests.testkit.django import SEEDED_USER_PK
from tests.testkit.provider_native import provider_native_test_arguments


@pytest.fixture
def api_client(db):
    client = APIClient()
    client.force_authenticate(user=_seeded_user())
    return client


@pytest.fixture
def anon_client():
    return APIClient()


@pytest.fixture
def fervis_foundation_reset(db):
    configure_host_api_context(
        HostApiContext(
            adapter=DjangoHostApiAdapter(sources=configured_django_source_scopes()),
            host_context=HostContext(
                organization_name="Fervis Test",
                about_api=(
                    "The Fervis test API helps operators work with generic "
                    "business records."
                ),
            ),
        )
    )
    reset_question_run_queue_for_tests()
    reset_runtime_for_tests(provider_backbone=_fervis_test_provider_backbone())

    yield

    reset_question_run_queue_for_tests()
    reset_runtime_for_tests(provider_backbone=_fervis_test_provider_backbone())


def _seeded_user():
    user_model = get_user_model()
    user, _ = user_model._default_manager.update_or_create(
        pk=SEEDED_USER_PK,
        defaults={
            "username": "fervis-test-user",
            "email": "fervis-test@example.com",
            "is_active": True,
            "is_staff": True,
            "is_superuser": True,
        },
    )
    return user


def _fervis_test_provider_backbone():
    adapter = _FervisPlannerTestAdapter()
    return build_test_provider_backbone(
        provider_name="anthropic",
        adapters={
            "anthropic": adapter,
            "openai": adapter,
            "openai_gpt_5_4_mini": adapter,
        },
    )


class _FervisPlannerTestAdapter:
    provider_name = "openai"

    def generate(
        self,
        *,
        prompt: str,
        max_thinking_tokens: int,
        output_mode=None,
        tool_specs=(),
    ):
        del max_thinking_tokens, output_mode
        tool_name = _first_tool_spec_name(tool_specs)
        if tool_name:
            return _planner_response(
                prompt,
                {
                    "tool": tool_name,
                    "arguments": provider_native_test_arguments(
                        tool_name=tool_name,
                        prompt=prompt,
                        tool_specs=tool_specs,
                    ),
                },
            )
        return _planner_response(prompt, {"answer": "Test adapter response."})


def _first_tool_spec_name(tool_specs) -> str:
    for spec in tool_specs or ():
        name = getattr(spec, "name", None)
        if name:
            return str(name)
        if isinstance(spec, dict):
            if spec.get("name"):
                return str(spec["name"])
            function = spec.get("function")
            if isinstance(function, dict) and function.get("name"):
                return str(function["name"])
    return ""


def _planner_response(prompt: str, answer: dict) -> dict:
    return {
        "provider": "openai",
        "answer": json.dumps(answer),
        "toolRequests": [],
        "usage": {
            "inputTokens": max(1, len(prompt.split())),
            "outputTokens": 12,
            "thinkingTokens": 8,
            "costUsd": 0.001,
            "inputCostUsd": 0.001,
            "outputCostUsd": 0,
            "thinkingCostUsd": 0,
        },
        "raw": {"sdk": "test-planner"},
    }
