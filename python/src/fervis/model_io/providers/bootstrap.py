"""Provider implementation registration boundary."""

from __future__ import annotations

_BOOTSTRAPPED = False


def bootstrap_default_providers() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    # Import lazily so provider SDK dependencies stay isolated.
    from fervis.model_io.providers.anthropic_adapter import (
        register_anthropic_provider,
    )
    from fervis.model_io.providers.openai_compatible_adapter import (
        register_openai_compatible_providers,
    )
    from fervis.model_io.providers.openai_responses_adapter import (
        register_openai_responses_provider,
    )

    register_anthropic_provider()
    register_openai_compatible_providers()
    register_openai_responses_provider()
    _BOOTSTRAPPED = True


def reset_provider_bootstrap_for_tests() -> None:
    global _BOOTSTRAPPED
    _BOOTSTRAPPED = False
