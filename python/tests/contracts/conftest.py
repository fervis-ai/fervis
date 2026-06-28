from __future__ import annotations

import pytest

from .adapters import CONTRACT_ADAPTERS


@pytest.fixture(params=CONTRACT_ADAPTERS, ids=lambda factory: factory.name)
def adapter(request, tmp_path, fervis_foundation_reset, api_client):
    # The Django adapter needs the foundation reset and seeded API user; the SQL
    # adapter ignores those fixtures and builds its own isolated project.
    del fervis_foundation_reset, api_client
    return request.param(tmp_path)
