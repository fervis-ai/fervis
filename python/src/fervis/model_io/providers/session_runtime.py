from __future__ import annotations

import uuid

from fervis.model_io.backbone.dto import SessionRef


class ProviderSessionRuntime:
    def __init__(self, *, provider_name: str) -> None:
        self.provider_name = str(provider_name or "").strip()
        self._map: dict[str, SessionRef] = {}

    def _provider_ref(self, session_id: str) -> SessionRef:
        existing = self._map.get(session_id)
        if existing is not None:
            return existing

        created = SessionRef(
            session_id=session_id,
            provider_session_id=f"{self.provider_name}:{session_id}",
            metadata={"provider": self.provider_name},
        )
        self._map[session_id] = created
        return created

    def continue_session(self, *, session_id: str | None) -> SessionRef:
        if not session_id:
            session_id = str(uuid.uuid4())
        return self._provider_ref(session_id)

    def resume_session(self, *, session_id: str) -> SessionRef:
        return self._provider_ref(session_id)

    def fork_session(
        self, *, session_id: str, branch_point_event_id: str
    ) -> SessionRef:
        forked = str(uuid.uuid4())
        ref = SessionRef(
            session_id=forked,
            provider_session_id=f"{self.provider_name}:{forked}",
            metadata={
                "provider": self.provider_name,
                "parentSessionId": session_id,
                "branchPointEventId": branch_point_event_id,
            },
        )
        self._map[forked] = ref
        return ref
