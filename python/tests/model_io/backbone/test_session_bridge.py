from fervis.model_io.providers.session_runtime import ProviderSessionRuntime


def test_resume_session_uses_provider_checkpoint_mapping(fervis_foundation_reset):
    runtime = ProviderSessionRuntime(provider_name="test_provider")
    created = runtime.continue_session(session_id=None)
    resumed = runtime.resume_session(session_id=created.session_id)

    assert resumed.session_id == created.session_id
    assert resumed.provider_session_id == created.provider_session_id


def test_branch_updates_do_not_mutate_parent_branch(fervis_foundation_reset):
    runtime = ProviderSessionRuntime(provider_name="test_provider")
    parent = runtime.continue_session(session_id=None)
    child = runtime.fork_session(
        session_id=parent.session_id, branch_point_event_id="run:branch"
    )

    assert child.metadata["parentSessionId"] == parent.session_id


def test_fork_session_creates_distinct_child_session(fervis_foundation_reset):
    runtime = ProviderSessionRuntime(provider_name="test_provider")
    parent = runtime.continue_session(session_id=None)
    child = runtime.fork_session(
        session_id=parent.session_id, branch_point_event_id="run:branch"
    )

    assert child.session_id != parent.session_id


def test_fork_session_does_not_mutate_parent_metadata(fervis_foundation_reset):
    runtime = ProviderSessionRuntime(provider_name="test_provider")
    parent = runtime.continue_session(session_id=None)
    runtime.fork_session(
        session_id=parent.session_id, branch_point_event_id="run:branch"
    )

    assert parent.metadata.get("parentSessionId") is None
