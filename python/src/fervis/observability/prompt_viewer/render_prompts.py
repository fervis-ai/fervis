"""Export captured Fervis model-turn prompts.

Use `fervis inspect prompts --run-id <run-id>` to write raw JSON.
Use `--viewer-format html` to render the browser viewer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
import html
import json
from pathlib import Path
import re
import shutil
from typing import Any

from fervis.lineage.enums import ArtifactKind, ModelCallStatus
from fervis.observability.prompt_captures import (
    ModelTurnPromptCapture,
    PromptCaptureQueryPort,
    prompt_capture_artifacts_by_kind,
)


PROMPT_HEADING_RE = re.compile(r"^([A-Z][A-Za-z0-9 _/().-]{1,80}):\s*$")


@dataclass(frozen=True)
class RunOption:
    """A run selectable from the prompt-render home page."""

    run_id: str


@dataclass(frozen=True)
class ModelTurnCapture:
    """Captured prompt material for one model turn."""

    run_id: str
    sequence: int
    event_type: str
    purpose: str
    provider: str = ""
    model_key: str = ""
    selected_tool_name: str = ""
    raw_system_prompt: str = ""
    raw_prompt: str = ""
    raw_schema: Any = None
    raw_tool_specs: Any = None
    arguments: Any = None
    parsed_arguments: Any = None
    usage: dict[str, Any] = field(default_factory=dict)
    prompt_frame: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        label = self.purpose or self.event_type or "model_turn"
        return f"{self.sequence:04d}-{slugify(label)}"

    @property
    def title(self) -> str:
        label = self.purpose or self.event_type or "model turn"
        return f"{self.sequence}. {label.replace('_', ' ')}"


@dataclass(frozen=True)
class PromptSection:
    title: str
    content: str
    value: Any = None


class PromptInspectionFormat(StrEnum):
    RAW = "raw"
    HTML = "html"


@dataclass(frozen=True)
class PromptInspectionRun:
    run_id: str
    turns: tuple[ModelTurnCapture, ...]


@dataclass(frozen=True)
class PromptInspectionDocument:
    runs: tuple[PromptInspectionRun, ...]
    generated_at: str


@dataclass(frozen=True)
class PromptViewerRequest:
    run_id: str
    output_dir: Path
    title: str = "Fervis Prompt Viewer"
    output_format: PromptInspectionFormat = PromptInspectionFormat.RAW
    open_browser: bool = False


@dataclass(frozen=True)
class PromptViewerResult:
    run_count: int
    index_path: Path
    output_format: PromptInspectionFormat = PromptInspectionFormat.RAW


def render_prompt_viewer(
    request: PromptViewerRequest,
    *,
    prompt_capture_query: PromptCaptureQueryPort,
) -> PromptViewerResult:
    output_dir = request.output_dir.resolve()
    runs = (RunOption(run_id=request.run_id),)
    document = build_prompt_inspection_document(
        runs=runs,
        turn_loader=lambda run_id: load_model_turn_captures(
            run_id,
            prompt_capture_query=prompt_capture_query,
        ),
    )

    index_path = render_prompt_inspection(
        document=document,
        output_dir=output_dir,
        title=request.title,
        output_format=request.output_format,
    )
    if request.open_browser and request.output_format is PromptInspectionFormat.HTML:
        import webbrowser

        webbrowser.open(index_path.as_uri())
    return PromptViewerResult(
        run_count=len(runs),
        index_path=index_path,
        output_format=request.output_format,
    )


def load_model_turn_captures(
    run_id: str,
    *,
    prompt_capture_query: PromptCaptureQueryPort,
) -> list[ModelTurnCapture]:
    return [
        _capture_from_lineage(row)
        for row in prompt_capture_query.model_turn_prompt_captures_for_run(run_id)
    ]


def build_prompt_inspection_document(
    *,
    runs: Sequence[RunOption],
    turn_loader: Callable[[str], Sequence[ModelTurnCapture]],
) -> PromptInspectionDocument:
    return PromptInspectionDocument(
        runs=tuple(
            PromptInspectionRun(
                run_id=run.run_id,
                turns=tuple(turn_loader(run.run_id)),
            )
            for run in runs
        ),
        generated_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
    )


def render_prompt_inspection(
    *,
    document: PromptInspectionDocument,
    output_dir: Path,
    title: str,
    output_format: PromptInspectionFormat,
) -> Path:
    if output_format is PromptInspectionFormat.HTML:
        return render_html_site(document=document, output_dir=output_dir, title=title)
    if output_format is PromptInspectionFormat.RAW:
        return render_raw_prompt_json(document=document, output_dir=output_dir)
    raise ValueError(f"unsupported prompt inspection format: {output_format}")


def render_raw_prompt_json(
    *, document: PromptInspectionDocument, output_dir: Path
) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps(_raw_document_json(document), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return index_path


def render_html_site(
    *, document: PromptInspectionDocument, output_dir: Path, title: str
) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "assets").mkdir(parents=True)
    (output_dir / "runs").mkdir()
    write_assets(output_dir)

    run_pages: list[tuple[RunOption, tuple[ModelTurnCapture, ...], str]] = []
    for run in document.runs:
        run_option = RunOption(run_id=run.run_id)
        run_dir = output_dir / "runs" / slugify(run.run_id)
        run_dir.mkdir(parents=True)
        for turn in run.turns:
            call_path = run_dir / f"{turn.slug}.html"
            call_path.write_text(
                render_call_page(run=run_option, turn=turn, title=title),
                encoding="utf-8",
            )
        (run_dir / "index.html").write_text(
            render_run_page(run=run_option, turns=run.turns, title=title),
            encoding="utf-8",
        )
        run_pages.append(
            (run_option, run.turns, f"runs/{slugify(run.run_id)}/index.html")
        )

    index_path = output_dir / "index.html"
    index_path.write_text(
        render_index_page(run_pages=run_pages, title=title),
        encoding="utf-8",
    )
    return index_path


def render_static_site(
    *,
    runs: Sequence[RunOption],
    output_dir: Path,
    title: str,
    turn_loader: Callable[[str], Sequence[ModelTurnCapture]],
) -> None:
    document = build_prompt_inspection_document(
        runs=runs,
        turn_loader=turn_loader,
    )
    render_html_site(document=document, output_dir=output_dir, title=title)


def render_index_page(
    *,
    run_pages: Sequence[tuple[RunOption, Sequence[ModelTurnCapture], str]],
    title: str,
) -> str:
    rows = []
    for run, turns, href in run_pages:
        calls = ", ".join(
            html.escape(turn.purpose or turn.event_type) for turn in turns
        )
        rows.append(
            f"""
            <tr>
              <td><a href="{href}">{escape(run.run_id)}</a></td>
              <td>{escape(calls or "No model turns captured")}</td>
              <td>{len(turns)}</td>
            </tr>
            """
        )
    return page_shell(
        title=title,
        heading=title,
        breadcrumbs="",
        body=f"""
        <section class="toolbar">
          <input id="filter" type="search" placeholder="Filter runs and calls..." autofocus>
          <span>{len(run_pages)} run(s)</span>
        </section>
        <section class="table-wrap">
          <table id="runs">
            <thead>
              <tr>
                <th>Run</th>
                <th>Calls</th>
                <th>Model Turns</th>
              </tr>
            </thead>
            <tbody>
              {"".join(rows)}
            </tbody>
          </table>
        </section>
        """,
        asset_prefix="",
    )


def render_run_page(
    *, run: RunOption, turns: Sequence[ModelTurnCapture], title: str
) -> str:
    run_payload = {
        "runId": run.run_id,
        "modelTurns": len(turns),
    }
    cards = []
    for turn in turns:
        cards.append(
            f"""
            <article class="call-card">
              <div>
                <div class="eyebrow">Sequence {turn.sequence}</div>
                <h2>{escape(turn.purpose or turn.event_type)}</h2>
                <p>{escape(turn.provider)} {escape(turn.model_key)}</p>
              </div>
              <dl>
                <dt>Prompt</dt><dd>{turn.metadata.get("promptChars") or len(turn.raw_prompt)} chars</dd>
                <dt>Schema</dt><dd>{turn.metadata.get("schemaChars") or char_count(turn.raw_schema)} chars</dd>
                <dt>Tool Specs</dt><dd>{turn.metadata.get("toolSpecChars") or char_count(turn.raw_tool_specs)} chars</dd>
              </dl>
              <a class="button" href="{turn.slug}.html">Open Full Page</a>
            </article>
            """
        )
    body = f"""
      <section class="summary-grid">
        {json_details("Run Summary", run_payload)}
      </section>
      <section class="call-list">
        {"".join(cards) if cards else '<p class="empty">No model-turn captures were found for this run.</p>'}
      </section>
    """
    return page_shell(
        title=f"{run.run_id} - {title}",
        heading=run.run_id,
        breadcrumbs='<a href="../../index.html">All runs</a>',
        body=body,
        asset_prefix="../../",
    )


def render_call_page(*, run: RunOption, turn: ModelTurnCapture, title: str) -> str:
    metadata = dict(turn.metadata)
    metadata.update(
        {
            "runId": turn.run_id,
            "sequence": turn.sequence,
            "eventType": turn.event_type,
            "purpose": turn.purpose,
            "provider": turn.provider,
            "modelKey": turn.model_key,
            "selectedToolName": turn.selected_tool_name,
        }
    )
    prompt_sections = "".join(
        prompt_section_details(section)
        for section in split_prompt_sections(turn.raw_prompt)
    )
    body = f"""
      <section class="call-actions">
        <button type="button" data-expand="all">Expand all</button>
        <button type="button" data-collapse="all">Collapse all</button>
      </section>
      <section class="summary-grid">
        {json_details("Call Metadata", metadata)}
        {json_details("Usage", turn.usage)}
        {json_details("Prompt Frame", turn.prompt_frame)}
      </section>
      <section>
        <h2>System Prompt</h2>
        {text_details("Full System Prompt", turn.raw_system_prompt)}
      </section>
      <section>
        <h2>Prompt As Sent</h2>
        {prompt_sections}
        {text_details("Full Raw Prompt", turn.raw_prompt)}
      </section>
      <section>
        <h2>Typed Grammar / Schema</h2>
        {json_details("Raw Schema", turn.raw_schema)}
        {json_details("Raw Tool Specs", turn.raw_tool_specs)}
      </section>
      <section>
        <h2>Model Output</h2>
        {json_details("Submitted Arguments", turn.arguments)}
        {json_details("Parsed Arguments", turn.parsed_arguments)}
      </section>
    """
    return page_shell(
        title=f"{turn.purpose or turn.event_type} - {title}",
        heading=turn.title,
        breadcrumbs=(
            f'<a href="../../index.html">All runs</a>'
            f' <span>/</span> <a href="index.html">{escape(run.run_id)}</a>'
        ),
        body=body,
        asset_prefix="../../",
    )


def split_prompt_sections(raw_prompt: str) -> list[PromptSection]:
    if not raw_prompt:
        return [PromptSection(title="Prompt", content="")]
    lines = raw_prompt.splitlines()
    sections: list[PromptSection] = []
    current_title = "Preamble"
    current_lines: list[str] = []

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        if content or not sections:
            sections.append(
                PromptSection(current_title, content, parse_jsonish(content))
            )

    for line in lines:
        match = PROMPT_HEADING_RE.match(line)
        if match:
            flush()
            current_title = match.group(1)
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return sections


def prompt_section_details(section: PromptSection) -> str:
    if section.value is not None:
        return json_details(section.title, section.value)
    return text_details(section.title, section.content)


def json_details(title: str, value: Any) -> str:
    return f"""
      <details class="panel">
        <summary>{escape(title)} <span>{json_summary(value)}</span></summary>
        <div class="panel-tools"><button type="button" data-copy>Copy</button></div>
        <pre>{escape(pretty_json(value))}</pre>
      </details>
    """


def text_details(title: str, value: str) -> str:
    text = value or ""
    return f"""
      <details class="panel">
        <summary>{escape(title)} <span>{len(text)} chars</span></summary>
        <div class="panel-tools"><button type="button" data-copy>Copy</button></div>
        <pre>{escape(text)}</pre>
      </details>
    """


def page_shell(
    *, title: str, heading: str, breadcrumbs: str, body: str, asset_prefix: str
) -> str:
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <link rel="stylesheet" href="{asset_prefix}assets/style.css">
</head>
<body>
  <header>
    <div class="marks">+ + + +</div>
    <div class="docline">Fervis Prompt Render · Memo No. 001 · Generated {generated_at}</div>
    <div class="rule"></div>
    <nav>{breadcrumbs}</nav>
    <h1>{escape(heading)}</h1>
    <p>Raw prompt captures, typed grammars, tool schemas, and submitted model arguments.</p>
  </header>
  <main>
    {body}
  </main>
  <script src="{asset_prefix}assets/app.js"></script>
</body>
</html>
"""


def write_assets(output_dir: Path) -> None:
    (output_dir / "assets" / "style.css").write_text(STYLE_CSS, encoding="utf-8")
    (output_dir / "assets" / "app.js").write_text(APP_JS, encoding="utf-8")


def _capture_from_lineage(row: ModelTurnPromptCapture) -> ModelTurnCapture:
    artifacts = prompt_capture_artifacts_by_kind(row)
    return ModelTurnCapture(
        run_id=row.run_id,
        sequence=row.sequence,
        event_type=_event_type(row.status),
        purpose=row.step_key.value,
        provider=row.provider,
        model_key=row.model_key,
        selected_tool_name=_selected_tool_name(row),
        raw_system_prompt=_artifact_text(artifacts, ArtifactKind.SYSTEM_PROMPT),
        raw_prompt=_artifact_text(artifacts, ArtifactKind.PROMPT),
        raw_schema=_artifact_json(artifacts, ArtifactKind.SCHEMA),
        raw_tool_specs=_artifact_json(artifacts, ArtifactKind.TOOL_SPEC),
        arguments=_artifact_json(artifacts, ArtifactKind.SUBMITTED_PAYLOAD),
        parsed_arguments=_artifact_json(artifacts, ArtifactKind.PARSED_PAYLOAD),
        usage=_usage_payload(row),
        prompt_frame=_prompt_frame(row),
        metadata=_metadata_from_lineage(row),
    )


def _artifact_text(
    artifacts: dict[ArtifactKind, Any], artifact_kind: ArtifactKind
) -> str:
    artifact = artifacts.get(artifact_kind)
    if artifact is None:
        return ""
    return artifact.content


def _artifact_json(
    artifacts: dict[ArtifactKind, Any], artifact_kind: ArtifactKind
) -> Any:
    return parse_jsonish(_artifact_text(artifacts, artifact_kind))


def _event_type(status: ModelCallStatus) -> str:
    if status is ModelCallStatus.SUCCEEDED:
        return "model_turn.completed"
    return "model_turn.failed"


def _selected_tool_name(row: ModelTurnPromptCapture) -> str:
    value = row.step_output_summary.get(
        "selectedToolName"
    ) or row.step_output_summary.get("selected_tool_name")
    return str(value or "")


def _prompt_frame(row: ModelTurnPromptCapture) -> dict[str, Any]:
    value = row.step_input_summary.get("promptFrame") or row.step_input_summary.get(
        "prompt_frame"
    )
    if isinstance(value, dict):
        return dict(value)
    return {}


def _usage_payload(row: ModelTurnPromptCapture) -> dict[str, Any]:
    return {
        usage.provider_usage_key or usage.usage_kind.value: usage.quantity
        for usage in row.usage_rows
    }


def _metadata_from_lineage(row: ModelTurnPromptCapture) -> dict[str, Any]:
    output = {
        "providerRequestId": row.provider_request_id,
        "finishReason": row.finish_reason,
        "durationMs": row.duration_ms,
        "promptChars": row.prompt_chars,
        "schemaChars": row.schema_chars,
        "toolSpecChars": row.tool_spec_chars,
        "submittedPayloadChars": row.submitted_payload_chars,
        "rawOutputChars": row.raw_output_chars,
        "error": row.error_json,
    }
    return {key: value for key, value in output.items() if value not in ("", None, {})}


def _raw_document_json(document: PromptInspectionDocument) -> dict[str, Any]:
    return {
        "generated_at": document.generated_at,
        "runs": [_raw_run_json(run) for run in document.runs],
    }


def _raw_run_json(run: PromptInspectionRun) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "model_turn_count": len(run.turns),
        "model_turns": [_raw_turn_json(turn) for turn in run.turns],
    }


def _raw_turn_json(turn: ModelTurnCapture) -> dict[str, Any]:
    return {
        "run_id": turn.run_id,
        "sequence": turn.sequence,
        "event_type": turn.event_type,
        "purpose": turn.purpose,
        "provider": turn.provider,
        "model_key": turn.model_key,
        "selected_tool_name": turn.selected_tool_name,
        "error": turn.metadata.get("error"),
        "metadata": turn.metadata,
        "usage": turn.usage,
        "prompt_frame": turn.prompt_frame,
        "system_prompt": turn.raw_system_prompt,
        "prompt": turn.raw_prompt,
        "prompt_sections": [
            {
                "title": section.title,
                "content": section.content,
                "parsed_value": section.value,
            }
            for section in split_prompt_sections(turn.raw_prompt)
        ],
        "schema": turn.raw_schema,
        "tool_specs": turn.raw_tool_specs,
        "submitted_arguments": turn.arguments,
        "parsed_arguments": turn.parsed_arguments,
    }


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    if stripped[0] not in "{[":
        return None
    try:
        parsed, _index = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    return parsed


def pretty_json(value: Any) -> str:
    if isinstance(value, str):
        parsed = parse_jsonish(value)
        if parsed is not None:
            value = parsed
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str)


def json_summary(value: Any) -> str:
    if isinstance(value, dict):
        return f"{len(value)} keys"
    if isinstance(value, list):
        return f"{len(value)} items"
    if value is None:
        return "empty"
    return type(value).__name__


def char_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    return len(pretty_json(value))


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "item"


def escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


STYLE_CSS = """
:root {
  color-scheme: light;
  --bg: #f5f1e8;
  --panel: #fffdf6;
  --panel-soft: #ebe4d5;
  --text: #101010;
  --muted: #4c463c;
  --line: #111111;
  --accent: #0057ff;
  --ok: #007a3d;
  --bad: #d40000;
  --neutral: #111111;
  --shadow: 5px 5px 0 #111111;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: "Times New Roman", Times, serif;
}
header {
  background: var(--panel);
  border-bottom: 4px solid var(--line);
  padding: 18px 32px 20px;
  position: sticky;
  top: 0;
  z-index: 2;
}
.marks {
  font-family: Arial, Helvetica, sans-serif;
  font-size: 22px;
  font-weight: 900;
  letter-spacing: 12px;
  line-height: 1;
}
.docline {
  margin-top: 12px;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0;
}
.rule {
  border-top: 3px solid var(--line);
  margin: 14px 0 10px;
}
nav {
  min-height: 22px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
}
nav a, a {
  color: var(--accent);
  text-decoration: none;
}
nav a:hover, a:hover { text-decoration: underline; }
h1 {
  font-size: 48px;
  line-height: 0.98;
  margin: 6px 0 8px;
  max-width: 980px;
}
h2 {
  font-family: Arial, Helvetica, sans-serif;
  font-size: 16px;
  line-height: 1.15;
  margin: 30px 0 12px;
  text-transform: uppercase;
  border-top: 3px solid var(--line);
  padding-top: 12px;
}
header p {
  margin: 0;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  font-family: Arial, Helvetica, sans-serif;
}
main {
  max-width: 1360px;
  margin: 0 auto;
  padding: 24px 32px 48px;
}
.toolbar {
  display: flex;
  gap: 16px;
  align-items: center;
  margin-bottom: 16px;
}
input[type="search"] {
  width: min(560px, 100%);
  border: 3px solid var(--line);
  border-radius: 0;
  padding: 11px 12px;
  font-size: 14px;
  font-family: Arial, Helvetica, sans-serif;
  background: var(--panel);
  box-shadow: 3px 3px 0 var(--line);
}
.table-wrap {
  overflow: auto;
  background: var(--panel);
  border: 3px solid var(--line);
  border-radius: 0;
  box-shadow: var(--shadow);
}
table {
  width: 100%;
  border-collapse: collapse;
  min-width: 920px;
}
th, td {
  text-align: left;
  vertical-align: top;
  padding: 12px 14px;
  border-bottom: 2px solid var(--line);
  border-right: 2px solid var(--line);
  font-size: 13px;
  font-family: Arial, Helvetica, sans-serif;
}
th:last-child, td:last-child { border-right: 0; }
th {
  background: var(--panel-soft);
  color: var(--muted);
  font-weight: 900;
  text-transform: uppercase;
}
tr:last-child td { border-bottom: 0; }
.mono, pre {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
}
.pill {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  border-radius: 0;
  padding: 2px 10px;
  font-size: 12px;
  font-weight: 900;
  border: 2px solid currentColor;
  background: #ffffff;
  font-family: Arial, Helvetica, sans-serif;
}
.pill.ok { color: var(--ok); }
.pill.bad { color: var(--bad); }
.pill.neutral { color: var(--neutral); }
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 14px;
  margin-bottom: 22px;
}
.panel, .call-card {
  background: var(--panel);
  border: 3px solid var(--line);
  border-radius: 0;
  box-shadow: var(--shadow);
}
.panel {
  margin-bottom: 12px;
  overflow: hidden;
}
summary {
  font-family: Arial, Helvetica, sans-serif;
  cursor: pointer;
  padding: 14px 16px;
  font-weight: 900;
  list-style-position: inside;
  text-transform: uppercase;
}
summary span {
  color: var(--muted);
  font-weight: 800;
  margin-left: 8px;
  text-transform: none;
}
.panel-tools {
  display: flex;
  justify-content: flex-end;
  padding: 8px 12px;
  background: var(--panel-soft);
  border-top: 2px solid var(--line);
  border-bottom: 2px solid var(--line);
}
button, .button {
  border: 3px solid var(--line);
  border-radius: 0;
  background: var(--panel);
  color: var(--text);
  font-size: 13px;
  font-weight: 900;
  padding: 8px 10px;
  cursor: pointer;
  text-decoration: none;
  text-transform: uppercase;
  box-shadow: 3px 3px 0 var(--line);
  font-family: Arial, Helvetica, sans-serif;
}
button:hover, .button:hover {
  background: var(--accent);
  color: #ffffff;
  text-decoration: none;
}
pre {
  margin: 0;
  padding: 16px;
  overflow: auto;
  max-height: 72vh;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  line-height: 1.5;
  background: #fbfaf4;
}
.call-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 14px;
}
.call-card {
  display: grid;
  gap: 14px;
  padding: 16px;
  align-content: start;
}
.call-card h2 {
  margin: 2px 0 4px;
  border-top: 0;
  padding-top: 0;
}
.call-card p {
  color: var(--muted);
  margin: 0;
  font-family: Arial, Helvetica, sans-serif;
}
.eyebrow {
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0;
  font-weight: 900;
  font-family: Arial, Helvetica, sans-serif;
}
dl {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 6px 12px;
  margin: 0;
  font-size: 13px;
  font-family: Arial, Helvetica, sans-serif;
}
dt { color: var(--muted); }
dd { margin: 0; }
.call-actions {
  display: flex;
  gap: 10px;
  margin-bottom: 16px;
}
.empty {
  color: var(--muted);
}
@media (max-width: 720px) {
  header { padding: 18px 18px 14px; }
  main { padding: 18px; }
  .toolbar { align-items: stretch; flex-direction: column; }
  h1 { font-size: 22px; }
}
"""


APP_JS = """
const filter = document.querySelector("#filter");
if (filter) {
  filter.addEventListener("input", () => {
    const term = filter.value.toLowerCase();
    document.querySelectorAll("#runs tbody tr").forEach((row) => {
      row.style.display = row.textContent.toLowerCase().includes(term) ? "" : "none";
    });
  });
}

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const pre = button.closest(".panel")?.querySelector("pre");
    if (!pre) return;
    await navigator.clipboard.writeText(pre.textContent);
    const original = button.textContent;
    button.textContent = "Copied";
    setTimeout(() => { button.textContent = original; }, 900);
  });
});

const setAllDetails = (open) => {
  document.querySelectorAll("details").forEach((details) => {
    details.open = open;
  });
};
document.querySelector("[data-expand='all']")?.addEventListener("click", () => setAllDetails(true));
document.querySelector("[data-collapse='all']")?.addEventListener("click", () => setAllDetails(false));
"""
