# FastAPI

Install Fervis:

```bash
uv add "fervis[fastapi]"
```

For a normal app object:

```bash
fervis init --framework fastapi --yes
```

For a factory app:

```bash
fervis init --framework fastapi \
  --app-factory app.main:create_app \
  --path-prefixes /api/ \
  --yes
```

Then:

```bash
fervis migrate
fervis doctor
```

If `doctor` blocks, follow every reported `next_actions` item, then rerun `fervis doctor` until it passes.

Fervis inspects the live FastAPI routes and their declared response models. It
does not require a generated OpenAPI document.

If the host API uses a FastAPI dependency for the current user:

```bash
fervis auth configure \
  --framework fastapi \
  --transport-mode in_process \
  --principal-dependency app.api.deps:get_current_user \
  --principal-id-attr id \
  --principal-resolver app.users:get_user_by_id
```

Then:

```bash
fervis doctor
fervis runtime ask "How many orders happened this month?"
```

Only run `runtime ask` after `fervis doctor` passes.
