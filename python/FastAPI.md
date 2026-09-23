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

Fervis discovers the live FastAPI routes and uses their declared response models
when present. Selected routes without response models can be inspected at runtime
under the caller's authority, provided the read needs no additional invocation
inputs and returns JSON objects or arrays of objects. Observed values do not
establish keys, relationships, or closed enums. Existing declared contracts remain
authoritative.

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
fervis runtime ask "How many orders happened this month?" \
  --tenant-id <tenant-id> --principal-id <principal-id> --wait
```

Only run `runtime ask` after `fervis doctor` passes.
