# Django / DRF

Install Fervis:

```bash
uv add "fervis[django]"
```

Run from the Django project root:

```bash
fervis init --framework django --yes
fervis migrate
fervis doctor
```

`init` adds Fervis to Django settings/URLs when it can do so safely. If it blocks, follow its `next_actions`.
If `doctor` blocks, follow every reported `next_actions` item, then rerun `fervis doctor` until it passes.

If the host API uses header auth for reads:

```bash
fervis auth configure \
  --framework django-drf \
  --transport-mode http \
  --base-url-env FERVIS_HOST_API_BASE_URL \
  --capture-credential-header Authorization
```

Then:

```bash
fervis doctor
fervis runtime ask "How many orders happened this month?"
```

Only run `runtime ask` after `fervis doctor` passes.
