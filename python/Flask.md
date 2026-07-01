# Fervis Flask Host API Setup Checklist

Use this file to make a Flask API fully readable by Fervis. Fervis can only answer factual questions reliably when every configured GET route exposes a deterministic route contract: path params, query params, operation identity, JSON response schema, and correct object-vs-array response cardinality. Follow the steps in order, keep existing OpenAPI/Swagger-style surfaces when they already exist, add schema metadata where routes are plain Flask, then run Fervis doctor until every configured GET route is certified read-eligible.

This checklist must not change business logic or host API functionality; its sole purpose is to enrich endpoint contracts with metadata that supports Fervis' factual answer runtime.

## Step 0: Create a local working checklist.
- [ ] 0.1 Copy this file into the host repository as a temporary working checklist, such as `FERVIS_FLASK_SETUP_CHECKLIST.md`.
- [ ] 0.2 Check off each item in the local copy immediately after completing it.
- [ ] 0.3 Keep the local copy until Fervis doctor, host tests, smoke tests, and runtime ask all pass.

## Step 1: Select the route-contract surface.
- [ ] 1.1 Confirm the Fervis CLI will run with Python 3.11 or newer; run `python --version`, `python3.12 --version`, or `uv python list`.
- [ ] 1.2 Use a project-local Python 3.11+ environment for Fervis commands when the host app itself runs on an older Python version.
- [ ] 1.3 Check whether the app already exposes OpenAPI/Swagger, `flask-smorest`, `flask-apispec`, `flask-restx`, `Flask-AppBuilder`, `flask-rest-jsonapi-next`, Connexion, or RESTPlus.
- [ ] 1.4 Keep the existing surface when one exists.
- [ ] 1.5 Treat Connexion and RESTPlus apps as OpenAPI/Swagger apps when they expose a Swagger document.
- [ ] 1.6 Choose `flask-smorest` + `marshmallow` + `apispec` only when the app has plain Flask routes and no existing contract surface.
- [ ] 1.7 Open package links when needed: <https://pypi.org/project/flask-smorest/>, <https://pypi.org/project/marshmallow/>, <https://pypi.org/project/apispec/>, <https://pypi.org/project/flask-apispec/>, <https://pypi.org/project/flask-restx/>, <https://pypi.org/project/Flask-AppBuilder/>, <https://pypi.org/project/flask-rest-jsonapi-next/>.

## Step 2: Install the selected package family.
- [ ] 2.1 Use the newest Python version supported by both Fervis and the host app's pinned dependencies.
- [ ] 2.2 Install the host app's runtime dependencies first, using `uv sync`, `pip install -r requirements.txt`, `poetry install`, or the app's documented runtime package list.
- [ ] 2.3 Run no schema-package install when the app already exposes complete OpenAPI/Swagger.
- [ ] 2.4 Run `uv add flask-smorest marshmallow apispec` for smorest.
- [ ] 2.5 Run `uv add flask-apispec marshmallow apispec` for flask-apispec decorators.
- [ ] 2.6 Run `uv add flask-restx` for RESTX.
- [ ] 2.7 Run `uv add flask-rest-jsonapi-next marshmallow-jsonapi` for JSON:API.
- [ ] 2.8 Run `uv add Flask-AppBuilder` for AppBuilder.
- [ ] 2.9 Replace `uv add` with `pip install` when the project uses pip.
- [ ] 2.10 Replace `uv add` with `poetry add` when the project uses Poetry.
- [ ] 2.11 Run `python -c "from <module> import <app_or_factory>"` for the chosen Flask app target.
- [ ] 2.12 Fix every import or dependency error before running Fervis commands.

## Step 3: Annotate every Fervis-exposed GET route.
- [ ] 3.1 Declare every path parameter in the route path, such as `/orders/<int:order_id>`.
- [ ] 3.2 Declare every query parameter with schema or argument metadata.
- [ ] 3.3 Attach the JSON response schema to the route's existing documented success response.
- [ ] 3.4 Mark collection responses with `many=True` or an OpenAPI array schema.
- [ ] 3.5 Name operations semantically, such as `list_orders` or `get_order`.
- [ ] 3.6 Declare every `request.args` parameter and every `Schema().dump(...)` response shape in route metadata.

## Step 4: Add smorest contracts when no existing surface exists.
- [ ] 4.1 Create a Marshmallow schema such as `OrderSchema` with fields like `id = fields.Int(required=True)`.
- [ ] 4.2 Add `@blp.response(<success_status>, OrderSchema)` to each detail GET route, using the route's existing success status.
- [ ] 4.3 Add `@blp.response(<success_status>, OrderSchema(many=True))` to each list GET route, using the route's existing success status.
- [ ] 4.4 Add `@blp.arguments(OrderQuerySchema, location="query")` when the route accepts query parameters.

## Step 5: Add flask-apispec contracts when the app already uses decorators.
- [ ] 5.1 Add `@use_kwargs({"status": fields.Str()}, location="query")` for query parameters.
- [ ] 5.2 Add `@marshal_with(OrderSchema)` for detail GET routes.
- [ ] 5.3 Add `@marshal_with(OrderSchema(many=True))` for list GET routes.

## Step 6: Validate existing OpenAPI/Swagger.
- [ ] 6.1 Confirm every configured GET operation declares parameters.
- [ ] 6.2 Confirm every configured GET operation attaches a JSON schema to its documented success response.
- [ ] 6.3 Confirm list endpoints use an array schema.
- [ ] 6.4 Rename generic operation IDs like `get` or `wrapper` to semantic names such as `list_orders`, `get_order`, or `list_inventory_items`, using the route's business resource name.
- [ ] 6.5 Expose the generated OpenAPI/Swagger document before running doctor.

## Step 7: Fix doctor blockers.
- [ ] 7.1 Add a response schema when doctor reports missing response fields.
- [ ] 7.2 Add an argument/query schema when doctor reports missing query parameters.
- [ ] 7.3 Fix object-vs-array metadata when doctor reports that the declared schema cardinality does not match the runtime JSON response.
- [ ] 7.4 Fix every configured GET route before certifying the app.
- [ ] 7.5 Pass doctor only when every configured GET route is read-eligible.

## Step 8: Install and initialize Fervis.
- [ ] 8.1 Run `uv add "fervis[flask]"` or `pip install "fervis[flask]"` or `poetry add "fervis[flask]"`.
- [ ] 8.2 Find the Flask app import target, such as `src.app:app`, `autoapp:app`, or `app:create_app`.
- [ ] 8.3 Expose wrapper apps as a top-level Flask app, such as `app = connex_app.app`, after the wrapper registers API routes.
- [ ] 8.4 Find the API path prefix to expose, such as `/api/`, `/api/v1/`, or `/v1/`.
- [ ] 8.5 Run `fervis init --framework flask --app <module:app_or_factory> --source-prefix <api-prefix> --yes`.
- [ ] 8.6 Add `--blueprint <name>` only when exposing one specific blueprint.
- [ ] 8.7 Create or identify importable Flask principal functions; use project-root `fervis_auth_support.py` when no auth module exists.
- [ ] 8.8 Add `capture(request)` and `resolve(key, tenant_id=None)` to that module.
- [ ] 8.9 Add one `--capture-credential-header <Header-Name>` for each auth header required by GET reads, such as `Authorization`, `API-TOKEN`, or `X-API-Key`.
- [ ] 8.10 Run `fervis auth configure --framework flask --principal-source callable --principal-dependency fervis_auth_support:capture --principal-resolver fervis_auth_support:resolve --transport-mode in_process [--capture-credential-header API-TOKEN]`.
- [ ] 8.11 Run `fervis migrate`.
- [ ] 8.12 Run `fervis doctor`.
- [ ] 8.13 If doctor returns `"status": "blocked"`, run the listed `next_actions`, then rerun `fervis doctor`.

## Step 9: Verify the host API still works.
- [ ] 9.1 Run the host app's existing test suite.
- [ ] 9.2 Start the host API with its normal local command.
- [ ] 9.3 Smoke-test representative existing GET endpoints that were exposed to Fervis.
- [ ] 9.4 Smoke-test one existing authenticated GET endpoint when the API has authenticated reads.
- [ ] 9.5 Fix every host API regression before treating Fervis setup as complete.

## Step 10: Ask a factual question through Fervis.
- [ ] 10.1 Run `fervis runtime ask "How many records are available?" --wait` after doctor succeeds.
- [ ] 10.2 Verify Fervis answers from the configured Flask API.
