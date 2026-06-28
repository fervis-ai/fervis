# Factual Answer Engine for your API

Fervis's goal is to provide a backend-owned deterministic runtime for answering factual questions via framework-native adapters.  

## How to install

Python projects should install Fervis with `uv`.

```bash
uv add fervis
```

Install the framework extra for your API:

```bash
uv add "fervis[django]"
uv add "fervis[fastapi]"
uv add "fervis[flask]"
```

During local development from this repository:

```bash
cd python
uv sync --extra django --extra fastapi --extra flask --extra dev
uv run fervis --help
```

## Tested Providers

We recommend using Openai's API for now as it seems to have the most reliable structured outputs with large enough size.  
We would recommend `gpt-5.4-mini` for testing.

## Production Usage

At the moment, we do NOT recommend using Fervis in production as it is in alpha. 


## License

Fervis is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
