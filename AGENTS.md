# AGENTS.md

Guidance for Codex and other AI coding agents working in this repository.

## Project Identity

This repository is **Pharma-Agent / Pharma-RAG Assistant**.

It is a **biomedical AI agent prototype** that combines local retrieval-augmented generation, a FastAPI service layer, a LangGraph-based agent workflow, PubMed retrieval, and limited data-analysis tooling for mock clinical data.

Do not describe this project as clinical-grade, production-ready, validated medical software, or suitable for direct clinical decision-making. Keep capability descriptions accurate to the current source code.

## Core Files

Important files and their expected roles:

- `README.md`: Public project overview, quick start notes, and API usage examples.
- `api.py`: FastAPI application exposing the `/agent/chat` endpoint and static `clinical_data` mount.
- `agent_engine.py`: LangGraph agent orchestration, tool definitions, PubMed lookup, CSV preview, and Python execution sandbox.
- `core.py`: Core biomedical RAG assistant using DeepSeek-compatible chat completion, BGE-M3 embeddings, and ChromaDB.
- `app.py`: Legacy Gradio UI for streaming RAG chat.
- `main.py`: CLI-style evaluation entry point for building or querying the local vector database.
- `generate_mock_data.py`: Utility for generating mock clinical data used by the analytics workflow.
- `requirements.txt`: Python dependency pin list.
- `Dockerfile`: Container build and FastAPI startup definition.
- `.gitignore`: Repository ignore rules for secrets, generated databases, data folders, and generated clinical plots.

## Safety Rules

- Never read, print, summarize, expose, or modify `.env`.
- Never expose API keys, tokens, credentials, authorization headers, or auth-related files.
- Do not include real secrets or personal private information in code, documentation, logs, examples, commits, or generated files.
- Treat biomedical outputs as prototype research assistance only. Avoid wording that implies diagnosis, treatment recommendation, regulatory approval, or clinical validation.
- PubMed and external retrieval behavior may depend on network availability and API behavior; do not imply guaranteed real-time coverage unless the code and runtime environment support it.

## Repository Hygiene

Do not commit generated or local runtime artifacts:

- `chroma_db_bge_m3/`
- `chroma_db/`
- `data/`
- `clinical_data/*.png`

The mock CSV dataset may be kept:

- `clinical_data/lung_cancer_mock_data.csv`

Before changing ignore rules or generated-data handling, inspect `.gitignore` and keep these boundaries intact unless the user explicitly asks for a scoped change.

## Review Mode

Default review mode is read-only.

In review mode:

- Inspect only the files needed for the review.
- Report bugs, risks, behavioral regressions, and missing tests.
- Do not edit files.
- Do not run services, install dependencies, or perform git write operations.

## Modification Mode

When asked to modify the repository:

- Work in small, scoped steps.
- Limit edits to the files requested by the user.
- Do not opportunistically refactor unrelated code.
- Prefer existing project patterns over new abstractions.
- Keep documentation aligned with the implemented source behavior.
- If README capability claims drift beyond the code, make them more precise rather than more promotional.
- Do not run tests, services, dependency installation, or network commands unless the user explicitly permits them for the task.

## Git Rules

- Do not automatically run `git add`.
- Do not automatically run `git commit`.
- Do not automatically run `git push`.
- Do not run `git reset`, `git restore`, or `git clean` unless the user explicitly asks and confirms the intended scope.
- Use read-only git commands, such as `git status` and `git diff`, when useful.

## Command Policy

Allowed read-only commands:

- `dir`
- `Get-ChildItem`
- `Get-Content`
- `Select-String`
- `git status`
- `git diff`
- `git show`
- `git log`

Forbidden unless explicitly approved by the user:

- Any command that reads, prints, summarizes, modifies, or exposes `.env` or secret-bearing files.
- Network commands, package installation, dependency updates, or model downloads.
- Service startup commands, including FastAPI, Uvicorn, Gradio, Docker run, or Docker compose commands.
- Test execution commands.
- File deletion, cleanup, reset, restore, or bulk move commands.
- `git add`
- `git commit`
- `git push`
- `git reset`
- `git restore`
- `git clean`

## Windows Codex Notes

This project may be used with native Windows Codex.

- The Codex shell environment policy may use `inherit = all` while excluding common secret variables.
- Common secret variables to exclude include API keys, tokens, credentials, and auth headers.
- A PowerShell language mode warning can be treated as a non-blocking warning if it does not prevent normal file reading or repository inspection.
- Prefer PowerShell-native read-only commands when inspecting files on Windows.

## Documentation Standards

Documentation should stay conservative and source-aligned:

- Describe the project as a biomedical AI agent prototype.
- Avoid clinical-grade, production-ready, or enterprise-ready claims unless the repository contains direct supporting implementation and validation evidence.
- Keep examples free of real patient data, real secrets, and personal private information.
- Clearly distinguish mock clinical data from real clinical evidence.
