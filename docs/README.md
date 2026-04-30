# Documentation Index

This folder contains project-level technical documentation for the local Document AI system.

## Document Metadata

- Author: Ali Salem
- Last Updated: 2026-04-30

## Read First

1. `architecture.md`  
   Full architecture, workflows, module breakdown, data flow, decision logic, limitations, and roadmap.

2. `prompts_log.md`  
   Prompt inventory with usage context for RAG QA, structured extraction, and structured-answer flows.

## Suggested Reading Path

1. Start with `architecture.md` sections:
   - Project Overview
   - System Architecture
   - Workflow
   - Decision Logic
2. Then review `prompts_log.md` to understand:
   - which prompt is used for each document type
   - when strict JSON extraction is triggered
   - how anti-hallucination behavior is enforced

## Scope

These docs reflect the current implementation in:

- `backend/`
- `ui/`
- `data/`
- `vectorstore/`

