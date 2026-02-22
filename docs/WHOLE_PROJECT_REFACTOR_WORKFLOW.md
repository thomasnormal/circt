# Whole-Project Refactor Workflow

This is the unified continuation workflow for the refactor workstream.

## Canonical Continue Prompt

Use this exact prompt when asking an agent to continue:

`continue according to docs/WHOLE_PROJECT_REFACTOR_PLAN.md, keeping track of progress using docs/WHOLE_PROJECT_REFACTOR_TODO.md`

## Unified Helper Command

Use `utils/refactor_continue.sh` to avoid repeating long instructions.

- Show status + next task + canonical prompt:
  `utils/refactor_continue.sh --status`
- Show one recommended next task only:
  `utils/refactor_continue.sh --next`
- Print canonical continue prompt only:
  `utils/refactor_continue.sh --prompt`

## Session Rules

1. Read `docs/WHOLE_PROJECT_REFACTOR_PLAN.md` and `docs/WHOLE_PROJECT_REFACTOR_TODO.md` first.
2. Consult `docs/WHOLE_PROJECT_OWNERSHIP_MAP.md` and `docs/WHOLE_PROJECT_MAINTENANCE_PLAYBOOK.md` for ownership + maintenance routing.
3. Prefer completing one behavior-preserving slice at a time.
4. Run focused validation for the touched domain (mutation/formal/sim).
5. Update `docs/WHOLE_PROJECT_REFACTOR_TODO.md` status in the same change.
6. Add a concise `CHANGELOG.md` entry with tests run and outcomes.
