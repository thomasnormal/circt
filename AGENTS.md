# Repository Guidelines

## Project Structure & Module Organization
- `include/` and `lib/` hold core CIRCT C++ headers and implementations.
- `tools/` contains binaries such as `circt-verilog`, `circt-opt`, and `circt-bmc`.
- `frontends/` hosts language frontends and import pipelines.
- `test/` contains FileCheck-based regression tests grouped by dialect/feature.
- `unittests/` contains GoogleTest-based unit tests for runtime and utilities.
- `docs/`, `README.md`, and `PROJECT_PLAN.md` document design and direction.
- `build/` is the default out-of-tree build directory.

## Build, Test, and Development Commands
- Configure: follow `README.md`; typical LLVM-style setup is `cmake -G Ninja -B build -S llvm/llvm`.
- Build tools: `ninja -C build circt-verilog` or `ninja -C build circt-opt`.
- Regression tests: `ninja -C build check-circt`.
- Unit tests: `ninja -C build check-circt-unittests` or `ctest --test-dir build`.
- For local integration suites (if available): `~/mbit/*avip*`, `~/sv-tests/`.

## Coding Style & Naming Conventions
- Follow LLVM/CIRCT style: 2-space indentation, K&R braces, descriptive identifiers.
- Keep TableGen (`.td`) patterns consistent with nearby files.
- Name tests by feature area, e.g. `test/Conversion/ImportVerilog/<topic>.sv`.
- Use ASCII-only text unless the file already contains Unicode.

## Testing Guidelines
- Add a regression test in `test/` for functional changes; use `// CHECK:` patterns.
- Add or update `unittests/` for library/runtime behavior.
- Keep tests minimal and focused; prefer a new test over extending an unrelated one.

## Commit & Pull Request Guidelines
- No strict format, but use short, specific subjects and optional subsystem tags (e.g., `[ImportVerilog]`).
- Include rationale, test commands + outcomes, and issue links where applicable.
- Keep changes scoped; split unrelated work across commits/PRs.

## Project-Specific Notes
- Update `CHANGELOG.md` for significant features or fixes.
- Merge regularly with upstream/main to keep worktrees aligned.
