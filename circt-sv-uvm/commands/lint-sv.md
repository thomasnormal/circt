---
name: lint-sv
description: Run SystemVerilog linting and static analysis on files
argument-hint: "<file_or_directory>"
allowed-tools:
  - Bash
  - Read
  - Glob
---

# SystemVerilog Linting Command

Perform static analysis and linting on SystemVerilog files to identify code quality issues, potential bugs, and style violations.

## Instructions

1. **Identify target files**: If the user provides a file path, lint that file. If a directory, find all `.sv` and `.v` files recursively.

2. **Check for available linting tools** in this order:
   - `slang` (preferred - CIRCT's parser)
   - `verilator --lint-only`
   - `iverilog -t null` (basic syntax check)

3. **Run linting**:
   ```bash
   # For slang (if available)
   slang --lint-only <files>

   # For verilator
   verilator --lint-only -Wall <files>

   # For iverilog
   iverilog -t null -Wall <files>
   ```

4. **Parse and present results**:
   - Group errors and warnings by file
   - Highlight critical issues first
   - Suggest fixes for common problems

5. **Common issues to check**:
   - Undriven signals
   - Width mismatches
   - Unused variables/parameters
   - Missing case items
   - Blocking vs non-blocking assignment misuse
   - Clock domain crossing issues (if detectable)

## Example Usage

```
/lint-sv src/rtl/
/lint-sv my_module.sv
/lint-sv .  (lint current directory)
```

## Output Format

Present results grouped by severity:
- **ERRORS**: Must be fixed for compilation
- **WARNINGS**: Potential issues to review
- **INFO**: Style suggestions and notes
