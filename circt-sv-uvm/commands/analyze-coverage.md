---
name: analyze-coverage
description: Analyze functional coverage results from simulation
argument-hint: "<coverage_database_or_report>"
allowed-tools:
  - Bash
  - Read
  - Glob
  - Grep
---

# Coverage Analysis Command

Analyze functional coverage results from UVM simulations to identify coverage gaps and suggest improvements.

## Instructions

1. **Identify coverage data format**:
   - UCDB (Questa/ModelSim)
   - VDB (VCS)
   - IMC database (Cadence)
   - Text/HTML coverage reports
   - Merged coverage files

2. **For text/HTML reports**, read and parse:
   - Overall coverage percentage
   - Covergroup/coverpoint breakdown
   - Cross coverage
   - Assertion coverage
   - Code coverage (line, branch, toggle, FSM)

3. **Identify coverage holes**:
   - Coverpoints at 0% or low coverage
   - Missing cross coverage bins
   - Uncovered assertions
   - Unreached code branches

4. **Provide analysis**:
   - Summary statistics (total %, by category)
   - Top 10 uncovered items
   - Recommendations for improving coverage
   - Potential test scenarios to add

5. **For tool-specific databases**, suggest commands:
   ```bash
   # Questa
   vcover report -details <ucdb>

   # VCS
   urg -dir <vdb> -report coverage

   # Xcelium
   imc -load <database> -report
   ```

## Example Usage

```
/analyze-coverage coverage.txt
/analyze-coverage sim_results/coverage/
/analyze-coverage merged_cov.ucdb
```

## Coverage Improvement Suggestions

When coverage is low, suggest:
- New sequences to hit uncovered bins
- Constraint modifications for constrained-random
- Directed tests for corner cases
- Cross coverage simplification if too sparse
