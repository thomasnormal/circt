// RUN: rm -rf %t && mkdir -p %t

// Create a test coverage database in JSON format with various coverage types
// RUN: echo '{"version": 1, "coverage_points": [{"name": "module.v:10", "type": "line", "hits": 5, "goal": 1, "location": {"filename": "module.v", "line": 10, "column": 0}, "hierarchy": "top.sub", "description": "covered line"}, {"name": "module.v:20", "type": "line", "hits": 0, "goal": 1, "location": {"filename": "module.v", "line": 20, "column": 0}, "hierarchy": "top.sub", "description": "uncovered line"}, {"name": "sig1", "type": "toggle", "hits": 0, "goal": 1, "location": {"filename": "", "line": 0, "column": 0}, "hierarchy": "top", "description": "", "toggle_01": true, "toggle_10": false}, {"name": "branch1", "type": "branch", "hits": 0, "goal": 1, "location": {"filename": "ctrl.v", "line": 15, "column": 0}, "hierarchy": "top.ctrl", "description": "", "branch_true": true, "branch_false": true}], "groups": [{"name": "group1", "description": "Test group", "points": ["module.v:10", "module.v:20"]}], "exclusions": [{"name": "excluded_pt", "reason": "Dead code", "author": "test", "date": "2024-01-01"}], "trends": [{"timestamp": "2024-01-01T00:00:00Z", "run_id": "run1", "commit_hash": "abc123", "line_coverage": 50.0, "toggle_coverage": 50.0, "branch_coverage": 100.0, "overall_coverage": 62.5, "total_points": 4, "covered_points": 2}], "summary": {"total_points": 4, "covered_points": 2, "overall_coverage": 62.5}}' > %t/test.json

// Test HTML report generation
// RUN: circt-cov --report %t/test.json --format=html -o %t/report.html
// RUN: cat %t/report.html | FileCheck %s --check-prefix=HTML

// Check basic HTML structure
// HTML: <!DOCTYPE html>
// HTML: CIRCT Coverage Report
// HTML: </html>

// Check summary section
// HTML: Overall Coverage
// HTML: Total Points
// HTML: Covered Points
// HTML: Uncovered Points

// Check coverage by type section
// HTML: Coverage by Type
// HTML: Line Coverage
// HTML: Toggle Coverage
// HTML: Branch Coverage

// Check hierarchy breakdown section
// HTML: Hierarchy Breakdown

// Check source file coverage section
// HTML: Source File Coverage
// HTML: module.v

// Check uncovered items section
// HTML: Uncovered Items

// Check coverage trends section (we have trend data)
// HTML: Coverage Trends

// Check exclusions section
// HTML: Exclusions

// Test HTML report with custom title
// RUN: circt-cov --report %t/test.json --format=html --title="My Custom Report" -o %t/custom.html
// RUN: cat %t/custom.html | FileCheck %s --check-prefix=CUSTOM-TITLE

// CUSTOM-TITLE: My Custom Report

// Test HTML report with uncovered-only filter
// RUN: circt-cov --report %t/test.json --format=html --uncovered-only -o %t/uncovered.html
// RUN: cat %t/uncovered.html | FileCheck %s --check-prefix=UNCOVERED

// UNCOVERED: Uncovered Items
// UNCOVERED: module.v:20

// Test HTML report with hierarchy filter
// RUN: circt-cov --report %t/test.json --format=html --hierarchy=top.sub -o %t/filtered.html
// RUN: cat %t/filtered.html | FileCheck %s --check-prefix=FILTERED

// FILTERED: module.v
