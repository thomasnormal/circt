// RUN: rm -rf %t && mkdir -p %t

// Create first test coverage database in JSON format
// RUN: echo '{"version": 1, "coverage_points": [{"name": "test.v:10", "type": "line", "hits": 5, "goal": 1, "location": {"filename": "test.v", "line": 10, "column": 0}, "hierarchy": "top", "description": ""}], "groups": [], "exclusions": [], "trends": [], "summary": {"total_points": 1, "covered_points": 1, "overall_coverage": 100.0, "line_coverage": 100.0, "toggle_coverage": 100.0, "branch_coverage": 100.0}}' > %t/run1.json

// Create second test coverage database in JSON format
// RUN: echo '{"version": 1, "coverage_points": [{"name": "test.v:10", "type": "line", "hits": 3, "goal": 1, "location": {"filename": "test.v", "line": 10, "column": 0}, "hierarchy": "top", "description": ""}, {"name": "test.v:20", "type": "line", "hits": 0, "goal": 1, "location": {"filename": "test.v", "line": 20, "column": 0}, "hierarchy": "top", "description": ""}], "groups": [], "exclusions": [], "trends": [], "summary": {"total_points": 2, "covered_points": 1, "overall_coverage": 50.0, "line_coverage": 50.0, "toggle_coverage": 100.0, "branch_coverage": 100.0}}' > %t/run2.json

// Test merge command
// RUN: circt-cov --merge %t/run1.json %t/run2.json -o %t/merged.json 2>&1 | FileCheck %s --check-prefix=MERGE
// MERGE: Merged 2 databases
// MERGE: Total coverage points: 2

// Test report command (text format)
// RUN: circt-cov --report %t/merged.json --format=text 2>&1 | FileCheck %s --check-prefix=REPORT
// REPORT: COVERAGE REPORT
// REPORT: SUMMARY
// REPORT: Overall Coverage:

// Test report command with verbose
// RUN: circt-cov --report %t/merged.json --format=text -v 2>&1 | FileCheck %s --check-prefix=VERBOSE
// VERBOSE: UNCOVERED POINTS
// VERBOSE: test.v:20

// Test report command (HTML format)
// RUN: circt-cov --report %t/merged.json --format=html -o %t/report.html 2>&1
// RUN: cat %t/report.html | FileCheck %s --check-prefix=HTML
// HTML: <!DOCTYPE html>
// HTML: CIRCT Coverage Report
// HTML: Coverage by Type

// Test convert command
// RUN: circt-cov --convert %t/merged.json -o %t/converted.cov --format=binary 2>&1 | FileCheck %s --check-prefix=CONVERT
// CONVERT: Converted
