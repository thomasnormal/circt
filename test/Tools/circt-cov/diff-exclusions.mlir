// RUN: rm -rf %t && mkdir -p %t

// Create old coverage database (baseline)
// RUN: echo '{"version": 1, "coverage_points": [{"name": "point1", "type": "line", "hits": 1, "goal": 1, "location": {"filename": "test.v", "line": 10, "column": 0}, "hierarchy": "top", "description": ""}, {"name": "point2", "type": "line", "hits": 0, "goal": 1, "location": {"filename": "test.v", "line": 20, "column": 0}, "hierarchy": "top", "description": ""}, {"name": "point3", "type": "line", "hits": 1, "goal": 1, "location": {"filename": "test.v", "line": 30, "column": 0}, "hierarchy": "top", "description": ""}], "groups": [], "exclusions": [], "trends": [], "summary": {"total_points": 3, "covered_points": 2, "overall_coverage": 66.67}}' > %t/old.json

// Create new coverage database (current)
// RUN: echo '{"version": 1, "coverage_points": [{"name": "point1", "type": "line", "hits": 1, "goal": 1, "location": {"filename": "test.v", "line": 10, "column": 0}, "hierarchy": "top", "description": ""}, {"name": "point2", "type": "line", "hits": 5, "goal": 1, "location": {"filename": "test.v", "line": 20, "column": 0}, "hierarchy": "top", "description": ""}, {"name": "point4", "type": "line", "hits": 1, "goal": 1, "location": {"filename": "test.v", "line": 40, "column": 0}, "hierarchy": "top", "description": ""}], "groups": [], "exclusions": [], "trends": [], "summary": {"total_points": 3, "covered_points": 3, "overall_coverage": 100.0}}' > %t/new.json

// Test diff command - shows newly covered points
// RUN: circt-cov --diff %t/new.json %t/old.json 2>&1 | FileCheck %s --check-prefix=DIFF
// DIFF: Coverage Comparison
// DIFF: Coverage delta: +
// DIFF: Newly covered (1):
// DIFF: point2

// Test diff with verbose - shows only in each database
// RUN: circt-cov --diff %t/new.json %t/old.json -v 2>&1 | FileCheck %s --check-prefix=DIFF-VERBOSE
// DIFF-VERBOSE: Only in {{.*}}new.json{{.*}}:
// DIFF-VERBOSE: point4
// DIFF-VERBOSE: Only in {{.*}}old.json{{.*}}:
// DIFF-VERBOSE: point3

// Create exclusions file
// RUN: echo '[{"name": "point2", "reason": "Known false positive", "author": "test", "date": "2024-01-15", "ticket": "BUG-123"}]' > %t/exclusions.json

// Test applying exclusions
// RUN: circt-cov --exclude %t/old.json --exclusions=%t/exclusions.json -o %t/excluded.json 2>&1 | FileCheck %s --check-prefix=EXCLUDE
// EXCLUDE: Applied 1 exclusions

// Verify exclusion is in the output
// RUN: circt-cov --report %t/excluded.json --format=text 2>&1 | FileCheck %s --check-prefix=EXCLUDE-REPORT
// EXCLUDE-REPORT: EXCLUSIONS
// EXCLUDE-REPORT: point2
// EXCLUDE-REPORT: Known false positive

// Test exclusions with report command
// RUN: circt-cov --report %t/old.json --exclusions=%t/exclusions.json --format=text 2>&1 | FileCheck %s --check-prefix=REPORT-EXCL
// REPORT-EXCL: EXCLUSIONS
// REPORT-EXCL: Known false positive
