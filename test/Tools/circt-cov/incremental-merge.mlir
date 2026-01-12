// RUN: rm -rf %t && mkdir -p %t

// Create initial coverage database
// RUN: echo '{"version": 1, "coverage_points": [{"name": "point1", "type": "line", "hits": 3, "goal": 1, "location": {"filename": "test.v", "line": 10, "column": 0}, "hierarchy": "top", "description": ""}], "groups": [], "exclusions": [], "trends": [], "summary": {"total_points": 1, "covered_points": 1, "overall_coverage": 100.0}}' > %t/run1.json

// Create second coverage run
// RUN: echo '{"version": 1, "coverage_points": [{"name": "point1", "type": "line", "hits": 2, "goal": 1, "location": {"filename": "test.v", "line": 10, "column": 0}, "hierarchy": "top", "description": ""}, {"name": "point2", "type": "line", "hits": 5, "goal": 1, "location": {"filename": "test.v", "line": 20, "column": 0}, "hierarchy": "top", "description": ""}], "groups": [], "exclusions": [], "trends": [], "summary": {"total_points": 2, "covered_points": 2, "overall_coverage": 100.0}}' > %t/run2.json

// Test incremental merge - first run creates new file
// RUN: circt-cov --merge --incremental %t/run1.json -o %t/merged.json 2>&1 | FileCheck %s --check-prefix=INCR-FIRST
// INCR-FIRST: Incrementally merged 1 database(s)
// INCR-FIRST: Total coverage points: 1

// Test incremental merge - second run adds to existing
// RUN: circt-cov --merge --incremental %t/run2.json -o %t/merged.json 2>&1 | FileCheck %s --check-prefix=INCR-SECOND
// INCR-SECOND: Incrementally merged 1 database(s)
// INCR-SECOND: Total coverage points: 2

// Verify merged content
// RUN: circt-cov --report %t/merged.json --format=text 2>&1 | FileCheck %s --check-prefix=MERGED
// MERGED: Total Points:{{.*}}2

// Create third run with same points
// RUN: echo '{"version": 1, "coverage_points": [{"name": "point1", "type": "line", "hits": 1, "goal": 1, "location": {"filename": "test.v", "line": 10, "column": 0}, "hierarchy": "top", "description": ""}], "groups": [], "exclusions": [], "trends": [], "summary": {"total_points": 1, "covered_points": 1, "overall_coverage": 100.0}}' > %t/run3.json

// Test incremental merge accumulates hits
// RUN: circt-cov --merge --incremental %t/run3.json -o %t/merged.json -v 2>&1 | FileCheck %s --check-prefix=INCR-ACCUM
// INCR-ACCUM: Loaded existing database
// INCR-ACCUM: Incrementally merged 1 database(s)

// Test standard merge still works
// RUN: circt-cov --merge %t/run1.json %t/run2.json -o %t/standard.json 2>&1 | FileCheck %s --check-prefix=STD-MERGE
// STD-MERGE: Merged 2 databases
// STD-MERGE: Total coverage points: 2
