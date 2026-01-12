// RUN: rm -rf %t && mkdir -p %t

// Create a coverage database without trends
// RUN: echo '{"version": 1, "coverage_points": [{"name": "point1", "type": "line", "hits": 1, "goal": 1, "location": {"filename": "test.v", "line": 10, "column": 0}, "hierarchy": "top", "description": ""}, {"name": "point2", "type": "line", "hits": 0, "goal": 1, "location": {"filename": "test.v", "line": 20, "column": 0}, "hierarchy": "top", "description": ""}], "groups": [], "exclusions": [], "trends": [], "summary": {"total_points": 2, "covered_points": 1, "overall_coverage": 50.0}}' > %t/coverage.json

// Test adding a trend point
// RUN: circt-cov --trend %t/coverage.json --add-point --run-id=ci-build-123 --commit=abc123def -o %t/with-trend.json 2>&1 | FileCheck %s --check-prefix=ADD-TREND
// ADD-TREND: Added trend point
// ADD-TREND: Run ID: ci-build-123
// ADD-TREND: Commit: abc123def
// ADD-TREND: Overall:{{.*}}50.00%

// Test showing trend history
// RUN: circt-cov --trend %t/with-trend.json 2>&1 | FileCheck %s --check-prefix=SHOW-TREND
// SHOW-TREND: Coverage Trend History
// SHOW-TREND: ci-build-123
// SHOW-TREND: 50.00%

// Add another trend point
// RUN: echo '{"version": 1, "coverage_points": [{"name": "point1", "type": "line", "hits": 1, "goal": 1, "location": {"filename": "test.v", "line": 10, "column": 0}, "hierarchy": "top", "description": ""}, {"name": "point2", "type": "line", "hits": 3, "goal": 1, "location": {"filename": "test.v", "line": 20, "column": 0}, "hierarchy": "top", "description": ""}], "groups": [], "exclusions": [], "trends": [], "summary": {"total_points": 2, "covered_points": 2, "overall_coverage": 100.0}}' > %t/improved.json

// Test incremental trend tracking (merge new coverage and add trend)
// RUN: circt-cov --merge --incremental %t/improved.json -o %t/with-trend.json 2>&1
// RUN: circt-cov --trend %t/with-trend.json --add-point --run-id=ci-build-124 -o %t/with-trend.json 2>&1 | FileCheck %s --check-prefix=ADD-TREND2
// ADD-TREND2: Added trend point
// ADD-TREND2: Run ID: ci-build-124
// ADD-TREND2: Overall:{{.*}}100.00%

// Verify both trend points are shown
// RUN: circt-cov --trend %t/with-trend.json 2>&1 | FileCheck %s --check-prefix=MULTI-TREND
// MULTI-TREND: Coverage Trend History
// MULTI-TREND: ci-build-123
// MULTI-TREND: ci-build-124

// Test trend in HTML report
// RUN: circt-cov --report %t/with-trend.json --format=html --show-trends -o %t/trend-report.html
// RUN: cat %t/trend-report.html | FileCheck %s --check-prefix=HTML-TREND
// HTML-TREND: Coverage Trends
// HTML-TREND: ci-build-123
// HTML-TREND: ci-build-124
