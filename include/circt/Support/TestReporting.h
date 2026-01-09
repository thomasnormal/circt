//===- TestReporting.h - Test result reporting utilities ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for test result reporting in various formats,
// including JUnit XML for CI/CD integration.
//
// Example JUnit XML output:
//
// ```xml
// <?xml version="1.0" encoding="UTF-8"?>
// <testsuites name="circt-lint" tests="10" failures="2" errors="0" time="1.5">
//   <testsuite name="naming_convention" tests="5" failures="1" time="0.5">
//     <testcase name="module_name_check" classname="lint.naming" time="0.1">
//     </testcase>
//     <testcase name="signal_name_check" classname="lint.naming" time="0.2">
//       <failure message="Signal 'BadName' violates naming convention">
//         rtl/counter.sv:15:5: signal 'BadName' should match pattern ^[a-z][a-z0-9_]*$
//       </failure>
//     </testcase>
//   </testsuite>
// </testsuites>
// ```
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_TESTREPORTING_H
#define CIRCT_SUPPORT_TESTREPORTING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace circt {

//===----------------------------------------------------------------------===//
// Test Result Types
//===----------------------------------------------------------------------===//

/// Represents the outcome of a single test case.
enum class TestOutcome {
  Passed,  /// Test passed successfully
  Failed,  /// Test failed with an assertion failure
  Error,   /// Test encountered an unexpected error
  Skipped, /// Test was skipped
};

//===----------------------------------------------------------------------===//
// TestCase
//===----------------------------------------------------------------------===//

/// Represents a single test case result.
struct TestCase {
  /// The name of the test case.
  std::string name;

  /// The class/category name (used for grouping in reports).
  std::string className;

  /// The file where the test case is defined/run.
  std::string file;

  /// The line number in the file.
  unsigned line = 0;

  /// Test execution time in seconds.
  double timeSeconds = 0.0;

  /// The outcome of the test.
  TestOutcome outcome = TestOutcome::Passed;

  /// Failure/error message (if applicable).
  std::string message;

  /// Detailed failure/error output.
  std::string output;

  /// Type of failure (e.g., "AssertionError", "CompileError").
  std::string failureType;

  /// Standard output captured during test execution.
  std::string stdout;

  /// Standard error captured during test execution.
  std::string stderr;

  TestCase() = default;
  TestCase(llvm::StringRef name, llvm::StringRef className = "")
      : name(name.str()), className(className.str()) {}

  /// Mark the test as passed.
  void pass() { outcome = TestOutcome::Passed; }

  /// Mark the test as failed with a message.
  void fail(llvm::StringRef msg, llvm::StringRef detail = "",
            llvm::StringRef type = "AssertionError");

  /// Mark the test as errored.
  void error(llvm::StringRef msg, llvm::StringRef detail = "",
             llvm::StringRef type = "Error");

  /// Mark the test as skipped.
  void skip(llvm::StringRef reason = "");

  /// Check if the test passed.
  bool passed() const { return outcome == TestOutcome::Passed; }

  /// Check if the test failed.
  bool failed() const { return outcome == TestOutcome::Failed; }

  /// Check if the test errored.
  bool errored() const { return outcome == TestOutcome::Error; }

  /// Check if the test was skipped.
  bool skipped() const { return outcome == TestOutcome::Skipped; }
};

//===----------------------------------------------------------------------===//
// TestSuite
//===----------------------------------------------------------------------===//

/// Represents a collection of related test cases.
class TestSuite {
public:
  TestSuite(llvm::StringRef name);

  /// Get the suite name.
  llvm::StringRef getName() const { return name; }

  /// Add a test case to the suite.
  TestCase &addTestCase(llvm::StringRef name, llvm::StringRef className = "");

  /// Add an existing test case.
  void addTestCase(TestCase testCase);

  /// Get all test cases.
  llvm::ArrayRef<TestCase> getTestCases() const { return testCases; }

  /// Get the total number of tests.
  size_t getTestCount() const { return testCases.size(); }

  /// Get the number of passed tests.
  size_t getPassedCount() const;

  /// Get the number of failed tests.
  size_t getFailedCount() const;

  /// Get the number of errored tests.
  size_t getErrorCount() const;

  /// Get the number of skipped tests.
  size_t getSkippedCount() const;

  /// Get the total execution time.
  double getTotalTime() const;

  /// Set the suite-level timestamp.
  void setTimestamp(llvm::StringRef ts) { timestamp = ts.str(); }

  /// Get the timestamp.
  llvm::StringRef getTimestamp() const { return timestamp; }

  /// Set suite properties.
  void setProperty(llvm::StringRef key, llvm::StringRef value);

  /// Get suite properties.
  const llvm::SmallVector<std::pair<std::string, std::string>> &
  getProperties() const {
    return properties;
  }

private:
  std::string name;
  std::vector<TestCase> testCases;
  std::string timestamp;
  llvm::SmallVector<std::pair<std::string, std::string>> properties;
};

//===----------------------------------------------------------------------===//
// TestReport
//===----------------------------------------------------------------------===//

/// Represents a complete test report containing multiple test suites.
class TestReport {
public:
  TestReport(llvm::StringRef name = "");

  /// Get the report name.
  llvm::StringRef getName() const { return name; }

  /// Set the report name.
  void setName(llvm::StringRef n) { name = n.str(); }

  /// Add a test suite.
  TestSuite &addSuite(llvm::StringRef name);

  /// Add an existing test suite.
  void addSuite(TestSuite suite);

  /// Get all test suites.
  llvm::ArrayRef<TestSuite> getSuites() const { return suites; }

  /// Get the total number of tests across all suites.
  size_t getTotalTestCount() const;

  /// Get the total number of failures across all suites.
  size_t getTotalFailureCount() const;

  /// Get the total number of errors across all suites.
  size_t getTotalErrorCount() const;

  /// Get the total number of skipped tests.
  size_t getTotalSkippedCount() const;

  /// Get the total execution time.
  double getTotalTime() const;

  /// Check if all tests passed.
  bool allPassed() const {
    return getTotalFailureCount() == 0 && getTotalErrorCount() == 0;
  }

private:
  std::string name;
  std::vector<TestSuite> suites;
};

//===----------------------------------------------------------------------===//
// Test Report Writers
//===----------------------------------------------------------------------===//

/// Write a test report in JUnit XML format.
void writeJUnitXML(llvm::raw_ostream &os, const TestReport &report);

/// Write a test report in JUnit XML format to a file.
llvm::Error writeJUnitXMLFile(llvm::StringRef filename,
                               const TestReport &report);

/// Write a test report in JSON format (for custom tooling).
void writeTestReportJSON(llvm::raw_ostream &os, const TestReport &report);

/// Write a test report as plain text (human-readable).
void writeTestReportText(llvm::raw_ostream &os, const TestReport &report,
                         bool verbose = false);

//===----------------------------------------------------------------------===//
// Test Timer Utility
//===----------------------------------------------------------------------===//

/// Simple RAII timer for measuring test execution time.
class TestTimer {
public:
  TestTimer();

  /// Get elapsed time in seconds.
  double getElapsedSeconds() const;

  /// Stop the timer and return elapsed time.
  double stop();

  /// Reset the timer.
  void reset();

private:
  std::chrono::high_resolution_clock::time_point startTime;
  std::optional<std::chrono::high_resolution_clock::time_point> endTime;
};

//===----------------------------------------------------------------------===//
// Lint Result to Test Report Conversion
//===----------------------------------------------------------------------===//

/// Convert lint diagnostics to a test report.
/// Each lint rule becomes a test case, and violations become failures.
TestReport createLintTestReport(llvm::StringRef reportName);

} // namespace circt

#endif // CIRCT_SUPPORT_TESTREPORTING_H
