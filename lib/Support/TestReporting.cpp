//===- TestReporting.cpp - Test result reporting utilities ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/TestReporting.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <ctime>
#include <iomanip>
#include <sstream>

using namespace circt;

//===----------------------------------------------------------------------===//
// TestCase Implementation
//===----------------------------------------------------------------------===//

void TestCase::fail(llvm::StringRef msg, llvm::StringRef detail,
                    llvm::StringRef type) {
  outcome = TestOutcome::Failed;
  message = msg.str();
  output = detail.str();
  failureType = type.str();
}

void TestCase::error(llvm::StringRef msg, llvm::StringRef detail,
                     llvm::StringRef type) {
  outcome = TestOutcome::Error;
  message = msg.str();
  output = detail.str();
  failureType = type.str();
}

void TestCase::skip(llvm::StringRef reason) {
  outcome = TestOutcome::Skipped;
  message = reason.str();
}

//===----------------------------------------------------------------------===//
// TestSuite Implementation
//===----------------------------------------------------------------------===//

TestSuite::TestSuite(llvm::StringRef name) : name(name.str()) {
  // Set default timestamp to current time
  auto now = std::time(nullptr);
  std::ostringstream oss;
  oss << std::put_time(std::gmtime(&now), "%Y-%m-%dT%H:%M:%S");
  timestamp = oss.str();
}

TestCase &TestSuite::addTestCase(llvm::StringRef name,
                                 llvm::StringRef className) {
  testCases.emplace_back(name, className);
  return testCases.back();
}

void TestSuite::addTestCase(TestCase testCase) {
  testCases.push_back(std::move(testCase));
}

size_t TestSuite::getPassedCount() const {
  size_t count = 0;
  for (const auto &tc : testCases)
    if (tc.passed())
      count++;
  return count;
}

size_t TestSuite::getFailedCount() const {
  size_t count = 0;
  for (const auto &tc : testCases)
    if (tc.failed())
      count++;
  return count;
}

size_t TestSuite::getErrorCount() const {
  size_t count = 0;
  for (const auto &tc : testCases)
    if (tc.errored())
      count++;
  return count;
}

size_t TestSuite::getSkippedCount() const {
  size_t count = 0;
  for (const auto &tc : testCases)
    if (tc.skipped())
      count++;
  return count;
}

double TestSuite::getTotalTime() const {
  double total = 0.0;
  for (const auto &tc : testCases)
    total += tc.timeSeconds;
  return total;
}

void TestSuite::setProperty(llvm::StringRef key, llvm::StringRef value) {
  properties.emplace_back(key.str(), value.str());
}

//===----------------------------------------------------------------------===//
// TestReport Implementation
//===----------------------------------------------------------------------===//

TestReport::TestReport(llvm::StringRef name) : name(name.str()) {}

TestSuite &TestReport::addSuite(llvm::StringRef name) {
  suites.emplace_back(name);
  return suites.back();
}

void TestReport::addSuite(TestSuite suite) {
  suites.push_back(std::move(suite));
}

size_t TestReport::getTotalTestCount() const {
  size_t total = 0;
  for (const auto &suite : suites)
    total += suite.getTestCount();
  return total;
}

size_t TestReport::getTotalFailureCount() const {
  size_t total = 0;
  for (const auto &suite : suites)
    total += suite.getFailedCount();
  return total;
}

size_t TestReport::getTotalErrorCount() const {
  size_t total = 0;
  for (const auto &suite : suites)
    total += suite.getErrorCount();
  return total;
}

size_t TestReport::getTotalSkippedCount() const {
  size_t total = 0;
  for (const auto &suite : suites)
    total += suite.getSkippedCount();
  return total;
}

double TestReport::getTotalTime() const {
  double total = 0.0;
  for (const auto &suite : suites)
    total += suite.getTotalTime();
  return total;
}

//===----------------------------------------------------------------------===//
// XML Escaping Utility
//===----------------------------------------------------------------------===//

namespace {

/// Escape special characters for XML output.
std::string escapeXML(llvm::StringRef str) {
  std::string result;
  result.reserve(str.size());

  for (char c : str) {
    switch (c) {
    case '&':
      result += "&amp;";
      break;
    case '<':
      result += "&lt;";
      break;
    case '>':
      result += "&gt;";
      break;
    case '"':
      result += "&quot;";
      break;
    case '\'':
      result += "&apos;";
      break;
    default:
      // Filter out control characters except for common whitespace
      if (c >= 0x20 || c == '\n' || c == '\r' || c == '\t')
        result += c;
      break;
    }
  }

  return result;
}

/// Format a double with fixed precision.
std::string formatTime(double seconds) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3) << seconds;
  return oss.str();
}

} // namespace

//===----------------------------------------------------------------------===//
// JUnit XML Writer
//===----------------------------------------------------------------------===//

void circt::writeJUnitXML(llvm::raw_ostream &os, const TestReport &report) {
  os << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";

  // Write testsuites element
  os << "<testsuites";
  if (!report.getName().empty())
    os << " name=\"" << escapeXML(report.getName()) << "\"";
  os << " tests=\"" << report.getTotalTestCount() << "\"";
  os << " failures=\"" << report.getTotalFailureCount() << "\"";
  os << " errors=\"" << report.getTotalErrorCount() << "\"";
  os << " skipped=\"" << report.getTotalSkippedCount() << "\"";
  os << " time=\"" << formatTime(report.getTotalTime()) << "\"";
  os << ">\n";

  // Write each test suite
  for (const auto &suite : report.getSuites()) {
    os << "  <testsuite";
    os << " name=\"" << escapeXML(suite.getName()) << "\"";
    os << " tests=\"" << suite.getTestCount() << "\"";
    os << " failures=\"" << suite.getFailedCount() << "\"";
    os << " errors=\"" << suite.getErrorCount() << "\"";
    os << " skipped=\"" << suite.getSkippedCount() << "\"";
    os << " time=\"" << formatTime(suite.getTotalTime()) << "\"";
    if (!suite.getTimestamp().empty())
      os << " timestamp=\"" << escapeXML(suite.getTimestamp()) << "\"";
    os << ">\n";

    // Write properties if any
    const auto &props = suite.getProperties();
    if (!props.empty()) {
      os << "    <properties>\n";
      for (const auto &prop : props) {
        os << "      <property name=\"" << escapeXML(prop.first) << "\" value=\""
           << escapeXML(prop.second) << "\"/>\n";
      }
      os << "    </properties>\n";
    }

    // Write test cases
    for (const auto &tc : suite.getTestCases()) {
      os << "    <testcase";
      os << " name=\"" << escapeXML(tc.name) << "\"";
      if (!tc.className.empty())
        os << " classname=\"" << escapeXML(tc.className) << "\"";
      os << " time=\"" << formatTime(tc.timeSeconds) << "\"";

      if (tc.passed() && tc.stdout.empty() && tc.stderr.empty()) {
        // Simple passed test with no output
        os << "/>\n";
      } else {
        os << ">\n";

        // Write failure/error/skipped element
        if (tc.failed()) {
          os << "      <failure";
          if (!tc.failureType.empty())
            os << " type=\"" << escapeXML(tc.failureType) << "\"";
          if (!tc.message.empty())
            os << " message=\"" << escapeXML(tc.message) << "\"";
          os << ">";
          if (!tc.output.empty())
            os << escapeXML(tc.output);
          os << "</failure>\n";
        } else if (tc.errored()) {
          os << "      <error";
          if (!tc.failureType.empty())
            os << " type=\"" << escapeXML(tc.failureType) << "\"";
          if (!tc.message.empty())
            os << " message=\"" << escapeXML(tc.message) << "\"";
          os << ">";
          if (!tc.output.empty())
            os << escapeXML(tc.output);
          os << "</error>\n";
        } else if (tc.skipped()) {
          os << "      <skipped";
          if (!tc.message.empty())
            os << " message=\"" << escapeXML(tc.message) << "\"";
          os << "/>\n";
        }

        // Write stdout/stderr if present
        if (!tc.stdout.empty()) {
          os << "      <system-out>" << escapeXML(tc.stdout)
             << "</system-out>\n";
        }
        if (!tc.stderr.empty()) {
          os << "      <system-err>" << escapeXML(tc.stderr)
             << "</system-err>\n";
        }

        os << "    </testcase>\n";
      }
    }

    os << "  </testsuite>\n";
  }

  os << "</testsuites>\n";
}

llvm::Error circt::writeJUnitXMLFile(llvm::StringRef filename,
                                     const TestReport &report) {
  std::error_code ec;
  llvm::raw_fd_ostream os(filename, ec);
  if (ec)
    return llvm::createStringError(ec, "failed to open file: %s",
                                   filename.str().c_str());

  writeJUnitXML(os, report);
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// JSON Writer
//===----------------------------------------------------------------------===//

void circt::writeTestReportJSON(llvm::raw_ostream &os,
                                const TestReport &report) {
  llvm::json::Object root;
  root["name"] = report.getName();
  root["tests"] = static_cast<int64_t>(report.getTotalTestCount());
  root["failures"] = static_cast<int64_t>(report.getTotalFailureCount());
  root["errors"] = static_cast<int64_t>(report.getTotalErrorCount());
  root["skipped"] = static_cast<int64_t>(report.getTotalSkippedCount());
  root["time"] = report.getTotalTime();

  llvm::json::Array suitesArr;
  for (const auto &suite : report.getSuites()) {
    llvm::json::Object suiteObj;
    suiteObj["name"] = suite.getName();
    suiteObj["tests"] = static_cast<int64_t>(suite.getTestCount());
    suiteObj["failures"] = static_cast<int64_t>(suite.getFailedCount());
    suiteObj["errors"] = static_cast<int64_t>(suite.getErrorCount());
    suiteObj["skipped"] = static_cast<int64_t>(suite.getSkippedCount());
    suiteObj["time"] = suite.getTotalTime();

    llvm::json::Array casesArr;
    for (const auto &tc : suite.getTestCases()) {
      llvm::json::Object tcObj;
      tcObj["name"] = tc.name;
      if (!tc.className.empty())
        tcObj["classname"] = tc.className;
      tcObj["time"] = tc.timeSeconds;

      switch (tc.outcome) {
      case TestOutcome::Passed:
        tcObj["status"] = "passed";
        break;
      case TestOutcome::Failed:
        tcObj["status"] = "failed";
        if (!tc.message.empty())
          tcObj["message"] = tc.message;
        if (!tc.output.empty())
          tcObj["output"] = tc.output;
        break;
      case TestOutcome::Error:
        tcObj["status"] = "error";
        if (!tc.message.empty())
          tcObj["message"] = tc.message;
        if (!tc.output.empty())
          tcObj["output"] = tc.output;
        break;
      case TestOutcome::Skipped:
        tcObj["status"] = "skipped";
        if (!tc.message.empty())
          tcObj["reason"] = tc.message;
        break;
      }

      if (!tc.file.empty()) {
        tcObj["file"] = tc.file;
        tcObj["line"] = static_cast<int64_t>(tc.line);
      }

      casesArr.push_back(std::move(tcObj));
    }
    suiteObj["testcases"] = std::move(casesArr);

    suitesArr.push_back(std::move(suiteObj));
  }
  root["testsuites"] = std::move(suitesArr);

  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
}

//===----------------------------------------------------------------------===//
// Text Writer
//===----------------------------------------------------------------------===//

void circt::writeTestReportText(llvm::raw_ostream &os, const TestReport &report,
                                bool verbose) {
  os << "Test Report: " << report.getName() << "\n";
  os << "============================================================\n\n";

  size_t totalTests = report.getTotalTestCount();
  size_t totalPassed = totalTests - report.getTotalFailureCount() -
                       report.getTotalErrorCount() -
                       report.getTotalSkippedCount();

  os << "Summary:\n";
  os << "  Total:   " << totalTests << "\n";
  os << "  Passed:  " << totalPassed << "\n";
  os << "  Failed:  " << report.getTotalFailureCount() << "\n";
  os << "  Errors:  " << report.getTotalErrorCount() << "\n";
  os << "  Skipped: " << report.getTotalSkippedCount() << "\n";
  os << "  Time:    " << formatTime(report.getTotalTime()) << "s\n\n";

  for (const auto &suite : report.getSuites()) {
    os << "Suite: " << suite.getName() << "\n";
    os << "------------------------------------------------------------\n";

    for (const auto &tc : suite.getTestCases()) {
      switch (tc.outcome) {
      case TestOutcome::Passed:
        os << "  [PASS] ";
        break;
      case TestOutcome::Failed:
        os << "  [FAIL] ";
        break;
      case TestOutcome::Error:
        os << "  [ERR]  ";
        break;
      case TestOutcome::Skipped:
        os << "  [SKIP] ";
        break;
      }

      os << tc.name;
      if (!tc.className.empty())
        os << " (" << tc.className << ")";
      os << " [" << formatTime(tc.timeSeconds) << "s]\n";

      if (verbose || !tc.passed()) {
        if (!tc.message.empty())
          os << "         Message: " << tc.message << "\n";
        if (!tc.output.empty())
          os << "         Details: " << tc.output << "\n";
        if (!tc.file.empty())
          os << "         Location: " << tc.file << ":" << tc.line << "\n";
      }
    }
    os << "\n";
  }

  // Final result
  if (report.allPassed()) {
    os << "Result: ALL TESTS PASSED\n";
  } else {
    os << "Result: TESTS FAILED\n";
  }
}

//===----------------------------------------------------------------------===//
// TestTimer Implementation
//===----------------------------------------------------------------------===//

TestTimer::TestTimer()
    : startTime(std::chrono::high_resolution_clock::now()), endTime() {}

double TestTimer::getElapsedSeconds() const {
  auto end = endTime.value_or(std::chrono::high_resolution_clock::now());
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end - startTime);
  return duration.count() / 1000000.0;
}

double TestTimer::stop() {
  if (!endTime)
    endTime = std::chrono::high_resolution_clock::now();
  return getElapsedSeconds();
}

void TestTimer::reset() {
  startTime = std::chrono::high_resolution_clock::now();
  endTime = std::nullopt;
}

//===----------------------------------------------------------------------===//
// Lint Test Report Creation
//===----------------------------------------------------------------------===//

TestReport circt::createLintTestReport(llvm::StringRef reportName) {
  TestReport report(reportName);
  // This is a placeholder - actual implementation would integrate
  // with the lint infrastructure to create test cases from lint results
  return report;
}
