//===- TestReportingTest.cpp - Unit tests for TestReporting ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/TestReporting.h"
#include "gtest/gtest.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;

namespace {

//===----------------------------------------------------------------------===//
// TestCase Tests
//===----------------------------------------------------------------------===//

TEST(TestCaseTest, DefaultState) {
  TestCase tc("test_name", "test_class");
  EXPECT_EQ(tc.name, "test_name");
  EXPECT_EQ(tc.className, "test_class");
  EXPECT_TRUE(tc.passed());
  EXPECT_FALSE(tc.failed());
  EXPECT_FALSE(tc.errored());
  EXPECT_FALSE(tc.skipped());
}

TEST(TestCaseTest, FailTest) {
  TestCase tc("test_name");
  tc.fail("Test failed", "Details here", "AssertionError");

  EXPECT_FALSE(tc.passed());
  EXPECT_TRUE(tc.failed());
  EXPECT_EQ(tc.message, "Test failed");
  EXPECT_EQ(tc.output, "Details here");
  EXPECT_EQ(tc.failureType, "AssertionError");
}

TEST(TestCaseTest, ErrorTest) {
  TestCase tc("test_name");
  tc.error("Test errored", "Error details");

  EXPECT_FALSE(tc.passed());
  EXPECT_TRUE(tc.errored());
  EXPECT_EQ(tc.message, "Test errored");
}

TEST(TestCaseTest, SkipTest) {
  TestCase tc("test_name");
  tc.skip("Not implemented");

  EXPECT_FALSE(tc.passed());
  EXPECT_TRUE(tc.skipped());
  EXPECT_EQ(tc.message, "Not implemented");
}

//===----------------------------------------------------------------------===//
// TestSuite Tests
//===----------------------------------------------------------------------===//

TEST(TestSuiteTest, AddTestCases) {
  TestSuite suite("my_suite");
  EXPECT_EQ(suite.getName(), "my_suite");
  EXPECT_EQ(suite.getTestCount(), 0u);

  auto &tc1 = suite.addTestCase("test1", "class1");
  tc1.timeSeconds = 0.5;

  auto &tc2 = suite.addTestCase("test2", "class1");
  tc2.fail("Failed");
  tc2.timeSeconds = 0.3;

  auto &tc3 = suite.addTestCase("test3", "class2");
  tc3.skip("Skipped");
  tc3.timeSeconds = 0.0;

  EXPECT_EQ(suite.getTestCount(), 3u);
  EXPECT_EQ(suite.getPassedCount(), 1u);
  EXPECT_EQ(suite.getFailedCount(), 1u);
  EXPECT_EQ(suite.getSkippedCount(), 1u);
  EXPECT_EQ(suite.getErrorCount(), 0u);
  EXPECT_NEAR(suite.getTotalTime(), 0.8, 0.001);
}

TEST(TestSuiteTest, Properties) {
  TestSuite suite("suite");
  suite.setProperty("version", "1.0");
  suite.setProperty("platform", "linux");

  const auto &props = suite.getProperties();
  ASSERT_EQ(props.size(), 2u);
  EXPECT_EQ(props[0].first, "version");
  EXPECT_EQ(props[0].second, "1.0");
}

//===----------------------------------------------------------------------===//
// TestReport Tests
//===----------------------------------------------------------------------===//

TEST(TestReportTest, MultiSuiteReport) {
  TestReport report("my_report");

  auto &suite1 = report.addSuite("suite1");
  suite1.addTestCase("test1").timeSeconds = 0.1;
  suite1.addTestCase("test2").fail("failure");
  suite1.getTestCases().back().timeSeconds = 0.2;

  auto &suite2 = report.addSuite("suite2");
  suite2.addTestCase("test3").timeSeconds = 0.15;
  suite2.addTestCase("test4").error("error");

  EXPECT_EQ(report.getTotalTestCount(), 4u);
  EXPECT_EQ(report.getTotalFailureCount(), 1u);
  EXPECT_EQ(report.getTotalErrorCount(), 1u);
  EXPECT_FALSE(report.allPassed());
}

TEST(TestReportTest, AllPassed) {
  TestReport report("passing");

  auto &suite = report.addSuite("suite");
  suite.addTestCase("test1");
  suite.addTestCase("test2");

  EXPECT_TRUE(report.allPassed());
}

//===----------------------------------------------------------------------===//
// JUnit XML Writer Tests
//===----------------------------------------------------------------------===//

TEST(JUnitXMLTest, EmptyReport) {
  TestReport report("empty");
  std::string output;
  llvm::raw_string_ostream os(output);
  writeJUnitXML(os, report);

  EXPECT_NE(output.find("<?xml version=\"1.0\""), std::string::npos);
  EXPECT_NE(output.find("<testsuites"), std::string::npos);
  EXPECT_NE(output.find("name=\"empty\""), std::string::npos);
  EXPECT_NE(output.find("tests=\"0\""), std::string::npos);
}

TEST(JUnitXMLTest, WithTestCases) {
  TestReport report("test_report");

  auto &suite = report.addSuite("lint_tests");
  suite.addTestCase("naming_check", "lint.naming").timeSeconds = 0.1;

  auto &failedCase = suite.addTestCase("unused_signal", "lint.unused");
  failedCase.fail("Signal 'x' is unused", "rtl/module.sv:10:5", "LintError");
  failedCase.timeSeconds = 0.2;

  std::string output;
  llvm::raw_string_ostream os(output);
  writeJUnitXML(os, report);

  EXPECT_NE(output.find("tests=\"2\""), std::string::npos);
  EXPECT_NE(output.find("failures=\"1\""), std::string::npos);
  EXPECT_NE(output.find("<testsuite"), std::string::npos);
  EXPECT_NE(output.find("name=\"lint_tests\""), std::string::npos);
  EXPECT_NE(output.find("classname=\"lint.naming\""), std::string::npos);
  EXPECT_NE(output.find("<failure"), std::string::npos);
  EXPECT_NE(output.find("type=\"LintError\""), std::string::npos);
}

TEST(JUnitXMLTest, XMLEscaping) {
  TestReport report("escaping");

  auto &suite = report.addSuite("suite");
  auto &tc = suite.addTestCase("test<>\"'&");
  tc.fail("Error with <special> & \"chars\"");

  std::string output;
  llvm::raw_string_ostream os(output);
  writeJUnitXML(os, report);

  EXPECT_NE(output.find("&lt;"), std::string::npos);
  EXPECT_NE(output.find("&gt;"), std::string::npos);
  EXPECT_NE(output.find("&amp;"), std::string::npos);
  EXPECT_NE(output.find("&quot;"), std::string::npos);
}

TEST(JUnitXMLTest, SkippedTests) {
  TestReport report("skipped");

  auto &suite = report.addSuite("suite");
  auto &tc = suite.addTestCase("skipped_test");
  tc.skip("Not implemented");

  std::string output;
  llvm::raw_string_ostream os(output);
  writeJUnitXML(os, report);

  EXPECT_NE(output.find("skipped=\"1\""), std::string::npos);
  EXPECT_NE(output.find("<skipped"), std::string::npos);
  EXPECT_NE(output.find("Not implemented"), std::string::npos);
}

//===----------------------------------------------------------------------===//
// JSON Writer Tests
//===----------------------------------------------------------------------===//

TEST(JSONWriterTest, BasicReport) {
  TestReport report("json_test");

  auto &suite = report.addSuite("suite");
  suite.addTestCase("test1");

  std::string output;
  llvm::raw_string_ostream os(output);
  writeTestReportJSON(os, report);

  EXPECT_NE(output.find("\"name\": \"json_test\""), std::string::npos);
  EXPECT_NE(output.find("\"status\": \"passed\""), std::string::npos);
}

//===----------------------------------------------------------------------===//
// Text Writer Tests
//===----------------------------------------------------------------------===//

TEST(TextWriterTest, BasicReport) {
  TestReport report("text_test");

  auto &suite = report.addSuite("suite");
  suite.addTestCase("passing_test").timeSeconds = 0.1;
  suite.addTestCase("failing_test").fail("Failed!");

  std::string output;
  llvm::raw_string_ostream os(output);
  writeTestReportText(os, report, false);

  EXPECT_NE(output.find("Test Report: text_test"), std::string::npos);
  EXPECT_NE(output.find("[PASS]"), std::string::npos);
  EXPECT_NE(output.find("[FAIL]"), std::string::npos);
  EXPECT_NE(output.find("TESTS FAILED"), std::string::npos);
}

//===----------------------------------------------------------------------===//
// Timer Tests
//===----------------------------------------------------------------------===//

TEST(TestTimerTest, BasicTiming) {
  TestTimer timer;

  // Sleep briefly
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  double elapsed = timer.getElapsedSeconds();
  EXPECT_GT(elapsed, 0.005); // At least 5ms (accounting for timing variations)
}

TEST(TestTimerTest, StopAndReset) {
  TestTimer timer;
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  double stopped = timer.stop();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Time should not change after stop
  EXPECT_DOUBLE_EQ(stopped, timer.getElapsedSeconds());

  timer.reset();
  EXPECT_LT(timer.getElapsedSeconds(), 0.001);
}

} // namespace
