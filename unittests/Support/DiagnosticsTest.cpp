//===- DiagnosticsTest.cpp - Diagnostics unit tests -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/Diagnostics.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace circt;
using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// SourceSpan Tests
//===----------------------------------------------------------------------===//

TEST(SourceSpanTest, BasicConstruction) {
  SourceSpan span("test.sv", 10, 5, 15, "test label", true);
  EXPECT_EQ(span.filename, "test.sv");
  EXPECT_EQ(span.line, 10u);
  EXPECT_EQ(span.startColumn, 5u);
  EXPECT_EQ(span.endColumn, 15u);
  EXPECT_EQ(span.label, "test label");
  EXPECT_TRUE(span.isPrimary);
  EXPECT_TRUE(span.isValid());
}

TEST(SourceSpanTest, InvalidSpan) {
  SourceSpan span;
  EXPECT_FALSE(span.isValid());
}

TEST(SourceSpanTest, FromFileLineColLoc) {
  MLIRContext ctx;
  auto loc = FileLineColLoc::get(&ctx, "test.sv", 42, 10);
  auto span = SourceSpan::fromLocation(loc, "test label", true);

  ASSERT_TRUE(span.has_value());
  EXPECT_EQ(span->filename, "test.sv");
  EXPECT_EQ(span->line, 42u);
  EXPECT_EQ(span->startColumn, 10u);
  EXPECT_EQ(span->label, "test label");
  EXPECT_TRUE(span->isPrimary);
}

TEST(SourceSpanTest, FromUnknownLoc) {
  MLIRContext ctx;
  auto loc = UnknownLoc::get(&ctx);
  auto span = SourceSpan::fromLocation(loc);

  EXPECT_FALSE(span.has_value());
}

TEST(SourceSpanTest, FromNameLoc) {
  MLIRContext ctx;
  auto innerLoc = FileLineColLoc::get(&ctx, "test.sv", 5, 3);
  auto loc = NameLoc::get(StringAttr::get(&ctx, "myVar"), innerLoc);
  auto span = SourceSpan::fromLocation(loc);

  ASSERT_TRUE(span.has_value());
  EXPECT_EQ(span->filename, "test.sv");
  EXPECT_EQ(span->line, 5u);
  EXPECT_EQ(span->startColumn, 3u);
}

//===----------------------------------------------------------------------===//
// RichDiagnostic Tests
//===----------------------------------------------------------------------===//

TEST(RichDiagnosticTest, BasicConstruction) {
  RichDiagnostic diag(DiagSeverity::Error, "test error message");

  EXPECT_EQ(diag.getSeverity(), DiagSeverity::Error);
  EXPECT_EQ(diag.getMessage(), "test error message");
  EXPECT_TRUE(diag.getSpans().empty());
  EXPECT_TRUE(diag.getNotes().empty());
  EXPECT_TRUE(diag.getHelps().empty());
  EXPECT_TRUE(diag.getFixes().empty());
}

TEST(RichDiagnosticTest, AddPrimarySpan) {
  RichDiagnostic diag(DiagSeverity::Warning, "test warning");
  SourceSpan span("file.sv", 1, 1, 10, "primary", false);

  diag.addPrimarySpan(span);

  EXPECT_EQ(diag.getSpans().size(), 1u);
  EXPECT_TRUE(diag.getSpans()[0].isPrimary);
}

TEST(RichDiagnosticTest, AddSecondarySpan) {
  RichDiagnostic diag(DiagSeverity::Note, "test note");
  SourceSpan span("file.sv", 2, 5, 15, "secondary", true);

  diag.addSecondarySpan(span);

  EXPECT_EQ(diag.getSpans().size(), 1u);
  EXPECT_FALSE(diag.getSpans()[0].isPrimary);
}

TEST(RichDiagnosticTest, AddNotes) {
  RichDiagnostic diag(DiagSeverity::Error, "error");
  diag.addNote("first note");
  diag.addNote("second note");

  EXPECT_EQ(diag.getNotes().size(), 2u);
  EXPECT_EQ(diag.getNotes()[0], "first note");
  EXPECT_EQ(diag.getNotes()[1], "second note");
}

TEST(RichDiagnosticTest, AddHelps) {
  RichDiagnostic diag(DiagSeverity::Error, "error");
  diag.addHelp("try this");
  diag.addHelp("or this");

  EXPECT_EQ(diag.getHelps().size(), 2u);
  EXPECT_EQ(diag.getHelps()[0], "try this");
  EXPECT_EQ(diag.getHelps()[1], "or this");
}

TEST(RichDiagnosticTest, AddFixes) {
  RichDiagnostic diag(DiagSeverity::Error, "error");
  SourceSpan span("file.sv", 1, 5, 10);
  SuggestedFix fix("replace foo with bar", span, "bar");

  diag.addFix(fix);

  EXPECT_EQ(diag.getFixes().size(), 1u);
  EXPECT_EQ(diag.getFixes()[0].message, "replace foo with bar");
  EXPECT_EQ(diag.getFixes()[0].replacement, "bar");
}

TEST(RichDiagnosticTest, ChainedMethods) {
  RichDiagnostic diag(DiagSeverity::Error, "chained");
  SourceSpan span("file.sv", 1, 1, 5);

  diag.addPrimarySpan(span)
      .addNote("note1")
      .addHelp("help1")
      .addFix(SuggestedFix("fix", span, "fixed"));

  EXPECT_EQ(diag.getSpans().size(), 1u);
  EXPECT_EQ(diag.getNotes().size(), 1u);
  EXPECT_EQ(diag.getHelps().size(), 1u);
  EXPECT_EQ(diag.getFixes().size(), 1u);
}

//===----------------------------------------------------------------------===//
// DiagnosticPrinter Tests
//===----------------------------------------------------------------------===//

class DiagnosticPrinterTest : public ::testing::Test {
protected:
  std::string output;
  llvm::raw_string_ostream os{output};
  llvm::SourceMgr sourceMgr;

  void SetUp() override {
    // Add a test buffer to the source manager.
    std::string testSource = "module test;\n"
                             "  wire [7:0] count;\n"
                             "  wire [31:0] data_in;\n"
                             "  assign count = data_in;\n"
                             "endmodule\n";
    auto buffer = llvm::MemoryBuffer::getMemBufferCopy(testSource, "test.sv");
    sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  }

  void TearDown() override { output.clear(); }
};

TEST_F(DiagnosticPrinterTest, PlainFormatError) {
  DiagnosticPrinter printer(os, DiagnosticOutputFormat::Plain, &sourceMgr);

  RichDiagnostic diag(DiagSeverity::Error, "width mismatch in assignment");
  diag.addPrimarySpan(SourceSpan("test.sv", 4, 10, 14, "target is 8 bits"));
  diag.addNote("24 bits will be truncated");
  diag.addHelp("use explicit slice: data_in[7:0]");

  printer.print(diag);

  // Check that the output contains expected elements.
  EXPECT_TRUE(output.find("error") != std::string::npos);
  EXPECT_TRUE(output.find("width mismatch") != std::string::npos);
  EXPECT_TRUE(output.find("test.sv:4:10") != std::string::npos);
  EXPECT_TRUE(output.find("note:") != std::string::npos);
  EXPECT_TRUE(output.find("24 bits will be truncated") != std::string::npos);
  EXPECT_TRUE(output.find("help:") != std::string::npos);
  EXPECT_TRUE(output.find("data_in[7:0]") != std::string::npos);
}

TEST_F(DiagnosticPrinterTest, PlainFormatWarning) {
  DiagnosticPrinter printer(os, DiagnosticOutputFormat::Plain, &sourceMgr);

  RichDiagnostic diag(DiagSeverity::Warning, "unused variable");
  diag.addPrimarySpan(SourceSpan("test.sv", 2, 14, 18, "declared here"));

  printer.print(diag);

  EXPECT_TRUE(output.find("warning") != std::string::npos);
  EXPECT_TRUE(output.find("unused variable") != std::string::npos);
}

TEST_F(DiagnosticPrinterTest, JSONFormat) {
  DiagnosticPrinter printer(os, DiagnosticOutputFormat::JSON, &sourceMgr);

  RichDiagnostic diag(DiagSeverity::Error, "test error");
  diag.addPrimarySpan(SourceSpan("test.sv", 1, 1, 5));
  diag.addNote("test note");

  printer.print(diag);
  printer.flush();

  // Check that we got valid JSON-like output.
  EXPECT_TRUE(output.find("\"severity\"") != std::string::npos);
  EXPECT_TRUE(output.find("\"error\"") != std::string::npos);
  EXPECT_TRUE(output.find("\"message\"") != std::string::npos);
  EXPECT_TRUE(output.find("\"test error\"") != std::string::npos);
  EXPECT_TRUE(output.find("\"locations\"") != std::string::npos);
  EXPECT_TRUE(output.find("\"notes\"") != std::string::npos);
}

TEST_F(DiagnosticPrinterTest, SARIFFormat) {
  DiagnosticPrinter printer(os, DiagnosticOutputFormat::SARIF, &sourceMgr);

  RichDiagnostic diag(DiagSeverity::Warning, "test warning");
  diag.addPrimarySpan(SourceSpan("test.sv", 2, 3, 10));

  printer.print(diag);
  printer.flush();

  // Check for SARIF-specific structure.
  EXPECT_TRUE(output.find("\"version\"") != std::string::npos);
  EXPECT_TRUE(output.find("\"2.1.0\"") != std::string::npos);
  EXPECT_TRUE(output.find("\"$schema\"") != std::string::npos);
  EXPECT_TRUE(output.find("\"runs\"") != std::string::npos);
  EXPECT_TRUE(output.find("\"results\"") != std::string::npos);
  EXPECT_TRUE(output.find("\"level\"") != std::string::npos);
  EXPECT_TRUE(output.find("\"warning\"") != std::string::npos);
}

TEST_F(DiagnosticPrinterTest, ErrorCounting) {
  DiagnosticPrinter printer(os, DiagnosticOutputFormat::Plain, &sourceMgr);

  EXPECT_EQ(printer.getNumErrors(), 0u);
  EXPECT_EQ(printer.getNumWarnings(), 0u);

  printer.print(RichDiagnostic(DiagSeverity::Error, "error 1"));
  printer.print(RichDiagnostic(DiagSeverity::Error, "error 2"));
  printer.print(RichDiagnostic(DiagSeverity::Warning, "warning 1"));
  printer.print(RichDiagnostic(DiagSeverity::Note, "note 1"));

  EXPECT_EQ(printer.getNumErrors(), 2u);
  EXPECT_EQ(printer.getNumWarnings(), 1u);

  printer.resetCounts();

  EXPECT_EQ(printer.getNumErrors(), 0u);
  EXPECT_EQ(printer.getNumWarnings(), 0u);
}

TEST_F(DiagnosticPrinterTest, ConvenienceMethods) {
  MLIRContext ctx;
  DiagnosticPrinter printer(os, DiagnosticOutputFormat::Plain, &sourceMgr);

  auto loc = FileLineColLoc::get(&ctx, "test.sv", 1, 1);

  printer.emitError(loc, "test error");
  printer.emitWarning(loc, "test warning");
  printer.emitNote(loc, "test note");
  printer.emitHint(loc, "test hint");

  EXPECT_TRUE(output.find("error") != std::string::npos);
  EXPECT_TRUE(output.find("warning") != std::string::npos);
  EXPECT_TRUE(output.find("note") != std::string::npos);
  EXPECT_TRUE(output.find("help") != std::string::npos);
}

//===----------------------------------------------------------------------===//
// Utility Function Tests
//===----------------------------------------------------------------------===//

TEST(DiagnosticUtilsTest, ParseOutputFormat) {
  EXPECT_EQ(*parseDiagnosticOutputFormat("terminal"),
            DiagnosticOutputFormat::Terminal);
  EXPECT_EQ(*parseDiagnosticOutputFormat("term"),
            DiagnosticOutputFormat::Terminal);
  EXPECT_EQ(*parseDiagnosticOutputFormat("plain"),
            DiagnosticOutputFormat::Plain);
  EXPECT_EQ(*parseDiagnosticOutputFormat("text"),
            DiagnosticOutputFormat::Plain);
  EXPECT_EQ(*parseDiagnosticOutputFormat("json"), DiagnosticOutputFormat::JSON);
  EXPECT_EQ(*parseDiagnosticOutputFormat("sarif"),
            DiagnosticOutputFormat::SARIF);
  EXPECT_FALSE(parseDiagnosticOutputFormat("invalid").has_value());
}

TEST(DiagnosticUtilsTest, GetOutputFormatString) {
  EXPECT_EQ(getDiagnosticOutputFormatString(DiagnosticOutputFormat::Terminal),
            "terminal");
  EXPECT_EQ(getDiagnosticOutputFormatString(DiagnosticOutputFormat::Plain),
            "plain");
  EXPECT_EQ(getDiagnosticOutputFormatString(DiagnosticOutputFormat::JSON),
            "json");
  EXPECT_EQ(getDiagnosticOutputFormatString(DiagnosticOutputFormat::SARIF),
            "sarif");
}

TEST(DiagnosticUtilsTest, CreateSpanWithLength) {
  MLIRContext ctx;
  auto loc = FileLineColLoc::get(&ctx, "test.sv", 10, 5);
  auto span = createSpan(loc, 8, "test", true);

  EXPECT_EQ(span.filename, "test.sv");
  EXPECT_EQ(span.line, 10u);
  EXPECT_EQ(span.startColumn, 5u);
  EXPECT_EQ(span.endColumn, 12u); // 5 + 8 - 1
  EXPECT_EQ(span.label, "test");
  EXPECT_TRUE(span.isPrimary);
}

TEST(DiagnosticUtilsTest, CreateSpanFromRange) {
  MLIRContext ctx;
  auto startLoc = FileLineColLoc::get(&ctx, "test.sv", 10, 5);
  auto endLoc = FileLineColLoc::get(&ctx, "test.sv", 10, 15);
  auto span = createSpan(startLoc, endLoc, "range", false);

  EXPECT_EQ(span.filename, "test.sv");
  EXPECT_EQ(span.line, 10u);
  EXPECT_EQ(span.startColumn, 5u);
  EXPECT_EQ(span.endColumn, 15u);
  EXPECT_EQ(span.label, "range");
  EXPECT_FALSE(span.isPrimary);
}

//===----------------------------------------------------------------------===//
// SuggestedFix Tests
//===----------------------------------------------------------------------===//

TEST(SuggestedFixTest, BasicConstruction) {
  SourceSpan span("test.sv", 5, 10, 15);
  SuggestedFix fix("replace old with new", span, "new_value");

  EXPECT_EQ(fix.message, "replace old with new");
  EXPECT_EQ(fix.span.filename, "test.sv");
  EXPECT_EQ(fix.span.line, 5u);
  EXPECT_EQ(fix.replacement, "new_value");
}

//===----------------------------------------------------------------------===//
// Integration Test with MLIR Context
//===----------------------------------------------------------------------===//

TEST(RichDiagnosticHandlerTest, HandleMLIRDiagnostic) {
  std::string output;
  llvm::raw_string_ostream os(output);
  DiagnosticPrinter printer(os, DiagnosticOutputFormat::Plain);

  MLIRContext ctx;
  ctx.loadDialect<mlir::BuiltinDialect>();

  RichDiagnosticHandler handler(&ctx, printer);

  // Emit a diagnostic through MLIR.
  auto loc = FileLineColLoc::get(&ctx, "test.sv", 5, 10);
  emitError(loc, "test MLIR error");

  // Check that it was handled.
  EXPECT_TRUE(output.find("error") != std::string::npos);
  EXPECT_TRUE(output.find("test MLIR error") != std::string::npos);
  EXPECT_EQ(printer.getNumErrors(), 1u);
}

} // namespace
