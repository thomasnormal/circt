//===- Diagnostics.h - Rich diagnostic formatting -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for rich diagnostic formatting with modern
// compiler UX features like caret-style diagnostics, multi-span highlighting,
// and suggested fixes. Inspired by Rust/Clang diagnostic formatting.
//
// Example output format:
//   error: width mismatch in assignment
//      --> rtl/counter.sv:15:5
//       |
//    15 |     count <= data_in;
//       |     ^^^^^ --- ^^^^^^^ source is 32 bits
//       |     |
//       |     target is 8 bits
//       |
//       = note: 24 bits will be truncated
//       = help: use explicit slice: data_in[7:0]
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_DIAGNOSTICS_H
#define CIRCT_SUPPORT_DIAGNOSTICS_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <optional>

namespace circt {

//===----------------------------------------------------------------------===//
// Diagnostic Severity
//===----------------------------------------------------------------------===//

/// Severity levels for diagnostics.
enum class DiagSeverity { Error, Warning, Note, Hint };

//===----------------------------------------------------------------------===//
// Source Span
//===----------------------------------------------------------------------===//

/// Represents a span of source code with an optional label.
struct SourceSpan {
  /// File path (can be empty for unknown locations).
  std::string filename;
  /// 1-based line number.
  unsigned line = 0;
  /// 1-based column number (start of span).
  unsigned startColumn = 0;
  /// 1-based column number (end of span, inclusive).
  unsigned endColumn = 0;
  /// Optional label to display with this span.
  std::string label;
  /// Whether this is a primary span (vs secondary/related).
  bool isPrimary = true;

  SourceSpan() = default;
  SourceSpan(StringRef filename, unsigned line, unsigned startCol,
             unsigned endCol, StringRef label = "", bool isPrimary = true)
      : filename(filename.str()), line(line), startColumn(startCol),
        endColumn(endCol), label(label.str()), isPrimary(isPrimary) {}

  /// Create a SourceSpan from an MLIR Location.
  static std::optional<SourceSpan> fromLocation(mlir::Location loc,
                                                StringRef label = "",
                                                bool isPrimary = true);

  /// Check if this span is valid.
  bool isValid() const { return line > 0 && startColumn > 0; }
};

//===----------------------------------------------------------------------===//
// Suggested Fix
//===----------------------------------------------------------------------===//

/// Represents a suggested code fix.
struct SuggestedFix {
  /// Description of the fix.
  std::string message;
  /// The span of code to replace.
  SourceSpan span;
  /// The replacement text.
  std::string replacement;

  SuggestedFix() = default;
  SuggestedFix(StringRef message, SourceSpan span, StringRef replacement)
      : message(message.str()), span(span), replacement(replacement.str()) {}
};

//===----------------------------------------------------------------------===//
// Rich Diagnostic
//===----------------------------------------------------------------------===//

/// A rich diagnostic with source spans, notes, and suggested fixes.
class RichDiagnostic {
public:
  RichDiagnostic(DiagSeverity severity, StringRef message);

  /// Get the severity level.
  DiagSeverity getSeverity() const { return severity; }

  /// Get the main message.
  StringRef getMessage() const { return message; }

  /// Add a primary source span.
  RichDiagnostic &addPrimarySpan(SourceSpan span);

  /// Add a secondary/related source span.
  RichDiagnostic &addSecondarySpan(SourceSpan span);

  /// Add a note message.
  RichDiagnostic &addNote(StringRef note);

  /// Add a help/hint message.
  RichDiagnostic &addHelp(StringRef help);

  /// Add a suggested fix.
  RichDiagnostic &addFix(SuggestedFix fix);

  /// Get the source spans.
  ArrayRef<SourceSpan> getSpans() const { return spans; }

  /// Get the notes.
  ArrayRef<std::string> getNotes() const { return notes; }

  /// Get the help messages.
  ArrayRef<std::string> getHelps() const { return helps; }

  /// Get the suggested fixes.
  ArrayRef<SuggestedFix> getFixes() const { return fixes; }

private:
  DiagSeverity severity;
  std::string message;
  SmallVector<SourceSpan> spans;
  SmallVector<std::string> notes;
  SmallVector<std::string> helps;
  SmallVector<SuggestedFix> fixes;
};

//===----------------------------------------------------------------------===//
// Output Format
//===----------------------------------------------------------------------===//

/// Output format for diagnostics.
enum class DiagnosticOutputFormat {
  /// Terminal output with ANSI colors.
  Terminal,
  /// Plain text without colors.
  Plain,
  /// JSON format for IDE integration.
  JSON,
  /// SARIF format for CI/CD integration.
  SARIF
};

//===----------------------------------------------------------------------===//
// Diagnostic Printer
//===----------------------------------------------------------------------===//

/// A printer for rich diagnostics with multiple output format support.
class DiagnosticPrinter {
public:
  /// Create a diagnostic printer with the given output stream and format.
  explicit DiagnosticPrinter(
      llvm::raw_ostream &os,
      DiagnosticOutputFormat format = DiagnosticOutputFormat::Terminal,
      llvm::SourceMgr *sourceMgr = nullptr);

  ~DiagnosticPrinter();

  /// Set the output format.
  void setFormat(DiagnosticOutputFormat format) { this->format = format; }

  /// Set whether to use colors (only affects Terminal format).
  void setUseColors(bool useColors) { this->useColors = useColors; }

  /// Set the source manager for retrieving source lines.
  void setSourceManager(llvm::SourceMgr *mgr) { sourceMgr = mgr; }

  /// Print a rich diagnostic.
  void print(const RichDiagnostic &diag);

  /// Convenience methods to emit diagnostics directly.
  void emitError(mlir::Location loc, StringRef message);
  void emitWarning(mlir::Location loc, StringRef message);
  void emitNote(mlir::Location loc, StringRef message);
  void emitHint(mlir::Location loc, StringRef suggestion);

  /// Emit a diagnostic with multiple spans.
  void emitDiagnostic(DiagSeverity severity, StringRef message,
                      ArrayRef<SourceSpan> spans, ArrayRef<StringRef> notes = {},
                      ArrayRef<StringRef> helps = {});

  /// Highlight a range of source code with a label.
  void highlightRange(SourceSpan span);

  /// Suggest a replacement for source code.
  void suggestReplacement(SourceSpan span, StringRef oldText,
                          StringRef newText);

  /// Flush any buffered output (for JSON/SARIF formats).
  void flush();

  /// Get the number of errors emitted.
  unsigned getNumErrors() const { return numErrors; }

  /// Get the number of warnings emitted.
  unsigned getNumWarnings() const { return numWarnings; }

  /// Reset error/warning counts.
  void resetCounts() {
    numErrors = 0;
    numWarnings = 0;
  }

private:
  /// Print a diagnostic in terminal format.
  void printTerminal(const RichDiagnostic &diag);

  /// Print a diagnostic in plain text format.
  void printPlain(const RichDiagnostic &diag);

  /// Print a diagnostic in JSON format.
  void printJSON(const RichDiagnostic &diag);

  /// Print a diagnostic in SARIF format.
  void printSARIF(const RichDiagnostic &diag);

  /// Print the header line with severity and message.
  void printHeader(const RichDiagnostic &diag);

  /// Print source code with span highlighting.
  void printSourceWithSpans(ArrayRef<SourceSpan> spans);

  /// Print a single source line with highlighting.
  void printSourceLine(StringRef filename, unsigned lineNum, StringRef line,
                       ArrayRef<SourceSpan> lineSpans);

  /// Print the caret line showing span locations.
  void printCaretLine(unsigned gutterWidth, ArrayRef<SourceSpan> lineSpans,
                      unsigned lineLength);

  /// Get the source line text from the source manager.
  std::optional<StringRef> getSourceLine(StringRef filename, unsigned line);

  /// Get the ANSI color code for a severity.
  StringRef getSeverityColor(DiagSeverity severity);

  /// Get the string representation of a severity.
  StringRef getSeverityString(DiagSeverity severity);

  /// Print with optional ANSI color.
  void printColored(StringRef text, StringRef color);

  llvm::raw_ostream &os;
  DiagnosticOutputFormat format;
  llvm::SourceMgr *sourceMgr;
  bool useColors;
  unsigned numErrors = 0;
  unsigned numWarnings = 0;

  /// Buffer for JSON/SARIF output.
  SmallVector<llvm::json::Value> jsonBuffer;
};

//===----------------------------------------------------------------------===//
// Diagnostic Handler Integration
//===----------------------------------------------------------------------===//

/// A diagnostic handler that uses the DiagnosticPrinter for rich output.
/// This can be registered with an MLIRContext to handle diagnostics.
class RichDiagnosticHandler {
public:
  RichDiagnosticHandler(mlir::MLIRContext *ctx, DiagnosticPrinter &printer);
  ~RichDiagnosticHandler();

  /// Process an MLIR diagnostic and emit it with rich formatting.
  void handleDiagnostic(mlir::Diagnostic &diag);

private:
  mlir::MLIRContext *context;
  DiagnosticPrinter &printer;
  mlir::DiagnosticEngine::HandlerID handlerID;
};

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Create a SourceSpan from an MLIR FileLineColLoc.
SourceSpan createSpan(mlir::FileLineColLoc loc, unsigned length = 1,
                      StringRef label = "", bool isPrimary = true);

/// Create a SourceSpan covering a range between two FileLineColLocs.
SourceSpan createSpan(mlir::FileLineColLoc start, mlir::FileLineColLoc end,
                      StringRef label = "", bool isPrimary = true);

/// Parse a DiagnosticOutputFormat from a string.
std::optional<DiagnosticOutputFormat>
parseDiagnosticOutputFormat(StringRef str);

/// Get the string representation of a DiagnosticOutputFormat.
StringRef getDiagnosticOutputFormatString(DiagnosticOutputFormat format);

} // namespace circt

#endif // CIRCT_SUPPORT_DIAGNOSTICS_H
