//===- Diagnostics.cpp - Rich diagnostic formatting -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements rich diagnostic formatting utilities.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/Diagnostics.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>

using namespace circt;
using namespace mlir;

//===----------------------------------------------------------------------===//
// ANSI Color Codes
//===----------------------------------------------------------------------===//

namespace {
// ANSI color codes for terminal output.
constexpr StringLiteral kColorReset = "\033[0m";
constexpr StringLiteral kColorBold = "\033[1m";
constexpr StringLiteral kColorRed = "\033[31m";
constexpr StringLiteral kColorYellow = "\033[33m";
constexpr StringLiteral kColorBlue = "\033[34m";
constexpr StringLiteral kColorCyan = "\033[36m";
constexpr StringLiteral kColorGreen = "\033[32m";
constexpr StringLiteral kColorMagenta = "\033[35m";
constexpr StringLiteral kColorBoldRed = "\033[1;31m";
constexpr StringLiteral kColorBoldYellow = "\033[1;33m";
constexpr StringLiteral kColorBoldBlue = "\033[1;34m";
constexpr StringLiteral kColorBoldCyan = "\033[1;36m";
} // namespace

//===----------------------------------------------------------------------===//
// SourceSpan
//===----------------------------------------------------------------------===//

std::optional<SourceSpan> SourceSpan::fromLocation(Location loc, StringRef label,
                                                   bool isPrimary) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
    return SourceSpan(fileLoc.getFilename(), fileLoc.getLine(),
                      fileLoc.getColumn(), fileLoc.getColumn(), label,
                      isPrimary);
  }
  // Handle FusedLoc by extracting the first FileLineColLoc.
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    for (Location subLoc : fusedLoc.getLocations()) {
      if (auto span = fromLocation(subLoc, label, isPrimary))
        return span;
    }
  }
  // Handle NameLoc by getting its child location.
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    return fromLocation(nameLoc.getChildLoc(), label, isPrimary);
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// RichDiagnostic
//===----------------------------------------------------------------------===//

RichDiagnostic::RichDiagnostic(DiagSeverity severity, StringRef message)
    : severity(severity), message(message.str()) {}

RichDiagnostic &RichDiagnostic::addPrimarySpan(SourceSpan span) {
  span.isPrimary = true;
  spans.push_back(std::move(span));
  return *this;
}

RichDiagnostic &RichDiagnostic::addSecondarySpan(SourceSpan span) {
  span.isPrimary = false;
  spans.push_back(std::move(span));
  return *this;
}

RichDiagnostic &RichDiagnostic::addNote(StringRef note) {
  notes.push_back(note.str());
  return *this;
}

RichDiagnostic &RichDiagnostic::addHelp(StringRef help) {
  helps.push_back(help.str());
  return *this;
}

RichDiagnostic &RichDiagnostic::addFix(SuggestedFix fix) {
  fixes.push_back(std::move(fix));
  return *this;
}

//===----------------------------------------------------------------------===//
// DiagnosticPrinter
//===----------------------------------------------------------------------===//

DiagnosticPrinter::DiagnosticPrinter(llvm::raw_ostream &os,
                                     DiagnosticOutputFormat format,
                                     llvm::SourceMgr *sourceMgr)
    : os(os), format(format), sourceMgr(sourceMgr),
      useColors(format == DiagnosticOutputFormat::Terminal &&
                os.has_colors()) {}

DiagnosticPrinter::~DiagnosticPrinter() { flush(); }

void DiagnosticPrinter::print(const RichDiagnostic &diag) {
  // Update counters.
  if (diag.getSeverity() == DiagSeverity::Error)
    ++numErrors;
  else if (diag.getSeverity() == DiagSeverity::Warning)
    ++numWarnings;

  switch (format) {
  case DiagnosticOutputFormat::Terminal:
    printTerminal(diag);
    break;
  case DiagnosticOutputFormat::Plain:
    printPlain(diag);
    break;
  case DiagnosticOutputFormat::JSON:
    printJSON(diag);
    break;
  case DiagnosticOutputFormat::SARIF:
    printSARIF(diag);
    break;
  }
}

void DiagnosticPrinter::emitError(Location loc, StringRef message) {
  RichDiagnostic diag(DiagSeverity::Error, message);
  if (auto span = SourceSpan::fromLocation(loc))
    diag.addPrimarySpan(*span);
  print(diag);
}

void DiagnosticPrinter::emitWarning(Location loc, StringRef message) {
  RichDiagnostic diag(DiagSeverity::Warning, message);
  if (auto span = SourceSpan::fromLocation(loc))
    diag.addPrimarySpan(*span);
  print(diag);
}

void DiagnosticPrinter::emitNote(Location loc, StringRef message) {
  RichDiagnostic diag(DiagSeverity::Note, message);
  if (auto span = SourceSpan::fromLocation(loc))
    diag.addPrimarySpan(*span);
  print(diag);
}

void DiagnosticPrinter::emitHint(Location loc, StringRef suggestion) {
  RichDiagnostic diag(DiagSeverity::Hint, suggestion);
  if (auto span = SourceSpan::fromLocation(loc))
    diag.addPrimarySpan(*span);
  print(diag);
}

void DiagnosticPrinter::emitDiagnostic(DiagSeverity severity, StringRef message,
                                        ArrayRef<SourceSpan> spans,
                                        ArrayRef<StringRef> notes,
                                        ArrayRef<StringRef> helps) {
  RichDiagnostic diag(severity, message);
  for (const auto &span : spans)
    diag.addPrimarySpan(span);
  for (StringRef note : notes)
    diag.addNote(note);
  for (StringRef help : helps)
    diag.addHelp(help);
  print(diag);
}

void DiagnosticPrinter::highlightRange(SourceSpan span) {
  printSourceWithSpans({span});
}

void DiagnosticPrinter::suggestReplacement(SourceSpan span, StringRef oldText,
                                            StringRef newText) {
  RichDiagnostic diag(DiagSeverity::Hint, "suggested replacement");
  diag.addPrimarySpan(span);
  diag.addFix(SuggestedFix(
      llvm::formatv("replace `{0}` with `{1}`", oldText, newText), span,
      newText));
  print(diag);
}

void DiagnosticPrinter::flush() {
  if (format == DiagnosticOutputFormat::JSON && !jsonBuffer.empty()) {
    llvm::json::Array arr;
    for (auto &val : jsonBuffer)
      arr.push_back(std::move(val));
    os << llvm::json::Value(std::move(arr)) << "\n";
    jsonBuffer.clear();
  } else if (format == DiagnosticOutputFormat::SARIF && !jsonBuffer.empty()) {
    // Emit SARIF wrapper with results.
    llvm::json::Object sarif;
    sarif["version"] = "2.1.0";
    sarif["$schema"] =
        "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json";

    llvm::json::Object run;
    llvm::json::Object tool;
    llvm::json::Object driver;
    driver["name"] = "circt";
    driver["informationUri"] = "https://circt.llvm.org/";
    tool["driver"] = std::move(driver);
    run["tool"] = std::move(tool);

    llvm::json::Array results;
    for (auto &val : jsonBuffer)
      results.push_back(std::move(val));
    run["results"] = std::move(results);

    llvm::json::Array runs;
    runs.push_back(std::move(run));
    sarif["runs"] = std::move(runs);

    os << llvm::json::Value(std::move(sarif)) << "\n";
    jsonBuffer.clear();
  }
}

StringRef DiagnosticPrinter::getSeverityColor(DiagSeverity severity) {
  switch (severity) {
  case DiagSeverity::Error:
    return kColorBoldRed;
  case DiagSeverity::Warning:
    return kColorBoldYellow;
  case DiagSeverity::Note:
    return kColorBoldBlue;
  case DiagSeverity::Hint:
    return kColorBoldCyan;
  }
  return kColorReset;
}

StringRef DiagnosticPrinter::getSeverityString(DiagSeverity severity) {
  switch (severity) {
  case DiagSeverity::Error:
    return "error";
  case DiagSeverity::Warning:
    return "warning";
  case DiagSeverity::Note:
    return "note";
  case DiagSeverity::Hint:
    return "help";
  }
  return "unknown";
}

void DiagnosticPrinter::printColored(StringRef text, StringRef color) {
  if (useColors)
    os << color;
  os << text;
  if (useColors)
    os << kColorReset;
}

void DiagnosticPrinter::printHeader(const RichDiagnostic &diag) {
  printColored(getSeverityString(diag.getSeverity()),
               getSeverityColor(diag.getSeverity()));
  os << ": ";
  if (useColors)
    os << kColorBold;
  os << diag.getMessage();
  if (useColors)
    os << kColorReset;
  os << "\n";
}

std::optional<StringRef> DiagnosticPrinter::getSourceLine(StringRef filename,
                                                          unsigned line) {
  if (!sourceMgr)
    return std::nullopt;

  // Find the buffer with this filename.
  for (unsigned i = 1, e = sourceMgr->getNumBuffers(); i <= e; ++i) {
    auto *buf = sourceMgr->getMemoryBuffer(i);
    if (buf->getBufferIdentifier() == filename) {
      StringRef content = buf->getBuffer();
      unsigned currentLine = 1;
      size_t lineStart = 0;

      for (size_t i = 0; i < content.size(); ++i) {
        if (currentLine == line) {
          size_t lineEnd = content.find('\n', i);
          if (lineEnd == StringRef::npos)
            lineEnd = content.size();
          return content.substr(lineStart, lineEnd - lineStart);
        }
        if (content[i] == '\n') {
          ++currentLine;
          lineStart = i + 1;
        }
      }
      break;
    }
  }
  return std::nullopt;
}

void DiagnosticPrinter::printSourceLine(StringRef filename, unsigned lineNum,
                                        StringRef line,
                                        ArrayRef<SourceSpan> lineSpans) {
  // Calculate gutter width based on line number.
  unsigned gutterWidth = std::max(3u, (unsigned)std::to_string(lineNum).size());

  // Print line number and gutter.
  if (useColors)
    os << kColorBlue;
  os << llvm::format_decimal(lineNum, gutterWidth) << " | ";
  if (useColors)
    os << kColorReset;

  // Print the source line.
  os << line << "\n";

  // Print caret line.
  printCaretLine(gutterWidth, lineSpans, line.size());
}

void DiagnosticPrinter::printCaretLine(unsigned gutterWidth,
                                       ArrayRef<SourceSpan> lineSpans,
                                       unsigned lineLength) {
  if (lineSpans.empty())
    return;

  // Print empty gutter.
  if (useColors)
    os << kColorBlue;
  os << std::string(gutterWidth, ' ') << " | ";
  if (useColors)
    os << kColorReset;

  // Build the caret line.
  std::string caretLine(lineLength + 1, ' ');
  std::string labelLine;
  bool hasLabel = false;

  for (const auto &span : lineSpans) {
    unsigned startCol = span.startColumn > 0 ? span.startColumn - 1 : 0;
    unsigned endCol =
        span.endColumn > 0 ? std::min(span.endColumn, (unsigned)lineLength) : startCol + 1;

    // Fill in carets.
    char caretChar = span.isPrimary ? '^' : '-';
    for (unsigned i = startCol; i < endCol && i < caretLine.size(); ++i)
      caretLine[i] = caretChar;

    if (!span.label.empty())
      hasLabel = true;
  }

  // Print caret line with colors.
  for (size_t i = 0; i < caretLine.size(); ++i) {
    if (caretLine[i] == '^') {
      printColored(StringRef(&caretLine[i], 1), kColorBoldRed);
    } else if (caretLine[i] == '-') {
      printColored(StringRef(&caretLine[i], 1), kColorBoldBlue);
    } else {
      os << caretLine[i];
    }
  }
  os << "\n";

  // Print labels if any span has one.
  if (hasLabel) {
    for (const auto &span : lineSpans) {
      if (span.label.empty())
        continue;

      if (useColors)
        os << kColorBlue;
      os << std::string(gutterWidth, ' ') << " | ";
      if (useColors)
        os << kColorReset;

      unsigned startCol = span.startColumn > 0 ? span.startColumn - 1 : 0;
      os << std::string(startCol, ' ');

      if (span.isPrimary)
        printColored(span.label, kColorBoldRed);
      else
        printColored(span.label, kColorBoldBlue);
      os << "\n";
    }
  }
}

void DiagnosticPrinter::printSourceWithSpans(ArrayRef<SourceSpan> spans) {
  if (spans.empty())
    return;

  // Group spans by file and line.
  llvm::StringMap<llvm::SmallDenseMap<unsigned, SmallVector<SourceSpan>>>
      spansByFileAndLine;
  for (const auto &span : spans) {
    if (span.isValid())
      spansByFileAndLine[span.filename][span.line].push_back(span);
  }

  // Print each file's spans.
  for (auto &[filename, lineSpans] : spansByFileAndLine) {
    // Print file location header.
    if (!filename.empty()) {
      if (useColors)
        os << kColorBlue;
      os << "   --> ";
      if (useColors)
        os << kColorReset;

      // Get the first span for location info.
      auto firstLineIt = lineSpans.begin();
      if (firstLineIt != lineSpans.end() && !firstLineIt->second.empty()) {
        const auto &firstSpan = firstLineIt->second[0];
        os << filename << ":" << firstSpan.line << ":" << firstSpan.startColumn;
      } else {
        os << filename;
      }
      os << "\n";

      // Print separator line.
      if (useColors)
        os << kColorBlue;
      os << "    |\n";
      if (useColors)
        os << kColorReset;
    }

    // Print each line with its spans.
    for (auto &[lineNum, lineSpansVec] : lineSpans) {
      if (auto sourceLine = getSourceLine(filename, lineNum)) {
        printSourceLine(filename, lineNum, *sourceLine, lineSpansVec);
      }
    }

    // Print closing separator.
    if (useColors)
      os << kColorBlue;
    os << "    |\n";
    if (useColors)
      os << kColorReset;
  }
}

void DiagnosticPrinter::printTerminal(const RichDiagnostic &diag) {
  printHeader(diag);
  printSourceWithSpans(diag.getSpans());

  // Print notes.
  for (const auto &note : diag.getNotes()) {
    if (useColors)
      os << kColorBlue;
    os << "    = ";
    if (useColors)
      os << kColorReset;
    os << "note: " << note << "\n";
  }

  // Print help messages.
  for (const auto &help : diag.getHelps()) {
    if (useColors)
      os << kColorCyan;
    os << "    = ";
    if (useColors)
      os << kColorReset;
    os << "help: " << help << "\n";
  }

  // Print suggested fixes.
  for (const auto &fix : diag.getFixes()) {
    if (useColors)
      os << kColorGreen;
    os << "    = ";
    if (useColors)
      os << kColorReset;
    os << "fix: " << fix.message;
    if (!fix.replacement.empty())
      os << " -> `" << fix.replacement << "`";
    os << "\n";
  }

  os << "\n";
}

void DiagnosticPrinter::printPlain(const RichDiagnostic &diag) {
  // Simple plain text format without colors.
  bool savedUseColors = useColors;
  useColors = false;
  printTerminal(diag);
  useColors = savedUseColors;
}

void DiagnosticPrinter::printJSON(const RichDiagnostic &diag) {
  llvm::json::Object obj;
  obj["severity"] = getSeverityString(diag.getSeverity());
  obj["message"] = diag.getMessage();

  // Add locations.
  if (!diag.getSpans().empty()) {
    llvm::json::Array locations;
    for (const auto &span : diag.getSpans()) {
      llvm::json::Object loc;
      loc["file"] = span.filename;
      loc["line"] = span.line;
      loc["startColumn"] = span.startColumn;
      loc["endColumn"] = span.endColumn;
      if (!span.label.empty())
        loc["label"] = span.label;
      loc["primary"] = span.isPrimary;
      locations.push_back(std::move(loc));
    }
    obj["locations"] = std::move(locations);
  }

  // Add notes.
  if (!diag.getNotes().empty()) {
    llvm::json::Array notes;
    for (const auto &note : diag.getNotes())
      notes.push_back(note);
    obj["notes"] = std::move(notes);
  }

  // Add helps.
  if (!diag.getHelps().empty()) {
    llvm::json::Array helps;
    for (const auto &help : diag.getHelps())
      helps.push_back(help);
    obj["helps"] = std::move(helps);
  }

  // Add fixes.
  if (!diag.getFixes().empty()) {
    llvm::json::Array fixes;
    for (const auto &fix : diag.getFixes()) {
      llvm::json::Object fixObj;
      fixObj["message"] = fix.message;
      if (fix.span.isValid()) {
        llvm::json::Object spanObj;
        spanObj["file"] = fix.span.filename;
        spanObj["line"] = fix.span.line;
        spanObj["startColumn"] = fix.span.startColumn;
        spanObj["endColumn"] = fix.span.endColumn;
        fixObj["span"] = std::move(spanObj);
      }
      fixObj["replacement"] = fix.replacement;
      fixes.push_back(std::move(fixObj));
    }
    obj["fixes"] = std::move(fixes);
  }

  jsonBuffer.push_back(std::move(obj));
}

void DiagnosticPrinter::printSARIF(const RichDiagnostic &diag) {
  llvm::json::Object result;

  // Map severity to SARIF level.
  StringRef level;
  switch (diag.getSeverity()) {
  case DiagSeverity::Error:
    level = "error";
    break;
  case DiagSeverity::Warning:
    level = "warning";
    break;
  case DiagSeverity::Note:
    level = "note";
    break;
  case DiagSeverity::Hint:
    level = "note";
    break;
  }
  result["level"] = level;

  // Set message.
  llvm::json::Object message;
  message["text"] = diag.getMessage();
  result["message"] = std::move(message);

  // Add locations.
  if (!diag.getSpans().empty()) {
    llvm::json::Array locations;
    for (const auto &span : diag.getSpans()) {
      if (!span.isValid())
        continue;

      llvm::json::Object location;
      llvm::json::Object physicalLocation;
      llvm::json::Object artifactLocation;
      artifactLocation["uri"] = span.filename;
      physicalLocation["artifactLocation"] = std::move(artifactLocation);

      llvm::json::Object region;
      region["startLine"] = span.line;
      region["startColumn"] = span.startColumn;
      region["endLine"] = span.line;
      region["endColumn"] = span.endColumn;
      physicalLocation["region"] = std::move(region);

      location["physicalLocation"] = std::move(physicalLocation);

      if (!span.label.empty()) {
        llvm::json::Object msg;
        msg["text"] = span.label;
        location["message"] = std::move(msg);
      }

      locations.push_back(std::move(location));
    }
    result["locations"] = std::move(locations);
  }

  // Add related locations for notes.
  if (!diag.getNotes().empty()) {
    llvm::json::Array relatedLocations;
    int id = 0;
    for (const auto &note : diag.getNotes()) {
      llvm::json::Object related;
      related["id"] = id++;
      llvm::json::Object msg;
      msg["text"] = note;
      related["message"] = std::move(msg);
      relatedLocations.push_back(std::move(related));
    }
    result["relatedLocations"] = std::move(relatedLocations);
  }

  // Add fixes.
  if (!diag.getFixes().empty()) {
    llvm::json::Array fixes;
    for (const auto &fix : diag.getFixes()) {
      llvm::json::Object fixObj;

      llvm::json::Object description;
      description["text"] = fix.message;
      fixObj["description"] = std::move(description);

      if (fix.span.isValid()) {
        llvm::json::Array artifactChanges;
        llvm::json::Object artifactChange;

        llvm::json::Object artifactLocation;
        artifactLocation["uri"] = fix.span.filename;
        artifactChange["artifactLocation"] = std::move(artifactLocation);

        llvm::json::Array replacements;
        llvm::json::Object replacement;

        llvm::json::Object deletedRegion;
        deletedRegion["startLine"] = fix.span.line;
        deletedRegion["startColumn"] = fix.span.startColumn;
        deletedRegion["endLine"] = fix.span.line;
        deletedRegion["endColumn"] = fix.span.endColumn;
        replacement["deletedRegion"] = std::move(deletedRegion);

        llvm::json::Object insertedContent;
        insertedContent["text"] = fix.replacement;
        replacement["insertedContent"] = std::move(insertedContent);

        replacements.push_back(std::move(replacement));
        artifactChange["replacements"] = std::move(replacements);
        artifactChanges.push_back(std::move(artifactChange));
        fixObj["artifactChanges"] = std::move(artifactChanges);
      }

      fixes.push_back(std::move(fixObj));
    }
    result["fixes"] = std::move(fixes);
  }

  jsonBuffer.push_back(std::move(result));
}

//===----------------------------------------------------------------------===//
// RichDiagnosticHandler
//===----------------------------------------------------------------------===//

RichDiagnosticHandler::RichDiagnosticHandler(MLIRContext *ctx,
                                             DiagnosticPrinter &printer)
    : context(ctx), printer(printer) {
  // Register the handler with the MLIR context.
  handlerID = ctx->getDiagEngine().registerHandler(
      [this](Diagnostic &diag) { handleDiagnostic(diag); return success(); });
}

RichDiagnosticHandler::~RichDiagnosticHandler() {
  context->getDiagEngine().eraseHandler(handlerID);
}

void RichDiagnosticHandler::handleDiagnostic(Diagnostic &diag) {
  // Map MLIR severity to our severity.
  DiagSeverity severity;
  switch (diag.getSeverity()) {
  case mlir::DiagnosticSeverity::Error:
    severity = DiagSeverity::Error;
    break;
  case mlir::DiagnosticSeverity::Warning:
    severity = DiagSeverity::Warning;
    break;
  case mlir::DiagnosticSeverity::Remark:
    severity = DiagSeverity::Note;
    break;
  case mlir::DiagnosticSeverity::Note:
    severity = DiagSeverity::Note;
    break;
  }

  // Create rich diagnostic.
  std::string message;
  llvm::raw_string_ostream msgStream(message);
  msgStream << diag;

  RichDiagnostic richDiag(severity, message);

  // Add primary location.
  if (auto span = SourceSpan::fromLocation(diag.getLocation()))
    richDiag.addPrimarySpan(*span);

  // Add notes from attached notes.
  for (auto &note : diag.getNotes()) {
    std::string noteMsg;
    llvm::raw_string_ostream noteMsgStream(noteMsg);
    noteMsgStream << note;

    // If the note has a location, add it as a secondary span.
    if (auto span = SourceSpan::fromLocation(note.getLocation(), noteMsg, false))
      richDiag.addSecondarySpan(*span);
    else
      richDiag.addNote(noteMsg);
  }

  printer.print(richDiag);
}

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

SourceSpan circt::createSpan(FileLineColLoc loc, unsigned length,
                             StringRef label, bool isPrimary) {
  return SourceSpan(loc.getFilename(), loc.getLine(), loc.getColumn(),
                    loc.getColumn() + length - 1, label, isPrimary);
}

SourceSpan circt::createSpan(FileLineColLoc start, FileLineColLoc end,
                             StringRef label, bool isPrimary) {
  // Note: This assumes start and end are on the same line.
  return SourceSpan(start.getFilename(), start.getLine(), start.getColumn(),
                    end.getColumn(), label, isPrimary);
}

std::optional<DiagnosticOutputFormat>
circt::parseDiagnosticOutputFormat(StringRef str) {
  if (str == "terminal" || str == "term")
    return DiagnosticOutputFormat::Terminal;
  if (str == "plain" || str == "text")
    return DiagnosticOutputFormat::Plain;
  if (str == "json")
    return DiagnosticOutputFormat::JSON;
  if (str == "sarif")
    return DiagnosticOutputFormat::SARIF;
  return std::nullopt;
}

StringRef circt::getDiagnosticOutputFormatString(DiagnosticOutputFormat format) {
  switch (format) {
  case DiagnosticOutputFormat::Terminal:
    return "terminal";
  case DiagnosticOutputFormat::Plain:
    return "plain";
  case DiagnosticOutputFormat::JSON:
    return "json";
  case DiagnosticOutputFormat::SARIF:
    return "sarif";
  }
  return "unknown";
}
