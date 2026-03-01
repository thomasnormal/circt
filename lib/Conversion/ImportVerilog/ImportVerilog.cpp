//===- ImportVerilog.cpp - Slang Verilog frontend integration -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements bridging from the slang Verilog frontend to CIRCT dialects.
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "circt/Conversion/MooreToCore.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/Passes.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include <cctype>

#include "slang/ast/Compilation.h"
#include "slang/analysis/AnalysisManager.h"
#include "slang/ast/SystemSubroutine.h"
#include "slang/diagnostics/AnalysisDiags.h"
#include "slang/diagnostics/DiagnosticClient.h"
#include "slang/diagnostics/DeclarationsDiags.h"
#include "slang/diagnostics/ExpressionsDiags.h"
#include "slang/diagnostics/PreprocessorDiags.h"
#include "slang/diagnostics/StatementsDiags.h"
#include "slang/driver/Driver.h"
#include "slang/parsing/Preprocessor.h"
#include "slang/syntax/SyntaxPrinter.h"
#include "slang/util/Bag.h"
#include "slang/util/VersionInfo.h"

using namespace mlir;
using namespace circt;
using namespace ImportVerilog;

using llvm::SourceMgr;

std::string circt::getSlangVersion() {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  os << "slang version ";
  os << slang::VersionInfo::getMajor() << ".";
  os << slang::VersionInfo::getMinor() << ".";
  os << slang::VersionInfo::getPatch() << "+";
  os << slang::VersionInfo::getHash();
  return buffer;
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

/// Convert a slang `SourceLocation` to an MLIR `Location`.
static Location convertLocation(MLIRContext *context,
                                const slang::SourceManager &sourceManager,
                                slang::SourceLocation loc) {
  if (loc && loc.buffer() != slang::SourceLocation::NoLocation.buffer()) {
    auto fileName = sourceManager.getFileName(loc);
    auto line = sourceManager.getLineNumber(loc);
    auto column = sourceManager.getColumnNumber(loc);
    return FileLineColLoc::get(context, fileName, line, column);
  }
  return UnknownLoc::get(context);
}

Location Context::convertLocation(slang::SourceLocation loc) {
  return ::convertLocation(getContext(), sourceManager, loc);
}

Location Context::convertLocation(slang::SourceRange range) {
  return convertLocation(range.start());
}

namespace {
static bool isIdentifierChar(char c) {
  unsigned char uc = static_cast<unsigned char>(c);
  return std::isalnum(uc) || c == '_' || c == '$';
}

static bool isIdentifierStartChar(char c) {
  unsigned char uc = static_cast<unsigned char>(c);
  return std::isalpha(uc) || c == '_' || c == '$';
}

/// Skip whitespace and comments. Returns std::nullopt for unterminated block
/// comments.
static std::optional<size_t> skipTrivia(StringRef text, size_t i) {
  while (i < text.size()) {
    if (std::isspace(static_cast<unsigned char>(text[i]))) {
      ++i;
      continue;
    }
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      i += 2;
      while (i < text.size() && text[i] != '\n')
        ++i;
      continue;
    }
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      i += 2;
      while (i + 1 < text.size() &&
             !(text[i] == '*' && text[i + 1] == '/'))
        ++i;
      if (i + 1 >= text.size())
        return std::nullopt;
      i += 2;
      continue;
    }
    break;
  }
  return i;
}

/// Scan from `openIdx` (which must point at `openChar`) to the matching
/// `closeChar`, skipping over strings and comments. Returns std::nullopt on
/// malformed / unterminated input.
static std::optional<size_t> scanBalanced(StringRef text, size_t openIdx,
                                          char openChar, char closeChar) {
  if (openIdx >= text.size() || text[openIdx] != openChar)
    return std::nullopt;

  size_t depth = 1;
  for (size_t i = openIdx + 1; i < text.size(); ++i) {
    char c = text[i];
    if (c == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      i += 2;
      while (i < text.size() && text[i] != '\n')
        ++i;
      continue;
    }
    if (c == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      i += 2;
      while (i + 1 < text.size() &&
             !(text[i] == '*' && text[i + 1] == '/'))
        ++i;
      if (i + 1 >= text.size())
        return std::nullopt;
      ++i;
      continue;
    }
    if (c == '"') {
      ++i;
      while (i < text.size()) {
        if (text[i] == '\\' && i + 1 < text.size()) {
          i += 2;
          continue;
        }
        if (text[i] == '"')
          break;
        ++i;
      }
      if (i >= text.size())
        return std::nullopt;
      continue;
    }

    if (c == openChar) {
      ++depth;
      continue;
    }
    if (c == closeChar) {
      if (--depth == 0)
        return i;
    }
  }
  return std::nullopt;
}

/// Return true for format specifiers where slang rejects width modifiers but
/// major simulators accept and ignore them.
static bool isWidthIgnoredFormatSpecifier(char c) {
  switch (std::tolower(static_cast<unsigned char>(c))) {
  case 'c':
  case 'p':
  case 'u':
  case 'z':
  case 'v':
  case 'm':
  case 'l':
    return true;
  default:
    return false;
  }
}

/// Map compatibility-only format specifiers to a supported equivalent.
static char mapFormatSpecifierCompat(char c) {
  switch (c) {
  case 'n':
    return 'u';
  case 'N':
    return 'U';
  default:
    return c;
  }
}

/// Rewrite unsupported width / alignment modifiers for a subset of format
/// specifiers inside a *string literal body*.
static std::string rewriteFormatWidthCompatLiteral(StringRef literal,
                                                   bool &changed) {
  std::string out;
  out.reserve(literal.size());

  for (size_t i = 0; i < literal.size();) {
    if (literal[i] != '%') {
      out.push_back(literal[i]);
      ++i;
      continue;
    }

    size_t start = i++;
    if (i >= literal.size()) {
      out.push_back('%');
      break;
    }
    if (literal[i] == '%') {
      out.append("%%");
      ++i;
      continue;
    }

    bool leftJustify = false;
    bool zeroPad = false;
    while (i < literal.size()) {
      if (literal[i] == '-' && !leftJustify) {
        leftJustify = true;
        ++i;
        continue;
      }
      if (literal[i] == '0' && !zeroPad) {
        zeroPad = true;
        ++i;
        continue;
      }
      break;
    }

    size_t widthStart = i;
    while (i < literal.size() &&
           std::isdigit(static_cast<unsigned char>(literal[i])))
      ++i;
    bool hasWidth = i > widthStart;

    bool hasPrecision = false;
    if (i < literal.size() && literal[i] == '.') {
      hasPrecision = true;
      ++i;
      while (i < literal.size() &&
             std::isdigit(static_cast<unsigned char>(literal[i])))
        ++i;
    }

    if (i >= literal.size()) {
      out.append(literal.substr(start));
      break;
    }

    char spec = literal[i];
    char compatSpec = mapFormatSpecifierCompat(spec);
    bool mappedSpecifier = compatSpec != spec;
    if (mappedSpecifier)
      changed = true;
    spec = compatSpec;

    if (!hasPrecision && isWidthIgnoredFormatSpecifier(spec) &&
        (leftJustify || zeroPad || hasWidth)) {
      changed = true;
      out.push_back('%');
      out.push_back(spec);
      ++i;
      continue;
    }

    if (mappedSpecifier) {
      out.append(literal.substr(start, i - start));
      out.push_back(spec);
      ++i;
      continue;
    }

    out.append(literal.substr(start, i - start + 1));
    ++i;
  }

  return out;
}

/// Rewrite format-width compatibility in all string literals in `text`.
static std::string rewriteFormatWidthCompatLiterals(StringRef text,
                                                    bool &changed) {
  std::string out;
  out.reserve(text.size());

  for (size_t i = 0; i < text.size();) {
    char c = text[i];

    if (c == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      size_t start = i;
      i += 2;
      while (i < text.size() && text[i] != '\n')
        ++i;
      out.append(text.substr(start, i - start));
      continue;
    }
    if (c == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      size_t start = i;
      i += 2;
      while (i + 1 < text.size() && !(text[i] == '*' && text[i + 1] == '/'))
        ++i;
      if (i + 1 < text.size())
        i += 2;
      out.append(text.substr(start, i - start));
      continue;
    }
    if (c != '"') {
      out.push_back(c);
      ++i;
      continue;
    }

    size_t strStart = i++;
    while (i < text.size()) {
      if (text[i] == '\\' && i + 1 < text.size()) {
        i += 2;
        continue;
      }
      if (text[i] == '"')
        break;
      ++i;
    }
    if (i >= text.size()) {
      out.append(text.substr(strStart));
      break;
    }

    size_t strEnd = i;
    bool literalChanged = false;
    StringRef literal = text.slice(strStart + 1, strEnd);
    auto rewrittenLiteral = rewriteFormatWidthCompatLiteral(literal,
                                                            literalChanged);
    if (!literalChanged) {
      out.append(text.substr(strStart, strEnd - strStart + 1));
    } else {
      changed = true;
      out.push_back('"');
      out += rewrittenLiteral;
      out.push_back('"');
    }

    i = strEnd + 1;
  }

  return out;
}

static bool isFormatSystemTaskName(StringRef name) {
  return llvm::StringSwitch<bool>(name.lower())
      .Cases({"display", "displayb", "displayh", "displayo"}, true)
      .Cases({"write", "writeb", "writeh", "writeo"}, true)
      .Cases({"strobe", "strobeb", "strobeh", "strobeo"}, true)
      .Cases({"monitor", "monitorb", "monitorh", "monitoro"}, true)
      .Cases({"fdisplay", "fdisplayb", "fdisplayh", "fdisplayo"}, true)
      .Cases({"fwrite", "fwriteb", "fwriteh", "fwriteo"}, true)
      .Cases({"fstrobe", "fstrobeb", "fstrobeh", "fstrobeo"}, true)
      .Cases({"fmonitor", "fmonitorb", "fmonitorh", "fmonitoro"}, true)
      .Cases({"swrite", "sformat", "sformatf"}, true)
      .Default(false);
}

/// Apply format-width compatibility rewrites to argument lists of format
/// system calls only, preserving all other string literals unchanged.
static std::string rewriteFormatWidthCompat(StringRef text, bool &changed) {
  std::string out;
  out.reserve(text.size());

  for (size_t i = 0; i < text.size();) {
    char c = text[i];

    if (c == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      size_t start = i;
      i += 2;
      while (i < text.size() && text[i] != '\n')
        ++i;
      out.append(text.substr(start, i - start));
      continue;
    }
    if (c == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      size_t start = i;
      i += 2;
      while (i + 1 < text.size() && !(text[i] == '*' && text[i + 1] == '/'))
        ++i;
      if (i + 1 < text.size())
        i += 2;
      out.append(text.substr(start, i - start));
      continue;
    }
    if (c == '"') {
      size_t start = i++;
      while (i < text.size()) {
        if (text[i] == '\\' && i + 1 < text.size()) {
          i += 2;
          continue;
        }
        if (text[i] == '"') {
          ++i;
          break;
        }
        ++i;
      }
      out.append(text.substr(start, i - start));
      continue;
    }
    if (c != '$') {
      out.push_back(c);
      ++i;
      continue;
    }

    size_t nameStart = i + 1;
    if (nameStart >= text.size() || !isIdentifierChar(text[nameStart])) {
      out.push_back(c);
      ++i;
      continue;
    }
    size_t nameEnd = nameStart + 1;
    while (nameEnd < text.size() && isIdentifierChar(text[nameEnd]))
      ++nameEnd;

    StringRef callName = text.slice(nameStart, nameEnd);
    if (!isFormatSystemTaskName(callName)) {
      out.append(text.substr(i, nameEnd - i));
      i = nameEnd;
      continue;
    }

    auto maybeOpenParen = skipTrivia(text, nameEnd);
    if (!maybeOpenParen || *maybeOpenParen >= text.size() ||
        text[*maybeOpenParen] != '(') {
      out.append(text.substr(i, nameEnd - i));
      i = nameEnd;
      continue;
    }

    auto maybeCloseParen = scanBalanced(text, *maybeOpenParen, '(', ')');
    if (!maybeCloseParen) {
      out.append(text.substr(i));
      break;
    }

    out.append(text.substr(i, *maybeOpenParen + 1 - i));
    bool argsChanged = false;
    StringRef args = text.slice(*maybeOpenParen + 1, *maybeCloseParen);
    auto rewrittenArgs = rewriteFormatWidthCompatLiterals(args, argsChanged);
    if (!argsChanged) {
      out.append(args);
    } else {
      changed = true;
      out += rewrittenArgs;
    }
    out.push_back(')');
    i = *maybeCloseParen + 1;
  }

  return out;
}

/// Rewrite UDP table `z` symbols to `x` for parser compatibility.
///
/// IEEE 1800-2023 §29.3.5 describes `z` in UDP table entries as illegal and
/// states that `z` values passed to UDP inputs are treated like `x`. Some
/// simulators accept `z` in table rows as a compatibility extension. Slang
/// rejects these rows during parsing, so canonicalize `z/Z` to `x/X` only
/// within `primitive ... table ... endtable ... endprimitive` regions.
static std::string rewriteUDPZCompat(StringRef text, bool &changed) {
  auto isWordStart = [](char c) {
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
  };
  auto isWordBody = [](char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$';
  };

  std::string out;
  out.reserve(text.size());

  bool inPrimitive = false;
  bool inTable = false;

  for (size_t i = 0; i < text.size();) {
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      size_t start = i;
      i += 2;
      while (i < text.size() && text[i] != '\n')
        ++i;
      out.append(text.substr(start, i - start));
      continue;
    }
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      size_t start = i;
      i += 2;
      while (i + 1 < text.size() && !(text[i] == '*' && text[i + 1] == '/'))
        ++i;
      if (i + 1 < text.size())
        i += 2;
      out.append(text.substr(start, i - start));
      continue;
    }
    if (text[i] == '"') {
      size_t start = i++;
      while (i < text.size()) {
        if (text[i] == '\\' && i + 1 < text.size()) {
          i += 2;
          continue;
        }
        if (text[i] == '"') {
          ++i;
          break;
        }
        ++i;
      }
      out.append(text.substr(start, i - start));
      continue;
    }

    if (isWordStart(text[i])) {
      size_t start = i++;
      while (i < text.size() && isWordBody(text[i]))
        ++i;
      StringRef word = text.slice(start, i);
      if (word == "primitive") {
        inPrimitive = true;
        inTable = false;
      } else if (word == "endprimitive") {
        inPrimitive = false;
        inTable = false;
      } else if (inPrimitive && word == "table") {
        inTable = true;
      } else if (inPrimitive && word == "endtable") {
        inTable = false;
      }
      if (inPrimitive && inTable && (word == "z" || word == "Z")) {
        out.push_back(word.front() == 'z' ? 'x' : 'X');
        changed = true;
      } else {
        out.append(word);
      }
      continue;
    }

    char c = text[i++];
    if (inPrimitive && inTable && (c == 'z' || c == 'Z')) {
      out.push_back(c == 'z' ? 'x' : 'X');
      changed = true;
    } else {
      out.push_back(c);
    }
  }

  return out;
}

/// Rewrite `this.` references inside a randomize-with inline constraint body.
/// This is a compatibility workaround for class randomize calls on array
/// elements where slang currently resolves `this` to the array symbol instead
/// of the element handle.
static std::string rewriteConstraintThisDot(StringRef body, bool &changed) {
  std::string out;
  out.reserve(body.size());

  for (size_t i = 0; i < body.size();) {
    char c = body[i];

    if (c == '/' && i + 1 < body.size() && body[i + 1] == '/') {
      size_t start = i;
      i += 2;
      while (i < body.size() && body[i] != '\n')
        ++i;
      out.append(body.substr(start, i - start));
      continue;
    }
    if (c == '/' && i + 1 < body.size() && body[i + 1] == '*') {
      size_t start = i;
      i += 2;
      while (i + 1 < body.size() &&
             !(body[i] == '*' && body[i + 1] == '/'))
        ++i;
      if (i + 1 < body.size())
        i += 2;
      out.append(body.substr(start, i - start));
      continue;
    }
    if (c == '"') {
      size_t start = i++;
      while (i < body.size()) {
        if (body[i] == '\\' && i + 1 < body.size()) {
          i += 2;
          continue;
        }
        if (body[i] == '"') {
          ++i;
          break;
        }
        ++i;
      }
      out.append(body.substr(start, i - start));
      continue;
    }

    if (i + 4 <= body.size() && body.substr(i, 4) == "this" &&
        (i == 0 || !isIdentifierChar(body[i - 1])) &&
        (i + 4 == body.size() || !isIdentifierChar(body[i + 4]))) {
      auto maybeJ = skipTrivia(body, i + 4);
      if (maybeJ && *maybeJ < body.size() && body[*maybeJ] == '.') {
        changed = true;
        i = *maybeJ + 1;
        continue;
      }
    }

    out.push_back(c);
    ++i;
  }

  return out;
}

/// Apply `this.` inline-constraint compatibility rewrites for class randomize
/// calls, i.e. `obj.randomize(...) with { ... }`.
static std::string rewriteRandomizeInlineConstraints(StringRef text,
                                                     bool &changed) {
  std::string out;
  out.reserve(text.size());

  auto matchRandomizeAfterDot = [&](size_t dotIdx) -> std::optional<size_t> {
    if (dotIdx >= text.size() || text[dotIdx] != '.')
      return std::nullopt;
    auto maybeNameStart = skipTrivia(text, dotIdx + 1);
    if (!maybeNameStart)
      return std::nullopt;
    size_t nameStart = *maybeNameStart;
    constexpr StringLiteral token = "randomize";
    if (nameStart + token.size() > text.size())
      return std::nullopt;
    if (text.substr(nameStart, token.size()) != token)
      return std::nullopt;
    size_t end = nameStart + token.size();
    if (end < text.size() && isIdentifierChar(text[end]))
      return std::nullopt;
    return end;
  };

  for (size_t i = 0; i < text.size();) {
    auto maybeAfterName = matchRandomizeAfterDot(i);
    if (!maybeAfterName) {
      out.push_back(text[i]);
      ++i;
      continue;
    }

    size_t afterName = *maybeAfterName;
    auto maybeOpenParen = skipTrivia(text, afterName);
    if (!maybeOpenParen) {
      out.push_back(text[i]);
      ++i;
      continue;
    }
    size_t openParen = *maybeOpenParen;
    if (openParen >= text.size() || text[openParen] != '(') {
      out.push_back(text[i]);
      ++i;
      continue;
    }

    auto closeParen = scanBalanced(text, openParen, '(', ')');
    if (!closeParen) {
      out.push_back(text[i]);
      ++i;
      continue;
    }

    auto maybeAfterCall = skipTrivia(text, *closeParen + 1);
    if (!maybeAfterCall) {
      out.push_back(text[i]);
      ++i;
      continue;
    }
    size_t afterCall = *maybeAfterCall;
    if (afterCall + 4 > text.size() || text.substr(afterCall, 4) != "with" ||
        (afterCall + 4 < text.size() && isIdentifierChar(text[afterCall + 4]))) {
      out.append(text.substr(i, *closeParen + 1 - i));
      i = *closeParen + 1;
      continue;
    }

    auto maybeOpenBrace = skipTrivia(text, afterCall + 4);
    if (!maybeOpenBrace) {
      out.push_back(text[i]);
      ++i;
      continue;
    }
    size_t openBrace = *maybeOpenBrace;
    if (openBrace >= text.size() || text[openBrace] != '{') {
      out.append(text.substr(i, *closeParen + 1 - i));
      i = *closeParen + 1;
      continue;
    }

    auto closeBrace = scanBalanced(text, openBrace, '{', '}');
    if (!closeBrace) {
      out.append(text.substr(i, openBrace + 1 - i));
      i = openBrace + 1;
      continue;
    }

    out.append(text.substr(i, openBrace + 1 - i));
    bool blockChanged = false;
    auto body = text.slice(openBrace + 1, *closeBrace);
    out += rewriteConstraintThisDot(body, blockChanged);
    changed |= blockChanged;
    out.push_back('}');
    i = *closeBrace + 1;
  }

  return out;
}

static bool isConfigKeywordIdentifierCompat(StringRef word) {
  return word == "cell" || word == "config" || word == "design" ||
         word == "endconfig" || word == "incdir" || word == "include" ||
         word == "instance" || word == "liblist" || word == "library" ||
         word == "use";
}

/// Return true if this source likely contains configuration / library
/// compilation-unit syntax where config keywords must retain keyword meaning.
static bool hasProbableConfigCompilationUnit(StringRef text) {
  auto startsWithUnitKeyword = [](StringRef line, StringRef keyword) {
    if (!line.starts_with(keyword))
      return false;
    if (line.size() == keyword.size())
      return true;
    return std::isspace(static_cast<unsigned char>(line[keyword.size()])) != 0;
  };
  auto isLikelyCompilationUnitKeywordUse = [&](StringRef line,
                                               StringRef keyword) {
    if (!startsWithUnitKeyword(line, keyword))
      return false;

    line = line.drop_front(keyword.size()).ltrim();
    if (line.empty())
      return true;

    // Procedural uses like `config = ...` / `library <= ...` should not be
    // treated as compilation-unit directives.
    char first = line.front();
    if (first == '=' || first == ',' || first == '.' || first == ':' ||
        first == ')' || first == ']' || first == '}' || first == '+' ||
        first == '-' || first == '*' || first == '/' || first == '%' ||
        first == '&' || first == '|' || first == '^' || first == '?')
      return false;
    if (first == '<' && line.size() > 1 && line[1] == '=')
      return false;

    // Guard against assignment-looking lines.
    size_t semi = line.find(';');
    size_t eq = line.find('=');
    if (eq != StringRef::npos && (semi == StringRef::npos || eq < semi))
      return false;

    if (keyword == "include")
      return first == '"' || first == '<' || isIdentifierStartChar(first) ||
             first == '\\';

    return isIdentifierStartChar(first) || first == '\\';
  };
  size_t lineStart = 0;
  while (lineStart <= text.size()) {
    size_t lineEnd = text.find('\n', lineStart);
    if (lineEnd == StringRef::npos)
      lineEnd = text.size();
    auto line = text.slice(lineStart, lineEnd).ltrim();
    if (isLikelyCompilationUnitKeywordUse(line, "config") ||
        isLikelyCompilationUnitKeywordUse(line, "library") ||
        isLikelyCompilationUnitKeywordUse(line, "include"))
      return true;
    if (lineEnd == text.size())
      break;
    lineStart = lineEnd + 1;
  }
  return false;
}

/// Rewrite config-reserved keywords that are used as identifiers in design
/// source, matching mainstream simulator compatibility behavior.
static std::string rewriteConfigKeywordIdentifiersCompat(StringRef text,
                                                         bool &changed) {
  if (hasProbableConfigCompilationUnit(text))
    return text.str();

  std::string out;
  out.reserve(text.size());

  for (size_t i = 0; i < text.size();) {
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      size_t start = i;
      i += 2;
      while (i < text.size() && text[i] != '\n')
        ++i;
      out.append(text.slice(start, i));
      continue;
    }
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      size_t start = i;
      i += 2;
      while (i + 1 < text.size() && !(text[i] == '*' && text[i + 1] == '/'))
        ++i;
      if (i + 1 < text.size())
        i += 2;
      out.append(text.slice(start, i));
      continue;
    }
    if (text[i] == '"') {
      size_t start = i++;
      while (i < text.size()) {
        if (text[i] == '\\' && i + 1 < text.size()) {
          i += 2;
          continue;
        }
        if (text[i] == '"') {
          ++i;
          break;
        }
        ++i;
      }
      out.append(text.slice(start, i));
      continue;
    }
    if (text[i] == '`') {
      size_t start = i++;
      while (i < text.size() && text[i] != '\n')
        ++i;
      out.append(text.slice(start, i));
      continue;
    }
    // Escaped identifiers are already explicit and should remain untouched.
    if (text[i] == '\\') {
      size_t start = i++;
      while (i < text.size() &&
             !std::isspace(static_cast<unsigned char>(text[i])))
        ++i;
      out.append(text.slice(start, i));
      continue;
    }
    if (isIdentifierStartChar(text[i])) {
      size_t start = i++;
      while (i < text.size() && isIdentifierChar(text[i]))
        ++i;
      StringRef word = text.slice(start, i);
      if (isConfigKeywordIdentifierCompat(word)) {
        out.append("__circt_cfgkw_");
        out.append(word);
        changed = true;
      } else {
        out.append(word);
      }
      continue;
    }
    out.push_back(text[i++]);
  }

  return out;
}

/// Rewrite unspecialized generic class scope references `C::x` into
/// `C#()::x`, where `C` is declared as a parameterized class.
static std::string rewriteGenericClassScopeCompat(StringRef text,
                                                  bool &changed) {
  llvm::StringSet<> parameterizedClassNames;

  auto skipCommentOrString = [&](size_t &i) -> bool {
    if (i >= text.size())
      return false;
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      i += 2;
      while (i < text.size() && text[i] != '\n')
        ++i;
      return true;
    }
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      i += 2;
      while (i + 1 < text.size() &&
             !(text[i] == '*' && text[i + 1] == '/'))
        ++i;
      if (i + 1 < text.size())
        i += 2;
      return true;
    }
    if (text[i] == '"') {
      ++i;
      while (i < text.size()) {
        if (text[i] == '\\' && i + 1 < text.size()) {
          i += 2;
          continue;
        }
        if (text[i] == '"') {
          ++i;
          break;
        }
        ++i;
      }
      return true;
    }
    return false;
  };

  auto scanIdentifier = [&](size_t start,
                            std::optional<StringRef> &ident) -> size_t {
    ident = std::nullopt;
    if (start >= text.size() || !isIdentifierStartChar(text[start]))
      return start;
    size_t end = start + 1;
    while (end < text.size() && isIdentifierChar(text[end]))
      ++end;
    ident = text.slice(start, end);
    return end;
  };

  // Pass 1: collect parameterized class names declared in this source.
  for (size_t i = 0; i < text.size();) {
    if (skipCommentOrString(i))
      continue;
    if (!isIdentifierStartChar(text[i])) {
      ++i;
      continue;
    }
    size_t wordStart = i;
    while (i < text.size() && isIdentifierChar(text[i]))
      ++i;
    StringRef word = text.slice(wordStart, i);
    if (word != "class")
      continue;

    auto maybeNameStart = skipTrivia(text, i);
    if (!maybeNameStart)
      continue;
    size_t cursor = *maybeNameStart;

    std::optional<StringRef> token;
    cursor = scanIdentifier(cursor, token);
    if (!token)
      continue;
    if (*token == "static" || *token == "automatic") {
      auto maybeAfterLifetime = skipTrivia(text, cursor);
      if (!maybeAfterLifetime)
        continue;
      cursor = scanIdentifier(*maybeAfterLifetime, token);
      if (!token)
        continue;
    }

    StringRef className = *token;
    auto maybeAfterName = skipTrivia(text, cursor);
    if (!maybeAfterName || *maybeAfterName >= text.size())
      continue;
    if (text[*maybeAfterName] == '#')
      parameterizedClassNames.insert(className);
  }

  if (parameterizedClassNames.empty())
    return text.str();

  // Pass 2: rewrite unspecialized generic class scope references.
  std::string out;
  out.reserve(text.size());
  for (size_t i = 0; i < text.size();) {
    size_t before = i;
    if (skipCommentOrString(i)) {
      out.append(text.slice(before, i));
      continue;
    }
    if (!isIdentifierStartChar(text[i])) {
      out.push_back(text[i]);
      ++i;
      continue;
    }

    size_t identStart = i;
    while (i < text.size() && isIdentifierChar(text[i]))
      ++i;
    StringRef ident = text.slice(identStart, i);
    out.append(ident);

    if (!parameterizedClassNames.contains(ident))
      continue;

    auto maybeAfterIdent = skipTrivia(text, i);
    if (!maybeAfterIdent)
      continue;
    size_t afterIdent = *maybeAfterIdent;
    if (afterIdent + 1 >= text.size())
      continue;
    if (text[afterIdent] == '#' || text[afterIdent] != ':' ||
        text[afterIdent + 1] != ':')
      continue;

    out.append("#()");
    changed = true;
  }

  return out;
}

/// Split a comma-separated argument list while honoring nested delimiters,
/// comments, and strings.
static bool
splitTopLevelArgumentRanges(StringRef text,
                            SmallVectorImpl<std::pair<size_t, size_t>> &ranges) {
  size_t argStart = 0;
  unsigned parenDepth = 0;
  unsigned bracketDepth = 0;
  unsigned braceDepth = 0;

  for (size_t i = 0; i < text.size(); ++i) {
    char c = text[i];

    if (c == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      i += 2;
      while (i < text.size() && text[i] != '\n')
        ++i;
      continue;
    }
    if (c == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      i += 2;
      while (i + 1 < text.size() && !(text[i] == '*' && text[i + 1] == '/'))
        ++i;
      if (i + 1 >= text.size())
        return false;
      ++i;
      continue;
    }
    if (c == '"') {
      ++i;
      while (i < text.size()) {
        if (text[i] == '\\' && i + 1 < text.size()) {
          i += 2;
          continue;
        }
        if (text[i] == '"')
          break;
        ++i;
      }
      if (i >= text.size())
        return false;
      continue;
    }

    switch (c) {
    case '(':
      ++parenDepth;
      break;
    case ')':
      if (parenDepth == 0)
        return false;
      --parenDepth;
      break;
    case '[':
      ++bracketDepth;
      break;
    case ']':
      if (bracketDepth == 0)
        return false;
      --bracketDepth;
      break;
    case '{':
      ++braceDepth;
      break;
    case '}':
      if (braceDepth == 0)
        return false;
      --braceDepth;
      break;
    case ',':
      if (parenDepth == 0 && bracketDepth == 0 && braceDepth == 0) {
        ranges.emplace_back(argStart, i);
        argStart = i + 1;
      }
      break;
    default:
      break;
    }
  }

  if (parenDepth != 0 || bracketDepth != 0 || braceDepth != 0)
    return false;
  ranges.emplace_back(argStart, text.size());
  return true;
}

/// Rewrite compatibility form `$past(expr, ticks, @(clock))` into standard
/// `$past(expr, ticks, , @(clock))`.
static std::string rewritePastClockingArgCompat(StringRef text, bool &changed) {
  std::string out;
  out.reserve(text.size());

  auto isPastCallAt = [&](size_t idx) {
    if (idx + 5 > text.size() || text.substr(idx, 5) != "$past")
      return false;
    return idx + 5 == text.size() || !isIdentifierChar(text[idx + 5]);
  };

  auto isClockingArgument = [&](StringRef arg) {
    auto maybe = skipTrivia(arg, 0);
    return maybe && *maybe < arg.size() && arg[*maybe] == '@';
  };

  for (size_t i = 0; i < text.size();) {
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      size_t start = i;
      i += 2;
      while (i < text.size() && text[i] != '\n')
        ++i;
      out.append(text.slice(start, i));
      continue;
    }
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      size_t start = i;
      i += 2;
      while (i + 1 < text.size() && !(text[i] == '*' && text[i + 1] == '/'))
        ++i;
      if (i + 1 < text.size())
        i += 2;
      out.append(text.slice(start, i));
      continue;
    }
    if (text[i] == '"') {
      size_t start = i++;
      while (i < text.size()) {
        if (text[i] == '\\' && i + 1 < text.size()) {
          i += 2;
          continue;
        }
        if (text[i] == '"') {
          ++i;
          break;
        }
        ++i;
      }
      out.append(text.slice(start, i));
      continue;
    }

    if (!isPastCallAt(i)) {
      out.push_back(text[i]);
      ++i;
      continue;
    }

    size_t pastEnd = i + 5;
    auto maybeOpenParen = skipTrivia(text, pastEnd);
    if (!maybeOpenParen || *maybeOpenParen >= text.size() ||
        text[*maybeOpenParen] != '(') {
      out.append(text.slice(i, pastEnd));
      i = pastEnd;
      continue;
    }
    auto maybeCloseParen = scanBalanced(text, *maybeOpenParen, '(', ')');
    if (!maybeCloseParen) {
      out.append(text.drop_front(i));
      break;
    }

    StringRef args = text.slice(*maybeOpenParen + 1, *maybeCloseParen);
    SmallVector<std::pair<size_t, size_t>, 8> argRanges;
    bool isCompatForm = splitTopLevelArgumentRanges(args, argRanges) &&
                        argRanges.size() == 3 &&
                        isClockingArgument(args.slice(argRanges[2].first,
                                                      argRanges[2].second));

    out.append(text.slice(i, *maybeOpenParen + 1));
    if (!isCompatForm) {
      out.append(args);
    } else {
      out.append(args.slice(0, argRanges[2].first));
      out.append(", ");
      out.append(args.drop_front(argRanges[2].first));
      changed = true;
    }
    out.push_back(')');
    i = *maybeCloseParen + 1;
  }

  return out;
}

/// Rewrite compatibility forms like `@posedge (clk)` into standard
/// `@(posedge (clk))` event controls.
static std::string rewriteEventControlParenCompat(StringRef text,
                                                  bool &changed) {
  std::string out;
  out.reserve(text.size());

  auto matchKeyword = [&](size_t start) -> std::optional<StringRef> {
    auto matchOne = [&](StringRef kw) {
      if (start + kw.size() > text.size() || text.substr(start, kw.size()) != kw)
        return false;
      size_t end = start + kw.size();
      return end == text.size() || !isIdentifierChar(text[end]);
    };
    if (matchOne("posedge"))
      return StringRef("posedge");
    if (matchOne("negedge"))
      return StringRef("negedge");
    if (matchOne("edge"))
      return StringRef("edge");
    return std::nullopt;
  };

  for (size_t i = 0; i < text.size();) {
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      size_t start = i;
      i += 2;
      while (i < text.size() && text[i] != '\n')
        ++i;
      out.append(text.slice(start, i));
      continue;
    }
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      size_t start = i;
      i += 2;
      while (i + 1 < text.size() && !(text[i] == '*' && text[i + 1] == '/'))
        ++i;
      if (i + 1 < text.size())
        i += 2;
      out.append(text.slice(start, i));
      continue;
    }
    if (text[i] == '"') {
      size_t start = i++;
      while (i < text.size()) {
        if (text[i] == '\\' && i + 1 < text.size()) {
          i += 2;
          continue;
        }
        if (text[i] == '"') {
          ++i;
          break;
        }
        ++i;
      }
      out.append(text.slice(start, i));
      continue;
    }

    if (text[i] != '@') {
      out.push_back(text[i]);
      ++i;
      continue;
    }

    auto maybeKeywordStart = skipTrivia(text, i + 1);
    if (!maybeKeywordStart) {
      out.push_back(text[i]);
      ++i;
      continue;
    }
    size_t keywordStart = *maybeKeywordStart;
    auto keyword = matchKeyword(keywordStart);
    if (!keyword) {
      out.push_back(text[i]);
      ++i;
      continue;
    }

    auto maybeOpenParen = skipTrivia(text, keywordStart + keyword->size());
    if (!maybeOpenParen || *maybeOpenParen >= text.size() ||
        text[*maybeOpenParen] != '(') {
      out.push_back(text[i]);
      ++i;
      continue;
    }
    auto maybeCloseParen = scanBalanced(text, *maybeOpenParen, '(', ')');
    if (!maybeCloseParen) {
      out.append(text.drop_front(i));
      break;
    }

    out.append("@(");
    out.append(text.slice(keywordStart, *maybeCloseParen + 1));
    out.push_back(')');
    changed = true;
    i = *maybeCloseParen + 1;
  }

  return out;
}

/// Collect simple `property <name>` declarations in this source text.
static void collectPropertyDeclNames(StringRef text,
                                     llvm::StringSet<> &propertyNames) {
  for (size_t i = 0; i < text.size();) {
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      i += 2;
      while (i < text.size() && text[i] != '\n')
        ++i;
      continue;
    }
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      i += 2;
      while (i + 1 < text.size() && !(text[i] == '*' && text[i + 1] == '/'))
        ++i;
      if (i + 1 < text.size())
        i += 2;
      continue;
    }
    if (text[i] == '"') {
      ++i;
      while (i < text.size()) {
        if (text[i] == '\\' && i + 1 < text.size()) {
          i += 2;
          continue;
        }
        if (text[i] == '"') {
          ++i;
          break;
        }
        ++i;
      }
      continue;
    }
    if (!isIdentifierStartChar(text[i])) {
      ++i;
      continue;
    }

    size_t wordStart = i++;
    while (i < text.size() && isIdentifierChar(text[i]))
      ++i;
    StringRef word = text.slice(wordStart, i);
    if (word != "property")
      continue;

    auto maybeNameStart = skipTrivia(text, i);
    if (!maybeNameStart || *maybeNameStart >= text.size())
      continue;
    size_t nameStart = *maybeNameStart;
    if (!isIdentifierStartChar(text[nameStart]))
      continue; // skip `assert property (...)`, etc.
    size_t nameEnd = nameStart + 1;
    while (nameEnd < text.size() && isIdentifierChar(text[nameEnd]))
      ++nameEnd;
    propertyNames.insert(text.slice(nameStart, nameEnd));
  }
}

/// Rewrite open-range unary SVA forms rejected by this slang revision:
/// - eventually[m:$]
/// - s_always[m:$]
/// - nexttime[m:$]
/// - s_nexttime[m:$]
///
/// The rewrite targets common operand forms used in UVM / SVA tests where the
/// operand is a simple identifier or a parenthesized expression.
static std::string rewriteOpenRangeUnarySVACompat(StringRef text,
                                                  bool &changed) {
  llvm::StringSet<> propertyNames;
  collectPropertyDeclNames(text, propertyNames);

  auto parseOpenRange = [&](size_t afterKeyword, StringRef &lowerBound,
                            size_t &afterRange) -> bool {
    auto maybeOpen = skipTrivia(text, afterKeyword);
    if (!maybeOpen || *maybeOpen >= text.size() || text[*maybeOpen] != '[')
      return false;
    size_t open = *maybeOpen;

    auto maybeLowerStart = skipTrivia(text, open + 1);
    if (!maybeLowerStart || *maybeLowerStart >= text.size())
      return false;
    size_t lowerStart = *maybeLowerStart;
    size_t lowerEnd = lowerStart;
    while (lowerEnd < text.size() &&
           (std::isdigit(static_cast<unsigned char>(text[lowerEnd])) ||
            text[lowerEnd] == '_'))
      ++lowerEnd;
    if (lowerEnd == lowerStart)
      return false;

    auto maybeColon = skipTrivia(text, lowerEnd);
    if (!maybeColon || *maybeColon >= text.size() || text[*maybeColon] != ':')
      return false;
    auto maybeDollar = skipTrivia(text, *maybeColon + 1);
    if (!maybeDollar || *maybeDollar >= text.size() || text[*maybeDollar] != '$')
      return false;
    auto maybeClose = skipTrivia(text, *maybeDollar + 1);
    if (!maybeClose || *maybeClose >= text.size() || text[*maybeClose] != ']')
      return false;

    lowerBound = text.slice(lowerStart, lowerEnd);
    afterRange = *maybeClose + 1;
    return true;
  };

  auto parseOperand = [&](size_t start, size_t &end, bool &isIdentifier,
                          StringRef &identifier) -> bool {
    isIdentifier = false;
    identifier = StringRef();
    if (start >= text.size())
      return false;
    if (isIdentifierStartChar(text[start])) {
      size_t identEnd = start + 1;
      while (identEnd < text.size() && isIdentifierChar(text[identEnd]))
        ++identEnd;
      end = identEnd;
      isIdentifier = true;
      identifier = text.slice(start, identEnd);
      return true;
    }
    if (text[start] == '(') {
      auto close = scanBalanced(text, start, '(', ')');
      if (!close)
        return false;
      end = *close + 1;
      return true;
    }
    return false;
  };

  std::string out;
  out.reserve(text.size());

  for (size_t i = 0; i < text.size();) {
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      size_t start = i;
      i += 2;
      while (i < text.size() && text[i] != '\n')
        ++i;
      out.append(text.slice(start, i));
      continue;
    }
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      size_t start = i;
      i += 2;
      while (i + 1 < text.size() && !(text[i] == '*' && text[i + 1] == '/'))
        ++i;
      if (i + 1 < text.size())
        i += 2;
      out.append(text.slice(start, i));
      continue;
    }
    if (text[i] == '"') {
      size_t start = i++;
      while (i < text.size()) {
        if (text[i] == '\\' && i + 1 < text.size()) {
          i += 2;
          continue;
        }
        if (text[i] == '"') {
          ++i;
          break;
        }
        ++i;
      }
      out.append(text.slice(start, i));
      continue;
    }

    if (!isIdentifierStartChar(text[i])) {
      out.push_back(text[i]);
      ++i;
      continue;
    }

    size_t wordStart = i++;
    while (i < text.size() && isIdentifierChar(text[i]))
      ++i;
    StringRef keyword = text.slice(wordStart, i);
    bool isInteresting = keyword == "eventually" || keyword == "s_always" ||
                         keyword == "nexttime" || keyword == "s_nexttime";
    if (!isInteresting) {
      out.append(keyword);
      continue;
    }

    StringRef lowerBound;
    size_t afterRange = i;
    if (!parseOpenRange(i, lowerBound, afterRange)) {
      out.append(keyword);
      continue;
    }

    auto maybeOperandStart = skipTrivia(text, afterRange);
    if (!maybeOperandStart || *maybeOperandStart >= text.size()) {
      out.append(keyword);
      continue;
    }

    size_t operandStart = *maybeOperandStart;
    size_t operandEnd = operandStart;
    bool operandIsIdentifier = false;
    StringRef operandIdentifier;
    if (!parseOperand(operandStart, operandEnd, operandIsIdentifier,
                      operandIdentifier)) {
      out.append(keyword);
      continue;
    }

    StringRef operand = text.slice(operandStart, operandEnd);
    bool operandIsProperty =
        operandIsIdentifier && propertyNames.contains(operandIdentifier);

    if (keyword == "s_nexttime") {
      out.append("s_eventually [");
      out.append(lowerBound);
      out.append(":$] ");
      out.append(operand);
    } else if (keyword == "nexttime" || keyword == "eventually") {
      if (operandIsProperty) {
        out.append("(not always [");
        out.append(lowerBound);
        out.append(":$] (not ");
        out.append(operand);
        out.append("))");
      } else {
        out.append("##[");
        out.append(lowerBound);
        out.append(":$] ");
        out.append(operand);
      }
    } else if (keyword == "s_always") {
      if (operandIsProperty) {
        out.append("(not s_eventually (not (s_nexttime [");
        out.append(lowerBound);
        out.append("] ");
        out.append(operand);
        out.append(")))");
      } else {
        out.append("(not s_eventually (not ((##[");
        out.append(lowerBound);
        out.append("] 1'b1) and ((##[");
        out.append(lowerBound);
        out.append("] 1'b1) |-> ");
        out.append(operand);
        out.append("))))");
      }
    } else {
      llvm_unreachable("covered by isInteresting");
    }

    changed = true;
    i = operandEnd;
  }

  return out;
}

static std::optional<size_t> findTopLevelDefaultEquals(StringRef text) {
  unsigned parenDepth = 0;
  unsigned bracketDepth = 0;
  unsigned braceDepth = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    char c = text[i];

    if (c == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      i += 2;
      while (i < text.size() && text[i] != '\n')
        ++i;
      continue;
    }
    if (c == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      i += 2;
      while (i + 1 < text.size() && !(text[i] == '*' && text[i + 1] == '/'))
        ++i;
      if (i + 1 >= text.size())
        return std::nullopt;
      ++i;
      continue;
    }
    if (c == '"') {
      ++i;
      while (i < text.size()) {
        if (text[i] == '\\' && i + 1 < text.size()) {
          i += 2;
          continue;
        }
        if (text[i] == '"')
          break;
        ++i;
      }
      if (i >= text.size())
        return std::nullopt;
      continue;
    }

    switch (c) {
    case '(':
      ++parenDepth;
      continue;
    case ')':
      if (parenDepth == 0)
        return std::nullopt;
      --parenDepth;
      continue;
    case '[':
      ++bracketDepth;
      continue;
    case ']':
      if (bracketDepth == 0)
        return std::nullopt;
      --bracketDepth;
      continue;
    case '{':
      ++braceDepth;
      continue;
    case '}':
      if (braceDepth == 0)
        return std::nullopt;
      --braceDepth;
      continue;
    case '=':
      break;
    default:
      continue;
    }

    if (parenDepth || bracketDepth || braceDepth)
      continue;
    auto prevPos = skipTrivia(text.slice(0, i), 0);
    char prev = 0;
    if (prevPos) {
      StringRef prefix = text.slice(0, i).rtrim();
      if (!prefix.empty())
        prev = prefix.back();
    }
    auto maybeNext = skipTrivia(text, i + 1);
    char next = 0;
    if (maybeNext && *maybeNext < text.size())
      next = text[*maybeNext];
    if (prev == '=' || prev == '!' || prev == '<' || prev == '>')
      continue;
    if (next == '=' || next == '>')
      continue;
    return i;
  }
  return std::nullopt;
}

/// Compatibility rewrite: strip default value from `uvm_comparer` argument in
/// `do_compare` declarations/definitions.
///
/// Some UVM AVIP code writes `do_compare(..., uvm_comparer comparer = null)`
/// while the base method omits a default. Slang diagnoses this mismatch as an
/// error; mainstream simulators commonly accept it.
static std::string rewriteUvmDoCompareComparerDefaultCompat(StringRef text,
                                                            bool &changed) {
  std::string out;
  out.reserve(text.size());

  auto isDoCompareAt = [&](size_t idx) {
    constexpr StringRef kName = "do_compare";
    if (idx + kName.size() > text.size() ||
        text.substr(idx, kName.size()) != kName)
      return false;
    if (idx > 0 && isIdentifierChar(text[idx - 1]))
      return false;
    size_t end = idx + kName.size();
    return end == text.size() || !isIdentifierChar(text[end]);
  };

  for (size_t i = 0; i < text.size();) {
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      size_t start = i;
      i += 2;
      while (i < text.size() && text[i] != '\n')
        ++i;
      out.append(text.slice(start, i));
      continue;
    }
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      size_t start = i;
      i += 2;
      while (i + 1 < text.size() && !(text[i] == '*' && text[i + 1] == '/'))
        ++i;
      if (i + 1 < text.size())
        i += 2;
      out.append(text.slice(start, i));
      continue;
    }
    if (text[i] == '"') {
      size_t start = i++;
      while (i < text.size()) {
        if (text[i] == '\\' && i + 1 < text.size()) {
          i += 2;
          continue;
        }
        if (text[i] == '"') {
          ++i;
          break;
        }
        ++i;
      }
      out.append(text.slice(start, i));
      continue;
    }

    if (!isDoCompareAt(i)) {
      out.push_back(text[i]);
      ++i;
      continue;
    }

    constexpr size_t kNameLen = sizeof("do_compare") - 1;
    auto maybeOpenParen = skipTrivia(text, i + kNameLen);
    if (!maybeOpenParen || *maybeOpenParen >= text.size() ||
        text[*maybeOpenParen] != '(') {
      out.append(text.slice(i, i + kNameLen));
      i += kNameLen;
      continue;
    }
    auto maybeCloseParen = scanBalanced(text, *maybeOpenParen, '(', ')');
    if (!maybeCloseParen) {
      out.append(text.drop_front(i));
      break;
    }

    StringRef args = text.slice(*maybeOpenParen + 1, *maybeCloseParen);
    SmallVector<std::pair<size_t, size_t>, 8> argRanges;
    if (!splitTopLevelArgumentRanges(args, argRanges)) {
      out.append(text.slice(i, *maybeCloseParen + 1));
      i = *maybeCloseParen + 1;
      continue;
    }

    bool rewrittenSignature = false;
    std::string rewrittenArgs;
    rewrittenArgs.reserve(args.size());
    for (size_t argIdx = 0; argIdx < argRanges.size(); ++argIdx) {
      StringRef arg = args.slice(argRanges[argIdx].first, argRanges[argIdx].second);
      StringRef trimmed = arg.trim();
      std::string argText = arg.str();
      if (trimmed.contains("uvm_comparer")) {
        if (auto eqPos = findTopLevelDefaultEquals(arg)) {
          argText = StringRef(argText).slice(0, *eqPos).rtrim().str();
          rewrittenSignature = true;
        }
      }
      if (argIdx)
        rewrittenArgs.append(", ");
      rewrittenArgs.append(argText);
    }

    out.append(text.slice(i, *maybeOpenParen + 1));
    if (rewrittenSignature) {
      out.append(rewrittenArgs);
      changed = true;
    } else {
      out.append(args);
    }
    out.push_back(')');
    i = *maybeCloseParen + 1;
  }

  return out;
}

/// Compatibility rewrite: avoid name lookup collision in forms like
/// `bins BAUD_4800 = {BAUD_4800};` where the bin identifier shadows the enum
/// literal and slang rejects the RHS expression.
static std::string rewriteCovergroupBinSelfNameCompat(StringRef text,
                                                      bool &changed) {
  std::string out;
  out.reserve(text.size());

  auto isBinsAt = [&](size_t idx) {
    constexpr StringRef kBins = "bins";
    if (idx + kBins.size() > text.size() ||
        text.substr(idx, kBins.size()) != kBins)
      return false;
    if (idx > 0 && isIdentifierChar(text[idx - 1]))
      return false;
    size_t end = idx + kBins.size();
    return end == text.size() || !isIdentifierChar(text[end]);
  };

  for (size_t i = 0; i < text.size();) {
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      size_t start = i;
      i += 2;
      while (i < text.size() && text[i] != '\n')
        ++i;
      out.append(text.slice(start, i));
      continue;
    }
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      size_t start = i;
      i += 2;
      while (i + 1 < text.size() && !(text[i] == '*' && text[i + 1] == '/'))
        ++i;
      if (i + 1 < text.size())
        i += 2;
      out.append(text.slice(start, i));
      continue;
    }
    if (text[i] == '"') {
      size_t start = i++;
      while (i < text.size()) {
        if (text[i] == '\\' && i + 1 < text.size()) {
          i += 2;
          continue;
        }
        if (text[i] == '"') {
          ++i;
          break;
        }
        ++i;
      }
      out.append(text.slice(start, i));
      continue;
    }

    if (!isBinsAt(i)) {
      out.push_back(text[i]);
      ++i;
      continue;
    }

    size_t binsEnd = i + 4;
    auto maybeNameStart = skipTrivia(text, binsEnd);
    if (!maybeNameStart || *maybeNameStart >= text.size() ||
        !isIdentifierStartChar(text[*maybeNameStart])) {
      out.append(text.slice(i, binsEnd));
      i = binsEnd;
      continue;
    }

    size_t nameStart = *maybeNameStart;
    size_t nameEnd = nameStart + 1;
    while (nameEnd < text.size() && isIdentifierChar(text[nameEnd]))
      ++nameEnd;
    StringRef binName = text.slice(nameStart, nameEnd);

    size_t afterName = nameEnd;
    auto maybeArrayOpen = skipTrivia(text, afterName);
    if (maybeArrayOpen && *maybeArrayOpen < text.size() &&
        text[*maybeArrayOpen] == '[') {
      auto maybeArrayClose = scanBalanced(text, *maybeArrayOpen, '[', ']');
      if (!maybeArrayClose) {
        out.append(text.slice(i, binsEnd));
        i = binsEnd;
        continue;
      }
      afterName = *maybeArrayClose + 1;
    }

    auto maybeEq = skipTrivia(text, afterName);
    if (!maybeEq || *maybeEq >= text.size() || text[*maybeEq] != '=') {
      out.append(text.slice(i, binsEnd));
      i = binsEnd;
      continue;
    }
    auto maybeOpenBrace = skipTrivia(text, *maybeEq + 1);
    if (!maybeOpenBrace || *maybeOpenBrace >= text.size() ||
        text[*maybeOpenBrace] != '{') {
      out.append(text.slice(i, binsEnd));
      i = binsEnd;
      continue;
    }
    auto maybeCloseBrace = scanBalanced(text, *maybeOpenBrace, '{', '}');
    if (!maybeCloseBrace) {
      out.append(text.slice(i, binsEnd));
      i = binsEnd;
      continue;
    }

    StringRef valueExpr =
        text.slice(*maybeOpenBrace + 1, *maybeCloseBrace).trim();
    if (valueExpr != binName) {
      out.append(text.slice(i, binsEnd));
      i = binsEnd;
      continue;
    }

    out.append(text.slice(i, nameStart));
    out.append("__circt_bin_");
    out.append(binName);
    out.append(text.slice(nameEnd, *maybeCloseBrace + 1));
    changed = true;
    i = *maybeCloseBrace + 1;
  }

  return out;
}

/// A converter that can be plugged into a slang `DiagnosticEngine` as a client
/// that will map slang diagnostics to their MLIR counterpart and emit them.
class MlirDiagnosticClient : public slang::DiagnosticClient {
public:
  MlirDiagnosticClient(MLIRContext *context) : context(context) {}

  void report(const slang::ReportedDiagnostic &diag) override {
    // Generate the primary MLIR diagnostic.
    auto &diagEngine = context->getDiagEngine();
    auto mlirDiag = diagEngine.emit(convertLocation(diag.location),
                                    getSeverity(diag.severity));
    mlirDiag << diag.formattedMessage;

    // Append the name of the option that can be used to control this
    // diagnostic.
    auto optionName = engine->getOptionName(diag.originalDiagnostic.code);
    if (!optionName.empty())
      mlirDiag << " [-W" << optionName << "]";

    // Write out macro expansions, if we have any, in reverse order.
    for (auto loc : std::views::reverse(diag.expansionLocs)) {
      auto &note = mlirDiag.attachNote(
          convertLocation(sourceManager->getFullyOriginalLoc(loc)));
      auto macroName = sourceManager->getMacroName(loc);
      if (macroName.empty())
        note << "expanded from here";
      else
        note << "expanded from macro '" << macroName << "'";
    }

    // Write out the include stack.
    slang::SmallVector<slang::SourceLocation> includeStack;
    getIncludeStack(diag.location.buffer(), includeStack);
    for (auto &loc : std::views::reverse(includeStack))
      mlirDiag.attachNote(convertLocation(loc)) << "included from here";
  }

  /// Convert a slang `SourceLocation` to an MLIR `Location`.
  Location convertLocation(slang::SourceLocation loc) const {
    return ::convertLocation(context, *sourceManager, loc);
  }

  static DiagnosticSeverity getSeverity(slang::DiagnosticSeverity severity) {
    switch (severity) {
    case slang::DiagnosticSeverity::Fatal:
    case slang::DiagnosticSeverity::Error:
      return DiagnosticSeverity::Error;
    case slang::DiagnosticSeverity::Warning:
      return DiagnosticSeverity::Warning;
    case slang::DiagnosticSeverity::Ignored:
    case slang::DiagnosticSeverity::Note:
      return DiagnosticSeverity::Remark;
    }
    llvm_unreachable("all slang diagnostic severities should be handled");
    return DiagnosticSeverity::Error;
  }

private:
  MLIRContext *context;
};
} // namespace

// Allow for `slang::BufferID` to be used as hash map keys.
namespace llvm {
template <>
struct DenseMapInfo<slang::BufferID> {
  static slang::BufferID getEmptyKey() { return slang::BufferID(); }
  static slang::BufferID getTombstoneKey() {
    return slang::BufferID(UINT32_MAX - 1, ""sv);
    // UINT32_MAX is already used by `BufferID::getPlaceholder`.
  }
  static unsigned getHashValue(slang::BufferID id) {
    return llvm::hash_value(id.getId());
  }
  static bool isEqual(slang::BufferID a, slang::BufferID b) { return a == b; }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

namespace {
const static ImportVerilogOptions defaultOptions;

/// Run slang analysis only in explicit lint mode.
///
/// Slang analysis currently relies on an internal thread-pool implementation
/// that can hang or crash on large formal front-end workloads (notably UVM/SVA
/// imports used by BMC/LEC). Keep that phase out of normal lowering flows
/// until upstream analysis threading is hardened for these workloads.
static bool shouldRunSlangAnalysis(const ImportVerilogOptions &options) {
  if (options.runSlangAnalysis.has_value())
    return *options.runSlangAnalysis;
  return options.mode == ImportVerilogOptions::Mode::OnlyLint;
}

/// Keep slang's lint-mode behavior aligned with CIRCT mode selection.
static bool getRequestedSlangLintMode(const ImportVerilogOptions &options) {
  (void)options;
  // CIRCT keeps slang's lint-mode gate disabled so semantic legality checks
  // still run in all CIRCT modes, including `--lint-only`.
  return false;
}

static void
setDiagnosticSeverity(slang::driver::Driver &driver,
                      std::initializer_list<slang::DiagCode> diagnostics,
                      slang::DiagnosticSeverity severity) {
  for (auto code : diagnostics)
    driver.diagEngine.setSeverity(code, severity);
}

static bool
isBindUnknownModuleDiagnostic(const slang::Diagnostic &diag,
                              const slang::SourceManager &sourceManager) {
  if (diag.code != slang::diag::UnknownModule)
    return false;

  auto loc = sourceManager.getFullyOriginalLoc(diag.location);
  if (!loc)
    loc = diag.location;
  if (!loc || !loc.buffer().valid())
    return false;

  auto text = sourceManager.getSourceText(loc.buffer());
  if (text.empty())
    return false;
  auto offset = std::min<size_t>(loc.offset(), text.size() - 1);

  auto lineStart = text.rfind('\n', offset);
  if (lineStart == std::string_view::npos)
    lineStart = 0;
  else
    ++lineStart;
  auto lineEnd = text.find('\n', offset);
  if (lineEnd == std::string_view::npos)
    lineEnd = text.size();

  auto startsWithBindKeyword = [](StringRef fragment) {
    fragment = fragment.ltrim();
    if (!fragment.consume_front("bind"))
      return false;
    return fragment.empty() || !isIdentifierChar(fragment.front());
  };

  auto line = StringRef(text.data() + lineStart, lineEnd - lineStart);
  if (startsWithBindKeyword(line))
    return true;

  // Slang may place UnknownModule on the bind-target token instead of the
  // `bind` keyword (for example in multiline bind statements). Re-check the
  // statement prefix before the failing token.
  size_t stmtStart = text.rfind(';', offset);
  stmtStart = stmtStart == std::string_view::npos ? 0 : stmtStart + 1;
  size_t prefixEnd = offset < stmtStart ? stmtStart : offset;
  auto statementPrefix =
      StringRef(text.data() + stmtStart, prefixEnd - stmtStart);
  if (auto triviaEnd = skipTrivia(statementPrefix, 0))
    return startsWithBindKeyword(statementPrefix.drop_front(*triviaEnd));

  return false;
}

static void
applySlangDiagnosticSeverityPolicy(slang::driver::Driver &driver,
                                   const ImportVerilogOptions &options) {
  // Set DynamicNotProcedural severity AFTER processOptions() so it doesn't
  // get overwritten by the default warning option processing.
  if (options.allowNonProceduralDynamic.value_or(false))
    driver.diagEngine.setSeverity(slang::diag::DynamicNotProcedural,
                                  slang::DiagnosticSeverity::Warning);

  // Downgrade out-of-bounds index/range accesses from error to warning by
  // default. Most tools (VCS, Xcelium, yosys) accept these and handle them
  // at runtime, while slang treats them as hard errors.
  setDiagnosticSeverity(driver, {slang::diag::IndexOOB, slang::diag::RangeOOB},
                        slang::DiagnosticSeverity::Warning);
  // Allow standalone bind files that reference external design modules.
  // Non-bind unresolved modules are re-promoted to hard errors below.
  setDiagnosticSeverity(driver, {slang::diag::UnknownModule},
                        slang::DiagnosticSeverity::Warning);

  // CIRCT historically did not run slang's full analysis pass. Running it now
  // is needed for strict driver legality diagnostics, but we keep most newly
  // surfaced analysis diagnostics as warnings to avoid broad behavior churn.
  setDiagnosticSeverity(
      driver,
      {
          slang::diag::AlwaysWithoutTimingControl,
          slang::diag::AssertionFormalMultiAssign,
          slang::diag::AssertionFormalUnassigned,
          slang::diag::AssertionLocalUnassigned,
          slang::diag::AssertionNoClock,
          slang::diag::BlockingDelayInTask,
          slang::diag::ClockVarTargetAssign,
          slang::diag::DifferentClockInClockingBlock,
          slang::diag::EnumValueSizeMismatch,
          slang::diag::GFSVMatchItems,
          slang::diag::ImplicitConnNetInconsistent,
          slang::diag::InterconnectPortVar,
          slang::diag::InvalidMulticlockedSeqOp,
          slang::diag::MismatchedUserDefPortConn,
          slang::diag::MismatchedUserDefPortDir,
          slang::diag::MultipleContAssigns,
          slang::diag::MulticlockedInClockingBlock,
          slang::diag::MulticlockedSeqEmptyMatch,
          slang::diag::NTResolveArgModify,
          slang::diag::NoInferredClock,
          slang::diag::NoUniqueClock,
          slang::diag::RandCInSoft,
          slang::diag::RandCInSolveBefore,
          slang::diag::SampledValueFuncClock,
          slang::diag::SeqMethodEndClock,
          slang::diag::DirectiveInsideDesignElement,
          slang::diag::NewInterfaceClass,
          slang::diag::NewVirtualClass,
          slang::diag::UserDefPortMixedConcat,
          slang::diag::UserDefPortTwoSided,
          slang::diag::VirtualIfaceConfigRule,
          slang::diag::VirtualIfaceDefparam,
      },
      slang::DiagnosticSeverity::Warning);

  // Keep strict legality diagnostics as errors.
  setDiagnosticSeverity(
      driver,
      {
          slang::diag::InputPortAssign,
          slang::diag::MixedVarAssigns,
          slang::diag::MultipleAlwaysAssigns,
          slang::diag::MultipleUDNTDrivers,
          slang::diag::MultipleUWireDrivers,
      },
      slang::DiagnosticSeverity::Error);
}

/// Temporarily override a slang compilation flag and restore it on scope exit.
class ScopedCompilationFlagOverride {
public:
  ScopedCompilationFlagOverride(slang::driver::Driver &driver,
                                slang::ast::CompilationFlags flag, bool value)
      : driver(driver), flag(flag) {
    auto &flags = driver.options.compilationFlags;
    if (auto it = flags.find(flag); it != flags.end()) {
      hadOldEntry = true;
      oldValue = it->second;
    }
    flags[flag] = value;
  }

  ~ScopedCompilationFlagOverride() {
    auto &flags = driver.options.compilationFlags;
    if (hadOldEntry)
      flags[flag] = oldValue;
    else
      flags.erase(flag);
  }

private:
  slang::driver::Driver &driver;
  slang::ast::CompilationFlags flag;
  bool hadOldEntry = false;
  std::optional<bool> oldValue;
};

struct ImportDriver {
  ImportDriver(MLIRContext *mlirContext, TimingScope &ts,
               const ImportVerilogOptions *options)
      : mlirContext(mlirContext), ts(ts),
        options(options ? *options : defaultOptions) {}

  LogicalResult prepareDriver(SourceMgr &sourceMgr);
  LogicalResult importVerilog(ModuleOp module);
  LogicalResult preprocessVerilog(llvm::raw_ostream &os);

  MLIRContext *mlirContext;
  TimingScope &ts;
  const ImportVerilogOptions &options;

  // Use slang's driver which conveniently packages a lot of the things we
  // need for compilation.
  slang::driver::Driver driver;
};
} // namespace

/// Populate the Slang driver with source files from the given `sourceMgr`, and
/// configure driver options based on the `ImportVerilogOptions` passed to the
/// `ImportDriver` constructor.
LogicalResult ImportDriver::prepareDriver(SourceMgr &sourceMgr) {
  // Use slang's driver which conveniently packages a lot of the things we
  // need for compilation.
  auto diagClient = std::make_shared<MlirDiagnosticClient>(mlirContext);
  driver.diagEngine.addClient(diagClient);

  for (const auto &value : options.commandFiles)
    if (!driver.processCommandFiles(value, /*makeRelative=*/true,
                                    /*separateUnit=*/true))
      return failure();

  // Populate the source manager with the source files.
  // NOTE: This is a bit ugly since we're essentially copying the Verilog
  // source text in memory. At a later stage we'll want to extend slang's
  // SourceManager such that it can contain non-owned buffers. This will do
  // for now.
  DenseSet<StringRef> seenBuffers;
  for (unsigned i = 0, e = sourceMgr.getNumBuffers(); i < e; ++i) {
    const llvm::MemoryBuffer *mlirBuffer = sourceMgr.getMemoryBuffer(i + 1);
    auto name = mlirBuffer->getBufferIdentifier();
    if (!name.empty() && !seenBuffers.insert(name).second)
      continue; // Slang doesn't like listing the same buffer twice
    auto text = mlirBuffer->getBuffer();
    StringRef rewrittenText = text;
    std::string rewrittenStorage;
    auto applyRewrite = [&](auto &&rewriteFn, bool enabled) {
      if (!enabled)
        return;
      bool rewritten = false;
      auto candidate = rewriteFn(rewrittenText, rewritten);
      if (!rewritten)
        return;
      rewrittenStorage = std::move(candidate);
      rewrittenText = rewrittenStorage;
    };
    applyRewrite(rewriteRandomizeInlineConstraints,
                 rewrittenText.contains("randomize") &&
                     rewrittenText.contains("this"));
    applyRewrite(rewritePastClockingArgCompat,
                 rewrittenText.contains("$past") &&
                     rewrittenText.contains("@("));
    applyRewrite(rewriteEventControlParenCompat,
                 rewrittenText.contains("@posedge (") ||
                     rewrittenText.contains("@negedge (") ||
                     rewrittenText.contains("@edge ("));
    applyRewrite(rewriteOpenRangeUnarySVACompat,
                 rewrittenText.contains(":$") &&
                     (rewrittenText.contains("eventually") ||
                      rewrittenText.contains("s_always") ||
                      rewrittenText.contains("nexttime") ||
                      rewrittenText.contains("s_nexttime")));
    applyRewrite(rewriteConfigKeywordIdentifiersCompat,
                 rewrittenText.contains("cell") ||
                     rewrittenText.contains("config") ||
                     rewrittenText.contains("design") ||
                     rewrittenText.contains("endconfig") ||
                     rewrittenText.contains("incdir") ||
                     rewrittenText.contains("include") ||
                     rewrittenText.contains("instance") ||
                     rewrittenText.contains("liblist") ||
                     rewrittenText.contains("library") ||
                     rewrittenText.contains("use"));
    applyRewrite(rewriteGenericClassScopeCompat,
                 rewrittenText.contains("class") &&
                     rewrittenText.contains("::"));
    applyRewrite(rewriteUvmDoCompareComparerDefaultCompat,
                 rewrittenText.contains("do_compare") &&
                     rewrittenText.contains("uvm_comparer") &&
                     rewrittenText.contains('='));
    applyRewrite(rewriteCovergroupBinSelfNameCompat,
                 rewrittenText.contains("bins") &&
                     rewrittenText.contains('{') &&
                     rewrittenText.contains('='));
    applyRewrite(rewriteFormatWidthCompat,
                 rewrittenText.contains('"') && rewrittenText.contains('%'));
    applyRewrite(rewriteUDPZCompat,
                 rewrittenText.contains("primitive") &&
                     rewrittenText.contains("table") &&
                     (rewrittenText.contains('z') ||
                      rewrittenText.contains('Z')));
    text = rewrittenText;

    auto slangBuffer = driver.sourceManager.assignText(name, text);
    driver.sourceLoader.addBuffer(slangBuffer);
  }

  for (const auto &libDir : options.libDirs)
    driver.sourceLoader.addSearchDirectories(libDir);

  for (const auto &libExt : options.libExts)
    driver.sourceLoader.addSearchExtension(libExt);

  for (const auto &[i, f] : llvm::enumerate(options.libraryFiles)) {
    // Include a space to avoid conflicts with explicitly-specified names.
    auto libName = "library " + std::to_string(i);
    driver.sourceLoader.addLibraryFiles(libName, f);
  }

  for (const auto &includeDir : options.includeDirs)
    if (driver.sourceManager.addUserDirectories(includeDir))
      return failure();

  for (const auto &includeSystemDir : options.includeSystemDirs)
    if (driver.sourceManager.addSystemDirectories(includeSystemDir))
      return failure();

  // These options are exposed on circt-verilog and should behave like slang's
  // standard command line parser, including support for comma-separated values.
  auto forEachCommaValue =
      [](const std::vector<std::string> &values, auto &&processOne) {
        for (const auto &value : values) {
          StringRef remaining = value;
          while (!remaining.empty()) {
            auto [part, rest] = remaining.split(',');
            auto token = part.trim();
            if (!token.empty())
              processOne(token);
            remaining = rest;
          }
        }
      };

  forEachCommaValue(options.suppressWarningsPaths, [&](StringRef pathPattern) {
    if (auto ec = driver.diagEngine.addIgnorePaths(pathPattern))
      emitWarning(UnknownLoc::get(mlirContext))
          << "--suppress-warnings path '" << pathPattern
          << "': " << ec.message();
  });

  forEachCommaValue(options.suppressMacroWarningsPaths,
                    [&](StringRef pathPattern) {
                      if (auto ec =
                              driver.diagEngine.addIgnoreMacroPaths(pathPattern))
                        emitWarning(UnknownLoc::get(mlirContext))
                            << "--suppress-macro-warnings path '" << pathPattern
                            << "': " << ec.message();
                    });

  forEachCommaValue(options.libraryFiles, [&](StringRef libraryPattern) {
    StringRef libraryName;
    StringRef filePattern = libraryPattern;
    auto eqPos = libraryPattern.find('=');
    if (eqPos != StringRef::npos) {
      libraryName = libraryPattern.take_front(eqPos).trim();
      filePattern = libraryPattern.drop_front(eqPos + 1).trim();
    }
    driver.sourceLoader.addLibraryFiles(libraryName, filePattern);
  });

  forEachCommaValue(options.libraryMapFiles, [&](StringRef mapPattern) {
    driver.sourceLoader.addLibraryMaps(mapPattern, {}, slang::Bag{});
  });

  // Populate the driver options.
  driver.options.excludeExts.insert(options.excludeExts.begin(),
                                    options.excludeExts.end());
  driver.options.ignoreDirectives = options.ignoreDirectives;

  driver.options.maxIncludeDepth = options.maxIncludeDepth;
  driver.options.defines = options.defines;
  driver.options.undefines = options.undefines;
  driver.options.librariesInheritMacros = options.librariesInheritMacros;
  driver.options.disableLocalIncludes = options.disableLocalIncludes;
  driver.options.enableLegacyProtect = options.enableLegacyProtect;
  driver.options.translateOffOptions = options.translateOffOptions;
  for (const auto &mappingSpec : options.keywordVersionMappings) {
    StringRef spec = mappingSpec;
    auto plusPos = spec.find('+');
    if (plusPos == StringRef::npos) {
      emitError(UnknownLoc::get(mlirContext))
          << "--map-keyword-version expects "
             "<keyword-version>+<file-pattern>[,...], got '"
          << spec << "'";
      return failure();
    }
    StringRef versionText = spec.take_front(plusPos).trim();
    StringRef patternsText = spec.drop_front(plusPos + 1).trim();
    if (versionText.empty() || patternsText.empty()) {
      emitError(UnknownLoc::get(mlirContext))
          << "--map-keyword-version expects "
             "<keyword-version>+<file-pattern>[,...], got '"
          << spec << "'";
      return failure();
    }
    auto keywordVersion =
        slang::parsing::LexerFacts::getKeywordVersion(versionText);
    if (!keywordVersion) {
      emitError(UnknownLoc::get(mlirContext))
          << "--map-keyword-version has unknown keyword version '"
          << versionText << "'";
      return failure();
    }
    bool addedPattern = false;
    StringRef remaining = patternsText;
    while (!remaining.empty()) {
      auto [part, rest] = remaining.split(',');
      auto pattern = part.trim();
      if (!pattern.empty()) {
        driver.options.keywordMapping.emplace_back(pattern.str(),
                                                   *keywordVersion);
        addedPattern = true;
      }
      remaining = rest;
    }
    if (!addedPattern) {
      emitError(UnknownLoc::get(mlirContext))
          << "--map-keyword-version requires at least one non-empty file "
             "pattern in '"
          << spec << "'";
      return failure();
    }
  }

  driver.options.languageVersion = options.languageVersion;
  driver.options.maxParseDepth = options.maxParseDepth;
  driver.options.maxLexerErrors = options.maxLexerErrors;
  driver.options.numThreads = options.numThreads;
  driver.options.maxInstanceDepth = options.maxInstanceDepth;
  driver.options.maxGenerateSteps = options.maxGenerateSteps;
  driver.options.maxConstexprDepth = options.maxConstexprDepth;
  driver.options.maxConstexprSteps = options.maxConstexprSteps;
  driver.options.maxConstexprBacktrace = options.maxConstexprBacktrace;
  driver.options.maxInstanceArray = options.maxInstanceArray;
  driver.options.maxUDPCoverageNotes = options.maxUDPCoverageNotes;

  driver.options.timeScale = options.timeScale;
  if (options.minTypMax) {
    StringRef mtm = *options.minTypMax;
    if (mtm.equals_insensitive("min"))
      driver.options.minTypMax = slang::ast::MinTypMax::Min;
    else if (mtm.equals_insensitive("typ"))
      driver.options.minTypMax = slang::ast::MinTypMax::Typ;
    else if (mtm.equals_insensitive("max"))
      driver.options.minTypMax = slang::ast::MinTypMax::Max;
    else {
      emitError(UnknownLoc::get(mlirContext))
          << "--timing expects one of: min, typ, max";
      return failure();
    }
  }

  // Enable AllowUseBeforeDeclare by default — forward references are
  // ubiquitous in real SV code and accepted by VCS/Xcelium.  The explicit
  // CLI flag can still override this.
  driver.options.compilationFlags.emplace(
      slang::ast::CompilationFlags::AllowUseBeforeDeclare,
      options.allowUseBeforeDeclare.value_or(true));
  // Match mainstream simulator behavior for implicit enum conversions.
  driver.options.compilationFlags.emplace(
      slang::ast::CompilationFlags::RelaxEnumConversions,
      options.relaxEnumConversions.value_or(true));
  // Match mainstream simulator behavior for mixed string/integral contexts.
  driver.options.compilationFlags.emplace(
      slang::ast::CompilationFlags::RelaxStringConversions,
      options.relaxStringConversions.value_or(true));
  // Enable AllowUnnamedGenerate by default — references to implicit genblk
  // names are accepted by all major tools.
  driver.options.compilationFlags.emplace(
      slang::ast::CompilationFlags::AllowUnnamedGenerate, true);
  driver.options.compilationFlags.emplace(
      slang::ast::CompilationFlags::IgnoreUnknownModules,
      options.ignoreUnknownModules);
  // Handle compat option - set VCS compatibility flags directly
  // We don't use slang's native compat mode because CIRCT bypasses
  // addStandardArgs() which initializes the flags map
  if (options.compat.has_value()) {
    auto compatStr = *options.compat;
    if (compatStr == "vcs" || compatStr == "all") {
      // VCS compatibility flags (from slang's VCS_COMP_FLAGS)
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::AllowHierarchicalConst, true);
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::RelaxEnumConversions, true);
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::AllowUseBeforeDeclare, true);
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::RelaxStringConversions, true);
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::AllowRecursiveImplicitCall, true);
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::AllowBareValParamAssignment, true);
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::AllowSelfDeterminedStreamConcat, true);
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::AllowMergingAnsiPorts, true);
    }
    if (compatStr == "all") {
      // Additional flags for "all" compat mode
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::AllowTopLevelIfacePorts, true);
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::AllowUnnamedGenerate, true);
    }
  }

  auto applyCompilationFlagOverride =
      [&](slang::ast::CompilationFlags flag, std::optional<bool> value) {
        if (!value.has_value())
          return;
        driver.options.compilationFlags[flag] = *value;
      };
  applyCompilationFlagOverride(slang::ast::CompilationFlags::AllowHierarchicalConst,
                               options.allowHierarchicalConst);
  applyCompilationFlagOverride(slang::ast::CompilationFlags::RelaxEnumConversions,
                               options.relaxEnumConversions);
  applyCompilationFlagOverride(slang::ast::CompilationFlags::RelaxStringConversions,
                               options.relaxStringConversions);
  applyCompilationFlagOverride(
      slang::ast::CompilationFlags::AllowRecursiveImplicitCall,
      options.allowRecursiveImplicitCall);
  applyCompilationFlagOverride(
      slang::ast::CompilationFlags::AllowBareValParamAssignment,
      options.allowBareValParamAssignment);
  applyCompilationFlagOverride(
      slang::ast::CompilationFlags::AllowSelfDeterminedStreamConcat,
      options.allowSelfDeterminedStreamConcat);
  applyCompilationFlagOverride(slang::ast::CompilationFlags::AllowMergingAnsiPorts,
                               options.allowMergingAnsiPorts);
  applyCompilationFlagOverride(
      slang::ast::CompilationFlags::AllowTopLevelIfacePorts,
      options.allowTopLevelIfacePorts);
  // `AllowVirtualIfaceWithOverride` exists only in newer slang versions.
  // Use dependent lookup so older versions compile without this enum member.
  auto applyAllowVirtualIfaceWithOverride = [&]<typename FlagEnum>() {
    if constexpr (requires { FlagEnum::AllowVirtualIfaceWithOverride; }) {
      // Enable by default for compatibility with mainstream simulator behavior
      // on bind/defparam-targeted interface instances.
      driver.options.compilationFlags[FlagEnum::AllowVirtualIfaceWithOverride] =
          options.allowVirtualIfaceWithOverride.value_or(true);
    }
  };
  applyAllowVirtualIfaceWithOverride
      .template operator()<slang::ast::CompilationFlags>();

  // CIRCT mode controls the baseline slang lint-mode setting; analysis can
  // still temporarily override this to force semantic legality checks.
  driver.options.compilationFlags[slang::ast::CompilationFlags::LintMode] =
      getRequestedSlangLintMode(options);
  driver.options.compilationFlags[slang::ast::CompilationFlags::DisableInstanceCaching] =
      false;
  driver.options.topModules = options.topModules;
  driver.options.paramOverrides = options.paramOverrides;
  forEachCommaValue(options.libraryOrder, [&](StringRef libraryName) {
    driver.options.libraryOrder.emplace_back(libraryName.str());
  });
  driver.options.defaultLibName = options.defaultLibName;

  driver.options.errorLimit = options.errorLimit;
  driver.options.warningOptions = options.warningOptions;

  driver.options.singleUnit = options.singleUnit;

  if (!driver.processOptions())
    return failure();

  applySlangDiagnosticSeverityPolicy(driver, options);

  return success();
}

/// Parse and elaborate the prepared source files, and populate the given MLIR
/// `module` with corresponding operations.
LogicalResult ImportDriver::importVerilog(ModuleOp module) {
  // Parse the input.
  auto parseTimer = ts.nest("Verilog parser");
  bool parseSuccess = driver.parseAllSources();
  parseTimer.stop();

  // Elaborate the input.
  auto compileTimer = ts.nest("Verilog elaboration");
  auto compilation = driver.createCompilation();

  // Register vendor-specific system functions as stubs.
  // $get_initial_random_seed is a Verilator extension that returns the
  // initial random seed. We stub it to return int (value 0 at runtime).
  compilation->addSystemSubroutine(
      std::make_shared<slang::ast::NonConstantFunction>(
          "$get_initial_random_seed", compilation->getIntType()));

  // $initstate is used in formal verification to indicate the initial state.
  // It returns 1 during the initial/reset state and 0 otherwise.
  // We stub it to return bit (value 0) since in simulation we are never in
  // the formal initial state.
  compilation->addSystemSubroutine(
      std::make_shared<slang::ast::NonConstantFunction>(
          "$initstate", compilation->getBitType()));

  bool hasBlockingDiagnostics = !parseSuccess;
  for (auto &diag : compilation->getAllDiagnostics()) {
    if (diag.code == slang::diag::UnknownModule &&
        !isBindUnknownModuleDiagnostic(diag, driver.sourceManager)) {
      mlir::emitError(convertLocation(mlirContext, driver.sourceManager,
                                      diag.location))
          << driver.diagEngine.formatMessage(diag);
      hasBlockingDiagnostics = true;
      continue;
    }
    driver.diagEngine.issue(diag);
  }
  if (hasBlockingDiagnostics || driver.diagEngine.getNumErrors() > 0)
    return failure();

  // Run slang's semantic analysis checks in all modes except parse-only.
  // This catches illegal driver combinations (for example, multi-driver
  // always_comb / always_ff conflicts and input-port assignments) that are
  // diagnosed during analysis rather than initial elaboration.
  if (shouldRunSlangAnalysis(options)) {
#if defined(__EMSCRIPTEN__)
    // Slang semantic analysis currently attempts to spawn worker threads in our
    // wasm builds, which aborts in single-threaded runtimes. Skip this phase
    // on emscripten instead of hard-aborting the entire import.
#else
    auto analysisTimer = ts.nest("Verilog semantic analysis");
    ScopedCompilationFlagOverride forceAnalysisMode(
        driver, slang::ast::CompilationFlags::LintMode, false);
    (void)driver.runAnalysis(*compilation);
    analysisTimer.stop();
    if (driver.diagEngine.getNumErrors() > 0)
      return failure();
#endif
  }

  compileTimer.stop();

  // If we were only supposed to lint or parse the input, return here. This
  // leaves the module empty, but any Slang messages got reported as
  // diagnostics.
  if (options.mode == ImportVerilogOptions::Mode::OnlyLint ||
      options.mode == ImportVerilogOptions::Mode::OnlyParse)
    return success();

  // Traverse the parsed Verilog AST and map it to the equivalent CIRCT ops.
  mlirContext
      ->loadDialect<moore::MooreDialect, hw::HWDialect, cf::ControlFlowDialect,
                    func::FuncDialect, verif::VerifDialect, ltl::LTLDialect,
                    debug::DebugDialect, mlir::LLVM::LLVMDialect>();
  auto conversionTimer = ts.nest("Verilog to dialect mapping");
  Context context(options, *compilation, module, driver.sourceManager);
  if (failed(context.convertCompilation()))
    return failure();
  conversionTimer.stop();

  // Run the verifier on the constructed module to ensure it is clean.
  auto verifierTimer = ts.nest("Post-parse verification");
  if (failed(verify(module))) {
    // The verifier should have emitted diagnostics, but add a summary message
    // in case it didn't for some reason.
    mlir::emitError(UnknownLoc::get(mlirContext))
        << "generated MLIR module failed to verify; "
           "this is likely a bug in circt-verilog";
    return failure();
  }
  return success();
}

/// Preprocess the prepared source files and print them to the given output
/// stream.
LogicalResult ImportDriver::preprocessVerilog(llvm::raw_ostream &os) {
  auto parseTimer = ts.nest("Verilog preprocessing");

  // Run the preprocessor to completion across all sources previously added with
  // `pushSource`, report diagnostics, and print the output.
  auto preprocessAndPrint = [&](slang::parsing::Preprocessor &preprocessor) {
    slang::syntax::SyntaxPrinter output;
    output.setIncludeComments(false);
    while (true) {
      slang::parsing::Token token = preprocessor.next();
      output.print(token);
      if (token.kind == slang::parsing::TokenKind::EndOfFile)
        break;
    }

    for (auto &diag : preprocessor.getDiagnostics()) {
      if (diag.isError()) {
        driver.diagEngine.issue(diag);
        return failure();
      }
    }
    os << output.str();
    return success();
  };

  // Depending on whether the single-unit option is set, either add all source
  // files to a single preprocessor such that they share define macros and
  // directives, or create a separate preprocessor for each, such that each
  // source file is in its own compilation unit.
  auto optionBag = driver.createOptionBag();
  if (driver.options.singleUnit == true) {
    slang::BumpAllocator alloc;
    slang::Diagnostics diagnostics;
    slang::parsing::Preprocessor preprocessor(driver.sourceManager, alloc,
                                              diagnostics, optionBag);
    // Sources have to be pushed in reverse, as they form a stack in the
    // preprocessor. Last pushed source is processed first.
    auto sources = driver.sourceLoader.loadSources();
    for (auto &buffer : std::views::reverse(sources))
      preprocessor.pushSource(buffer);
    if (failed(preprocessAndPrint(preprocessor)))
      return failure();
  } else {
    for (auto &buffer : driver.sourceLoader.loadSources()) {
      slang::BumpAllocator alloc;
      slang::Diagnostics diagnostics;
      slang::parsing::Preprocessor preprocessor(driver.sourceManager, alloc,
                                                diagnostics, optionBag);
      preprocessor.pushSource(buffer);
      if (failed(preprocessAndPrint(preprocessor)))
        return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Entry Points
//===----------------------------------------------------------------------===//

/// Parse the specified Verilog inputs into the specified MLIR context.
LogicalResult circt::importVerilog(SourceMgr &sourceMgr,
                                   MLIRContext *mlirContext, TimingScope &ts,
                                   ModuleOp module,
                                   const ImportVerilogOptions *options) {
  ImportDriver importDriver(mlirContext, ts, options);
  if (failed(importDriver.prepareDriver(sourceMgr)))
    return failure();
  return importDriver.importVerilog(module);
}

/// Run the files in a source manager through Slang's Verilog preprocessor and
/// emit the result to the given output stream.
LogicalResult circt::preprocessVerilog(SourceMgr &sourceMgr,
                                       MLIRContext *mlirContext,
                                       TimingScope &ts, llvm::raw_ostream &os,
                                       const ImportVerilogOptions *options) {
  ImportDriver importDriver(mlirContext, ts, options);
  if (failed(importDriver.prepareDriver(sourceMgr)))
    return failure();
  return importDriver.preprocessVerilog(os);
}

/// Entry point as an MLIR translation.
void circt::registerFromVerilogTranslation() {
  static TranslateToMLIRRegistration fromVerilog(
      "import-verilog", "import Verilog or SystemVerilog",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        TimingScope ts;
        OwningOpRef<ModuleOp> module(
            ModuleOp::create(UnknownLoc::get(context)));
        ImportVerilogOptions options;
        options.debugInfo = true;
        options.warningOptions.push_back("no-missing-top");
        if (failed(
                importVerilog(sourceMgr, context, ts, module.get(), &options)))
          module = {};
        return module;
      });
}

//===----------------------------------------------------------------------===//
// Pass Pipeline
//===----------------------------------------------------------------------===//

/// Optimize and simplify the Moore dialect IR.
void circt::populateVerilogToMoorePipeline(OpPassManager &pm) {
  {
    // Perform an initial cleanup and preprocessing across all
    // modules/functions.
    auto &anyPM = pm.nestAny();
    anyPM.addPass(mlir::createCSEPass());
    anyPM.addPass(mlir::createCanonicalizerPass());
  }

  pm.addPass(moore::createVTablesPass());

  // Remove unused symbols.
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(moore::createLowerConcatRefPass());

  {
    // Perform module-specific transformations.
    auto &modulePM = pm.nest<moore::SVModuleOp>();
    modulePM.addPass(moore::createSimplifyProceduresPass());
    modulePM.addPass(mlir::createSROA());
  }

  {
    // Perform a final cleanup across all modules/functions.
    auto &anyPM = pm.nestAny();
    anyPM.addPass(mlir::createMem2Reg());
    anyPM.addPass(mlir::createCSEPass());
    anyPM.addPass(mlir::createCanonicalizerPass());
  }
}

/// Convert Moore dialect IR into core dialect IR
void circt::populateMooreToCorePipeline(OpPassManager &pm,
                                        bool skipPostCleanup) {
  // Perform the conversion.
  pm.addPass(createConvertMooreToCorePass());

  if (skipPostCleanup)
    return;

  {
    // Conversion to the core dialects likely uncovers new canonicalization
    // opportunities.
    auto &anyPM = pm.nestAny();
    anyPM.addPass(mlir::createCSEPass());
    // Use top-down traversal here to avoid a bottom-up comb canonicalization
    // ordering issue that can introduce non-dominating values in large nested
    // mux/boolean expressions.
    anyPM.addPass(circt::createSimpleCanonicalizerPass());
  }
}

/// Convert LLHD dialect IR into core dialect IR
void circt::populateLlhdToCorePipeline(
    OpPassManager &pm, const LlhdToCorePipelineOptions &options) {
  // Inline function calls and lower SCF to CF.
  pm.addNestedPass<hw::HWModuleOp>(llhd::createWrapProceduralOpsPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(llhd::createInlineCallsPass());

  // Run MooreToCore again to convert moore.wait_event/detect_event ops that
  // were in functions (e.g., interface tasks) and have now been inlined into
  // llhd.process ops. The first MooreToCore pass leaves these unconverted when
  // they're in func.func, using dynamic legality rules. After inlining, they
  // are now inside llhd.process and can be converted.
  pm.addPass(createConvertMooreToCorePass());

  pm.addPass(mlir::createSymbolDCEPass());

  // Simplify processes, replace signals with process results, and detect
  // registers.
  auto &modulePM = pm.nest<hw::HWModuleOp>();
  // See https://github.com/llvm/circt/issues/8804.
  if (options.sroa) {
    modulePM.addPass(mlir::createSROA());
  }
  modulePM.addPass(llhd::createMem2RegPass());
  modulePM.addPass(llhd::createHoistSignalsPass());
  modulePM.addPass(llhd::createDeseqPass());
  modulePM.addPass(llhd::createLowerProcessesPass());
  modulePM.addPass(mlir::createCSEPass());
  modulePM.addPass(circt::createBottomUpSimpleCanonicalizerPass());

  // Unroll loops and remove control flow.
  modulePM.addPass(llhd::createUnrollLoopsPass());
  modulePM.addPass(mlir::createCSEPass());
  modulePM.addPass(circt::createBottomUpSimpleCanonicalizerPass());
  modulePM.addPass(llhd::createRemoveControlFlowPass());
  modulePM.addPass(mlir::createCSEPass());
  modulePM.addPass(circt::createBottomUpSimpleCanonicalizerPass());

  // Convert `arith.select` generated by some of the control flow canonicalizers
  // to `comb.mux`.
  modulePM.addPass(createMapArithToCombPass(true));

  // Simplify module-level signals.
  modulePM.addPass(llhd::createCombineDrivesPass());
  modulePM.addPass(llhd::createSig2Reg());
  modulePM.addPass(mlir::createCSEPass());
  modulePM.addPass(circt::createBottomUpSimpleCanonicalizerPass());

  // Map `seq.firreg` with array type and `hw.array_inject` self-feedback to
  // `seq.firmem` ops.
  if (options.detectMemories) {
    modulePM.addPass(seq::createRegOfVecToMem());
    modulePM.addPass(mlir::createCSEPass());
    modulePM.addPass(circt::createBottomUpSimpleCanonicalizerPass());
  }
}
