//===- NativeMutationPlanner.cpp - Native mutation planning --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeMutationPlanner.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>

using namespace llvm;

namespace circt::mut {

static constexpr const char *kNativeMutationOpsAll[] = {
    "EQ_TO_NEQ",        "NEQ_TO_EQ",      "LT_TO_LE",        "GT_TO_GE",
    "LE_TO_LT",         "GE_TO_GT",       "LT_TO_GT",        "GT_TO_LT",
    "LE_TO_GE",         "GE_TO_LE",       "AND_TO_OR",       "OR_TO_AND",
    "LAND_TO_BAND",     "LOR_TO_BOR",     "XOR_TO_OR",       "XOR_TO_XNOR",
    "XNOR_TO_XOR",      "REDAND_TO_REDOR", "REDOR_TO_REDAND",
    "REDXOR_TO_REDXNOR", "REDXNOR_TO_REDXOR", "BAND_TO_BOR",
    "BOR_TO_BAND",      "BAND_TO_LAND",   "BOR_TO_LOR",      "BA_TO_NBA",
    "NBA_TO_BA",        "ASSIGN_RHS_TO_CONST0", "ASSIGN_RHS_TO_CONST1",
    "ASSIGN_RHS_INVERT", "ASSIGN_RHS_PLUS_ONE", "ASSIGN_RHS_MINUS_ONE",
    "POSEDGE_TO_NEGEDGE", "NEGEDGE_TO_POSEDGE",
    "RESET_POSEDGE_TO_NEGEDGE", "RESET_NEGEDGE_TO_POSEDGE", "MUX_SWAP_ARMS",
    "MUX_FORCE_TRUE", "MUX_FORCE_FALSE",
    "CASE_TO_CASEZ", "CASEZ_TO_CASE", "CASE_TO_CASEX", "CASEX_TO_CASE",
    "CASEZ_TO_CASEX", "CASEX_TO_CASEZ",
    "IF_COND_NEGATE",   "RESET_COND_NEGATE", "RESET_COND_TRUE",
    "RESET_COND_FALSE", "IF_COND_TRUE",      "IF_COND_FALSE",
    "IF_ELSE_SWAP_ARMS", "CASE_ITEM_SWAP_ARMS", "UNARY_NOT_DROP",
    "UNARY_BNOT_DROP",
    "UNARY_MINUS_DROP", "CONST0_TO_1",    "CONST1_TO_0", "ADD_TO_SUB",
    "SUB_TO_ADD",       "MUL_TO_ADD",     "ADD_TO_MUL",  "DIV_TO_MUL",
    "MUL_TO_DIV",       "MOD_TO_DIV",     "DIV_TO_MOD",  "INC_TO_DEC",
    "DEC_TO_INC",       "PLUS_EQ_TO_MINUS_EQ", "MINUS_EQ_TO_PLUS_EQ",
    "MUL_EQ_TO_DIV_EQ", "DIV_EQ_TO_MUL_EQ",    "MOD_EQ_TO_DIV_EQ",
    "DIV_EQ_TO_MOD_EQ", "SHL_EQ_TO_SHR_EQ",    "SHR_EQ_TO_SHL_EQ",
    "SHR_EQ_TO_ASHR_EQ", "ASHR_EQ_TO_SHR_EQ",  "BAND_EQ_TO_BOR_EQ",
    "BOR_EQ_TO_BAND_EQ", "BXOR_EQ_TO_BOR_EQ",  "BXOR_EQ_TO_BAND_EQ",
    "BAND_EQ_TO_BXOR_EQ", "BOR_EQ_TO_BXOR_EQ",
    "SHL_TO_SHR",
    "SHR_TO_SHL",       "SHR_TO_ASHR",    "ASHR_TO_SHR", "CASEEQ_TO_EQ",
    "CASENEQ_TO_NEQ",
    "EQ_TO_CASEEQ",     "NEQ_TO_CASENEQ",
    "SIGNED_TO_UNSIGNED", "UNSIGNED_TO_SIGNED"};

namespace {

struct XorShift128 {
  uint32_t x = 123456789;
  uint32_t y = 0;
  uint32_t z = 0;
  uint32_t w = 0;

  explicit XorShift128(uint64_t seed) : w(static_cast<uint32_t>(seed)) {
    next();
    next();
    next();
  }

  uint32_t next() {
    uint32_t t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    w ^= (w >> 19) ^ t ^ (t >> 8);
    return w & 0x3fffffff;
  }

  uint32_t bounded(uint32_t n) {
    if (n < 2)
      return 0;
    while (true) {
      uint32_t k = next();
      uint32_t p = k % n;
      if ((k - p + n) <= 0x40000000)
        return p;
    }
  }
};

struct SiteInfo {
  size_t pos = StringRef::npos;
};

struct CaseItemSpan {
  size_t start = StringRef::npos;
  size_t end = StringRef::npos;
  bool isDefault = false;
};

struct CaseSwapSpan {
  size_t sitePos = StringRef::npos;
  size_t firstStart = StringRef::npos;
  size_t firstEnd = StringRef::npos;
  size_t secondStart = StringRef::npos;
  size_t secondEnd = StringRef::npos;
};

struct Candidate {
  std::string op;
  uint64_t siteIndex = 1;
  uint64_t siteCount = 1;

  std::string module;
  std::string srcKey;
  std::string family;

  std::string wireKey;
  std::string wireBitKey;
  std::string cellKey;

  std::string moduleWireKey;
  std::string moduleBitKey;
  std::string moduleCellKey;
  std::string moduleSrcKey;
  std::string contextKey;

  bool used = false;
};

struct CoverageDB {
  StringMap<uint64_t> srcCoverage;
  StringMap<uint64_t> wireCoverage;
  StringMap<uint64_t> wireBitCoverage;
  StringMap<uint64_t> familyCoverage;
  StringMap<uint64_t> opCoverage;
  StringMap<uint64_t> contextCoverage;

  static int contextRealismBonus(StringRef context) {
    if (context == "control")
      return 4;
    if (context == "assignment")
      return 3;
    if (context == "expression")
      return 1;
    if (context == "verification")
      return 0;
    return 0;
  }

  void insert(const Candidate &c) {
    srcCoverage.try_emplace(c.srcKey, 0);
    wireCoverage.try_emplace(c.wireKey, 0);
    wireBitCoverage.try_emplace(c.wireBitKey, 0);
    familyCoverage.try_emplace(c.family, 0);
    opCoverage.try_emplace(c.op, 0);
    contextCoverage.try_emplace(c.contextKey, 0);
  }

  void update(const Candidate &c) {
    ++srcCoverage[c.srcKey];
    ++wireCoverage[c.wireKey];
    ++wireBitCoverage[c.wireBitKey];
    ++familyCoverage[c.family];
    ++opCoverage[c.op];
    ++contextCoverage[c.contextKey];
  }

  int score(const Candidate &c) const {
    int thisScore = c.srcKey.empty() ? 0 : 1;
    if (auto it = wireCoverage.find(c.wireKey); it != wireCoverage.end())
      thisScore += it->second == 0 ? 5 : 0;
    if (auto it = wireBitCoverage.find(c.wireBitKey); it != wireBitCoverage.end())
      thisScore += it->second == 0 ? 1 : 0;
    if (auto it = srcCoverage.find(c.srcKey); it != srcCoverage.end())
      thisScore += it->second == 0 ? 5 : 0;
    if (auto it = familyCoverage.find(c.family); it != familyCoverage.end())
      thisScore += it->second == 0 ? 4 : 0;
    if (auto it = opCoverage.find(c.op); it != opCoverage.end())
      thisScore += it->second == 0 ? 3 : 0;
    if (auto it = contextCoverage.find(c.contextKey); it != contextCoverage.end())
      thisScore += it->second == 0 ? 2 : 0;
    thisScore += contextRealismBonus(c.contextKey);

    // Anti-dominance: keep schedules semantically distinct by discouraging
    // repeated operator/family/context picks once novelty has been consumed.
    if (auto it = familyCoverage.find(c.family); it != familyCoverage.end())
      thisScore -= static_cast<int>(std::min<uint64_t>(2, it->second));
    if (auto it = opCoverage.find(c.op); it != opCoverage.end())
      thisScore -= static_cast<int>(std::min<uint64_t>(3, it->second));
    if (auto it = contextCoverage.find(c.contextKey); it != contextCoverage.end())
      thisScore -= static_cast<int>(std::min<uint64_t>(1, it->second));
    return thisScore;
  }
};

enum class QueueKind {
  PrimaryWire,
  PrimaryBit,
  PrimaryCell,
  PrimarySrc,
  ModuleWire,
  ModuleBit,
  ModuleCell,
  ModuleSrc,
};

static bool parseIntValue(StringRef text, int &out) {
  int64_t parsed = 0;
  if (text.getAsInteger(10, parsed))
    return false;
  if (parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max())
    return false;
  out = static_cast<int>(parsed);
  return true;
}

static void buildCodeMask(StringRef text, SmallVectorImpl<uint8_t> &mask) {
  mask.assign(text.size(), 1);
  enum class State {
    Normal,
    LineComment,
    BlockComment,
    StringLiteral,
  };
  State state = State::Normal;
  bool escape = false;

  for (size_t i = 0, e = text.size(); i < e; ++i) {
    char ch = text[i];
    switch (state) {
    case State::Normal:
      if (ch == '/' && i + 1 < e && text[i + 1] == '/') {
        mask[i] = 0;
        mask[i + 1] = 0;
        ++i;
        state = State::LineComment;
        continue;
      }
      if (ch == '/' && i + 1 < e && text[i + 1] == '*') {
        mask[i] = 0;
        mask[i + 1] = 0;
        ++i;
        state = State::BlockComment;
        continue;
      }
      if (ch == '"') {
        mask[i] = 0;
        state = State::StringLiteral;
        escape = false;
      }
      continue;
    case State::LineComment:
      if (ch != '\n') {
        mask[i] = 0;
      } else {
        state = State::Normal;
      }
      continue;
    case State::BlockComment:
      mask[i] = 0;
      if (ch == '*' && i + 1 < e && text[i + 1] == '/') {
        mask[i + 1] = 0;
        ++i;
        state = State::Normal;
      }
      continue;
    case State::StringLiteral:
      mask[i] = 0;
      if (escape) {
        escape = false;
        continue;
      }
      if (ch == '\\') {
        escape = true;
        continue;
      }
      if (ch == '"')
        state = State::Normal;
      continue;
    }
  }

  // Preprocessor directives (for example `timescale, `define, `include) are
  // not semantic mutation targets and rewriting them can invalidate the file.
  // Mask directive lines after the lexical pass so operator scanners skip them.
  for (size_t lineStart = 0, e = text.size(); lineStart < e;) {
    size_t lineEnd = text.find('\n', lineStart);
    if (lineEnd == StringRef::npos)
      lineEnd = e;

    size_t i = lineStart;
    while (i < lineEnd) {
      if (!mask[i]) {
        ++i;
        continue;
      }
      if (std::isspace(static_cast<unsigned char>(text[i]))) {
        ++i;
        continue;
      }
      break;
    }

    if (i < lineEnd && mask[i] && text[i] == '`')
      for (size_t k = i; k < lineEnd; ++k)
        mask[k] = 0;

    lineStart = lineEnd < e ? lineEnd + 1 : e;
  }
}

static bool isCodeAt(ArrayRef<uint8_t> codeMask, size_t pos) {
  return pos < codeMask.size() && codeMask[pos];
}

static bool isCodeRange(ArrayRef<uint8_t> codeMask, size_t pos, size_t len) {
  if (len == 0 || pos == StringRef::npos || pos + len > codeMask.size())
    return false;
  for (size_t i = 0; i < len; ++i)
    if (!codeMask[pos + i])
      return false;
  return true;
}

static size_t findPrevCodeNonSpace(StringRef text, ArrayRef<uint8_t> codeMask,
                                   size_t pos);
static size_t findNextCodeNonSpace(StringRef text, ArrayRef<uint8_t> codeMask,
                                   size_t pos);
static size_t findStatementStart(StringRef text, ArrayRef<uint8_t> codeMask,
                                 size_t pos);
static size_t findMatchingParen(StringRef text, ArrayRef<uint8_t> codeMask,
                                size_t openPos);
static bool matchKeywordTokenAt(StringRef text, ArrayRef<uint8_t> codeMask,
                                size_t pos, StringRef keyword);
static bool isOperandEndChar(char c);
static bool isOperandStartChar(char c);
static bool isUnaryOperatorContext(char c);
static bool statementLooksLikeTypedDeclaration(StringRef text,
                                               ArrayRef<uint8_t> codeMask,
                                               size_t stmtStart, size_t pos);

static void collectLiteralTokenSites(StringRef text, StringRef token,
                                     ArrayRef<uint8_t> codeMask,
                                     SmallVectorImpl<SiteInfo> &sites) {
  if (token.empty())
    return;
  size_t pos = 0;
  while (true) {
    pos = text.find(token, pos);
    if (pos == StringRef::npos)
      break;
    if (isCodeRange(codeMask, pos, token.size()))
      sites.push_back({pos});
    pos += token.size();
  }
}

static void collectKeywordTokenSites(StringRef text, StringRef keyword,
                                     ArrayRef<uint8_t> codeMask,
                                     SmallVectorImpl<SiteInfo> &sites) {
  if (keyword.empty())
    return;
  size_t len = keyword.size();
  for (size_t i = 0, e = text.size(); i + len <= e; ++i) {
    if (!isCodeRange(codeMask, i, len))
      continue;
    if (!text.substr(i).starts_with(keyword))
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char next = (i + len < e && isCodeAt(codeMask, i + len)) ? text[i + len]
                                                              : '\0';
    bool prevBoundary = prev == '\0' || (!isAlnum(prev) && prev != '_' && prev != '$');
    bool nextBoundary = next == '\0' || (!isAlnum(next) && next != '_' && next != '$');
    if (!prevBoundary || !nextBoundary)
      continue;
    sites.push_back({i});
  }
}

static void collectLogicalTokenSites(StringRef text, StringRef token,
                                     ArrayRef<uint8_t> codeMask,
                                     SmallVectorImpl<SiteInfo> &sites) {
  assert((token == "&&" || token == "||") && "expected logical token");
  char marker = token[0];
  for (size_t i = 0, e = text.size(); i + 1 < e; ++i) {
    if (!isCodeRange(codeMask, i, token.size()))
      continue;
    if (!text.substr(i).starts_with(token))
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char next =
        (i + 2 < e && isCodeAt(codeMask, i + 2)) ? text[i + 2] : '\0';
    if (prev == marker || next == marker)
      continue;

    size_t prevSig = findPrevCodeNonSpace(text, codeMask, i);
    size_t nextSig = findNextCodeNonSpace(text, codeMask, i + 2);
    if (prevSig == StringRef::npos || nextSig == StringRef::npos)
      continue;
    char prevSigChar = text[prevSig];
    char nextSigChar = text[nextSig];
    if (!isOperandEndChar(prevSigChar) || !isOperandStartChar(nextSigChar))
      continue;

    sites.push_back({i});
  }
}

static void collectComparatorTokenSites(StringRef text, StringRef token,
                                        ArrayRef<uint8_t> codeMask,
                                        SmallVectorImpl<SiteInfo> &sites) {
  assert((token == "==" || token == "!=" || token == "===" ||
          token == "!==") &&
         "expected comparator token");
  size_t len = token.size();
  for (size_t i = 0, e = text.size(); i + len <= e; ++i) {
    if (!isCodeRange(codeMask, i, len))
      continue;
    if (!text.substr(i).starts_with(token))
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char next = (i + len < e && isCodeAt(codeMask, i + len)) ? text[i + len]
                                                              : '\0';
    if (token == "==") {
      if (prev == '=' || prev == '!' || prev == '<' || prev == '>')
        continue;
      if (next == '=')
        continue;
    } else if (token == "!=") {
      if (next == '=')
        continue;
    } else if (token == "===" || token == "!==") {
      if (prev == '=' || next == '=')
        continue;
    }
    sites.push_back({i});
  }
}

static void collectCastFunctionSites(StringRef text, StringRef name,
                                     ArrayRef<uint8_t> codeMask,
                                     SmallVectorImpl<SiteInfo> &sites) {
  assert((name == "signed" || name == "unsigned") &&
         "expected signed or unsigned cast helper");
  size_t nameLen = name.size();
  for (size_t i = 0, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i) || text[i] != '$')
      continue;
    if (!isCodeRange(codeMask, i + 1, nameLen))
      continue;
    if (!text.substr(i + 1).starts_with(name))
      continue;
    size_t endName = i + 1 + nameLen;
    char next = (endName < e && isCodeAt(codeMask, endName)) ? text[endName]
                                                              : '\0';
    if (isAlnum(next) || next == '_' || next == '$')
      continue;
    size_t j = endName;
    while (j < e && isCodeAt(codeMask, j) &&
           std::isspace(static_cast<unsigned char>(text[j])))
      ++j;
    if (j >= e || !isCodeAt(codeMask, j) || text[j] != '(')
      continue;
    sites.push_back({i});
  }
}

static void collectStandaloneCompareSites(StringRef text, char needle,
                                          ArrayRef<uint8_t> codeMask,
                                          SmallVectorImpl<SiteInfo> &sites) {
  auto isCmpNeighbor = [](char c) {
    return c == '<' || c == '>' || c == '=' || c == '!';
  };
  for (size_t i = 0, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    if (text[i] != needle)
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char next = (i + 1 < e && isCodeAt(codeMask, i + 1)) ? text[i + 1] : '\0';
    if (isCmpNeighbor(prev) || isCmpNeighbor(next))
      continue;
    sites.push_back({i});
  }
}

static void collectUnaryNotDropSites(StringRef text,
                                     ArrayRef<uint8_t> codeMask,
                                     SmallVectorImpl<SiteInfo> &sites) {
  for (size_t i = 0, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    if (text[i] != '!')
      continue;
    if (i + 1 < e && isCodeAt(codeMask, i + 1) && text[i + 1] == '=')
      continue;
    size_t j = i + 1;
    while (j < e && std::isspace(static_cast<unsigned char>(text[j])))
      ++j;
    if (j >= e || !isCodeAt(codeMask, j))
      continue;
    char next = text[j];
    if (isAlpha(next) || next == '_' || next == '(')
      sites.push_back({i});
  }
}

static void collectUnaryBitwiseNotDropSites(StringRef text,
                                            ArrayRef<uint8_t> codeMask,
                                            SmallVectorImpl<SiteInfo> &sites) {
  auto isIdentifierBody = [](char c) {
    return isAlnum(c) || c == '_' || c == '$';
  };
  auto isUnaryOperandStart = [](char c) {
    return isAlnum(c) || c == '_' || c == '(' || c == '[' || c == '{' ||
           c == '\'' || c == '~' || c == '!' || c == '$';
  };
  auto parseForwardIdentifier = [&](size_t pos, size_t &start,
                                    size_t &end) -> bool {
    if (pos >= text.size() || !isCodeAt(codeMask, pos))
      return false;
    char c = text[pos];
    if (!(isAlpha(c) || c == '_'))
      return false;
    start = pos;
    end = pos + 1;
    while (end < text.size() && isCodeAt(codeMask, end) &&
           isIdentifierBody(text[end]))
      ++end;
    return true;
  };
  auto parseBackwardIdentifier = [&](size_t pos, size_t &start,
                                     size_t &end) -> bool {
    if (pos == StringRef::npos || pos >= text.size() || !isCodeAt(codeMask, pos))
      return false;
    if (!isIdentifierBody(text[pos]))
      return false;
    end = pos + 1;
    start = pos;
    while (start > 0) {
      size_t prev = start - 1;
      if (!isCodeAt(codeMask, prev) || !isIdentifierBody(text[prev]))
        break;
      start = prev;
    }
    return true;
  };
  auto isSelfToggleAssignment = [&](size_t tildePos, size_t rhsStart) -> bool {
    size_t eqPos = findPrevCodeNonSpace(text, codeMask, tildePos);
    if (eqPos == StringRef::npos || text[eqPos] != '=')
      return false;
    size_t beforeEq = findPrevCodeNonSpace(text, codeMask, eqPos);
    if (beforeEq == StringRef::npos)
      return false;
    if (text[beforeEq] == '<' || text[beforeEq] == '>' || text[beforeEq] == '=' ||
        text[beforeEq] == '!')
      return false;

    size_t lhsStart = 0, lhsEnd = 0, rhsIdentStart = 0, rhsIdentEnd = 0;
    if (!parseBackwardIdentifier(beforeEq, lhsStart, lhsEnd) ||
        !parseForwardIdentifier(rhsStart, rhsIdentStart, rhsIdentEnd))
      return false;

    size_t rhsNext = findNextCodeNonSpace(text, codeMask, rhsIdentEnd);
    if (rhsNext != StringRef::npos &&
        (text[rhsNext] == '[' || text[rhsNext] == '.'))
      return false;

    return text.slice(lhsStart, lhsEnd) == text.slice(rhsIdentStart, rhsIdentEnd);
  };
  for (size_t i = 0, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    if (text[i] != '~')
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    if (prev == '^')
      continue;
    if (i + 1 < e && isCodeAt(codeMask, i + 1)) {
      char immediateNext = text[i + 1];
      if (immediateNext == '&' || immediateNext == '|' || immediateNext == '^' ||
          immediateNext == '=')
        continue;
    }
    size_t j = i + 1;
    while (j < e && std::isspace(static_cast<unsigned char>(text[j])))
      ++j;
    if (j >= e || !isCodeAt(codeMask, j))
      continue;
    char next = text[j];
    if (isUnaryOperandStart(next) && !isSelfToggleAssignment(i, j))
      sites.push_back({i});
  }
}

static void collectUnaryMinusDropSites(StringRef text, ArrayRef<uint8_t> codeMask,
                                       SmallVectorImpl<SiteInfo> &sites) {
  auto findPrevSig = [&](size_t pos) -> size_t {
    if (pos == 0)
      return StringRef::npos;
    size_t i = pos;
    while (i > 0) {
      --i;
      if (!isCodeAt(codeMask, i))
        continue;
      if (std::isspace(static_cast<unsigned char>(text[i])))
        continue;
      return i;
    }
    return StringRef::npos;
  };
  auto findNextSig = [&](size_t pos) -> size_t {
    for (size_t i = pos, e = text.size(); i < e; ++i) {
      if (!isCodeAt(codeMask, i))
        continue;
      if (std::isspace(static_cast<unsigned char>(text[i])))
        continue;
      return i;
    }
    return StringRef::npos;
  };
  for (size_t i = 0, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    if (text[i] != '-')
      continue;
    char prevImmediate =
        (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char nextImmediate =
        (i + 1 < e && isCodeAt(codeMask, i + 1)) ? text[i + 1] : '\0';
    if (prevImmediate == '-' || nextImmediate == '-')
      continue;
    if (nextImmediate == '>')
      continue;

    size_t prevSig = findPrevSig(i);
    if (prevSig != StringRef::npos &&
        !isUnaryOperatorContext(text[prevSig]))
      continue;

    size_t nextSig = findNextSig(i + 1);
    if (nextSig == StringRef::npos)
      continue;
    char next = text[nextSig];
    if (!(isAlnum(next) || next == '_' || next == '(' || next == '[' ||
          next == '{' || next == '\'' || next == '~' || next == '!' ||
          next == '$'))
      continue;
    sites.push_back({i});
  }
}

static size_t findPrevCodeNonSpace(StringRef text, ArrayRef<uint8_t> codeMask,
                                   size_t pos) {
  if (pos == 0)
    return StringRef::npos;
  size_t i = pos;
  while (i > 0) {
    --i;
    if (!isCodeAt(codeMask, i))
      continue;
    if (std::isspace(static_cast<unsigned char>(text[i])))
      continue;
    return i;
  }
  return StringRef::npos;
}

static size_t findNextCodeNonSpace(StringRef text, ArrayRef<uint8_t> codeMask,
                                   size_t pos) {
  for (size_t i = pos, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    if (std::isspace(static_cast<unsigned char>(text[i])))
      continue;
    return i;
  }
  return StringRef::npos;
}

static bool isOperandEndChar(char c) {
  return isAlnum(c) || c == '_' || c == ')' || c == ']' || c == '}' ||
         c == '\'';
}

static bool isOperandStartChar(char c) {
  return isAlnum(c) || c == '_' || c == '(' || c == '[' || c == '{' ||
         c == '\'' || c == '~' || c == '!' || c == '$';
}

static bool isUnaryOperatorContext(char prev) {
  return prev == '(' || prev == '[' || prev == '{' || prev == ':' ||
         prev == ';' || prev == ',' || prev == '?' || prev == '=' ||
         prev == '+' || prev == '-' || prev == '*' || prev == '/' ||
         prev == '%' || prev == '&' || prev == '|' || prev == '^' ||
         prev == '!' || prev == '~' || prev == '<' || prev == '>';
}

static void collectBinaryShiftSites(StringRef text, StringRef token,
                                    ArrayRef<uint8_t> codeMask,
                                    SmallVectorImpl<SiteInfo> &sites) {
  assert((token == "<<" || token == ">>") && "expected << or >> token");
  char marker = token[0];
  for (size_t i = 0, e = text.size(); i + 1 < e; ++i) {
    if (!isCodeRange(codeMask, i, token.size()))
      continue;
    if (!text.substr(i).starts_with(token))
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char next =
        (i + 2 < e && isCodeAt(codeMask, i + 2)) ? text[i + 2] : '\0';
    if (prev == marker || next == marker)
      continue;
    if (next == '=')
      continue;

    size_t prevSig = findPrevCodeNonSpace(text, codeMask, i);
    size_t nextSig = findNextCodeNonSpace(text, codeMask, i + 2);
    if (prevSig == StringRef::npos || nextSig == StringRef::npos)
      continue;
    char prevSigChar = text[prevSig];
    char nextSigChar = text[nextSig];
    if (!isOperandEndChar(prevSigChar) || !isOperandStartChar(nextSigChar))
      continue;

    sites.push_back({i});
  }
}

static void collectBinaryArithmeticRightShiftSites(
    StringRef text, ArrayRef<uint8_t> codeMask,
    SmallVectorImpl<SiteInfo> &sites) {
  for (size_t i = 0, e = text.size(); i + 2 < e; ++i) {
    if (!isCodeRange(codeMask, i, 3))
      continue;
    if (!text.substr(i).starts_with(">>>"))
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char next =
        (i + 3 < e && isCodeAt(codeMask, i + 3)) ? text[i + 3] : '\0';
    if (prev == '>' || next == '>')
      continue;
    if (next == '=')
      continue;

    size_t prevSig = findPrevCodeNonSpace(text, codeMask, i);
    size_t nextSig = findNextCodeNonSpace(text, codeMask, i + 3);
    if (prevSig == StringRef::npos || nextSig == StringRef::npos)
      continue;
    char prevSigChar = text[prevSig];
    char nextSigChar = text[nextSig];
    if (!isOperandEndChar(prevSigChar) || !isOperandStartChar(nextSigChar))
      continue;

    sites.push_back({i});
  }
}

static void collectBinaryXorSites(StringRef text, ArrayRef<uint8_t> codeMask,
                                  SmallVectorImpl<SiteInfo> &sites) {
  for (size_t i = 0, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    if (text[i] != '^')
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char next = (i + 1 < e && isCodeAt(codeMask, i + 1)) ? text[i + 1] : '\0';
    if (prev == '=' || next == '=')
      continue;
    if (prev == '~' || next == '~')
      continue;
    size_t prevSig = findPrevCodeNonSpace(text, codeMask, i);
    size_t nextSig = findNextCodeNonSpace(text, codeMask, i + 1);
    if (prevSig == StringRef::npos || nextSig == StringRef::npos)
      continue;
    char prevSigChar = text[prevSig];
    char nextSigChar = text[nextSig];
    if (!isOperandEndChar(prevSigChar) || !isOperandStartChar(nextSigChar))
      continue;
    sites.push_back({i});
  }
}

static void collectBinaryXnorSites(StringRef text, ArrayRef<uint8_t> codeMask,
                                   SmallVectorImpl<SiteInfo> &sites) {
  for (size_t i = 0, e = text.size(); i + 1 < e; ++i) {
    if (!isCodeRange(codeMask, i, 2))
      continue;
    if (!text.substr(i).starts_with("^~") && !text.substr(i).starts_with("~^"))
      continue;
    size_t prevSig = findPrevCodeNonSpace(text, codeMask, i);
    size_t nextSig = findNextCodeNonSpace(text, codeMask, i + 2);
    if (prevSig == StringRef::npos || nextSig == StringRef::npos)
      continue;
    char prevSigChar = text[prevSig];
    char nextSigChar = text[nextSig];
    if (!isOperandEndChar(prevSigChar) || !isOperandStartChar(nextSigChar))
      continue;
    sites.push_back({i});
  }
}

static void collectBinaryBitwiseSites(StringRef text, char needle,
                                      ArrayRef<uint8_t> codeMask,
                                      SmallVectorImpl<SiteInfo> &sites) {
  assert((needle == '&' || needle == '|') && "expected & or | token");
  for (size_t i = 0, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    if (text[i] != needle)
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char next = (i + 1 < e && isCodeAt(codeMask, i + 1)) ? text[i + 1] : '\0';
    if (prev == needle || next == needle)
      continue;
    if (prev == '=' || next == '=')
      continue;
    if (prev == '~')
      continue;
    size_t prevSig = findPrevCodeNonSpace(text, codeMask, i);
    size_t nextSig = findNextCodeNonSpace(text, codeMask, i + 1);
    if (prevSig == StringRef::npos || nextSig == StringRef::npos)
      continue;
    char prevSigChar = text[prevSig];
    char nextSigChar = text[nextSig];
    if (!isOperandEndChar(prevSigChar) || !isOperandStartChar(nextSigChar))
      continue;
    sites.push_back({i});
  }
}

static void collectUnaryReductionSites(StringRef text, char needle,
                                       ArrayRef<uint8_t> codeMask,
                                       SmallVectorImpl<SiteInfo> &sites) {
  assert((needle == '&' || needle == '|' || needle == '^') &&
         "expected unary reduction token");
  for (size_t i = 0, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    if (text[i] != needle)
      continue;

    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char next = (i + 1 < e && isCodeAt(codeMask, i + 1)) ? text[i + 1] : '\0';
    if (needle != '^' && (prev == needle || next == needle))
      continue;
    if (prev == '=' || next == '=')
      continue;
    if (needle == '^' && (prev == '~' || next == '~'))
      continue;
    if (prev == '~')
      continue;

    size_t prevSig = findPrevCodeNonSpace(text, codeMask, i);
    if (prevSig != StringRef::npos && !isUnaryOperatorContext(text[prevSig]))
      continue;

    size_t nextSig = findNextCodeNonSpace(text, codeMask, i + 1);
    if (nextSig == StringRef::npos)
      continue;
    if (!isOperandStartChar(text[nextSig]))
      continue;

    sites.push_back({i});
  }
}

static void collectUnaryReductionXnorSites(StringRef text,
                                           ArrayRef<uint8_t> codeMask,
                                           SmallVectorImpl<SiteInfo> &sites) {
  for (size_t i = 0, e = text.size(); i + 1 < e; ++i) {
    if (!isCodeRange(codeMask, i, 2))
      continue;
    if (!text.substr(i).starts_with("^~") && !text.substr(i).starts_with("~^"))
      continue;

    size_t prevSig = findPrevCodeNonSpace(text, codeMask, i);
    if (prevSig != StringRef::npos && !isUnaryOperatorContext(text[prevSig]))
      continue;

    size_t nextSig = findNextCodeNonSpace(text, codeMask, i + 2);
    if (nextSig == StringRef::npos)
      continue;
    if (!isOperandStartChar(text[nextSig]))
      continue;

    sites.push_back({i});
  }
}

static void collectBinaryArithmeticSites(StringRef text, char needle,
                                         ArrayRef<uint8_t> codeMask,
                                         SmallVectorImpl<SiteInfo> &sites) {
  assert((needle == '+' || needle == '-') &&
         "expected binary arithmetic token");
  int bracketDepth = 0;
  for (size_t i = 0, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    char ch = text[i];
    if (ch == '[') {
      ++bracketDepth;
      continue;
    }
    if (ch == ']') {
      if (bracketDepth > 0)
        --bracketDepth;
      continue;
    }
    if (ch != needle)
      continue;
    if (bracketDepth > 0)
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char next = (i + 1 < e && isCodeAt(codeMask, i + 1)) ? text[i + 1] : '\0';
    if (prev == needle || next == needle)
      continue;
    if (prev == '=' || next == '=')
      continue;
    if (needle == '-' && next == '>')
      continue;

    size_t prevSig = findPrevCodeNonSpace(text, codeMask, i);
    size_t nextSig = findNextCodeNonSpace(text, codeMask, i + 1);
    if (prevSig == StringRef::npos || nextSig == StringRef::npos)
      continue;
    char prevSigChar = text[prevSig];
    char nextSigChar = text[nextSig];
    if (!isOperandEndChar(prevSigChar) || !isOperandStartChar(nextSigChar))
      continue;

    sites.push_back({i});
  }
}

static void collectBinaryMulDivSites(StringRef text, char needle,
                                     ArrayRef<uint8_t> codeMask,
                                     SmallVectorImpl<SiteInfo> &sites) {
  assert((needle == '*' || needle == '/' || needle == '%') &&
         "expected * or / or % token");
  int bracketDepth = 0;
  for (size_t i = 0, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    char ch = text[i];
    if (ch == '[') {
      ++bracketDepth;
      continue;
    }
    if (ch == ']') {
      if (bracketDepth > 0)
        --bracketDepth;
      continue;
    }
    if (ch != needle)
      continue;
    if (bracketDepth > 0)
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char next = (i + 1 < e && isCodeAt(codeMask, i + 1)) ? text[i + 1] : '\0';
    if (needle == '*') {
      if (prev == '*' || next == '*')
        continue;
      if (next == '=')
        continue;
      if (prev == '(' && next == ')')
        continue;
    } else if (needle == '/') {
      if (prev == '/' || next == '/')
        continue;
      if (next == '=')
        continue;
    } else {
      if (next == '=')
        continue;
    }

    size_t prevSig = findPrevCodeNonSpace(text, codeMask, i);
    size_t nextSig = findNextCodeNonSpace(text, codeMask, i + 1);
    if (prevSig == StringRef::npos || nextSig == StringRef::npos)
      continue;
    char prevSigChar = text[prevSig];
    char nextSigChar = text[nextSig];
    if (!isOperandEndChar(prevSigChar) || !isOperandStartChar(nextSigChar))
      continue;

    sites.push_back({i});
  }
}

static bool isIdentifierChar(char c) {
  return isAlnum(c) || c == '_' || c == '$';
}

static bool isWithinForHeader(StringRef text, ArrayRef<uint8_t> codeMask,
                              size_t pos) {
  int depth = 0;
  for (size_t i = pos; i > 0; --i) {
    size_t idx = i - 1;
    if (!isCodeAt(codeMask, idx))
      continue;
    char ch = text[idx];
    if (ch == ')') {
      ++depth;
      continue;
    }
    if (ch == '(') {
      if (depth == 0) {
        size_t kwEnd = findPrevCodeNonSpace(text, codeMask, idx);
        if (kwEnd != StringRef::npos) {
          size_t kwStart = kwEnd;
          while (kwStart > 0 && isCodeAt(codeMask, kwStart - 1) &&
                 isIdentifierChar(text[kwStart - 1]))
            --kwStart;
          if (text.slice(kwStart, kwEnd + 1).equals_insensitive("for"))
            return true;
        }
      } else {
        --depth;
      }
      continue;
    }
    if (depth == 0 && (ch == '{' || ch == '}'))
      return false;
  }
  return false;
}

static void collectIncDecSites(StringRef text, StringRef token,
                               ArrayRef<uint8_t> codeMask,
                               SmallVectorImpl<SiteInfo> &sites) {
  assert((token == "++" || token == "--") && "expected ++ or -- token");
  for (size_t i = 0, e = text.size(); i + token.size() <= e; ++i) {
    if (!isCodeRange(codeMask, i, token.size()))
      continue;
    if (!text.substr(i).starts_with(token))
      continue;
    if (isWithinForHeader(text, codeMask, i))
      continue;
    sites.push_back({i});
  }
}

static void collectCompoundAssignSites(StringRef text, StringRef token,
                                       ArrayRef<uint8_t> codeMask,
                                       SmallVectorImpl<SiteInfo> &sites) {
  assert((token == "+=" || token == "-=" || token == "*=" || token == "/=" ||
          token == "%=" || token == "<<=" || token == ">>=" || token == ">>>=" ||
          token == "&=" || token == "|=") &&
         "expected +=, -=, *=, /=, %=, <<=, >>=, >>>=, &=, or |= compound assignment token");
  for (size_t i = 0, e = text.size(); i + token.size() <= e; ++i) {
    if (!isCodeRange(codeMask, i, token.size()))
      continue;
    if (!text.substr(i).starts_with(token))
      continue;

    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    if (token == "+=" && prev == '+')
      continue;
    if (token == "-=" && prev == '-')
      continue;
    if (token == "*=" && prev == '*')
      continue;
    if (token == "/=" && prev == '/')
      continue;
    if (token == "%=" && prev == '%')
      continue;
    if (token == "<<=" && prev == '<')
      continue;
    if (token == ">>=" && prev == '>')
      continue;
    if (token == ">>>=" && prev == '>')
      continue;
    if (token == "&=" && prev == '&')
      continue;
    if (token == "|=" && prev == '|')
      continue;

    size_t prevSig = findPrevCodeNonSpace(text, codeMask, i);
    size_t nextSig = findNextCodeNonSpace(text, codeMask, i + token.size());
    if (prevSig == StringRef::npos || nextSig == StringRef::npos)
      continue;
    if (!isOperandEndChar(text[prevSig]) || !isOperandStartChar(text[nextSig]))
      continue;

    size_t stmtStart = findStatementStart(text, codeMask, i);
    if (statementLooksLikeTypedDeclaration(text, codeMask, stmtStart, i))
      continue;

    sites.push_back({i});
  }
}

static void collectRelationalComparatorSites(StringRef text, StringRef token,
                                             ArrayRef<uint8_t> codeMask,
                                             SmallVectorImpl<SiteInfo> &sites) {
  assert((token == "<=" || token == ">=") &&
         "expected <= or >= relational token");
  int parenDepth = 0;
  int bracketDepth = 0;
  int braceDepth = 0;
  bool sawPlainAssign = false;
  auto decDepth = [](int &depth) {
    if (depth > 0)
      --depth;
  };

  for (size_t i = 0, e = text.size(); i + 1 < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    char ch = text[i];
    char next = isCodeAt(codeMask, i + 1) ? text[i + 1] : '\0';

    if (ch == ';') {
      sawPlainAssign = false;
      continue;
    }
    if (ch == '(')
      ++parenDepth;
    else if (ch == ')')
      decDepth(parenDepth);
    else if (ch == '[')
      ++bracketDepth;
    else if (ch == ']')
      decDepth(bracketDepth);
    else if (ch == '{')
      ++braceDepth;
    else if (ch == '}')
      decDepth(braceDepth);

    if (ch == '=') {
      char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
      if (prev != '=' && prev != '!' && prev != '<' && prev != '>' &&
          next != '=')
        sawPlainAssign = true;
    }

    if (!isCodeRange(codeMask, i, token.size()))
      continue;
    if (!text.substr(i).starts_with(token))
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    if (token == "<=" && prev == '<')
      continue;
    if (token == ">=" && prev == '>')
      continue;
    if (parenDepth > 0 || bracketDepth > 0 || braceDepth > 0 ||
        sawPlainAssign)
      sites.push_back({i});
  }
}

static size_t findStatementStart(StringRef text, ArrayRef<uint8_t> codeMask,
                                 size_t pos) {
  size_t i = pos;
  while (i > 0) {
    --i;
    if (!isCodeAt(codeMask, i))
      continue;
    char ch = text[i];
    if (ch == ';' || ch == '\n' || ch == '{' || ch == '}')
      return i + 1;
  }
  return 0;
}

static bool statementHasPlainAssignBefore(StringRef text,
                                          ArrayRef<uint8_t> codeMask,
                                          size_t stmtStart, size_t pos) {
  for (size_t i = stmtStart; i < pos; ++i) {
    if (!isCodeAt(codeMask, i) || text[i] != '=')
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char next =
        (i + 1 < text.size() && isCodeAt(codeMask, i + 1)) ? text[i + 1]
                                                            : '\0';
    if (prev == '=' || prev == '!' || prev == '<' || prev == '>' || next == '=' ||
        next == '>')
      continue;
    return true;
  }
  return false;
}

static bool statementHasAssignmentBefore(StringRef text,
                                         ArrayRef<uint8_t> codeMask,
                                         size_t stmtStart, size_t pos) {
  int parenDepth = 0;
  int bracketDepth = 0;
  int braceDepth = 0;
  auto decDepth = [](int &depth) {
    if (depth > 0)
      --depth;
  };

  for (size_t i = stmtStart; i < pos; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    char ch = text[i];
    if (ch == '(')
      ++parenDepth;
    else if (ch == ')')
      decDepth(parenDepth);
    else if (ch == '[')
      ++bracketDepth;
    else if (ch == ']')
      decDepth(bracketDepth);
    else if (ch == '{')
      ++braceDepth;
    else if (ch == '}')
      decDepth(braceDepth);

    if (parenDepth > 0 || bracketDepth > 0 || braceDepth > 0)
      continue;

    if (ch == '=') {
      char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
      char next = (i + 1 < text.size() && isCodeAt(codeMask, i + 1))
                      ? text[i + 1]
                      : '\0';
      if (prev == '=' || prev == '!' || prev == '<' || prev == '>' || next == '=' ||
          next == '>')
        continue;
      return true;
    }

    if (!isCodeRange(codeMask, i, 2) || !text.substr(i).starts_with("<="))
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char next = (i + 2 < text.size() && isCodeAt(codeMask, i + 2))
                    ? text[i + 2]
                    : '\0';
    if (prev == '<' || prev == '=' || prev == '!' || prev == '>' || next == '=' ||
        next == '>')
      continue;
    return true;
  }
  return false;
}

static bool statementHasDisqualifierToken(StringRef text,
                                          ArrayRef<uint8_t> codeMask,
                                          size_t stmtStart, size_t pos,
                                          ArrayRef<const char *> disqualifiers) {
  size_t i = stmtStart;
  while (i < pos) {
    if (!isCodeAt(codeMask, i) || !(isAlpha(text[i]) || text[i] == '_')) {
      ++i;
      continue;
    }
    size_t start = i;
    ++i;
    while (i < pos && isCodeAt(codeMask, i) &&
           (isAlnum(text[i]) || text[i] == '_' || text[i] == '$'))
      ++i;
    StringRef token = text.slice(start, i);
    for (const char *kw : disqualifiers)
      if (token.equals_insensitive(kw))
        return true;
  }
  return false;
}

static bool statementHasAssignmentDisqualifier(StringRef text,
                                               ArrayRef<uint8_t> codeMask,
                                               size_t stmtStart, size_t pos) {
  static constexpr const char *kDisqualifiers[] = {
      "assign",   "parameter", "localparam", "typedef", "input",
      "output",   "inout",     "wire",       "logic",   "reg",
      "bit",      "byte",      "shortint",   "int",     "longint",
      "integer",  "time",      "realtime",   "real",    "string",
      "enum",     "struct",    "union",      "genvar",  "module",
      "interface","package",   "class",      "function","task"};
  return statementHasDisqualifierToken(text, codeMask, stmtStart, pos,
                                       kDisqualifiers);
}

static bool statementHasDeclarativeDisqualifier(StringRef text,
                                                ArrayRef<uint8_t> codeMask,
                                                size_t stmtStart, size_t pos) {
  static constexpr const char *kDisqualifiers[] = {
      "parameter", "localparam", "typedef",  "input",   "output", "inout",
      "wire",      "logic",      "reg",      "bit",     "byte",   "shortint",
      "int",       "longint",    "integer",  "time",    "realtime",
      "real",      "string",     "enum",     "struct",  "union",  "genvar",
      "module",    "interface",  "package",  "class",   "function",
      "task"};
  return statementHasDisqualifierToken(text, codeMask, stmtStart, pos,
                                       kDisqualifiers);
}

static size_t skipCodeWhitespace(StringRef text, ArrayRef<uint8_t> codeMask,
                                 size_t i, size_t limit) {
  while (i < limit) {
    if (!isCodeAt(codeMask, i)) {
      ++i;
      continue;
    }
    if (std::isspace(static_cast<unsigned char>(text[i]))) {
      ++i;
      continue;
    }
    break;
  }
  return i;
}

static bool parseIdentifierToken(StringRef text, ArrayRef<uint8_t> codeMask,
                                 size_t &i, size_t limit) {
  if (i >= limit || !isCodeAt(codeMask, i))
    return false;
  char ch = text[i];
  if (!(isAlpha(ch) || ch == '_'))
    return false;
  ++i;
  while (i < limit && isCodeAt(codeMask, i) &&
         (isAlnum(text[i]) || text[i] == '_' || text[i] == '$'))
    ++i;
  return true;
}

static size_t skipBalancedGroup(StringRef text, ArrayRef<uint8_t> codeMask,
                                size_t i, size_t limit, char openCh,
                                char closeCh) {
  if (i >= limit || !isCodeAt(codeMask, i) || text[i] != openCh)
    return i;
  int depth = 0;
  while (i < limit) {
    if (!isCodeAt(codeMask, i)) {
      ++i;
      continue;
    }
    char ch = text[i];
    if (ch == openCh) {
      ++depth;
    } else if (ch == closeCh) {
      --depth;
      if (depth == 0)
        return i + 1;
    }
    ++i;
  }
  return limit;
}

static bool statementLooksLikeTypedDeclaration(StringRef text,
                                               ArrayRef<uint8_t> codeMask,
                                               size_t stmtStart, size_t pos) {
  size_t limit = std::min(pos, text.size());
  size_t i = skipCodeWhitespace(text, codeMask, stmtStart, limit);
  size_t firstStart = i;
  if (!parseIdentifierToken(text, codeMask, i, limit))
    return false;
  StringRef firstToken = text.slice(firstStart, i);
  if (firstToken.equals_insensitive("assign") ||
      firstToken.equals_insensitive("if") ||
      firstToken.equals_insensitive("for") ||
      firstToken.equals_insensitive("while") ||
      firstToken.equals_insensitive("case") ||
      firstToken.equals_insensitive("foreach") ||
      firstToken.equals_insensitive("return") ||
      firstToken.equals_insensitive("begin") ||
      firstToken.equals_insensitive("end"))
    return false;

  while (true) {
    i = skipCodeWhitespace(text, codeMask, i, limit);
    if (i + 1 < limit && isCodeRange(codeMask, i, 2) &&
        text.substr(i).starts_with("::")) {
      i = skipCodeWhitespace(text, codeMask, i + 2, limit);
      if (!parseIdentifierToken(text, codeMask, i, limit))
        return false;
      continue;
    }
    if (i < limit && isCodeAt(codeMask, i) && text[i] == '#') {
      i = skipCodeWhitespace(text, codeMask, i + 1, limit);
      i = skipBalancedGroup(text, codeMask, i, limit, '(', ')');
      continue;
    }
    if (i < limit && isCodeAt(codeMask, i) && text[i] == '[') {
      i = skipBalancedGroup(text, codeMask, i, limit, '[', ']');
      continue;
    }
    break;
  }

  i = skipCodeWhitespace(text, codeMask, i, limit);
  size_t secondStart = i;
  if (!parseIdentifierToken(text, codeMask, i, limit))
    return false;
  size_t prevSig = findPrevCodeNonSpace(text, codeMask, secondStart);
  if (prevSig != StringRef::npos && text[prevSig] == '.')
    return false;
  return true;
}

static bool
findNearestProceduralHeadBefore(StringRef text, ArrayRef<uint8_t> codeMask,
                                size_t anchor, StringRef &head,
                                size_t &headPos) {
  head = StringRef();
  headPos = StringRef::npos;
  if (anchor == StringRef::npos || anchor == 0)
    return false;

  static constexpr const char *kHeads[] = {"always_comb", "always_latch",
                                           "always_ff",   "always",
                                           "initial"};
  for (size_t off = anchor; off > 0; --off) {
    size_t i = off - 1;
    if (!isCodeAt(codeMask, i))
      continue;
    if (matchKeywordTokenAt(text, codeMask, i, "endmodule"))
      break;
    for (const char *kw : kHeads) {
      if (matchKeywordTokenAt(text, codeMask, i, kw)) {
        head = StringRef(kw);
        headPos = i;
        return true;
      }
    }
  }
  return false;
}

static bool shouldSkipTimingAssignmentMutation(StringRef text,
                                               ArrayRef<uint8_t> codeMask,
                                               size_t stmtStart,
                                               bool sourceIsNonblocking) {
  StringRef head;
  size_t headPos = StringRef::npos;
  if (!findNearestProceduralHeadBefore(text, codeMask, stmtStart, head,
                                       headPos))
    return false;

  // Mixed BA/NBA rewrites in always_comb/always_latch are non-portable and can
  // produce simulator-dependent behavior.
  if (head == "always_comb" || head == "always_latch")
    return true;

  // NBA->BA in initial blocks frequently creates same-edge races against
  // clocked logic. Keep these sites out of deterministic campaigns.
  if (sourceIsNonblocking && head == "initial")
    return true;

  return false;
}

static void collectProceduralBlockingAssignSites(
    StringRef text, ArrayRef<uint8_t> codeMask, SmallVectorImpl<SiteInfo> &sites) {
  int parenDepth = 0;
  int bracketDepth = 0;
  int braceDepth = 0;
  auto decDepth = [](int &depth) {
    if (depth > 0)
      --depth;
  };

  for (size_t i = 0, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    char ch = text[i];
    if (ch == '(') {
      ++parenDepth;
      continue;
    }
    if (ch == ')') {
      decDepth(parenDepth);
      continue;
    }
    if (ch == '[') {
      ++bracketDepth;
      continue;
    }
    if (ch == ']') {
      decDepth(bracketDepth);
      continue;
    }
    if (ch == '{') {
      ++braceDepth;
      continue;
    }
    if (ch == '}') {
      decDepth(braceDepth);
      continue;
    }
    if (ch != '=')
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char next =
        (i + 1 < e && isCodeAt(codeMask, i + 1)) ? text[i + 1] : '\0';
    if (prev == '=' || prev == '!' || prev == '<' || prev == '>' || next == '=' ||
        next == '>')
      continue;
    if (parenDepth > 0 || bracketDepth > 0 || braceDepth > 0)
      continue;

    size_t prevSig = findPrevCodeNonSpace(text, codeMask, i);
    size_t nextSig = findNextCodeNonSpace(text, codeMask, i + 1);
    if (prevSig == StringRef::npos || nextSig == StringRef::npos)
      continue;
    if (!isOperandEndChar(text[prevSig]) || !isOperandStartChar(text[nextSig]))
      continue;

    size_t stmtStart = findStatementStart(text, codeMask, i);
    if (statementHasAssignmentDisqualifier(text, codeMask, stmtStart, i))
      continue;
    if (statementLooksLikeTypedDeclaration(text, codeMask, stmtStart, i))
      continue;
    if (statementHasPlainAssignBefore(text, codeMask, stmtStart, i))
      continue;
    if (shouldSkipTimingAssignmentMutation(text, codeMask, stmtStart,
                                           /*sourceIsNonblocking=*/false))
      continue;

    sites.push_back({i});
  }
}

static void collectProceduralNonblockingAssignSites(
    StringRef text, ArrayRef<uint8_t> codeMask, SmallVectorImpl<SiteInfo> &sites) {
  int parenDepth = 0;
  int bracketDepth = 0;
  int braceDepth = 0;
  auto decDepth = [](int &depth) {
    if (depth > 0)
      --depth;
  };

  for (size_t i = 0, e = text.size(); i + 1 < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    char ch = text[i];
    if (ch == '(') {
      ++parenDepth;
      continue;
    }
    if (ch == ')') {
      decDepth(parenDepth);
      continue;
    }
    if (ch == '[') {
      ++bracketDepth;
      continue;
    }
    if (ch == ']') {
      decDepth(bracketDepth);
      continue;
    }
    if (ch == '{') {
      ++braceDepth;
      continue;
    }
    if (ch == '}') {
      decDepth(braceDepth);
      continue;
    }
    if (!isCodeRange(codeMask, i, 2) || !text.substr(i).starts_with("<="))
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char next =
        (i + 2 < e && isCodeAt(codeMask, i + 2)) ? text[i + 2] : '\0';
    if (prev == '<' || prev == '=' || prev == '!' || prev == '>' || next == '=' ||
        next == '>')
      continue;
    if (parenDepth > 0 || bracketDepth > 0 || braceDepth > 0)
      continue;

    size_t prevSig = findPrevCodeNonSpace(text, codeMask, i);
    size_t nextSig = findNextCodeNonSpace(text, codeMask, i + 2);
    if (prevSig == StringRef::npos || nextSig == StringRef::npos)
      continue;
    if (!isOperandEndChar(text[prevSig]) || !isOperandStartChar(text[nextSig]))
      continue;

    size_t stmtStart = findStatementStart(text, codeMask, i);
    if (statementHasAssignmentDisqualifier(text, codeMask, stmtStart, i))
      continue;
    if (statementLooksLikeTypedDeclaration(text, codeMask, stmtStart, i))
      continue;
    if (statementHasPlainAssignBefore(text, codeMask, stmtStart, i))
      continue;
    if (shouldSkipTimingAssignmentMutation(text, codeMask, stmtStart,
                                           /*sourceIsNonblocking=*/true))
      continue;

    sites.push_back({i});
  }
}

static size_t findStatementSemicolon(StringRef text, ArrayRef<uint8_t> codeMask,
                                     size_t start) {
  int parenDepth = 0;
  int bracketDepth = 0;
  int braceDepth = 0;
  auto decDepth = [](int &depth) {
    if (depth > 0)
      --depth;
  };
  for (size_t i = start, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    char ch = text[i];
    if (ch == '(') {
      ++parenDepth;
      continue;
    }
    if (ch == ')') {
      decDepth(parenDepth);
      continue;
    }
    if (ch == '[') {
      ++bracketDepth;
      continue;
    }
    if (ch == ']') {
      decDepth(bracketDepth);
      continue;
    }
    if (ch == '{') {
      ++braceDepth;
      continue;
    }
    if (ch == '}') {
      decDepth(braceDepth);
      continue;
    }
    if (parenDepth == 0 && bracketDepth == 0 && braceDepth == 0 && ch == ';')
      return i;
  }
  return StringRef::npos;
}

static bool findSimpleAssignmentRhsIdentifierSpan(StringRef text,
                                                  ArrayRef<uint8_t> codeMask,
                                                  size_t assignPos,
                                                  size_t &rhsStart,
                                                  size_t &rhsEnd) {
  if (assignPos >= text.size() || !isCodeAt(codeMask, assignPos))
    return false;

  size_t assignLen = 0;
  if (isCodeRange(codeMask, assignPos, 2) &&
      text.substr(assignPos).starts_with("<=")) {
    char prev =
        (assignPos == 0 || !isCodeAt(codeMask, assignPos - 1))
            ? '\0'
            : text[assignPos - 1];
    char next = (assignPos + 2 < text.size() && isCodeAt(codeMask, assignPos + 2))
                    ? text[assignPos + 2]
                    : '\0';
    if (prev == '<' || prev == '=' || prev == '!' || prev == '>' ||
        next == '=' || next == '>')
      return false;
    assignLen = 2;
  } else if (text[assignPos] == '=') {
    char prev =
        (assignPos == 0 || !isCodeAt(codeMask, assignPos - 1))
            ? '\0'
            : text[assignPos - 1];
    char next = (assignPos + 1 < text.size() && isCodeAt(codeMask, assignPos + 1))
                    ? text[assignPos + 1]
                    : '\0';
    if (prev == '=' || prev == '!' || prev == '<' || prev == '>' ||
        next == '=' || next == '>')
      return false;
    assignLen = 1;
  } else {
    return false;
  }

  size_t prevSig = findPrevCodeNonSpace(text, codeMask, assignPos);
  size_t nextSig = findNextCodeNonSpace(text, codeMask, assignPos + assignLen);
  if (prevSig == StringRef::npos || nextSig == StringRef::npos)
    return false;
  if (!isOperandEndChar(text[prevSig]) || !isOperandStartChar(text[nextSig]))
    return false;

  size_t stmtStart = findStatementStart(text, codeMask, assignPos);
  if (statementHasDeclarativeDisqualifier(text, codeMask, stmtStart, assignPos))
    return false;
  if (statementLooksLikeTypedDeclaration(text, codeMask, stmtStart, assignPos))
    return false;
  if (statementHasPlainAssignBefore(text, codeMask, stmtStart, assignPos))
    return false;

  rhsStart = findNextCodeNonSpace(text, codeMask, assignPos + assignLen);
  if (rhsStart == StringRef::npos)
    return false;

  size_t semiPos = findStatementSemicolon(text, codeMask, rhsStart);
  if (semiPos == StringRef::npos)
    return false;
  rhsEnd = findPrevCodeNonSpace(text, codeMask, semiPos);
  if (rhsEnd == StringRef::npos || rhsStart > rhsEnd)
    return false;

  size_t parsePos = rhsStart;
  size_t parseLimit = rhsEnd + 1;
  if (!parseIdentifierToken(text, codeMask, parsePos, parseLimit))
    return false;
  parsePos = skipCodeWhitespace(text, codeMask, parsePos, parseLimit);
  return parsePos == parseLimit;
}

static void collectAssignRhsIdentifierSites(
    StringRef text, ArrayRef<uint8_t> codeMask, SmallVectorImpl<SiteInfo> &sites) {
  int parenDepth = 0;
  int bracketDepth = 0;
  int braceDepth = 0;
  auto decDepth = [](int &depth) {
    if (depth > 0)
      --depth;
  };

  for (size_t i = 0, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    char ch = text[i];
    if (ch == '(') {
      ++parenDepth;
      continue;
    }
    if (ch == ')') {
      decDepth(parenDepth);
      continue;
    }
    if (ch == '[') {
      ++bracketDepth;
      continue;
    }
    if (ch == ']') {
      decDepth(bracketDepth);
      continue;
    }
    if (ch == '{') {
      ++braceDepth;
      continue;
    }
    if (ch == '}') {
      decDepth(braceDepth);
      continue;
    }

    bool isAssignToken = false;
    if (ch == '=')
      isAssignToken = true;
    else if (ch == '<' && i + 1 < e && isCodeRange(codeMask, i, 2) &&
             text.substr(i).starts_with("<="))
      isAssignToken = true;
    if (!isAssignToken)
      continue;
    if (parenDepth > 0 || bracketDepth > 0 || braceDepth > 0)
      continue;

    size_t rhsStart = StringRef::npos;
    size_t rhsEnd = StringRef::npos;
    if (!findSimpleAssignmentRhsIdentifierSpan(text, codeMask, i, rhsStart,
                                               rhsEnd))
      continue;

    sites.push_back({i});
  }
}

static size_t findMatchingTernaryColon(StringRef text, ArrayRef<uint8_t> codeMask,
                                       size_t questionPos) {
  int parenDepth = 0;
  int bracketDepth = 0;
  int braceDepth = 0;
  int nestedTernary = 0;
  auto decDepth = [](int &depth) {
    if (depth > 0)
      --depth;
  };

  for (size_t i = questionPos + 1, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    char ch = text[i];

    if (ch == '(') {
      ++parenDepth;
      continue;
    }
    if (ch == ')') {
      if (parenDepth == 0)
        break;
      decDepth(parenDepth);
      continue;
    }
    if (ch == '[') {
      ++bracketDepth;
      continue;
    }
    if (ch == ']') {
      if (bracketDepth == 0)
        break;
      decDepth(bracketDepth);
      continue;
    }
    if (ch == '{') {
      ++braceDepth;
      continue;
    }
    if (ch == '}') {
      if (braceDepth == 0)
        break;
      decDepth(braceDepth);
      continue;
    }

    if (parenDepth > 0 || bracketDepth > 0 || braceDepth > 0)
      continue;

    if (ch == '?') {
      ++nestedTernary;
      continue;
    }
    if (ch == ':') {
      if (nestedTernary == 0)
        return i;
      --nestedTernary;
      continue;
    }
    if (ch == ';' || ch == ',')
      break;
  }
  return StringRef::npos;
}

static void collectMuxSwapArmSites(StringRef text, ArrayRef<uint8_t> codeMask,
                                   SmallVectorImpl<SiteInfo> &sites) {
  for (size_t i = 0, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i) || text[i] != '?')
      continue;

    size_t prevSig = findPrevCodeNonSpace(text, codeMask, i);
    size_t nextSig = findNextCodeNonSpace(text, codeMask, i + 1);
    if (prevSig == StringRef::npos || nextSig == StringRef::npos)
      continue;
    if (!isOperandEndChar(text[prevSig]) || !isOperandStartChar(text[nextSig]))
      continue;

    size_t stmtStart = findStatementStart(text, codeMask, i);
    if (statementHasDeclarativeDisqualifier(text, codeMask, stmtStart, i))
      continue;
    if (statementLooksLikeTypedDeclaration(text, codeMask, stmtStart, i))
      continue;
    if (!statementHasAssignmentBefore(text, codeMask, stmtStart, i))
      continue;

    size_t colonPos = findMatchingTernaryColon(text, codeMask, i);
    if (colonPos == StringRef::npos)
      continue;

    size_t trueStart = findNextCodeNonSpace(text, codeMask, i + 1);
    size_t trueEnd = findPrevCodeNonSpace(text, codeMask, colonPos);
    size_t falseStart = findNextCodeNonSpace(text, codeMask, colonPos + 1);
    if (trueStart == StringRef::npos || trueEnd == StringRef::npos ||
        falseStart == StringRef::npos || trueStart > trueEnd)
      continue;

    sites.push_back({i});
  }
}

static bool isIdentifierBodyChar(char c) {
  return isAlnum(c) || c == '_' || c == '$';
}

static bool isResetLikeIdentifier(StringRef ident) {
  if (ident.empty())
    return false;
  std::string lower = ident.lower();
  StringRef lowerRef(lower);
  return lowerRef.starts_with("rst") || lowerRef.contains("reset");
}

static bool conditionContainsResetIdentifier(StringRef text,
                                             ArrayRef<uint8_t> codeMask,
                                             size_t openPos,
                                             size_t closePos) {
  if (openPos == StringRef::npos || closePos == StringRef::npos ||
      closePos <= openPos + 1 || closePos > text.size())
    return false;

  for (size_t i = openPos + 1; i < closePos;) {
    if (!isCodeAt(codeMask, i)) {
      ++i;
      continue;
    }
    char ch = text[i];
    if (ch == '\\') {
      size_t start = i + 1;
      ++i;
      while (i < closePos && isCodeAt(codeMask, i) &&
             !std::isspace(static_cast<unsigned char>(text[i])))
        ++i;
      if (i > start && isResetLikeIdentifier(text.slice(start, i)))
        return true;
      continue;
    }
    if (!(isAlpha(ch) || ch == '_')) {
      ++i;
      continue;
    }
    size_t start = i++;
    while (i < closePos && isCodeAt(codeMask, i) &&
           (isAlnum(text[i]) || text[i] == '_' || text[i] == '$'))
      ++i;
    if (isResetLikeIdentifier(text.slice(start, i)))
      return true;
  }
  return false;
}

static bool parseIdentifierAtOrAfter(StringRef text, ArrayRef<uint8_t> codeMask,
                                     size_t startPos, size_t limitPos,
                                     StringRef &identifier) {
  identifier = StringRef();
  if (startPos == StringRef::npos || limitPos == StringRef::npos ||
      startPos >= text.size() || startPos >= limitPos)
    return false;

  size_t i = startPos;
  while (i < limitPos) {
    if (!isCodeAt(codeMask, i)) {
      ++i;
      continue;
    }
    if (std::isspace(static_cast<unsigned char>(text[i]))) {
      ++i;
      continue;
    }
    break;
  }
  if (i >= limitPos || i >= text.size() || !isCodeAt(codeMask, i))
    return false;

  if (text[i] == '\\') {
    size_t begin = i + 1;
    ++i;
    while (i < limitPos && isCodeAt(codeMask, i) &&
           !std::isspace(static_cast<unsigned char>(text[i])))
      ++i;
    if (i <= begin)
      return false;
    identifier = text.slice(begin, i);
    return true;
  }

  if (!(isAlpha(text[i]) || text[i] == '_'))
    return false;
  size_t begin = i++;
  while (i < limitPos && isCodeAt(codeMask, i) &&
         (isAlnum(text[i]) || text[i] == '_' || text[i] == '$'))
    ++i;
  identifier = text.slice(begin, i);
  return !identifier.empty();
}

static bool parseIdentifierEndingAt(StringRef text, ArrayRef<uint8_t> codeMask,
                                    size_t endPos, StringRef &identifier) {
  identifier = StringRef();
  if (endPos == StringRef::npos || endPos >= text.size() ||
      !isCodeAt(codeMask, endPos))
    return false;
  if (!isIdentifierBodyChar(text[endPos]))
    return false;

  size_t begin = endPos;
  while (begin > 0 && isCodeAt(codeMask, begin - 1) &&
         isIdentifierBodyChar(text[begin - 1]))
    --begin;
  if (!(isAlpha(text[begin]) || text[begin] == '_'))
    return false;
  identifier = text.slice(begin, endPos + 1);
  return !identifier.empty();
}

static bool isAlwaysSensitivityKeyword(StringRef identifier) {
  return identifier == "always" || identifier == "always_ff" ||
         identifier == "always_comb" || identifier == "always_latch";
}

static bool findEventControlBoundsForEdgeSite(
    StringRef text, ArrayRef<uint8_t> codeMask, size_t edgeKeywordPos,
    size_t edgeKeywordLen, size_t &openParenPos, size_t &closeParenPos) {
  openParenPos = StringRef::npos;
  closeParenPos = StringRef::npos;
  if (edgeKeywordPos == StringRef::npos || edgeKeywordPos >= text.size())
    return false;

  int depth = 0;
  for (size_t i = edgeKeywordPos; i > 0;) {
    --i;
    if (!isCodeAt(codeMask, i))
      continue;
    char ch = text[i];
    if (ch == ')') {
      ++depth;
      continue;
    }
    if (ch == '(') {
      if (depth == 0) {
        openParenPos = i;
        break;
      }
      --depth;
      continue;
    }
    if (depth == 0 && ch == ';')
      break;
  }
  if (openParenPos == StringRef::npos)
    return false;

  closeParenPos = findMatchingParen(text, codeMask, openParenPos);
  if (closeParenPos == StringRef::npos)
    return false;
  if (edgeKeywordPos + edgeKeywordLen > closeParenPos)
    return false;
  return true;
}

static bool findAlwaysSensitivityEventBoundsForEdgeSite(
    StringRef text, ArrayRef<uint8_t> codeMask, size_t edgeKeywordPos,
    size_t edgeKeywordLen, size_t &openParenPos, size_t &closeParenPos) {
  if (!findEventControlBoundsForEdgeSite(text, codeMask, edgeKeywordPos,
                                         edgeKeywordLen, openParenPos,
                                         closeParenPos))
    return false;

  size_t atPos = findPrevCodeNonSpace(text, codeMask, openParenPos);
  if (atPos == StringRef::npos || text[atPos] != '@')
    return false;

  size_t procKeywordEnd = findPrevCodeNonSpace(text, codeMask, atPos);
  if (procKeywordEnd == StringRef::npos)
    return false;
  StringRef procKeyword;
  if (!parseIdentifierEndingAt(text, codeMask, procKeywordEnd, procKeyword))
    return false;
  if (!isAlwaysSensitivityKeyword(procKeyword))
    return false;
  return true;
}

static bool hasNonAlwaysEventControlEdgeOnSignal(StringRef text,
                                                 ArrayRef<uint8_t> codeMask,
                                                 StringRef edgeKeyword,
                                                 StringRef signalIdentifier) {
  if (edgeKeyword.empty() || signalIdentifier.empty())
    return false;

  SmallVector<SiteInfo, 8> edgeSites;
  collectKeywordTokenSites(text, edgeKeyword, codeMask, edgeSites);
  for (const SiteInfo &edgeSite : edgeSites) {
    size_t openParenPos = StringRef::npos;
    size_t closeParenPos = StringRef::npos;
    if (!findEventControlBoundsForEdgeSite(text, codeMask, edgeSite.pos,
                                           edgeKeyword.size(), openParenPos,
                                           closeParenPos))
      continue;

    size_t alwaysOpenParenPos = StringRef::npos;
    size_t alwaysCloseParenPos = StringRef::npos;
    if (findAlwaysSensitivityEventBoundsForEdgeSite(
            text, codeMask, edgeSite.pos, edgeKeyword.size(),
            alwaysOpenParenPos, alwaysCloseParenPos))
      continue;

    StringRef edgeSignal;
    if (!parseIdentifierAtOrAfter(text, codeMask,
                                  edgeSite.pos + edgeKeyword.size(),
                                  closeParenPos, edgeSignal))
      continue;
    if (edgeSignal == signalIdentifier)
      return true;
  }

  return false;
}

static void collectAlwaysSensitivityEdgeKeywordSites(
    StringRef text, StringRef keyword, StringRef targetKeyword,
    ArrayRef<uint8_t> codeMask, SmallVectorImpl<SiteInfo> &sites) {
  if (keyword.empty())
    return;
  SmallVector<SiteInfo, 8> keywordSites;
  collectKeywordTokenSites(text, keyword, codeMask, keywordSites);
  for (const SiteInfo &site : keywordSites) {
    size_t openParenPos = StringRef::npos;
    size_t closeParenPos = StringRef::npos;
    if (!findAlwaysSensitivityEventBoundsForEdgeSite(
            text, codeMask, site.pos, keyword.size(), openParenPos,
            closeParenPos))
      continue;

    // Avoid race-prone edge swaps that collapse a sequential process edge onto
    // a non-always wait edge on the same signal. These mutants are frequently
    // dominated by testbench scheduling artifacts rather than design behavior.
    if (!targetKeyword.empty()) {
      StringRef edgeSignal;
      if (parseIdentifierAtOrAfter(text, codeMask, site.pos + keyword.size(),
                                   closeParenPos, edgeSignal) &&
          hasNonAlwaysEventControlEdgeOnSignal(text, codeMask, targetKeyword,
                                               edgeSignal))
        continue;
    }
    sites.push_back(site);
  }
}

static void collectResetEdgeKeywordSites(StringRef text, StringRef keyword,
                                         ArrayRef<uint8_t> codeMask,
                                         SmallVectorImpl<SiteInfo> &sites) {
  SmallVector<SiteInfo, 8> keywordSites;
  collectKeywordTokenSites(text, keyword, codeMask, keywordSites);
  for (const SiteInfo &site : keywordSites) {
    size_t openParenPos = StringRef::npos;
    size_t closeParenPos = StringRef::npos;
    if (!findAlwaysSensitivityEventBoundsForEdgeSite(
            text, codeMask, site.pos, keyword.size(), openParenPos,
            closeParenPos))
      continue;
    StringRef identifier;
    size_t keywordEnd = site.pos + keyword.size();
    if (!parseIdentifierAtOrAfter(text, codeMask, keywordEnd, closeParenPos,
                                  identifier))
      continue;
    if (!isResetLikeIdentifier(identifier))
      continue;
    sites.push_back(site);
  }
}

static size_t findMatchingParen(StringRef text, ArrayRef<uint8_t> codeMask,
                                size_t openPos) {
  if (openPos == StringRef::npos || openPos >= text.size() ||
      !isCodeAt(codeMask, openPos) || text[openPos] != '(')
    return StringRef::npos;
  int depth = 0;
  for (size_t i = openPos, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    char ch = text[i];
    if (ch == '(') {
      ++depth;
      continue;
    }
    if (ch != ')')
      continue;
    --depth;
    if (depth == 0)
      return i;
    if (depth < 0)
      return StringRef::npos;
  }
  return StringRef::npos;
}

static bool matchKeywordTokenAt(StringRef text, ArrayRef<uint8_t> codeMask,
                                size_t pos, StringRef keyword) {
  size_t len = keyword.size();
  if (len == 0 || pos == StringRef::npos || pos + len > text.size())
    return false;
  if (!isCodeRange(codeMask, pos, len) || !text.substr(pos).starts_with(keyword))
    return false;
  char prev = (pos == 0 || !isCodeAt(codeMask, pos - 1)) ? '\0' : text[pos - 1];
  char next = (pos + len < text.size() && isCodeAt(codeMask, pos + len))
                  ? text[pos + len]
                  : '\0';
  bool prevBoundary = prev == '\0' || !isIdentifierBodyChar(prev);
  bool nextBoundary = next == '\0' || !isIdentifierBodyChar(next);
  return prevBoundary && nextBoundary;
}

static size_t findMatchingBeginEnd(StringRef text, ArrayRef<uint8_t> codeMask,
                                   size_t beginPos) {
  if (!matchKeywordTokenAt(text, codeMask, beginPos, "begin"))
    return StringRef::npos;
  int depth = 0;
  for (size_t i = beginPos, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    if (matchKeywordTokenAt(text, codeMask, i, "begin")) {
      ++depth;
      i += 4;
      continue;
    }
    if (!matchKeywordTokenAt(text, codeMask, i, "end"))
      continue;
    --depth;
    i += 2;
    if (depth == 0)
      return i + 1;
    if (depth < 0)
      return StringRef::npos;
  }
  return StringRef::npos;
}

static size_t findIfElseBranchEnd(StringRef text, ArrayRef<uint8_t> codeMask,
                                  size_t branchStart) {
  size_t i = skipCodeWhitespace(text, codeMask, branchStart, text.size());
  if (i == StringRef::npos || i >= text.size())
    return StringRef::npos;

  // Avoid dangling-else ambiguity; we only mutate explicit else branches with
  // non-if arms or begin/end blocks.
  if (matchKeywordTokenAt(text, codeMask, i, "if"))
    return StringRef::npos;

  if (matchKeywordTokenAt(text, codeMask, i, "begin"))
    return findMatchingBeginEnd(text, codeMask, i);

  int parenDepth = 0;
  int bracketDepth = 0;
  int braceDepth = 0;
  auto decDepth = [](int &depth) {
    if (depth > 0)
      --depth;
  };
  for (size_t e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    char ch = text[i];
    if (ch == '(') {
      ++parenDepth;
      continue;
    }
    if (ch == ')') {
      decDepth(parenDepth);
      continue;
    }
    if (ch == '[') {
      ++bracketDepth;
      continue;
    }
    if (ch == ']') {
      decDepth(bracketDepth);
      continue;
    }
    if (ch == '{') {
      ++braceDepth;
      continue;
    }
    if (ch == '}') {
      decDepth(braceDepth);
      continue;
    }
    if (parenDepth == 0 && bracketDepth == 0 && braceDepth == 0 && ch == ';')
      return i + 1;
  }
  return StringRef::npos;
}

static void collectIfConditionSites(StringRef text, ArrayRef<uint8_t> codeMask,
                                    SmallVectorImpl<SiteInfo> &sites,
                                    bool requireResetIdentifier) {
  for (size_t i = 0, e = text.size(); i + 1 < e; ++i) {
    if (!isCodeRange(codeMask, i, 2) || !text.substr(i).starts_with("if"))
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char next = (i + 2 < e && isCodeAt(codeMask, i + 2)) ? text[i + 2] : '\0';
    if ((prev != '\0' && isIdentifierBodyChar(prev)) ||
        (next != '\0' && isIdentifierBodyChar(next)))
      continue;

    size_t openPos = i + 2;
    while (openPos < e &&
           (!isCodeAt(codeMask, openPos) ||
            std::isspace(static_cast<unsigned char>(text[openPos]))))
      ++openPos;
    if (openPos >= e || !isCodeAt(codeMask, openPos) || text[openPos] != '(')
      continue;

    size_t closePos = findMatchingParen(text, codeMask, openPos);
    if (closePos == StringRef::npos)
      continue;

    if (requireResetIdentifier &&
        !conditionContainsResetIdentifier(text, codeMask, openPos, closePos))
      continue;

    sites.push_back({i});
  }
}

static void collectIfConditionNegateSites(
    StringRef text, ArrayRef<uint8_t> codeMask,
    SmallVectorImpl<SiteInfo> &sites) {
  collectIfConditionSites(text, codeMask, sites, /*requireResetIdentifier=*/false);
}

static void collectResetConditionNegateSites(
    StringRef text, ArrayRef<uint8_t> codeMask,
    SmallVectorImpl<SiteInfo> &sites) {
  collectIfConditionSites(text, codeMask, sites, /*requireResetIdentifier=*/true);
}

static void collectIfElseSwapArmSites(StringRef text, ArrayRef<uint8_t> codeMask,
                                      SmallVectorImpl<SiteInfo> &sites) {
  for (size_t i = 0, e = text.size(); i + 1 < e; ++i) {
    if (!matchKeywordTokenAt(text, codeMask, i, "if"))
      continue;

    size_t openPos = i + 2;
    while (openPos < e &&
           (!isCodeAt(codeMask, openPos) ||
            std::isspace(static_cast<unsigned char>(text[openPos]))))
      ++openPos;
    if (openPos >= e || !isCodeAt(codeMask, openPos) || text[openPos] != '(')
      continue;

    size_t closePos = findMatchingParen(text, codeMask, openPos);
    if (closePos == StringRef::npos)
      continue;

    size_t thenStart = skipCodeWhitespace(text, codeMask, closePos + 1, e);
    if (thenStart >= e)
      continue;
    size_t thenEnd = findIfElseBranchEnd(text, codeMask, thenStart);
    if (thenEnd == StringRef::npos)
      continue;

    size_t elsePos = skipCodeWhitespace(text, codeMask, thenEnd, e);
    if (elsePos >= e || !matchKeywordTokenAt(text, codeMask, elsePos, "else"))
      continue;

    size_t elseStart = skipCodeWhitespace(text, codeMask, elsePos + 4, e);
    if (elseStart >= e)
      continue;
    size_t elseEnd = findIfElseBranchEnd(text, codeMask, elseStart);
    if (elseEnd == StringRef::npos)
      continue;

    sites.push_back({i});
  }
}

static bool matchCaseKeywordTokenAt(StringRef text, ArrayRef<uint8_t> codeMask,
                                    size_t pos, size_t &keywordLen) {
  keywordLen = 0;
  if (matchKeywordTokenAt(text, codeMask, pos, "casez")) {
    keywordLen = 5;
    return true;
  }
  if (matchKeywordTokenAt(text, codeMask, pos, "casex")) {
    keywordLen = 5;
    return true;
  }
  if (matchKeywordTokenAt(text, codeMask, pos, "case")) {
    keywordLen = 4;
    return true;
  }
  return false;
}

static bool parseCaseBlockBoundsAt(StringRef text, ArrayRef<uint8_t> codeMask,
                                   size_t casePos, size_t &keywordLen,
                                   size_t &closeParenPos,
                                   size_t &endcasePos) {
  keywordLen = 0;
  closeParenPos = StringRef::npos;
  endcasePos = StringRef::npos;
  if (!matchCaseKeywordTokenAt(text, codeMask, casePos, keywordLen))
    return false;

  size_t openPos = skipCodeWhitespace(text, codeMask, casePos + keywordLen,
                                      text.size());
  if (openPos == StringRef::npos || openPos >= text.size() ||
      !isCodeAt(codeMask, openPos) || text[openPos] != '(')
    return false;

  closeParenPos = findMatchingParen(text, codeMask, openPos);
  if (closeParenPos == StringRef::npos)
    return false;

  int depth = 1;
  for (size_t i = closeParenPos + 1, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    if (matchKeywordTokenAt(text, codeMask, i, "endcase")) {
      --depth;
      if (depth == 0) {
        endcasePos = i;
        return true;
      }
      i += strlen("endcase") - 1;
      continue;
    }
    size_t nestedLen = 0;
    if (matchCaseKeywordTokenAt(text, codeMask, i, nestedLen)) {
      ++depth;
      i += nestedLen - 1;
      continue;
    }
  }
  return false;
}

static size_t findCaseItemHeaderColon(StringRef text, ArrayRef<uint8_t> codeMask,
                                      size_t itemStart, size_t endcasePos) {
  int parenDepth = 0;
  int bracketDepth = 0;
  int braceDepth = 0;
  int ternaryDepth = 0;
  auto decDepth = [](int &depth) {
    if (depth > 0)
      --depth;
  };

  for (size_t i = itemStart; i < endcasePos; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    char ch = text[i];
    if (ch == '(') {
      ++parenDepth;
      continue;
    }
    if (ch == ')') {
      decDepth(parenDepth);
      continue;
    }
    if (ch == '[') {
      ++bracketDepth;
      continue;
    }
    if (ch == ']') {
      decDepth(bracketDepth);
      continue;
    }
    if (ch == '{') {
      ++braceDepth;
      continue;
    }
    if (ch == '}') {
      decDepth(braceDepth);
      continue;
    }

    if (parenDepth != 0 || bracketDepth != 0 || braceDepth != 0)
      continue;

    if (ch == '?') {
      ++ternaryDepth;
      continue;
    }
    if (ch == ':') {
      if (ternaryDepth > 0) {
        --ternaryDepth;
        continue;
      }
      char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
      char next =
          (i + 1 < text.size() && isCodeAt(codeMask, i + 1)) ? text[i + 1] : '\0';
      if (prev == ':' || next == ':')
        continue;
      return i;
    }
    if (ch == ';' || matchKeywordTokenAt(text, codeMask, i, "endcase"))
      return StringRef::npos;
  }
  return StringRef::npos;
}

static size_t findSimpleCaseItemBodyEnd(StringRef text,
                                        ArrayRef<uint8_t> codeMask,
                                        size_t bodyStart, size_t endcasePos) {
  size_t i = skipCodeWhitespace(text, codeMask, bodyStart, endcasePos);
  if (i == StringRef::npos || i >= endcasePos)
    return StringRef::npos;

  if (matchKeywordTokenAt(text, codeMask, i, "begin"))
    return findMatchingBeginEnd(text, codeMask, i);

  // Keep swaps structurally conservative; skip complex procedural item bodies
  // without explicit begin/end wrapping.
  if (matchKeywordTokenAt(text, codeMask, i, "if") ||
      matchKeywordTokenAt(text, codeMask, i, "case") ||
      matchKeywordTokenAt(text, codeMask, i, "casez") ||
      matchKeywordTokenAt(text, codeMask, i, "casex"))
    return StringRef::npos;

  int parenDepth = 0;
  int bracketDepth = 0;
  int braceDepth = 0;
  auto decDepth = [](int &depth) {
    if (depth > 0)
      --depth;
  };
  for (size_t e = endcasePos; i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    char ch = text[i];
    if (ch == '(') {
      ++parenDepth;
      continue;
    }
    if (ch == ')') {
      decDepth(parenDepth);
      continue;
    }
    if (ch == '[') {
      ++bracketDepth;
      continue;
    }
    if (ch == ']') {
      decDepth(bracketDepth);
      continue;
    }
    if (ch == '{') {
      ++braceDepth;
      continue;
    }
    if (ch == '}') {
      decDepth(braceDepth);
      continue;
    }
    if (parenDepth == 0 && bracketDepth == 0 && braceDepth == 0 && ch == ';')
      return i + 1;
  }
  return StringRef::npos;
}

static void collectCaseItemsInBlock(StringRef text, ArrayRef<uint8_t> codeMask,
                                    size_t closeParenPos, size_t endcasePos,
                                    SmallVectorImpl<CaseItemSpan> &items) {
  items.clear();
  size_t i = skipCodeWhitespace(text, codeMask, closeParenPos + 1, endcasePos);
  while (i < endcasePos) {
    i = skipCodeWhitespace(text, codeMask, i, endcasePos);
    if (i == StringRef::npos || i >= endcasePos)
      break;

    size_t colonPos = findCaseItemHeaderColon(text, codeMask, i, endcasePos);
    if (colonPos == StringRef::npos)
      break;

    bool isDefault = matchKeywordTokenAt(text, codeMask, i, "default");
    size_t bodyStart = skipCodeWhitespace(text, codeMask, colonPos + 1, endcasePos);
    size_t bodyEnd = findSimpleCaseItemBodyEnd(text, codeMask, bodyStart,
                                               endcasePos);
    if (bodyEnd == StringRef::npos || bodyEnd <= i)
      break;

    items.push_back({i, bodyEnd, isDefault});
    i = bodyEnd;
  }
}

static void collectCaseItemSwapSpans(StringRef text, ArrayRef<uint8_t> codeMask,
                                     SmallVectorImpl<CaseSwapSpan> &swaps) {
  swaps.clear();
  for (size_t i = 0, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    size_t keywordLen = 0;
    size_t closeParenPos = StringRef::npos;
    size_t endcasePos = StringRef::npos;
    if (!parseCaseBlockBoundsAt(text, codeMask, i, keywordLen, closeParenPos,
                                endcasePos))
      continue;

    SmallVector<CaseItemSpan, 8> items;
    collectCaseItemsInBlock(text, codeMask, closeParenPos, endcasePos, items);
    for (size_t itemIndex = 1; itemIndex < items.size(); ++itemIndex) {
      const CaseItemSpan &lhs = items[itemIndex - 1];
      const CaseItemSpan &rhs = items[itemIndex];
      if (lhs.isDefault || rhs.isDefault)
        continue;
      swaps.push_back(
          {lhs.start, lhs.start, lhs.end, rhs.start, rhs.end});
    }

    i = endcasePos + strlen("endcase") - 1;
  }
}

static void collectCaseItemSwapArmSites(StringRef text,
                                        ArrayRef<uint8_t> codeMask,
                                        SmallVectorImpl<SiteInfo> &sites) {
  SmallVector<CaseSwapSpan, 8> swaps;
  collectCaseItemSwapSpans(text, codeMask, swaps);
  for (const CaseSwapSpan &swap : swaps)
    sites.push_back({swap.sitePos});
}

static void collectSitesForOp(StringRef designText, StringRef op,
                              ArrayRef<uint8_t> codeMask,
                              SmallVectorImpl<SiteInfo> &sites) {
  sites.clear();
  if (op == "EQ_TO_NEQ") {
    collectComparatorTokenSites(designText, "==", codeMask, sites);
    return;
  }
  if (op == "NEQ_TO_EQ") {
    collectComparatorTokenSites(designText, "!=", codeMask, sites);
    return;
  }
  if (op == "CASEEQ_TO_EQ") {
    collectComparatorTokenSites(designText, "===", codeMask, sites);
    return;
  }
  if (op == "CASENEQ_TO_NEQ") {
    collectComparatorTokenSites(designText, "!==", codeMask, sites);
    return;
  }
  if (op == "EQ_TO_CASEEQ") {
    collectComparatorTokenSites(designText, "==", codeMask, sites);
    return;
  }
  if (op == "NEQ_TO_CASENEQ") {
    collectComparatorTokenSites(designText, "!=", codeMask, sites);
    return;
  }
  if (op == "SIGNED_TO_UNSIGNED") {
    collectCastFunctionSites(designText, "signed", codeMask, sites);
    return;
  }
  if (op == "UNSIGNED_TO_SIGNED") {
    collectCastFunctionSites(designText, "unsigned", codeMask, sites);
    return;
  }
  if (op == "LT_TO_LE") {
    collectStandaloneCompareSites(designText, '<', codeMask, sites);
    return;
  }
  if (op == "GT_TO_GE") {
    collectStandaloneCompareSites(designText, '>', codeMask, sites);
    return;
  }
  if (op == "LE_TO_LT") {
    collectRelationalComparatorSites(designText, "<=", codeMask, sites);
    return;
  }
  if (op == "GE_TO_GT") {
    collectRelationalComparatorSites(designText, ">=", codeMask, sites);
    return;
  }
  if (op == "LT_TO_GT") {
    collectStandaloneCompareSites(designText, '<', codeMask, sites);
    return;
  }
  if (op == "GT_TO_LT") {
    collectStandaloneCompareSites(designText, '>', codeMask, sites);
    return;
  }
  if (op == "LE_TO_GE") {
    collectRelationalComparatorSites(designText, "<=", codeMask, sites);
    return;
  }
  if (op == "GE_TO_LE") {
    collectRelationalComparatorSites(designText, ">=", codeMask, sites);
    return;
  }
  if (op == "AND_TO_OR") {
    collectLiteralTokenSites(designText, "&&", codeMask, sites);
    return;
  }
  if (op == "OR_TO_AND") {
    collectLiteralTokenSites(designText, "||", codeMask, sites);
    return;
  }
  if (op == "LAND_TO_BAND") {
    collectLogicalTokenSites(designText, "&&", codeMask, sites);
    return;
  }
  if (op == "LOR_TO_BOR") {
    collectLogicalTokenSites(designText, "||", codeMask, sites);
    return;
  }
  if (op == "XOR_TO_OR") {
    collectBinaryXorSites(designText, codeMask, sites);
    return;
  }
  if (op == "XOR_TO_XNOR") {
    collectBinaryXorSites(designText, codeMask, sites);
    return;
  }
  if (op == "XNOR_TO_XOR") {
    collectBinaryXnorSites(designText, codeMask, sites);
    return;
  }
  if (op == "REDAND_TO_REDOR") {
    collectUnaryReductionSites(designText, '&', codeMask, sites);
    return;
  }
  if (op == "REDOR_TO_REDAND") {
    collectUnaryReductionSites(designText, '|', codeMask, sites);
    return;
  }
  if (op == "REDXOR_TO_REDXNOR") {
    collectUnaryReductionSites(designText, '^', codeMask, sites);
    return;
  }
  if (op == "REDXNOR_TO_REDXOR") {
    collectUnaryReductionXnorSites(designText, codeMask, sites);
    return;
  }
  if (op == "BAND_TO_BOR") {
    collectBinaryBitwiseSites(designText, '&', codeMask, sites);
    return;
  }
  if (op == "BOR_TO_BAND") {
    collectBinaryBitwiseSites(designText, '|', codeMask, sites);
    return;
  }
  if (op == "BAND_TO_LAND") {
    collectBinaryBitwiseSites(designText, '&', codeMask, sites);
    return;
  }
  if (op == "BOR_TO_LOR") {
    collectBinaryBitwiseSites(designText, '|', codeMask, sites);
    return;
  }
  if (op == "BA_TO_NBA") {
    collectProceduralBlockingAssignSites(designText, codeMask, sites);
    return;
  }
  if (op == "NBA_TO_BA") {
    collectProceduralNonblockingAssignSites(designText, codeMask, sites);
    return;
  }
  if (op == "ASSIGN_RHS_TO_CONST0" || op == "ASSIGN_RHS_TO_CONST1" ||
      op == "ASSIGN_RHS_INVERT" || op == "ASSIGN_RHS_PLUS_ONE" ||
      op == "ASSIGN_RHS_MINUS_ONE") {
    collectAssignRhsIdentifierSites(designText, codeMask, sites);
    return;
  }
  if (op == "POSEDGE_TO_NEGEDGE") {
    collectAlwaysSensitivityEdgeKeywordSites(designText, "posedge", "negedge",
                                             codeMask, sites);
    return;
  }
  if (op == "NEGEDGE_TO_POSEDGE") {
    collectAlwaysSensitivityEdgeKeywordSites(designText, "negedge", "posedge",
                                             codeMask, sites);
    return;
  }
  if (op == "RESET_POSEDGE_TO_NEGEDGE") {
    collectResetEdgeKeywordSites(designText, "posedge", codeMask, sites);
    return;
  }
  if (op == "RESET_NEGEDGE_TO_POSEDGE") {
    collectResetEdgeKeywordSites(designText, "negedge", codeMask, sites);
    return;
  }
  if (op == "CASE_TO_CASEZ") {
    collectKeywordTokenSites(designText, "case", codeMask, sites);
    return;
  }
  if (op == "CASEZ_TO_CASE") {
    collectKeywordTokenSites(designText, "casez", codeMask, sites);
    return;
  }
  if (op == "CASE_TO_CASEX") {
    collectKeywordTokenSites(designText, "case", codeMask, sites);
    return;
  }
  if (op == "CASEX_TO_CASE") {
    collectKeywordTokenSites(designText, "casex", codeMask, sites);
    return;
  }
  if (op == "CASEZ_TO_CASEX") {
    collectKeywordTokenSites(designText, "casez", codeMask, sites);
    return;
  }
  if (op == "CASEX_TO_CASEZ") {
    collectKeywordTokenSites(designText, "casex", codeMask, sites);
    return;
  }
  if (op == "MUX_SWAP_ARMS") {
    collectMuxSwapArmSites(designText, codeMask, sites);
    return;
  }
  if (op == "MUX_FORCE_TRUE") {
    collectMuxSwapArmSites(designText, codeMask, sites);
    return;
  }
  if (op == "MUX_FORCE_FALSE") {
    collectMuxSwapArmSites(designText, codeMask, sites);
    return;
  }
  if (op == "IF_COND_NEGATE") {
    collectIfConditionNegateSites(designText, codeMask, sites);
    return;
  }
  if (op == "RESET_COND_NEGATE") {
    collectResetConditionNegateSites(designText, codeMask, sites);
    return;
  }
  if (op == "RESET_COND_TRUE") {
    collectResetConditionNegateSites(designText, codeMask, sites);
    return;
  }
  if (op == "RESET_COND_FALSE") {
    collectResetConditionNegateSites(designText, codeMask, sites);
    return;
  }
  if (op == "IF_COND_TRUE") {
    collectIfConditionNegateSites(designText, codeMask, sites);
    return;
  }
  if (op == "IF_COND_FALSE") {
    collectIfConditionNegateSites(designText, codeMask, sites);
    return;
  }
  if (op == "IF_ELSE_SWAP_ARMS") {
    collectIfElseSwapArmSites(designText, codeMask, sites);
    return;
  }
  if (op == "CASE_ITEM_SWAP_ARMS") {
    collectCaseItemSwapArmSites(designText, codeMask, sites);
    return;
  }
  if (op == "UNARY_NOT_DROP") {
    collectUnaryNotDropSites(designText, codeMask, sites);
    return;
  }
  if (op == "UNARY_BNOT_DROP") {
    collectUnaryBitwiseNotDropSites(designText, codeMask, sites);
    return;
  }
  if (op == "UNARY_MINUS_DROP") {
    collectUnaryMinusDropSites(designText, codeMask, sites);
    return;
  }
  if (op == "CONST0_TO_1") {
    collectLiteralTokenSites(designText, "1'b0", codeMask, sites);
    collectLiteralTokenSites(designText, "1'd0", codeMask, sites);
    collectLiteralTokenSites(designText, "1'h0", codeMask, sites);
    collectLiteralTokenSites(designText, "'0", codeMask, sites);
    llvm::sort(sites, [](const SiteInfo &a, const SiteInfo &b) {
      return a.pos < b.pos;
    });
    return;
  }
  if (op == "CONST1_TO_0") {
    collectLiteralTokenSites(designText, "1'b1", codeMask, sites);
    collectLiteralTokenSites(designText, "1'd1", codeMask, sites);
    collectLiteralTokenSites(designText, "1'h1", codeMask, sites);
    collectLiteralTokenSites(designText, "'1", codeMask, sites);
    llvm::sort(sites, [](const SiteInfo &a, const SiteInfo &b) {
      return a.pos < b.pos;
    });
    return;
  }
  if (op == "ADD_TO_SUB") {
    collectBinaryArithmeticSites(designText, '+', codeMask, sites);
    return;
  }
  if (op == "SUB_TO_ADD") {
    collectBinaryArithmeticSites(designText, '-', codeMask, sites);
    return;
  }
  if (op == "MUL_TO_ADD") {
    collectBinaryMulDivSites(designText, '*', codeMask, sites);
    return;
  }
  if (op == "ADD_TO_MUL") {
    collectBinaryArithmeticSites(designText, '+', codeMask, sites);
    return;
  }
  if (op == "DIV_TO_MUL") {
    collectBinaryMulDivSites(designText, '/', codeMask, sites);
    return;
  }
  if (op == "MUL_TO_DIV") {
    collectBinaryMulDivSites(designText, '*', codeMask, sites);
    return;
  }
  if (op == "MOD_TO_DIV") {
    collectBinaryMulDivSites(designText, '%', codeMask, sites);
    return;
  }
  if (op == "DIV_TO_MOD") {
    collectBinaryMulDivSites(designText, '/', codeMask, sites);
    return;
  }
  if (op == "INC_TO_DEC") {
    collectIncDecSites(designText, "++", codeMask, sites);
    return;
  }
  if (op == "DEC_TO_INC") {
    collectIncDecSites(designText, "--", codeMask, sites);
    return;
  }
  if (op == "PLUS_EQ_TO_MINUS_EQ") {
    collectCompoundAssignSites(designText, "+=", codeMask, sites);
    return;
  }
  if (op == "MINUS_EQ_TO_PLUS_EQ") {
    collectCompoundAssignSites(designText, "-=", codeMask, sites);
    return;
  }
  if (op == "MUL_EQ_TO_DIV_EQ") {
    collectCompoundAssignSites(designText, "*=", codeMask, sites);
    return;
  }
  if (op == "DIV_EQ_TO_MUL_EQ") {
    collectCompoundAssignSites(designText, "/=", codeMask, sites);
    return;
  }
  if (op == "MOD_EQ_TO_DIV_EQ") {
    collectCompoundAssignSites(designText, "%=", codeMask, sites);
    return;
  }
  if (op == "DIV_EQ_TO_MOD_EQ") {
    collectCompoundAssignSites(designText, "/=", codeMask, sites);
    return;
  }
  if (op == "SHL_EQ_TO_SHR_EQ") {
    collectCompoundAssignSites(designText, "<<=", codeMask, sites);
    return;
  }
  if (op == "SHR_EQ_TO_SHL_EQ") {
    collectCompoundAssignSites(designText, ">>=", codeMask, sites);
    return;
  }
  if (op == "SHR_EQ_TO_ASHR_EQ") {
    collectCompoundAssignSites(designText, ">>=", codeMask, sites);
    return;
  }
  if (op == "ASHR_EQ_TO_SHR_EQ") {
    collectCompoundAssignSites(designText, ">>>=", codeMask, sites);
    return;
  }
  if (op == "BAND_EQ_TO_BOR_EQ") {
    collectCompoundAssignSites(designText, "&=", codeMask, sites);
    return;
  }
  if (op == "BOR_EQ_TO_BAND_EQ") {
    collectCompoundAssignSites(designText, "|=", codeMask, sites);
    return;
  }
  if (op == "BAND_EQ_TO_BXOR_EQ") {
    collectCompoundAssignSites(designText, "&=", codeMask, sites);
    return;
  }
  if (op == "BOR_EQ_TO_BXOR_EQ") {
    collectCompoundAssignSites(designText, "|=", codeMask, sites);
    return;
  }
  if (op == "BXOR_EQ_TO_BOR_EQ") {
    collectCompoundAssignSites(designText, "^=", codeMask, sites);
    return;
  }
  if (op == "BXOR_EQ_TO_BAND_EQ") {
    collectCompoundAssignSites(designText, "^=", codeMask, sites);
    return;
  }
  if (op == "SHL_TO_SHR") {
    collectBinaryShiftSites(designText, "<<", codeMask, sites);
    return;
  }
  if (op == "SHR_TO_SHL") {
    collectBinaryShiftSites(designText, ">>", codeMask, sites);
    return;
  }
  if (op == "SHR_TO_ASHR") {
    collectBinaryShiftSites(designText, ">>", codeMask, sites);
    return;
  }
  if (op == "ASHR_TO_SHR") {
    collectBinaryArithmeticRightShiftSites(designText, codeMask, sites);
    return;
  }
}

static uint64_t countNativeMutationSitesForOp(StringRef designText,
                                              ArrayRef<uint8_t> codeMask,
                                              StringRef op) {
  SmallVector<SiteInfo, 8> sites;
  collectSitesForOp(designText, op, codeMask, sites);
  return sites.size();
}

static bool hasNativeMutationPattern(StringRef designText,
                                     ArrayRef<uint8_t> codeMask,
                                     StringRef op) {
  return countNativeMutationSitesForOp(designText, codeMask, op) > 0;
}

static std::string getOpFamily(StringRef op) {
  if (op == "EQ_TO_NEQ" || op == "NEQ_TO_EQ" || op == "LT_TO_LE" ||
      op == "GT_TO_GE" || op == "LE_TO_LT" || op == "GE_TO_GT" ||
      op == "LT_TO_GT" || op == "GT_TO_LT" || op == "LE_TO_GE" ||
      op == "GE_TO_LE")
    return "compare";
  if (op == "CASEEQ_TO_EQ" || op == "CASENEQ_TO_NEQ" ||
      op == "EQ_TO_CASEEQ" || op == "NEQ_TO_CASENEQ")
    return "xcompare";
  if (op == "AND_TO_OR" || op == "OR_TO_AND" || op == "LAND_TO_BAND" ||
      op == "LOR_TO_BOR" || op == "XOR_TO_OR" || op == "XOR_TO_XNOR" ||
      op == "XNOR_TO_XOR" || op == "REDAND_TO_REDOR" ||
      op == "REDOR_TO_REDAND" || op == "REDXOR_TO_REDXNOR" ||
      op == "REDXNOR_TO_REDXOR" || op == "BAND_TO_BOR" ||
      op == "BOR_TO_BAND" || op == "BAND_TO_LAND" || op == "BOR_TO_LOR" ||
      op == "UNARY_NOT_DROP" || op == "UNARY_BNOT_DROP" ||
      op == "BAND_EQ_TO_BOR_EQ" || op == "BOR_EQ_TO_BAND_EQ" ||
      op == "BAND_EQ_TO_BXOR_EQ" || op == "BOR_EQ_TO_BXOR_EQ" ||
      op == "BXOR_EQ_TO_BOR_EQ" || op == "BXOR_EQ_TO_BAND_EQ")
    return "logic";
  if (op == "BA_TO_NBA" || op == "NBA_TO_BA" ||
      op == "POSEDGE_TO_NEGEDGE" || op == "NEGEDGE_TO_POSEDGE" ||
      op == "RESET_POSEDGE_TO_NEGEDGE" || op == "RESET_NEGEDGE_TO_POSEDGE")
    return "timing";
  if (op == "ASSIGN_RHS_TO_CONST0" || op == "ASSIGN_RHS_TO_CONST1" ||
      op == "ASSIGN_RHS_INVERT")
    return "connect";
  if (op == "MUX_SWAP_ARMS" || op == "MUX_FORCE_TRUE" ||
      op == "MUX_FORCE_FALSE")
    return "mux";
  if (op == "IF_COND_NEGATE" || op == "RESET_COND_NEGATE" ||
      op == "RESET_COND_TRUE" || op == "RESET_COND_FALSE" ||
      op == "IF_COND_TRUE" || op == "IF_COND_FALSE" ||
      op == "CASE_TO_CASEZ" || op == "CASEZ_TO_CASE" ||
      op == "CASE_TO_CASEX" || op == "CASEX_TO_CASE" ||
      op == "CASEZ_TO_CASEX" || op == "CASEX_TO_CASEZ" ||
      op == "IF_ELSE_SWAP_ARMS" || op == "CASE_ITEM_SWAP_ARMS")
    return "control";
  if (op == "CONST0_TO_1" || op == "CONST1_TO_0")
    return "constant";
  if (op == "ADD_TO_SUB" || op == "SUB_TO_ADD" || op == "MUL_TO_ADD" ||
      op == "ADD_TO_MUL" || op == "DIV_TO_MUL" || op == "MUL_TO_DIV" ||
      op == "MOD_TO_DIV" || op == "DIV_TO_MOD" ||
      op == "UNARY_MINUS_DROP" || op == "INC_TO_DEC" || op == "DEC_TO_INC" ||
      op == "ASSIGN_RHS_PLUS_ONE" || op == "ASSIGN_RHS_MINUS_ONE" ||
      op == "PLUS_EQ_TO_MINUS_EQ" || op == "MINUS_EQ_TO_PLUS_EQ" ||
      op == "MUL_EQ_TO_DIV_EQ" || op == "DIV_EQ_TO_MUL_EQ" ||
      op == "MOD_EQ_TO_DIV_EQ" || op == "DIV_EQ_TO_MOD_EQ")
    return "arithmetic";
  if (op == "SHL_TO_SHR" || op == "SHR_TO_SHL" || op == "SHR_TO_ASHR" ||
      op == "ASHR_TO_SHR" || op == "SHL_EQ_TO_SHR_EQ" ||
      op == "SHR_EQ_TO_SHL_EQ" || op == "SHR_EQ_TO_ASHR_EQ" ||
      op == "ASHR_EQ_TO_SHR_EQ")
    return "shift";
  if (op == "SIGNED_TO_UNSIGNED" || op == "UNSIGNED_TO_SIGNED")
    return "cast";
  return "misc";
}

static void collectLineStarts(StringRef text, SmallVectorImpl<size_t> &starts) {
  starts.clear();
  starts.push_back(0);
  for (size_t i = 0, e = text.size(); i < e; ++i)
    if (text[i] == '\n' && i + 1 < e)
      starts.push_back(i + 1);
}

static uint64_t getLineForPos(ArrayRef<size_t> lineStarts, size_t pos) {
  if (pos == StringRef::npos || lineStarts.empty())
    return 0;
  auto it = std::upper_bound(lineStarts.begin(), lineStarts.end(), pos);
  return static_cast<uint64_t>(it - lineStarts.begin());
}

static std::string classifyContextForPos(StringRef text,
                                         ArrayRef<size_t> lineStarts,
                                         size_t pos) {
  if (pos == StringRef::npos || lineStarts.empty())
    return "unknown";
  uint64_t line = getLineForPos(lineStarts, pos);
  if (line == 0 || line > lineStarts.size())
    return "unknown";
  size_t start = lineStarts[line - 1];
  size_t end = line < lineStarts.size() ? lineStarts[line] - 1 : text.size();
  if (end < start)
    end = start;
  StringRef lineText = text.slice(start, end);
  std::string lower = lineText.lower();
  StringRef lowerRef(lower);
  if (lowerRef.contains("if") || lowerRef.contains("case") ||
      lowerRef.contains("?"))
    return "control";
  if (lowerRef.contains("assert") || lowerRef.contains("cover"))
    return "verification";
  if (lowerRef.contains("assign") || lowerRef.contains("<=") ||
      lowerRef.contains("="))
    return "assignment";
  return "expression";
}

struct ModuleStart {
  size_t pos = 0;
  std::string name;
};

static void collectModuleStarts(StringRef text,
                                ArrayRef<uint8_t> codeMask,
                                SmallVectorImpl<ModuleStart> &modules) {
  modules.clear();
  modules.push_back({0, "$root"});

  size_t pos = 0;
  while (true) {
    pos = text.find("module", pos);
    if (pos == StringRef::npos)
      break;
    if (!isCodeRange(codeMask, pos, 6)) {
      ++pos;
      continue;
    }

    char prev = pos == 0 ? '\0' : text[pos - 1];
    char next = pos + 6 < text.size() ? text[pos + 6] : '\0';
    bool prevBoundary = !isAlnum(prev) && prev != '_';
    bool nextBoundary = !isAlnum(next) && next != '_';
    if (!prevBoundary || !nextBoundary) {
      ++pos;
      continue;
    }

    size_t i = pos + 6;
    while (i < text.size() &&
           (!isCodeAt(codeMask, i) ||
            std::isspace(static_cast<unsigned char>(text[i]))))
      ++i;

    size_t start = i;
    if (start >= text.size() || !isCodeAt(codeMask, start) ||
        !(isAlpha(text[start]) || text[start] == '_')) {
      ++pos;
      continue;
    }
    ++i;
    while (i < text.size() && isCodeAt(codeMask, i) &&
           (isAlnum(text[i]) || text[i] == '_' || text[i] == '$'))
      ++i;

    modules.push_back({pos, text.slice(start, i).str()});
    pos = i;
  }

  llvm::sort(modules, [](const ModuleStart &a, const ModuleStart &b) {
    return a.pos < b.pos;
  });
}

static std::string getModuleForPos(ArrayRef<ModuleStart> modules, size_t pos) {
  if (modules.empty() || pos == StringRef::npos)
    return "$root";

  size_t idx = 0;
  for (size_t i = 0; i < modules.size(); ++i) {
    if (modules[i].pos > pos)
      break;
    idx = i;
  }
  return modules[idx].name;
}

static std::string keyJoin(StringRef a, StringRef b) {
  return (Twine(a) + "|" + b).str();
}

static std::string keyJoin3(StringRef a, StringRef b, uint64_t c) {
  return (Twine(a) + "|" + b + "|" + Twine(c)).str();
}

static StringRef queueKeyFor(const Candidate &c, QueueKind kind) {
  switch (kind) {
  case QueueKind::PrimaryWire:
    return c.wireKey;
  case QueueKind::PrimaryBit:
    return c.wireBitKey;
  case QueueKind::PrimaryCell:
    return c.cellKey;
  case QueueKind::PrimarySrc:
    return c.srcKey;
  case QueueKind::ModuleWire:
    return c.moduleWireKey;
  case QueueKind::ModuleBit:
    return c.moduleBitKey;
  case QueueKind::ModuleCell:
    return c.moduleCellKey;
  case QueueKind::ModuleSrc:
    return c.moduleSrcKey;
  }
  return c.srcKey;
}

static int pickRandomCandidate(ArrayRef<int> candidates, XorShift128 &rng) {
  if (candidates.empty())
    return -1;
  return candidates[rng.bounded(candidates.size())];
}

static int pickCoverCandidate(ArrayRef<Candidate> db, CoverageDB &coverdb,
                              XorShift128 &rng) {
  int bestSrcCoverage = std::numeric_limits<int>::max();
  int bestNovelty = -1;
  SmallVector<int, 32> candidates;
  for (int i = 0, e = db.size(); i < e; ++i) {
    if (db[i].used || db[i].srcKey.empty())
      continue;
    auto it = coverdb.srcCoverage.find(db[i].srcKey);
    if (it == coverdb.srcCoverage.end())
      continue;
    int thisSrcCoverage = static_cast<int>(it->second);
    int noveltyScore = coverdb.score(db[i]);
    if (thisSrcCoverage < bestSrcCoverage ||
        (thisSrcCoverage == bestSrcCoverage && noveltyScore > bestNovelty)) {
      bestSrcCoverage = thisSrcCoverage;
      bestNovelty = noveltyScore;
      candidates.clear();
    }
    if (bestSrcCoverage == thisSrcCoverage && bestNovelty == noveltyScore)
      candidates.push_back(i);
  }
  return pickRandomCandidate(candidates, rng);
}

static int pickQueueCandidate(ArrayRef<Candidate> db, QueueKind kind,
                              CoverageDB &coverdb, XorShift128 &rng,
                              const NativeMutationPlannerConfig &config,
                              StringSet<> &usedKeys) {
  SmallVector<int, 32> rawCandidates;
  for (int i = 0, e = db.size(); i < e; ++i) {
    if (db[i].used)
      continue;
    StringRef key = queueKeyFor(db[i], kind);
    if (key.empty() || usedKeys.contains(key))
      continue;
    rawCandidates.push_back(i);
  }

  if (rawCandidates.empty()) {
    usedKeys.clear();
    for (int i = 0, e = db.size(); i < e; ++i) {
      if (db[i].used)
        continue;
      StringRef key = queueKeyFor(db[i], kind);
      if (key.empty())
        continue;
      rawCandidates.push_back(i);
    }
  }

  if (rawCandidates.empty())
    return -1;

  int picked = -1;
  if (rng.bounded(100) < static_cast<uint32_t>(std::max(0, config.pickCoverPercent))) {
    int bestScore = -1;
    SmallVector<int, 32> bestCandidates;
    for (int idx : rawCandidates) {
      int thisScore = coverdb.score(db[idx]);
      if (thisScore > bestScore) {
        bestScore = thisScore;
        bestCandidates.clear();
      }
      if (thisScore == bestScore)
        bestCandidates.push_back(idx);
    }
    picked = pickRandomCandidate(bestCandidates, rng);
  }

  if (picked < 0)
    picked = pickRandomCandidate(rawCandidates, rng);

  if (picked >= 0)
    usedKeys.insert(queueKeyFor(db[picked], kind));
  return picked;
}

static void emitLegacyNativeMutationPlan(ArrayRef<std::string> orderedOps,
                                         StringRef designText, uint64_t count,
                                         uint64_t seed, raw_ostream &out) {
  SmallVector<uint8_t, 0> codeMask;
  buildCodeMask(designText, codeMask);
  uint64_t opCount = orderedOps.size();
  uint64_t seedOffset = seed % opCount;
  StringMap<uint64_t> siteCounts;
  for (const std::string &op : orderedOps)
    siteCounts[op] =
        std::max<uint64_t>(1, countNativeMutationSitesForOp(designText, codeMask, op));

  for (uint64_t mid = 1; mid <= count; ++mid) {
    uint64_t rank = seedOffset + mid - 1;
    uint64_t opIdx = rank % opCount;
    uint64_t cycle = rank / opCount;
    const std::string &op = orderedOps[opIdx];
    uint64_t siteCount = siteCounts.lookup(op);
    uint64_t siteIndex = ((seed + cycle) % siteCount) + 1;

    out << mid << " NATIVE_" << op;
    if (siteCount > 1 || cycle > 0)
      out << "@" << siteIndex;
    out << "\n";
  }
}

static void emitWeightedNativeMutationPlan(ArrayRef<std::string> orderedOps,
                                           StringRef designText,
                                           uint64_t count, uint64_t seed,
                                           const NativeMutationPlannerConfig &config,
                                           raw_ostream &out) {
  SmallVector<uint8_t, 0> codeMask;
  buildCodeMask(designText, codeMask);
  SmallVector<size_t, 64> lineStarts;
  collectLineStarts(designText, lineStarts);

  SmallVector<ModuleStart, 16> modules;
  collectModuleStarts(designText, codeMask, modules);

  SmallVector<Candidate, 64> db;
  db.reserve(orderedOps.size() * 4);

  for (const std::string &op : orderedOps) {
    SmallVector<SiteInfo, 16> sites;
    collectSitesForOp(designText, op, codeMask, sites);
    if (sites.empty())
      continue;
    uint64_t siteCount = sites.size();

    for (uint64_t i = 0; i < siteCount; ++i) {
      size_t pos = sites[i].pos;
      std::string module = getModuleForPos(modules, pos);
      uint64_t line = getLineForPos(lineStarts, pos);

      Candidate c;
      c.op = op;
      c.siteIndex = i + 1;
      c.siteCount = siteCount;
      c.module = module;
      c.srcKey = (Twine("line:") + Twine(line)).str();
      c.family = getOpFamily(op);
      c.contextKey = classifyContextForPos(designText, lineStarts, pos);

      // Primary queues model global candidate families, while module queues add
      // module scoping on top of the same base keys.
      c.wireKey = c.op;
      c.wireBitKey = keyJoin3(c.op, c.family, c.siteIndex);
      c.cellKey = c.family;
      c.moduleWireKey = keyJoin(c.module, c.wireKey);
      c.moduleBitKey = keyJoin(c.module, c.wireBitKey);
      c.moduleCellKey = keyJoin(c.module, c.cellKey);
      c.moduleSrcKey = keyJoin(c.module, c.srcKey);

      db.push_back(std::move(c));
    }
  }

  if (db.empty()) {
    emitLegacyNativeMutationPlan(orderedOps, designText, count, seed, out);
    return;
  }

  CoverageDB coverdb;
  for (const Candidate &c : db)
    coverdb.insert(c);

  StringSet<> usedPrimaryWireKeys;
  StringSet<> usedPrimaryBitKeys;
  StringSet<> usedPrimaryCellKeys;
  StringSet<> usedPrimarySrcKeys;
  StringSet<> usedModuleWireKeys;
  StringSet<> usedModuleBitKeys;
  StringSet<> usedModuleCellKeys;
  StringSet<> usedModuleSrcKeys;

  int totalWeight = config.weightCover + config.weightPQW + config.weightPQB +
                    config.weightPQC + config.weightPQS + config.weightPQMW +
                    config.weightPQMB + config.weightPQMC + config.weightPQMS;
  if (totalWeight <= 0)
    totalWeight = 1;

  XorShift128 rng(seed);
  StringMap<uint64_t> opEmits;

  auto anyUnused = [&]() {
    for (const Candidate &c : db)
      if (!c.used)
        return true;
    return false;
  };

  for (uint64_t mid = 1; mid <= count; ++mid) {
    if (!anyUnused()) {
      for (Candidate &c : db)
        c.used = false;
    }

    int selected = -1;

    for (int attempts = 0; attempts < 12 && selected < 0; ++attempts) {
      int k = static_cast<int>(rng.bounded(static_cast<uint32_t>(totalWeight)));

      k -= config.weightCover;
      if (k < 0) {
        selected = pickCoverCandidate(db, coverdb, rng);
        if (selected >= 0)
          break;
      }

      k -= config.weightPQW;
      if (k < 0) {
        selected = pickQueueCandidate(db, QueueKind::PrimaryWire, coverdb, rng,
                                      config, usedPrimaryWireKeys);
        if (selected >= 0)
          break;
      }

      k -= config.weightPQB;
      if (k < 0) {
        selected = pickQueueCandidate(db, QueueKind::PrimaryBit, coverdb, rng,
                                      config, usedPrimaryBitKeys);
        if (selected >= 0)
          break;
      }

      k -= config.weightPQC;
      if (k < 0) {
        selected = pickQueueCandidate(db, QueueKind::PrimaryCell, coverdb, rng,
                                      config, usedPrimaryCellKeys);
        if (selected >= 0)
          break;
      }

      k -= config.weightPQS;
      if (k < 0) {
        selected = pickQueueCandidate(db, QueueKind::PrimarySrc, coverdb, rng,
                                      config, usedPrimarySrcKeys);
        if (selected >= 0)
          break;
      }

      k -= config.weightPQMW;
      if (k < 0) {
        selected = pickQueueCandidate(db, QueueKind::ModuleWire, coverdb, rng,
                                      config, usedModuleWireKeys);
        if (selected >= 0)
          break;
      }

      k -= config.weightPQMB;
      if (k < 0) {
        selected = pickQueueCandidate(db, QueueKind::ModuleBit, coverdb, rng,
                                      config, usedModuleBitKeys);
        if (selected >= 0)
          break;
      }

      k -= config.weightPQMC;
      if (k < 0) {
        selected = pickQueueCandidate(db, QueueKind::ModuleCell, coverdb, rng,
                                      config, usedModuleCellKeys);
        if (selected >= 0)
          break;
      }

      selected = pickQueueCandidate(db, QueueKind::ModuleSrc, coverdb, rng,
                                    config, usedModuleSrcKeys);
    }

    if (selected < 0) {
      SmallVector<int, 32> fallback;
      for (int i = 0, e = db.size(); i < e; ++i)
        if (!db[i].used)
          fallback.push_back(i);
      selected = pickRandomCandidate(fallback, rng);
      if (selected < 0)
        break;
    }

    Candidate &picked = db[selected];
    picked.used = true;
    coverdb.update(picked);

    uint64_t opCycle = opEmits[picked.op]++;

    out << mid << " NATIVE_" << picked.op;
    if (picked.siteCount > 1 || opCycle > 0)
      out << "@" << picked.siteIndex;
    out << "\n";
  }
}

} // namespace

bool parseNativeMutationPlannerConfig(ArrayRef<std::string> cfgEntries,
                                      NativeMutationPlannerConfig &config,
                                      std::string &error) {
  config = NativeMutationPlannerConfig();
  bool sawPlannerPolicy = false;
  bool sawWeightedKnob = false;

  for (const std::string &entry : cfgEntries) {
    StringRef ref(entry);
    auto split = ref.split('=');
    StringRef key = split.first.trim();
    StringRef value = split.second.trim();
    if (key.empty() || value == split.first) {
      error = (Twine("circt-mut generate: invalid --cfg entry: ") + entry +
               " (expected KEY=VALUE)")
                  .str();
      return false;
    }

    if (key == "planner_policy") {
      if (value == "legacy") {
        sawPlannerPolicy = true;
        config.policy = NativeMutationPlannerConfig::Policy::Legacy;
        continue;
      }
      if (value == "weighted") {
        sawPlannerPolicy = true;
        config.policy = NativeMutationPlannerConfig::Policy::Weighted;
        continue;
      }
      error = (Twine("circt-mut generate: unknown --cfg planner_policy value: ") +
               value + " (expected legacy|weighted)")
                  .str();
      return false;
    }

    int parsed = 0;
    if (!parseIntValue(value, parsed)) {
      error = (Twine("circt-mut generate: invalid --cfg value for ") + key +
               ": " + value + " (expected integer)")
                  .str();
      return false;
    }

    auto requireNonNegative = [&](StringRef name, int valueToCheck) {
      if (valueToCheck >= 0)
        return true;
      error = (Twine("circt-mut generate: --cfg ") + name +
               " must be >= 0 (got " + Twine(valueToCheck) + ")")
                  .str();
      return false;
    };

    if (key == "pick_cover_prcnt") {
      if (parsed < 0 || parsed > 100) {
        error = (Twine("circt-mut generate: --cfg pick_cover_prcnt must be in "
                       "range 0..100 (got ") +
                 Twine(parsed) + ")")
                    .str();
        return false;
      }
      sawWeightedKnob = true;
      config.pickCoverPercent = parsed;
      continue;
    }
    if (key == "weight_cover") {
      if (!requireNonNegative(key, parsed))
        return false;
      sawWeightedKnob = true;
      config.weightCover = parsed;
      continue;
    }
    if (key == "weight_pq_w") {
      if (!requireNonNegative(key, parsed))
        return false;
      sawWeightedKnob = true;
      config.weightPQW = parsed;
      continue;
    }
    if (key == "weight_pq_b") {
      if (!requireNonNegative(key, parsed))
        return false;
      sawWeightedKnob = true;
      config.weightPQB = parsed;
      continue;
    }
    if (key == "weight_pq_c") {
      if (!requireNonNegative(key, parsed))
        return false;
      sawWeightedKnob = true;
      config.weightPQC = parsed;
      continue;
    }
    if (key == "weight_pq_s") {
      if (!requireNonNegative(key, parsed))
        return false;
      sawWeightedKnob = true;
      config.weightPQS = parsed;
      continue;
    }
    if (key == "weight_pq_mw") {
      if (!requireNonNegative(key, parsed))
        return false;
      sawWeightedKnob = true;
      config.weightPQMW = parsed;
      continue;
    }
    if (key == "weight_pq_mb") {
      if (!requireNonNegative(key, parsed))
        return false;
      sawWeightedKnob = true;
      config.weightPQMB = parsed;
      continue;
    }
    if (key == "weight_pq_mc") {
      if (!requireNonNegative(key, parsed))
        return false;
      sawWeightedKnob = true;
      config.weightPQMC = parsed;
      continue;
    }
    if (key == "weight_pq_ms") {
      if (!requireNonNegative(key, parsed))
        return false;
      sawWeightedKnob = true;
      config.weightPQMS = parsed;
      continue;
    }

    error = (Twine("circt-mut generate: unknown --cfg option in CIRCT-only "
                   "mode: ") +
             key)
                .str();
    return false;
  }

  if (!sawPlannerPolicy && sawWeightedKnob)
    config.policy = NativeMutationPlannerConfig::Policy::Weighted;

  int totalWeight = config.weightCover + config.weightPQW + config.weightPQB +
                    config.weightPQC + config.weightPQS + config.weightPQMW +
                    config.weightPQMB + config.weightPQMC + config.weightPQMS;
  if (config.policy == NativeMutationPlannerConfig::Policy::Weighted &&
      totalWeight <= 0) {
    error = "circt-mut generate: weighted planner requires positive total weight across --cfg weight_* options";
    return false;
  }

  return true;
}

bool computeOrderedNativeMutationOps(StringRef designText,
                                     SmallVectorImpl<std::string> &orderedOps,
                                     std::string &error) {
  orderedOps.clear();
  SmallVector<uint8_t, 0> codeMask;
  buildCodeMask(designText, codeMask);

  SmallVector<std::string, 16> baseOps;
  for (const char *op : kNativeMutationOpsAll)
    baseOps.push_back(std::string(op));

  SmallVector<std::string, 16> applicableOps;
  for (const std::string &op : baseOps)
    if (hasNativeMutationPattern(designText, codeMask, op))
      applicableOps.push_back(op);

  StringSet<> seenOps;
  for (const std::string &op : applicableOps)
    if (seenOps.insert(op).second)
      orderedOps.push_back(op);
  for (const std::string &op : baseOps)
    if (seenOps.insert(op).second)
      orderedOps.push_back(op);

  if (!orderedOps.empty())
    return true;

  error = "circt-mut generate: native mutation operator set must not be empty";
  return false;
}

bool hasNativeMutationPatternForOp(StringRef designText, StringRef op) {
  SmallVector<uint8_t, 0> codeMask;
  buildCodeMask(designText, codeMask);
  return hasNativeMutationPattern(designText, codeMask, op);
}

void emitNativeMutationPlan(ArrayRef<std::string> orderedOps,
                            StringRef designText, uint64_t count,
                            uint64_t seed,
                            const NativeMutationPlannerConfig &config,
                            raw_ostream &out) {
  if (config.policy == NativeMutationPlannerConfig::Policy::Weighted) {
    emitWeightedNativeMutationPlan(orderedOps, designText, count, seed, config,
                                   out);
    return;
  }
  emitLegacyNativeMutationPlan(orderedOps, designText, count, seed, out);
}

static bool replaceSpan(std::string &text, size_t begin, size_t end,
                        StringRef replacement) {
  if (begin == StringRef::npos || end == StringRef::npos || begin > end ||
      end > text.size())
    return false;
  text = text.substr(0, begin) + replacement.str() + text.substr(end);
  return true;
}

static bool replaceTokenAt(std::string &text, size_t pos, size_t tokenLen,
                           StringRef replacement) {
  return replaceSpan(text, pos, pos + tokenLen, replacement);
}

static void parseMutationLabel(StringRef label, std::string &op,
                               uint64_t &siteIndex) {
  StringRef opRef = label;
  if (opRef.starts_with("NATIVE_"))
    opRef = opRef.drop_front(strlen("NATIVE_"));

  siteIndex = 1;
  size_t atPos = opRef.rfind('@');
  if (atPos != StringRef::npos) {
    StringRef suffix = opRef.drop_front(atPos + 1);
    uint64_t parsed = 0;
    if (!suffix.empty() && !suffix.getAsInteger(10, parsed) && parsed > 0) {
      opRef = opRef.take_front(atPos);
      siteIndex = parsed;
    }
  }
  op = opRef.str();
}

static bool findIfConditionBoundsAtSite(StringRef text, ArrayRef<uint8_t> codeMask,
                                        size_t ifPos, size_t &condOpen,
                                        size_t &condClose) {
  if (ifPos == StringRef::npos || ifPos + 2 > text.size() ||
      !matchKeywordTokenAt(text, codeMask, ifPos, "if"))
    return false;

  size_t openPos = ifPos + 2;
  while (openPos < text.size() &&
         (!isCodeAt(codeMask, openPos) ||
          std::isspace(static_cast<unsigned char>(text[openPos]))))
    ++openPos;
  if (openPos >= text.size() || !isCodeAt(codeMask, openPos) ||
      text[openPos] != '(')
    return false;

  size_t closePos = findMatchingParen(text, codeMask, openPos);
  if (closePos == StringRef::npos)
    return false;

  condOpen = openPos;
  condClose = closePos;
  return true;
}

static size_t findTernaryEndDelimiter(StringRef text, ArrayRef<uint8_t> codeMask,
                                      size_t colonPos) {
  if (colonPos == StringRef::npos || colonPos >= text.size())
    return StringRef::npos;

  int parenDepth = 0;
  int bracketDepth = 0;
  int braceDepth = 0;
  int nestedTernary = 0;
  auto decDepth = [](int &depth) {
    if (depth > 0)
      --depth;
  };

  for (size_t i = colonPos + 1, e = text.size(); i < e; ++i) {
    if (!isCodeAt(codeMask, i))
      continue;
    char ch = text[i];
    if (ch == '(') {
      ++parenDepth;
      continue;
    }
    if (ch == ')') {
      if (parenDepth == 0 && bracketDepth == 0 && braceDepth == 0 &&
          nestedTernary == 0)
        return i;
      decDepth(parenDepth);
      continue;
    }
    if (ch == '[') {
      ++bracketDepth;
      continue;
    }
    if (ch == ']') {
      if (parenDepth == 0 && bracketDepth == 0 && braceDepth == 0 &&
          nestedTernary == 0)
        return i;
      decDepth(bracketDepth);
      continue;
    }
    if (ch == '{') {
      ++braceDepth;
      continue;
    }
    if (ch == '}') {
      if (parenDepth == 0 && bracketDepth == 0 && braceDepth == 0 &&
          nestedTernary == 0)
        return i;
      decDepth(braceDepth);
      continue;
    }

    if (parenDepth == 0 && bracketDepth == 0 && braceDepth == 0) {
      if (ch == '?') {
        ++nestedTernary;
        continue;
      }
      if (ch == ':') {
        if (nestedTernary == 0)
          return i;
        --nestedTernary;
        continue;
      }
      if (nestedTernary == 0 && (ch == ';' || ch == ','))
        return i;
    }
  }
  return text.size();
}

static bool applyMuxSwapArmsAt(StringRef text, ArrayRef<uint8_t> codeMask,
                               size_t questionPos, std::string &mutatedText) {
  size_t colonPos = findMatchingTernaryColon(text, codeMask, questionPos);
  if (colonPos == StringRef::npos)
    return false;
  size_t endDelim = findTernaryEndDelimiter(text, codeMask, colonPos);
  if (endDelim == StringRef::npos)
    return false;

  size_t trueStart = findNextCodeNonSpace(text, codeMask, questionPos + 1);
  size_t trueEnd = findPrevCodeNonSpace(text, codeMask, colonPos);
  size_t falseStart = findNextCodeNonSpace(text, codeMask, colonPos + 1);
  size_t falseEnd = findPrevCodeNonSpace(text, codeMask, endDelim);
  if (trueStart == StringRef::npos || trueEnd == StringRef::npos ||
      falseStart == StringRef::npos || falseEnd == StringRef::npos ||
      trueStart > trueEnd || falseStart > falseEnd)
    return false;

  StringRef lhs = text.slice(questionPos + 1, trueStart);
  StringRef trueExpr = text.slice(trueStart, trueEnd + 1);
  StringRef trueToColonWS = text.slice(trueEnd + 1, colonPos);
  StringRef colonToFalseWS = text.slice(colonPos + 1, falseStart);
  StringRef falseExpr = text.slice(falseStart, falseEnd + 1);
  StringRef falseSuffixWS = text.slice(falseEnd + 1, endDelim);

  std::string swapped;
  swapped.reserve(text.size() + 8);
  swapped += text.slice(0, questionPos + 1).str();
  swapped += lhs.str();
  swapped += falseExpr.str();
  swapped += trueToColonWS.str();
  swapped.push_back(':');
  swapped += colonToFalseWS.str();
  swapped += trueExpr.str();
  swapped += falseSuffixWS.str();
  swapped += text.drop_front(endDelim).str();

  mutatedText = std::move(swapped);
  return true;
}

static bool applyMuxForceArmAt(StringRef text, ArrayRef<uint8_t> codeMask,
                               size_t questionPos, bool forceTrueArm,
                               std::string &mutatedText) {
  size_t colonPos = findMatchingTernaryColon(text, codeMask, questionPos);
  if (colonPos == StringRef::npos)
    return false;
  size_t endDelim = findTernaryEndDelimiter(text, codeMask, colonPos);
  if (endDelim == StringRef::npos)
    return false;

  size_t trueStart = findNextCodeNonSpace(text, codeMask, questionPos + 1);
  size_t trueEnd = findPrevCodeNonSpace(text, codeMask, colonPos);
  size_t falseStart = findNextCodeNonSpace(text, codeMask, colonPos + 1);
  size_t falseEnd = findPrevCodeNonSpace(text, codeMask, endDelim);
  if (trueStart == StringRef::npos || trueEnd == StringRef::npos ||
      falseStart == StringRef::npos || falseEnd == StringRef::npos ||
      trueStart > trueEnd || falseStart > falseEnd)
    return false;

  StringRef selectedExpr = forceTrueArm ? text.slice(trueStart, trueEnd + 1)
                                        : text.slice(falseStart, falseEnd + 1);
  StringRef trueToColonWS = text.slice(trueEnd + 1, colonPos);
  StringRef colonToFalseWS = text.slice(colonPos + 1, falseStart);
  StringRef falseSuffixWS = text.slice(falseEnd + 1, endDelim);

  std::string forced;
  forced.reserve(text.size() + 8);
  forced += text.slice(0, trueStart).str();
  forced += selectedExpr.str();
  forced += trueToColonWS.str();
  forced.push_back(':');
  forced += colonToFalseWS.str();
  forced += selectedExpr.str();
  forced += falseSuffixWS.str();
  forced += text.drop_front(endDelim).str();

  mutatedText = std::move(forced);
  return true;
}

static bool applyIfElseSwapAt(StringRef text, ArrayRef<uint8_t> codeMask,
                              size_t ifPos, std::string &mutatedText) {
  size_t condOpen = StringRef::npos;
  size_t condClose = StringRef::npos;
  if (!findIfConditionBoundsAtSite(text, codeMask, ifPos, condOpen, condClose))
    return false;

  size_t thenStart = skipCodeWhitespace(text, codeMask, condClose + 1, text.size());
  if (thenStart == StringRef::npos || thenStart >= text.size())
    return false;
  size_t thenEnd = findIfElseBranchEnd(text, codeMask, thenStart);
  if (thenEnd == StringRef::npos)
    return false;

  size_t elsePos = skipCodeWhitespace(text, codeMask, thenEnd, text.size());
  if (elsePos == StringRef::npos || elsePos >= text.size() ||
      !matchKeywordTokenAt(text, codeMask, elsePos, "else"))
    return false;

  size_t elseStart = skipCodeWhitespace(text, codeMask, elsePos + 4, text.size());
  if (elseStart == StringRef::npos || elseStart >= text.size())
    return false;
  size_t elseEnd = findIfElseBranchEnd(text, codeMask, elseStart);
  if (elseEnd == StringRef::npos)
    return false;

  StringRef thenArm = text.slice(thenStart, thenEnd);
  StringRef betweenArms = text.slice(thenEnd, elsePos);
  StringRef elseHeader = text.slice(elsePos, elseStart);
  StringRef elseArm = text.slice(elseStart, elseEnd);

  std::string swapped;
  swapped.reserve(text.size() + 8);
  swapped += text.slice(0, thenStart).str();
  swapped += elseArm.str();
  swapped += betweenArms.str();
  swapped += elseHeader.str();
  swapped += thenArm.str();
  swapped += text.drop_front(elseEnd).str();

  mutatedText = std::move(swapped);
  return true;
}

static bool applyCaseItemSwapAt(StringRef text, ArrayRef<uint8_t> codeMask,
                                size_t sitePos, std::string &mutatedText) {
  SmallVector<CaseSwapSpan, 8> swaps;
  collectCaseItemSwapSpans(text, codeMask, swaps);
  for (const CaseSwapSpan &swap : swaps) {
    if (swap.sitePos != sitePos)
      continue;

    if (swap.firstStart == StringRef::npos || swap.firstEnd == StringRef::npos ||
        swap.secondStart == StringRef::npos || swap.secondEnd == StringRef::npos ||
        swap.firstStart >= swap.firstEnd || swap.secondStart >= swap.secondEnd ||
        swap.firstEnd > swap.secondStart || swap.secondEnd > text.size())
      return false;

    StringRef firstItem = text.slice(swap.firstStart, swap.firstEnd);
    StringRef betweenItems = text.slice(swap.firstEnd, swap.secondStart);
    StringRef secondItem = text.slice(swap.secondStart, swap.secondEnd);

    std::string swapped;
    swapped.reserve(text.size() + 8);
    swapped += text.slice(0, swap.firstStart).str();
    swapped += secondItem.str();
    swapped += betweenItems.str();
    swapped += firstItem.str();
    swapped += text.drop_front(swap.secondEnd).str();

    mutatedText = std::move(swapped);
    return true;
  }
  return false;
}

static bool applyConstFlipAt(StringRef text, bool zeroToOne, size_t pos,
                             std::string &mutatedText) {
  auto apply = [&](StringRef needle, StringRef replacement) {
    if (pos + needle.size() > text.size() || !text.substr(pos).starts_with(needle))
      return false;
    return replaceTokenAt(mutatedText, pos, needle.size(), replacement);
  };

  if (zeroToOne) {
    return apply("1'b0", "1'b1") || apply("1'd0", "1'd1") ||
           apply("1'h0", "1'h1") || apply("'0", "'1");
  }
  return apply("1'b1", "1'b0") || apply("1'd1", "1'd0") ||
         apply("1'h1", "1'h0") || apply("'1", "'0");
}

static bool applyNativeMutationAtSite(StringRef text, ArrayRef<uint8_t> codeMask,
                                      StringRef op, size_t pos,
                                      std::string &mutatedText) {
  if (op == "EQ_TO_NEQ")
    return replaceTokenAt(mutatedText, pos, 2, "!=");
  if (op == "NEQ_TO_EQ")
    return replaceTokenAt(mutatedText, pos, 2, "==");
  if (op == "CASEEQ_TO_EQ")
    return replaceTokenAt(mutatedText, pos, 3, "==");
  if (op == "CASENEQ_TO_NEQ")
    return replaceTokenAt(mutatedText, pos, 3, "!=");
  if (op == "EQ_TO_CASEEQ")
    return replaceTokenAt(mutatedText, pos, 2, "===");
  if (op == "NEQ_TO_CASENEQ")
    return replaceTokenAt(mutatedText, pos, 2, "!==");
  if (op == "SIGNED_TO_UNSIGNED")
    return replaceTokenAt(mutatedText, pos, strlen("$signed"), "$unsigned");
  if (op == "UNSIGNED_TO_SIGNED")
    return replaceTokenAt(mutatedText, pos, strlen("$unsigned"), "$signed");
  if (op == "LT_TO_LE")
    return replaceTokenAt(mutatedText, pos, 1, "<=");
  if (op == "GT_TO_GE")
    return replaceTokenAt(mutatedText, pos, 1, ">=");
  if (op == "LE_TO_LT")
    return replaceTokenAt(mutatedText, pos, 2, "<");
  if (op == "GE_TO_GT")
    return replaceTokenAt(mutatedText, pos, 2, ">");
  if (op == "LT_TO_GT")
    return replaceTokenAt(mutatedText, pos, 1, ">");
  if (op == "GT_TO_LT")
    return replaceTokenAt(mutatedText, pos, 1, "<");
  if (op == "LE_TO_GE")
    return replaceTokenAt(mutatedText, pos, 2, ">=");
  if (op == "GE_TO_LE")
    return replaceTokenAt(mutatedText, pos, 2, "<=");
  if (op == "AND_TO_OR")
    return replaceTokenAt(mutatedText, pos, 2, "||");
  if (op == "OR_TO_AND")
    return replaceTokenAt(mutatedText, pos, 2, "&&");
  if (op == "LAND_TO_BAND")
    return replaceTokenAt(mutatedText, pos, 2, "&");
  if (op == "LOR_TO_BOR")
    return replaceTokenAt(mutatedText, pos, 2, "|");
  if (op == "XOR_TO_OR")
    return replaceTokenAt(mutatedText, pos, 1, "|");
  if (op == "XOR_TO_XNOR")
    return replaceTokenAt(mutatedText, pos, 1, "^~");
  if (op == "XNOR_TO_XOR")
    return replaceTokenAt(mutatedText, pos, 2, "^");
  if (op == "REDAND_TO_REDOR")
    return replaceTokenAt(mutatedText, pos, 1, "|");
  if (op == "REDOR_TO_REDAND")
    return replaceTokenAt(mutatedText, pos, 1, "&");
  if (op == "REDXOR_TO_REDXNOR")
    return replaceTokenAt(mutatedText, pos, 1, "^~");
  if (op == "REDXNOR_TO_REDXOR")
    return replaceTokenAt(mutatedText, pos, 2, "^");
  if (op == "BAND_TO_BOR")
    return replaceTokenAt(mutatedText, pos, 1, "|");
  if (op == "BOR_TO_BAND")
    return replaceTokenAt(mutatedText, pos, 1, "&");
  if (op == "BAND_TO_LAND")
    return replaceTokenAt(mutatedText, pos, 1, "&&");
  if (op == "BOR_TO_LOR")
    return replaceTokenAt(mutatedText, pos, 1, "||");
  if (op == "BA_TO_NBA")
    return replaceTokenAt(mutatedText, pos, 1, "<=");
  if (op == "NBA_TO_BA")
    return replaceTokenAt(mutatedText, pos, 2, "=");
  if (op == "ASSIGN_RHS_TO_CONST0" || op == "ASSIGN_RHS_TO_CONST1" ||
      op == "ASSIGN_RHS_INVERT" || op == "ASSIGN_RHS_PLUS_ONE" ||
      op == "ASSIGN_RHS_MINUS_ONE") {
    size_t rhsStart = StringRef::npos;
    size_t rhsEnd = StringRef::npos;
    if (!findSimpleAssignmentRhsIdentifierSpan(text, codeMask, pos, rhsStart,
                                               rhsEnd))
      return false;
    std::string replacement;
    if (op == "ASSIGN_RHS_TO_CONST0")
      replacement = "1'b0";
    else if (op == "ASSIGN_RHS_TO_CONST1")
      replacement = "1'b1";
    else if (op == "ASSIGN_RHS_PLUS_ONE")
      replacement =
          (Twine("(") + text.slice(rhsStart, rhsEnd + 1) + " + 1'b1)").str();
    else if (op == "ASSIGN_RHS_MINUS_ONE")
      replacement =
          (Twine("(") + text.slice(rhsStart, rhsEnd + 1) + " - 1'b1)").str();
    else
      replacement = (Twine("~(") + text.slice(rhsStart, rhsEnd + 1) + ")").str();
    return replaceSpan(mutatedText, rhsStart, rhsEnd + 1, replacement);
  }
  if (op == "POSEDGE_TO_NEGEDGE")
    return replaceTokenAt(mutatedText, pos, strlen("posedge"), "negedge");
  if (op == "NEGEDGE_TO_POSEDGE")
    return replaceTokenAt(mutatedText, pos, strlen("negedge"), "posedge");
  if (op == "RESET_POSEDGE_TO_NEGEDGE")
    return replaceTokenAt(mutatedText, pos, strlen("posedge"), "negedge");
  if (op == "RESET_NEGEDGE_TO_POSEDGE")
    return replaceTokenAt(mutatedText, pos, strlen("negedge"), "posedge");
  if (op == "CASE_TO_CASEZ")
    return replaceTokenAt(mutatedText, pos, strlen("case"), "casez");
  if (op == "CASEZ_TO_CASE")
    return replaceTokenAt(mutatedText, pos, strlen("casez"), "case");
  if (op == "CASE_TO_CASEX")
    return replaceTokenAt(mutatedText, pos, strlen("case"), "casex");
  if (op == "CASEX_TO_CASE")
    return replaceTokenAt(mutatedText, pos, strlen("casex"), "case");
  if (op == "CASEZ_TO_CASEX")
    return replaceTokenAt(mutatedText, pos, strlen("casez"), "casex");
  if (op == "CASEX_TO_CASEZ")
    return replaceTokenAt(mutatedText, pos, strlen("casex"), "casez");
  if (op == "MUX_SWAP_ARMS")
    return applyMuxSwapArmsAt(text, codeMask, pos, mutatedText);
  if (op == "MUX_FORCE_TRUE")
    return applyMuxForceArmAt(text, codeMask, pos, /*forceTrueArm=*/true,
                              mutatedText);
  if (op == "MUX_FORCE_FALSE")
    return applyMuxForceArmAt(text, codeMask, pos, /*forceTrueArm=*/false,
                              mutatedText);
  if (op == "IF_COND_NEGATE" || op == "RESET_COND_NEGATE" ||
      op == "RESET_COND_TRUE" || op == "RESET_COND_FALSE" ||
      op == "IF_COND_TRUE" || op == "IF_COND_FALSE") {
    size_t condOpen = StringRef::npos;
    size_t condClose = StringRef::npos;
    if (!findIfConditionBoundsAtSite(text, codeMask, pos, condOpen, condClose))
      return false;

    if (op == "IF_COND_TRUE" || op == "RESET_COND_TRUE")
      return replaceSpan(mutatedText, condOpen + 1, condClose, "1'b1");
    if (op == "IF_COND_FALSE" || op == "RESET_COND_FALSE")
      return replaceSpan(mutatedText, condOpen + 1, condClose, "1'b0");

    StringRef condExpr = text.slice(condOpen + 1, condClose);
    std::string replacement = (Twine("!(") + condExpr + ")").str();
    return replaceSpan(mutatedText, condOpen + 1, condClose, replacement);
  }
  if (op == "IF_ELSE_SWAP_ARMS")
    return applyIfElseSwapAt(text, codeMask, pos, mutatedText);
  if (op == "CASE_ITEM_SWAP_ARMS")
    return applyCaseItemSwapAt(text, codeMask, pos, mutatedText);
  if (op == "UNARY_NOT_DROP") {
    size_t end = pos + 1;
    while (end < text.size() &&
           std::isspace(static_cast<unsigned char>(text[end])))
      ++end;
    return replaceSpan(mutatedText, pos, end, "");
  }
  if (op == "UNARY_BNOT_DROP" || op == "UNARY_MINUS_DROP") {
    size_t end = pos + 1;
    while (end < text.size() && isCodeAt(codeMask, end) &&
           std::isspace(static_cast<unsigned char>(text[end])))
      ++end;
    return replaceSpan(mutatedText, pos, end, "");
  }
  if (op == "CONST0_TO_1")
    return applyConstFlipAt(text, /*zeroToOne=*/true, pos, mutatedText);
  if (op == "CONST1_TO_0")
    return applyConstFlipAt(text, /*zeroToOne=*/false, pos, mutatedText);
  if (op == "ADD_TO_SUB")
    return replaceTokenAt(mutatedText, pos, 1, "-");
  if (op == "SUB_TO_ADD")
    return replaceTokenAt(mutatedText, pos, 1, "+");
  if (op == "MUL_TO_ADD")
    return replaceTokenAt(mutatedText, pos, 1, "+");
  if (op == "ADD_TO_MUL")
    return replaceTokenAt(mutatedText, pos, 1, "*");
  if (op == "DIV_TO_MUL")
    return replaceTokenAt(mutatedText, pos, 1, "*");
  if (op == "MUL_TO_DIV")
    return replaceTokenAt(mutatedText, pos, 1, "/");
  if (op == "MOD_TO_DIV")
    return replaceTokenAt(mutatedText, pos, 1, "/");
  if (op == "DIV_TO_MOD")
    return replaceTokenAt(mutatedText, pos, 1, "%");
  if (op == "INC_TO_DEC")
    return replaceTokenAt(mutatedText, pos, 2, "--");
  if (op == "DEC_TO_INC")
    return replaceTokenAt(mutatedText, pos, 2, "++");
  if (op == "PLUS_EQ_TO_MINUS_EQ")
    return replaceTokenAt(mutatedText, pos, 2, "-=");
  if (op == "MINUS_EQ_TO_PLUS_EQ")
    return replaceTokenAt(mutatedText, pos, 2, "+=");
  if (op == "MUL_EQ_TO_DIV_EQ")
    return replaceTokenAt(mutatedText, pos, 2, "/=");
  if (op == "DIV_EQ_TO_MUL_EQ")
    return replaceTokenAt(mutatedText, pos, 2, "*=");
  if (op == "MOD_EQ_TO_DIV_EQ")
    return replaceTokenAt(mutatedText, pos, 2, "/=");
  if (op == "DIV_EQ_TO_MOD_EQ")
    return replaceTokenAt(mutatedText, pos, 2, "%=");
  if (op == "SHL_EQ_TO_SHR_EQ")
    return replaceTokenAt(mutatedText, pos, 3, ">>=");
  if (op == "SHR_EQ_TO_SHL_EQ")
    return replaceTokenAt(mutatedText, pos, 3, "<<=");
  if (op == "SHR_EQ_TO_ASHR_EQ")
    return replaceTokenAt(mutatedText, pos, 3, ">>>=");
  if (op == "ASHR_EQ_TO_SHR_EQ")
    return replaceTokenAt(mutatedText, pos, 4, ">>=");
  if (op == "BAND_EQ_TO_BOR_EQ")
    return replaceTokenAt(mutatedText, pos, 2, "|=");
  if (op == "BOR_EQ_TO_BAND_EQ")
    return replaceTokenAt(mutatedText, pos, 2, "&=");
  if (op == "BAND_EQ_TO_BXOR_EQ")
    return replaceTokenAt(mutatedText, pos, 2, "^=");
  if (op == "BOR_EQ_TO_BXOR_EQ")
    return replaceTokenAt(mutatedText, pos, 2, "^=");
  if (op == "BXOR_EQ_TO_BOR_EQ")
    return replaceTokenAt(mutatedText, pos, 2, "|=");
  if (op == "BXOR_EQ_TO_BAND_EQ")
    return replaceTokenAt(mutatedText, pos, 2, "&=");
  if (op == "SHL_TO_SHR")
    return replaceTokenAt(mutatedText, pos, 2, ">>");
  if (op == "SHR_TO_SHL")
    return replaceTokenAt(mutatedText, pos, 2, "<<");
  if (op == "SHR_TO_ASHR")
    return replaceTokenAt(mutatedText, pos, 2, ">>>");
  if (op == "ASHR_TO_SHR")
    return replaceTokenAt(mutatedText, pos, 3, ">>");
  return false;
}

bool applyNativeMutationLabel(StringRef designText, StringRef label,
                              std::string &mutatedText, bool &changed,
                              std::string &error) {
  error.clear();
  changed = false;
  mutatedText = designText.str();

  std::string op;
  uint64_t siteIndex = 1;
  parseMutationLabel(label, op, siteIndex);
  if (!op.empty()) {
    SmallVector<uint8_t, 0> codeMask;
    buildCodeMask(designText, codeMask);
    SmallVector<SiteInfo, 16> sites;
    collectSitesForOp(designText, op, codeMask, sites);
    if (siteIndex > 0 && siteIndex <= sites.size())
      changed = applyNativeMutationAtSite(designText, codeMask, op,
                                          sites[siteIndex - 1].pos,
                                          mutatedText);
  }

  if (!changed) {
    if (mutatedText.empty() || mutatedText.back() != '\n')
      mutatedText.push_back('\n');
    mutatedText += (Twine("// native_mutation_noop_fallback ") + label + "\n").str();

    const char *markerPath = std::getenv("CIRCT_MUT_NATIVE_NOOP_FALLBACK_MARKER");
    if (markerPath && *markerPath) {
      std::error_code ec;
      raw_fd_ostream marker(markerPath, ec, sys::fs::OF_Append | sys::fs::OF_Text);
      if (ec) {
        error = (Twine("circt-mut apply: failed to append noop fallback marker: ") +
                 markerPath + ": " + ec.message())
                    .str();
        return false;
      }
      marker << label << "\n";
    }
  }

  return true;
}

} // namespace circt::mut
