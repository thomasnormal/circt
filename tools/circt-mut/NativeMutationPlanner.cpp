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
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <limits>

using namespace llvm;

namespace circt::mut {

static constexpr const char *kNativeMutationOpsAll[] = {
    "EQ_TO_NEQ",        "NEQ_TO_EQ",      "LT_TO_LE",        "GT_TO_GE",
    "LE_TO_LT",         "GE_TO_GT",       "AND_TO_OR",       "OR_TO_AND",
    "XOR_TO_OR",        "BAND_TO_BOR",    "BOR_TO_BAND",     "UNARY_NOT_DROP",
    "UNARY_BNOT_DROP",  "CONST0_TO_1",    "CONST1_TO_0",     "ADD_TO_SUB",
    "SUB_TO_ADD",       "MUL_TO_ADD",     "ADD_TO_MUL",      "SHL_TO_SHR",
    "SHR_TO_SHL",       "CASEEQ_TO_EQ",   "CASENEQ_TO_NEQ",  "SIGNED_TO_UNSIGNED",
    "UNSIGNED_TO_SIGNED"};

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
  auto isUnaryOperandStart = [](char c) {
    return isAlnum(c) || c == '_' || c == '(' || c == '[' || c == '{' ||
           c == '\'' || c == '~' || c == '!' || c == '$';
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
    if (isUnaryOperandStart(next))
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

static void collectBinaryMultiplySites(StringRef text,
                                       ArrayRef<uint8_t> codeMask,
                                       SmallVectorImpl<SiteInfo> &sites) {
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
    if (ch != '*')
      continue;
    if (bracketDepth > 0)
      continue;
    char prev = (i == 0 || !isCodeAt(codeMask, i - 1)) ? '\0' : text[i - 1];
    char next = (i + 1 < e && isCodeAt(codeMask, i + 1)) ? text[i + 1] : '\0';
    if (prev == '*' || next == '*')
      continue;
    if (next == '=')
      continue;
    if (prev == '(' && next == ')')
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
  if (op == "AND_TO_OR") {
    collectLiteralTokenSites(designText, "&&", codeMask, sites);
    return;
  }
  if (op == "OR_TO_AND") {
    collectLiteralTokenSites(designText, "||", codeMask, sites);
    return;
  }
  if (op == "XOR_TO_OR") {
    collectBinaryXorSites(designText, codeMask, sites);
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
  if (op == "UNARY_NOT_DROP") {
    collectUnaryNotDropSites(designText, codeMask, sites);
    return;
  }
  if (op == "UNARY_BNOT_DROP") {
    collectUnaryBitwiseNotDropSites(designText, codeMask, sites);
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
    collectBinaryMultiplySites(designText, codeMask, sites);
    return;
  }
  if (op == "ADD_TO_MUL") {
    collectBinaryArithmeticSites(designText, '+', codeMask, sites);
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
      op == "GT_TO_GE" || op == "LE_TO_LT" || op == "GE_TO_GT")
    return "compare";
  if (op == "CASEEQ_TO_EQ" || op == "CASENEQ_TO_NEQ")
    return "xcompare";
  if (op == "AND_TO_OR" || op == "OR_TO_AND" || op == "XOR_TO_OR" ||
      op == "BAND_TO_BOR" || op == "BOR_TO_BAND" || op == "UNARY_NOT_DROP" ||
      op == "UNARY_BNOT_DROP")
    return "logic";
  if (op == "CONST0_TO_1" || op == "CONST1_TO_0")
    return "constant";
  if (op == "ADD_TO_SUB" || op == "SUB_TO_ADD" || op == "MUL_TO_ADD" ||
      op == "ADD_TO_MUL")
    return "arithmetic";
  if (op == "SHL_TO_SHR" || op == "SHR_TO_SHL")
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

} // namespace circt::mut
