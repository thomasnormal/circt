//===- VerifToSMT.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/VerifToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Support/LTLSequenceNFA.h"
#include "circt/Support/I1ValueSimplifier.h"
#include "circt/Support/CommutativeValueEquivalence.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Dialect/SMT/IR/SMTTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/APInt.h"
#include <cctype>
#include <memory>
#include <functional>
#include <limits>
#include <optional>

namespace circt {
#define GEN_PASS_DEF_CONVERTVERIFTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

constexpr const char kWeakEventuallyAttr[] = "ltl.weak";

static Value gatePropertyWithEnable(Value property, Value enable, bool isCover,
                                    OpBuilder &builder, Location loc) {
  if (!enable)
    return property;
  if (isCover)
    return ltl::AndOp::create(builder, loc,
                              SmallVector<Value, 2>{enable, property})
        .getResult();
  auto notEnable = ltl::NotOp::create(builder, loc, enable);
  return ltl::OrOp::create(builder, loc,
                           SmallVector<Value, 2>{notEnable, property})
      .getResult();
}

static Value gateSMTWithEnable(Value property, Value enable, bool isCover,
                               OpBuilder &builder, Location loc) {
  if (!enable)
    return property;
  if (isCover)
    return smt::AndOp::create(builder, loc, enable, property);
  auto notEnable = smt::NotOp::create(builder, loc, enable);
  return smt::OrOp::create(builder, loc, notEnable, property);
}

static bool isFourStateStruct(Type originalTy, int64_t &valueWidth,
                              int64_t &unknownWidth) {
  auto structTy = dyn_cast<hw::StructType>(originalTy);
  if (!structTy)
    return false;
  auto elements = structTy.getElements();
  if (elements.size() != 2)
    return false;
  if (!elements[0].name || !elements[1].name)
    return false;
  if (elements[0].name.getValue() != "value" ||
      elements[1].name.getValue() != "unknown")
    return false;
  valueWidth = hw::getBitWidth(elements[0].type);
  unknownWidth = hw::getBitWidth(elements[1].type);
  if (valueWidth <= 0 || unknownWidth <= 0 || valueWidth != unknownWidth)
    return false;
  return true;
}

static bool isFourStateStruct(Type originalTy) {
  int64_t valueWidth = 0;
  int64_t unknownWidth = 0;
  return isFourStateStruct(originalTy, valueWidth, unknownWidth);
}

static void maybeAssertKnownInput(Type originalTy, Value smtVal, Location loc,
                                  OpBuilder &builder) {
  int64_t valueWidth = 0;
  int64_t unknownWidth = 0;
  if (!isFourStateStruct(originalTy, valueWidth, unknownWidth))
    return;
  auto bvTy = dyn_cast<smt::BitVectorType>(smtVal.getType());
  if (!bvTy || bvTy.getWidth() !=
                   static_cast<unsigned>(valueWidth + unknownWidth))
    return;
  auto unkTy = smt::BitVectorType::get(builder.getContext(), unknownWidth);
  auto unknownBits = smt::ExtractOp::create(builder, loc, unkTy, 0, smtVal);
  auto zero = smt::BVConstantOp::create(builder, loc, 0, unknownWidth);
  auto isKnown = smt::EqOp::create(builder, loc, unknownBits, zero);
  smt::AssertOp::create(builder, loc, isKnown);
}

static Value buildXOptimisticDiff(Value lhs, Value rhs, Type originalTy,
                                  Location loc, OpBuilder &builder) {
  int64_t valueWidth = 0;
  int64_t unknownWidth = 0;
  if (!isFourStateStruct(originalTy, valueWidth, unknownWidth))
    return Value();
  auto bvTy = dyn_cast<smt::BitVectorType>(lhs.getType());
  if (!bvTy || bvTy.getWidth() !=
                   static_cast<unsigned>(valueWidth + unknownWidth))
    return Value();
  if (lhs.getType() != rhs.getType())
    return Value();
  auto valueTy = smt::BitVectorType::get(builder.getContext(), valueWidth);
  auto unknownTy = smt::BitVectorType::get(builder.getContext(), unknownWidth);
  Value lhsUnknown = smt::ExtractOp::create(builder, loc, unknownTy, 0, lhs);
  Value rhsUnknown = smt::ExtractOp::create(builder, loc, unknownTy, 0, rhs);
  Value lhsValue =
      smt::ExtractOp::create(builder, loc, valueTy, unknownWidth, lhs);
  Value rhsValue =
      smt::ExtractOp::create(builder, loc, valueTy, unknownWidth, rhs);
  Value diff = smt::BVXOrOp::create(builder, loc, lhsValue, rhsValue);
  Value unknownAny = smt::BVOrOp::create(builder, loc, lhsUnknown, rhsUnknown);
  Value knownMask = smt::BVNotOp::create(builder, loc, unknownAny);
  Value maskedDiff = smt::BVAndOp::create(builder, loc, diff, knownMask);
  Value zero =
      smt::BVConstantOp::create(builder, loc, 0, valueWidth);
  return smt::DistinctOp::create(builder, loc, maskedDiff, zero);
}

namespace {

struct ParsedNamedBoolExpr {
  enum class Kind {
    Name,
    Const,
    Not,
    BitwiseNot,
    ReduceAnd,
    ReduceOr,
    ReduceXor,
    ReduceNand,
    ReduceNor,
    ReduceXnor,
    And,
    Or,
    Xor,
    Eq,
    Ne
  };
  Kind kind = Kind::Name;
  std::string name;
  bool constValue = false;
  std::unique_ptr<ParsedNamedBoolExpr> lhs;
  std::unique_ptr<ParsedNamedBoolExpr> rhs;
};

struct ResolvedNamedBoolExpr {
  struct ArgSlice {
    unsigned lsb = 0;
    unsigned msb = 0;
    std::optional<unsigned> dynamicIndexArg;
    int32_t dynamicIndexSign = 1;
    int32_t dynamicIndexOffset = 0;
    unsigned dynamicWidth = 0;
  };
  enum class Kind {
    Arg,
    Const,
    Group,
    Not,
    BitwiseNot,
    ReduceAnd,
    ReduceOr,
    ReduceXor,
    ReduceNand,
    ReduceNor,
    ReduceXnor,
    And,
    Or,
    Xor,
    Implies,
    Iff,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne
  };
  Kind kind = Kind::Arg;
  unsigned argIndex = 0;
  std::optional<ArgSlice> argSlice;
  bool constValue = false;
  bool compareSigned = false;
  std::unique_ptr<ResolvedNamedBoolExpr> lhs;
  std::unique_ptr<ResolvedNamedBoolExpr> rhs;
};

enum class NamedBoolTokenKind {
  Eof,
  Identifier,
  LParen,
  RParen,
  Not,
  BitwiseNot,
  Nand,
  Nor,
  Xnor,
  And,
  Or,
  Xor,
  Eq,
  Ne
};

struct NamedBoolToken {
  NamedBoolTokenKind kind = NamedBoolTokenKind::Eof;
  StringRef text;
};

class NamedBoolExprParser {
public:
  explicit NamedBoolExprParser(StringRef input) : input(input) {
    consumeToken();
  }

  std::unique_ptr<ParsedNamedBoolExpr> parse() {
    auto expr = parseOr();
    if (!expr || token.kind != NamedBoolTokenKind::Eof)
      return {};
    return expr;
  }

private:
  StringRef input;
  size_t pos = 0;
  NamedBoolToken token;

  static bool isIdentifierStart(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '$';
  }

  static bool isIdentifierChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$' ||
           c == '.' || c == '[' || c == ']' || c == ':' || c == '+' ||
           c == '-' || c == '(' || c == ')';
  }

  static std::optional<bool> parseBoolLiteral(StringRef text) {
    if (text.equals_insensitive("true"))
      return true;
    if (text.equals_insensitive("false"))
      return false;
    if (text == "1" || text == "1'b1" || text == "1'h1")
      return true;
    if (text == "0" || text == "1'b0" || text == "1'h0")
      return false;
    return std::nullopt;
  }

  std::unique_ptr<ParsedNamedBoolExpr>
  makeUnary(ParsedNamedBoolExpr::Kind kind,
            std::unique_ptr<ParsedNamedBoolExpr> operand) {
    auto node = std::make_unique<ParsedNamedBoolExpr>();
    node->kind = kind;
    node->lhs = std::move(operand);
    return node;
  }

  std::unique_ptr<ParsedNamedBoolExpr>
  makeBinary(ParsedNamedBoolExpr::Kind kind,
             std::unique_ptr<ParsedNamedBoolExpr> lhs,
             std::unique_ptr<ParsedNamedBoolExpr> rhs) {
    auto node = std::make_unique<ParsedNamedBoolExpr>();
    node->kind = kind;
    node->lhs = std::move(lhs);
    node->rhs = std::move(rhs);
    return node;
  }

  std::unique_ptr<ParsedNamedBoolExpr> parsePrimary() {
    if (token.kind == NamedBoolTokenKind::LParen) {
      consumeToken();
      auto expr = parseOr();
      if (!expr || token.kind != NamedBoolTokenKind::RParen)
        return {};
      consumeToken();
      return expr;
    }
    if (token.kind != NamedBoolTokenKind::Identifier)
      return {};
    StringRef text = token.text;
    consumeToken();
    auto literal = parseBoolLiteral(text);
    if (literal) {
      auto node = std::make_unique<ParsedNamedBoolExpr>();
      node->kind = ParsedNamedBoolExpr::Kind::Const;
      node->constValue = *literal;
      return node;
    }
    auto node = std::make_unique<ParsedNamedBoolExpr>();
    node->kind = ParsedNamedBoolExpr::Kind::Name;
    node->name = text.str();
    return node;
  }

  std::unique_ptr<ParsedNamedBoolExpr> parseUnary() {
    if (token.kind == NamedBoolTokenKind::Not) {
      consumeToken();
      auto operand = parseUnary();
      if (!operand)
        return {};
      return makeUnary(ParsedNamedBoolExpr::Kind::Not, std::move(operand));
    }
    if (token.kind == NamedBoolTokenKind::BitwiseNot) {
      consumeToken();
      auto operand = parseUnary();
      if (!operand)
        return {};
      return makeUnary(ParsedNamedBoolExpr::Kind::BitwiseNot,
                       std::move(operand));
    }
    if (token.kind == NamedBoolTokenKind::Nand) {
      consumeToken();
      auto operand = parseUnary();
      if (!operand)
        return {};
      return makeUnary(ParsedNamedBoolExpr::Kind::ReduceNand,
                       std::move(operand));
    }
    if (token.kind == NamedBoolTokenKind::Nor) {
      consumeToken();
      auto operand = parseUnary();
      if (!operand)
        return {};
      return makeUnary(ParsedNamedBoolExpr::Kind::ReduceNor,
                       std::move(operand));
    }
    if (token.kind == NamedBoolTokenKind::Xnor) {
      consumeToken();
      auto operand = parseUnary();
      if (!operand)
        return {};
      return makeUnary(ParsedNamedBoolExpr::Kind::ReduceXnor,
                       std::move(operand));
    }
    if (token.kind == NamedBoolTokenKind::And) {
      consumeToken();
      auto operand = parseUnary();
      if (!operand)
        return {};
      return makeUnary(ParsedNamedBoolExpr::Kind::ReduceAnd,
                       std::move(operand));
    }
    if (token.kind == NamedBoolTokenKind::Or) {
      consumeToken();
      auto operand = parseUnary();
      if (!operand)
        return {};
      return makeUnary(ParsedNamedBoolExpr::Kind::ReduceOr, std::move(operand));
    }
    if (token.kind == NamedBoolTokenKind::Xor) {
      consumeToken();
      auto operand = parseUnary();
      if (!operand)
        return {};
      return makeUnary(ParsedNamedBoolExpr::Kind::ReduceXor,
                       std::move(operand));
    }
    return parsePrimary();
  }

  std::unique_ptr<ParsedNamedBoolExpr> parseEq() {
    auto lhs = parseUnary();
    if (!lhs)
      return {};
    while (token.kind == NamedBoolTokenKind::Eq ||
           token.kind == NamedBoolTokenKind::Ne) {
      auto op = token.kind;
      consumeToken();
      auto rhs = parseUnary();
      if (!rhs)
        return {};
      lhs = makeBinary(op == NamedBoolTokenKind::Eq
                           ? ParsedNamedBoolExpr::Kind::Eq
                           : ParsedNamedBoolExpr::Kind::Ne,
                       std::move(lhs), std::move(rhs));
    }
    return lhs;
  }

  std::unique_ptr<ParsedNamedBoolExpr> parseAnd() {
    auto lhs = parseEq();
    if (!lhs)
      return {};
    while (token.kind == NamedBoolTokenKind::And) {
      consumeToken();
      auto rhs = parseEq();
      if (!rhs)
        return {};
      lhs = makeBinary(ParsedNamedBoolExpr::Kind::And, std::move(lhs),
                       std::move(rhs));
    }
    return lhs;
  }

  std::unique_ptr<ParsedNamedBoolExpr> parseXor() {
    auto lhs = parseAnd();
    if (!lhs)
      return {};
    while (token.kind == NamedBoolTokenKind::Xor) {
      consumeToken();
      auto rhs = parseAnd();
      if (!rhs)
        return {};
      lhs = makeBinary(ParsedNamedBoolExpr::Kind::Xor, std::move(lhs),
                       std::move(rhs));
    }
    return lhs;
  }

  std::unique_ptr<ParsedNamedBoolExpr> parseOr() {
    auto lhs = parseXor();
    if (!lhs)
      return {};
    while (token.kind == NamedBoolTokenKind::Or) {
      consumeToken();
      auto rhs = parseXor();
      if (!rhs)
        return {};
      lhs = makeBinary(ParsedNamedBoolExpr::Kind::Or, std::move(lhs),
                       std::move(rhs));
    }
    return lhs;
  }

  void skipWhitespace() {
    while (pos < input.size() &&
           std::isspace(static_cast<unsigned char>(input[pos])))
      ++pos;
  }

  void consumeToken() {
    skipWhitespace();
    if (pos >= input.size()) {
      token = {NamedBoolTokenKind::Eof, StringRef{}};
      return;
    }
    size_t start = pos;
    auto peek = [&](char c) -> bool {
      return pos < input.size() && input[pos] == c;
    };
    if (peek('(')) {
      ++pos;
      token = {NamedBoolTokenKind::LParen, input.slice(start, pos)};
      return;
    }
    if (peek(')')) {
      ++pos;
      token = {NamedBoolTokenKind::RParen, input.slice(start, pos)};
      return;
    }
    if (peek('!')) {
      ++pos;
      if (peek('=')) {
        ++pos;
        token = {NamedBoolTokenKind::Ne, input.slice(start, pos)};
      } else {
        token = {NamedBoolTokenKind::Not, input.slice(start, pos)};
      }
      return;
    }
      if (peek('~')) {
        ++pos;
        if (peek('&')) {
          ++pos;
          token = {NamedBoolTokenKind::Nand, input.slice(start, pos)};
        return;
      }
      if (peek('|')) {
        ++pos;
        token = {NamedBoolTokenKind::Nor, input.slice(start, pos)};
        return;
      }
        if (peek('^')) {
          ++pos;
          token = {NamedBoolTokenKind::Xnor, input.slice(start, pos)};
          return;
        }
      token = {NamedBoolTokenKind::BitwiseNot, input.slice(start, pos)};
      return;
    }
    if (peek('&')) {
      ++pos;
      if (peek('&'))
        ++pos;
      token = {NamedBoolTokenKind::And, input.slice(start, pos)};
      return;
    }
    if (peek('|')) {
      ++pos;
      if (peek('|'))
        ++pos;
      token = {NamedBoolTokenKind::Or, input.slice(start, pos)};
      return;
    }
    if (peek('^')) {
      ++pos;
      if (peek('~')) {
        ++pos;
        token = {NamedBoolTokenKind::Xnor, input.slice(start, pos)};
        return;
      }
      token = {NamedBoolTokenKind::Xor, input.slice(start, pos)};
      return;
    }
    if (peek('=')) {
      ++pos;
      if (peek('=')) {
        ++pos;
        token = {NamedBoolTokenKind::Eq, input.slice(start, pos)};
        return;
      }
      token = {NamedBoolTokenKind::Eof, StringRef{}};
      return;
    }
    if (isIdentifierStart(input[pos]) || std::isdigit(input[pos])) {
      ++pos;
      while (pos < input.size() && isIdentifierChar(input[pos]))
        ++pos;
      if (pos < input.size() && input[pos] == '\'') {
        ++pos;
        if (pos < input.size() &&
            (input[pos] == 'b' || input[pos] == 'B' || input[pos] == 'h' ||
             input[pos] == 'H' || input[pos] == 'd' || input[pos] == 'D')) {
          ++pos;
          while (pos < input.size() &&
                 (std::isalnum(static_cast<unsigned char>(input[pos])) ||
                  input[pos] == '_'))
            ++pos;
        }
      }
      token = {NamedBoolTokenKind::Identifier, input.slice(start, pos)};
      return;
    }
    token = {NamedBoolTokenKind::Eof, StringRef{}};
  }
};

static std::unique_ptr<ParsedNamedBoolExpr>
parseNamedBoolExpr(StringRef text) {
  std::string compact;
  compact.reserve(text.size());
  for (char c : text)
    if (!std::isspace(static_cast<unsigned char>(c)))
      compact.push_back(c);
  NamedBoolExprParser parser(compact);
  return parser.parse();
}

static std::unique_ptr<ResolvedNamedBoolExpr>
resolveNamedBoolExpr(const ParsedNamedBoolExpr &expr,
                     const DenseMap<StringRef, unsigned> &inputNameToArgIndex,
                     size_t numNonStateArgs) {
  struct ParsedAffineIndex {
    StringRef name;
    int64_t scale = 1;
    int64_t offset = 0;
  };
  auto resolveArgName = [&](StringRef name)
      -> std::optional<std::pair<unsigned, std::optional<ResolvedNamedBoolExpr::ArgSlice>>> {
    auto stripOuterParens = [](StringRef text) -> StringRef {
      auto balancedOuterParens = [](StringRef s) -> bool {
        if (s.size() < 2 || s.front() != '(' || s.back() != ')')
          return false;
        int depth = 0;
        for (size_t i = 0; i < s.size(); ++i) {
          if (s[i] == '(')
            ++depth;
          else if (s[i] == ')')
            --depth;
          if (depth == 0 && i + 1 != s.size())
            return false;
          if (depth < 0)
            return false;
        }
        return depth == 0;
      };
      while (balancedOuterParens(text))
        text = text.slice(1, text.size() - 1);
      return text;
    };
    auto parseSignedInt = [&](StringRef text) -> std::optional<int64_t> {
      text = stripOuterParens(text);
      int64_t value = 0;
      if (text.empty() || text.getAsInteger(10, value))
        return std::nullopt;
      return value;
    };
    auto isNameChar = [](char c) {
      return std::isalnum(static_cast<unsigned char>(c)) || c == '_' ||
             c == '$' || c == '.';
    };
    auto parseSignedName = [&](StringRef text)
        -> std::optional<std::pair<StringRef, int64_t>> {
      text = stripOuterParens(text);
      if (text.empty())
        return std::nullopt;
      int64_t scale = 1;
      if (text.front() == '+' || text.front() == '-') {
        scale = text.front() == '-' ? -1 : 1;
        text = text.drop_front();
      }
      if (text.empty())
        return std::nullopt;
      if (!std::isalpha(static_cast<unsigned char>(text.front())) &&
          text.front() != '_' && text.front() != '$')
        return std::nullopt;
      if (llvm::any_of(text, [&](char c) { return !isNameChar(c); }))
        return std::nullopt;
      return std::pair<StringRef, int64_t>{text, scale};
    };
    auto parseAffineIndex = [&](StringRef text) -> std::optional<ParsedAffineIndex> {
      text = stripOuterParens(text);
      if (auto sym = parseSignedName(text))
        return ParsedAffineIndex{sym->first, sym->second, 0};

      size_t opPos = StringRef::npos;
      int depth = 0;
      for (size_t i = 1; i < text.size(); ++i) {
        if (text[i] == '(')
          ++depth;
        else if (text[i] == ')')
          --depth;
        if (depth == 0 && (text[i] == '+' || text[i] == '-')) {
          opPos = i;
          break;
        }
      }
      if (opPos == StringRef::npos)
        return std::nullopt;

      char op = text[opPos];
      StringRef lhs = text.take_front(opPos);
      StringRef rhs = text.drop_front(opPos + 1);
      if (lhs.empty() || rhs.empty())
        return std::nullopt;

      if (auto lhsSym = parseSignedName(lhs)) {
        if (auto rhsConst = parseSignedInt(rhs)) {
          int64_t offset = op == '+' ? *rhsConst : -*rhsConst;
          return ParsedAffineIndex{lhsSym->first, lhsSym->second, offset};
        }
      }

      if (auto lhsConst = parseSignedInt(lhs)) {
        if (auto rhsSym = parseSignedName(rhs)) {
          int64_t scale = rhsSym->second;
          int64_t offset = *lhsConst;
          if (op == '-')
            scale = -scale;
          return ParsedAffineIndex{rhsSym->first, scale, offset};
        }
      }
      return std::nullopt;
    };

    if (auto it = inputNameToArgIndex.find(name);
        it != inputNameToArgIndex.end() && it->second < numNonStateArgs)
      return std::pair<unsigned,
                       std::optional<ResolvedNamedBoolExpr::ArgSlice>>{
          it->second, std::nullopt};

    size_t lbracket = name.find_last_of('[');
    size_t rbracket = name.find_last_of(']');
    if (lbracket == StringRef::npos || rbracket == StringRef::npos ||
        lbracket + 1 >= rbracket || rbracket + 1 != name.size())
      return std::nullopt;
    StringRef base = name.take_front(lbracket);
    StringRef indexText = name.slice(lbracket + 1, rbracket);
    std::optional<ResolvedNamedBoolExpr::ArgSlice> slice;
    if (size_t upPos = indexText.find("+:"); upPos != StringRef::npos ||
        indexText.find("-:") != StringRef::npos) {
      size_t downPos = indexText.find("-:");
      bool indexedDown =
          downPos != StringRef::npos &&
          (upPos == StringRef::npos || downPos < upPos);
      size_t opPos = indexedDown ? downPos : upPos;
      StringRef startText = indexText.take_front(opPos);
      StringRef widthText = indexText.drop_front(opPos + 2);
      if (startText.empty() || widthText.empty())
        return std::nullopt;
      int64_t width = 0;
      if (widthText.getAsInteger(10, width) || width <= 0)
        return std::nullopt;
      int64_t downAdjust = indexedDown ? (width - 1) : 0;
      if (auto startConst = parseSignedInt(startText)) {
        int64_t lsb = *startConst - downAdjust;
        int64_t msb = lsb + width - 1;
        if (lsb < 0 || msb < lsb ||
            msb > std::numeric_limits<unsigned>::max())
          return std::nullopt;
        slice = ResolvedNamedBoolExpr::ArgSlice{static_cast<unsigned>(lsb),
                                                 static_cast<unsigned>(msb)};
      } else if (auto affine = parseAffineIndex(startText)) {
        auto dynIt = inputNameToArgIndex.find(affine->name);
        if (dynIt == inputNameToArgIndex.end() ||
            dynIt->second >= numNonStateArgs)
          return std::nullopt;
        if ((affine->scale != 1 && affine->scale != -1) ||
            affine->offset < std::numeric_limits<int32_t>::min() ||
            affine->offset > std::numeric_limits<int32_t>::max())
          return std::nullopt;
        ResolvedNamedBoolExpr::ArgSlice dynSlice;
        dynSlice.dynamicIndexArg = dynIt->second;
        dynSlice.dynamicIndexSign = static_cast<int32_t>(affine->scale);
        dynSlice.dynamicIndexOffset =
            static_cast<int32_t>(affine->offset - downAdjust);
        dynSlice.dynamicWidth = static_cast<unsigned>(width);
        slice = dynSlice;
      } else {
        return std::nullopt;
      }
    } else if (size_t colon = indexText.find(':'); colon != StringRef::npos) {
      StringRef msbText = indexText.take_front(colon);
      StringRef lsbText = indexText.drop_front(colon + 1);
      if (msbText.empty() || lsbText.empty())
        return std::nullopt;
      unsigned msb = 0;
      unsigned lsb = 0;
      if (msbText.getAsInteger(10, msb) || lsbText.getAsInteger(10, lsb) ||
          msb < lsb)
        return std::nullopt;
      slice = ResolvedNamedBoolExpr::ArgSlice{lsb, msb};
    } else {
      if (auto bitConst = parseSignedInt(indexText)) {
        if (*bitConst < 0 || *bitConst > std::numeric_limits<unsigned>::max())
          return std::nullopt;
        unsigned bit = static_cast<unsigned>(*bitConst);
        slice = ResolvedNamedBoolExpr::ArgSlice{bit, bit};
      } else if (auto affine = parseAffineIndex(indexText)) {
        auto dynIt = inputNameToArgIndex.find(affine->name);
        if (dynIt == inputNameToArgIndex.end() ||
            dynIt->second >= numNonStateArgs)
          return std::nullopt;
        if ((affine->scale != 1 && affine->scale != -1) ||
            affine->offset < std::numeric_limits<int32_t>::min() ||
            affine->offset > std::numeric_limits<int32_t>::max())
          return std::nullopt;
        ResolvedNamedBoolExpr::ArgSlice dynSlice;
        dynSlice.dynamicIndexArg = dynIt->second;
        dynSlice.dynamicIndexSign = static_cast<int32_t>(affine->scale);
        dynSlice.dynamicIndexOffset = static_cast<int32_t>(affine->offset);
        dynSlice.dynamicWidth = 1;
        slice = dynSlice;
      } else {
        return std::nullopt;
      }
    }
    auto it = inputNameToArgIndex.find(base);
    if (it == inputNameToArgIndex.end() || it->second >= numNonStateArgs)
      return std::nullopt;
    return std::pair<unsigned, std::optional<ResolvedNamedBoolExpr::ArgSlice>>{
        it->second, slice};
  };

  auto resolved = std::make_unique<ResolvedNamedBoolExpr>();
  switch (expr.kind) {
  case ParsedNamedBoolExpr::Kind::Const:
    resolved->kind = ResolvedNamedBoolExpr::Kind::Const;
    resolved->constValue = expr.constValue;
    return resolved;
  case ParsedNamedBoolExpr::Kind::Name: {
    auto argOr = resolveArgName(expr.name);
    if (!argOr)
      return {};
    resolved->kind = ResolvedNamedBoolExpr::Kind::Arg;
    resolved->argIndex = argOr->first;
    resolved->argSlice = argOr->second;
    return resolved;
  }
  case ParsedNamedBoolExpr::Kind::Not: {
    auto operand = resolveNamedBoolExpr(*expr.lhs, inputNameToArgIndex,
                                        numNonStateArgs);
    if (!operand)
      return {};
    resolved->kind = ResolvedNamedBoolExpr::Kind::Not;
    resolved->lhs = std::move(operand);
    return resolved;
  }
  case ParsedNamedBoolExpr::Kind::BitwiseNot: {
    auto operand = resolveNamedBoolExpr(*expr.lhs, inputNameToArgIndex,
                                        numNonStateArgs);
    if (!operand)
      return {};
    resolved->kind = ResolvedNamedBoolExpr::Kind::BitwiseNot;
    resolved->lhs = std::move(operand);
    return resolved;
  }
  case ParsedNamedBoolExpr::Kind::ReduceAnd:
  case ParsedNamedBoolExpr::Kind::ReduceOr:
  case ParsedNamedBoolExpr::Kind::ReduceXor:
  case ParsedNamedBoolExpr::Kind::ReduceNand:
  case ParsedNamedBoolExpr::Kind::ReduceNor:
  case ParsedNamedBoolExpr::Kind::ReduceXnor: {
    auto operand = resolveNamedBoolExpr(*expr.lhs, inputNameToArgIndex,
                                        numNonStateArgs);
    if (!operand)
      return {};
    switch (expr.kind) {
    case ParsedNamedBoolExpr::Kind::ReduceAnd:
      resolved->kind = ResolvedNamedBoolExpr::Kind::ReduceAnd;
      break;
    case ParsedNamedBoolExpr::Kind::ReduceOr:
      resolved->kind = ResolvedNamedBoolExpr::Kind::ReduceOr;
      break;
    case ParsedNamedBoolExpr::Kind::ReduceXor:
      resolved->kind = ResolvedNamedBoolExpr::Kind::ReduceXor;
      break;
    case ParsedNamedBoolExpr::Kind::ReduceNand:
      resolved->kind = ResolvedNamedBoolExpr::Kind::ReduceNand;
      break;
    case ParsedNamedBoolExpr::Kind::ReduceNor:
      resolved->kind = ResolvedNamedBoolExpr::Kind::ReduceNor;
      break;
    case ParsedNamedBoolExpr::Kind::ReduceXnor:
      resolved->kind = ResolvedNamedBoolExpr::Kind::ReduceXnor;
      break;
    default:
      break;
    }
    resolved->lhs = std::move(operand);
    return resolved;
  }
  case ParsedNamedBoolExpr::Kind::And:
  case ParsedNamedBoolExpr::Kind::Or:
  case ParsedNamedBoolExpr::Kind::Xor:
  case ParsedNamedBoolExpr::Kind::Eq:
  case ParsedNamedBoolExpr::Kind::Ne: {
    auto lhs = resolveNamedBoolExpr(*expr.lhs, inputNameToArgIndex,
                                    numNonStateArgs);
    auto rhs = resolveNamedBoolExpr(*expr.rhs, inputNameToArgIndex,
                                    numNonStateArgs);
    if (!lhs || !rhs)
      return {};
    switch (expr.kind) {
    case ParsedNamedBoolExpr::Kind::And:
      resolved->kind = ResolvedNamedBoolExpr::Kind::And;
      break;
    case ParsedNamedBoolExpr::Kind::Or:
      resolved->kind = ResolvedNamedBoolExpr::Kind::Or;
      break;
    case ParsedNamedBoolExpr::Kind::Xor:
      resolved->kind = ResolvedNamedBoolExpr::Kind::Xor;
      break;
    case ParsedNamedBoolExpr::Kind::Eq:
      resolved->kind = ResolvedNamedBoolExpr::Kind::Eq;
      break;
    case ParsedNamedBoolExpr::Kind::Ne:
      resolved->kind = ResolvedNamedBoolExpr::Kind::Ne;
      break;
    default:
      break;
    }
    resolved->lhs = std::move(lhs);
    resolved->rhs = std::move(rhs);
    return resolved;
  }
  }
  return {};
}

static bool resolveStructuredExprFromDetail(
    DictionaryAttr detail, StringRef prefix,
    const DenseMap<StringRef, unsigned> &inputNameToArgIndex,
    size_t numNonStateArgs, std::optional<unsigned> &argIndex,
    std::unique_ptr<ResolvedNamedBoolExpr> &resolvedExpr) {
  argIndex.reset();
  resolvedExpr.reset();
  auto key = [&](StringRef suffix) {
    std::string key = prefix.str();
    key += "_";
    key += suffix.str();
    return key;
  };
  auto parseBoolLikeAttr = [&](StringRef suffix) -> std::optional<bool> {
    Attribute any = detail.get(key(suffix));
    if (!any)
      return false;
    if (auto boolAttr = dyn_cast<BoolAttr>(any))
      return boolAttr.getValue();
    if (isa<UnitAttr>(any))
      return true;
    return std::nullopt;
  };
  auto materializeArgNode = [&](unsigned sourceIndex,
                                std::optional<ResolvedNamedBoolExpr::ArgSlice>
                                    argSlice) {
    auto argNode = std::make_unique<ResolvedNamedBoolExpr>();
    argNode->kind = ResolvedNamedBoolExpr::Kind::Arg;
    argNode->argIndex = sourceIndex;
    argNode->argSlice = argSlice;
    return argNode;
  };
  auto moveArgOrExprToNode = [&](std::optional<unsigned> &index,
                                 std::unique_ptr<ResolvedNamedBoolExpr> &expr)
      -> std::unique_ptr<ResolvedNamedBoolExpr> {
    if (expr)
      return std::move(expr);
    if (!index)
      return {};
    return materializeArgNode(*index, std::nullopt);
  };
  auto logicalNotAttr = parseBoolLikeAttr("logical_not");
  if (!logicalNotAttr)
    return false;
  bool logicalNot = *logicalNotAttr;
  auto groupAttr = parseBoolLikeAttr("group");
  if (!groupAttr)
    return false;
  bool grouped = *groupAttr;
  auto groupDepthAttr =
      dyn_cast_or_null<IntegerAttr>(detail.get(key("group_depth")));
  unsigned groupDepth = grouped ? 1u : 0u;
  if (groupDepthAttr) {
    int64_t depth = groupDepthAttr.getInt();
    if (depth < 0)
      return false;
    groupDepth = std::max<unsigned>(groupDepth, static_cast<unsigned>(depth));
  }
  auto maybeWrapGroup = [&](std::unique_ptr<ResolvedNamedBoolExpr> node) {
    if (groupDepth == 0)
      return node;
    for (unsigned i = 0; i < groupDepth; ++i) {
      auto groupNode = std::make_unique<ResolvedNamedBoolExpr>();
      groupNode->kind = ResolvedNamedBoolExpr::Kind::Group;
      groupNode->lhs = std::move(node);
      node = std::move(groupNode);
    }
    return node;
  };
  auto unaryOpAttr = dyn_cast_or_null<StringAttr>(detail.get(key("unary_op")));
  if (unaryOpAttr) {
    std::optional<ResolvedNamedBoolExpr::Kind> unaryKind;
    StringRef unaryOp = unaryOpAttr.getValue();
    if (unaryOp == "not")
      unaryKind = ResolvedNamedBoolExpr::Kind::Not;
    else if (unaryOp == "bitwise_not")
      unaryKind = ResolvedNamedBoolExpr::Kind::BitwiseNot;
    else if (unaryOp == "reduce_and")
      unaryKind = ResolvedNamedBoolExpr::Kind::ReduceAnd;
    else if (unaryOp == "reduce_or")
      unaryKind = ResolvedNamedBoolExpr::Kind::ReduceOr;
    else if (unaryOp == "reduce_xor")
      unaryKind = ResolvedNamedBoolExpr::Kind::ReduceXor;
    else if (unaryOp == "reduce_nand")
      unaryKind = ResolvedNamedBoolExpr::Kind::ReduceNand;
    else if (unaryOp == "reduce_nor")
      unaryKind = ResolvedNamedBoolExpr::Kind::ReduceNor;
    else if (unaryOp == "reduce_xnor")
      unaryKind = ResolvedNamedBoolExpr::Kind::ReduceXnor;
    else
      return false;

    std::string argPrefix = (prefix + "_arg").str();
    std::optional<unsigned> argArgIndex;
    std::unique_ptr<ResolvedNamedBoolExpr> argExpr;
    if (!resolveStructuredExprFromDetail(detail, argPrefix, inputNameToArgIndex,
                                         numNonStateArgs, argArgIndex, argExpr))
      return false;
    auto argNode = moveArgOrExprToNode(argArgIndex, argExpr);
    if (!argNode)
      return false;

    auto node = std::make_unique<ResolvedNamedBoolExpr>();
    node->kind = *unaryKind;
    node->lhs = std::move(argNode);
    resolvedExpr = maybeWrapGroup(std::move(node));
    return true;
  }

  auto binOpAttr = dyn_cast_or_null<StringAttr>(detail.get(key("bin_op")));
  if (binOpAttr) {
    std::optional<ResolvedNamedBoolExpr::Kind> binKind;
    StringRef binOp = binOpAttr.getValue();
    if (binOp == "and")
      binKind = ResolvedNamedBoolExpr::Kind::And;
    else if (binOp == "or")
      binKind = ResolvedNamedBoolExpr::Kind::Or;
    else if (binOp == "xor")
      binKind = ResolvedNamedBoolExpr::Kind::Xor;
    else if (binOp == "implies")
      binKind = ResolvedNamedBoolExpr::Kind::Implies;
    else if (binOp == "iff")
      binKind = ResolvedNamedBoolExpr::Kind::Iff;
    else if (binOp == "lt")
      binKind = ResolvedNamedBoolExpr::Kind::Lt;
    else if (binOp == "le")
      binKind = ResolvedNamedBoolExpr::Kind::Le;
    else if (binOp == "gt")
      binKind = ResolvedNamedBoolExpr::Kind::Gt;
    else if (binOp == "ge")
      binKind = ResolvedNamedBoolExpr::Kind::Ge;
    else if (binOp == "eq")
      binKind = ResolvedNamedBoolExpr::Kind::Eq;
    else if (binOp == "ne")
      binKind = ResolvedNamedBoolExpr::Kind::Ne;
    else
      return false;
    bool cmpSigned = false;
    if (*binKind == ResolvedNamedBoolExpr::Kind::Lt ||
        *binKind == ResolvedNamedBoolExpr::Kind::Le ||
        *binKind == ResolvedNamedBoolExpr::Kind::Gt ||
        *binKind == ResolvedNamedBoolExpr::Kind::Ge) {
      auto cmpSignedAttr = parseBoolLikeAttr("cmp_signed");
      if (!cmpSignedAttr)
        return false;
      cmpSigned = *cmpSignedAttr;
    }

    std::string lhsPrefix = (prefix + "_lhs").str();
    std::string rhsPrefix = (prefix + "_rhs").str();
    std::optional<unsigned> lhsArgIndex;
    std::optional<unsigned> rhsArgIndex;
    std::unique_ptr<ResolvedNamedBoolExpr> lhsExpr;
    std::unique_ptr<ResolvedNamedBoolExpr> rhsExpr;
    if (!resolveStructuredExprFromDetail(detail, lhsPrefix, inputNameToArgIndex,
                                         numNonStateArgs, lhsArgIndex, lhsExpr))
      return false;
    if (!resolveStructuredExprFromDetail(detail, rhsPrefix, inputNameToArgIndex,
                                         numNonStateArgs, rhsArgIndex, rhsExpr))
      return false;
    auto lhsNode = moveArgOrExprToNode(lhsArgIndex, lhsExpr);
    auto rhsNode = moveArgOrExprToNode(rhsArgIndex, rhsExpr);
    if (!lhsNode || !rhsNode)
      return false;

    auto node = std::make_unique<ResolvedNamedBoolExpr>();
    node->kind = *binKind;
    node->compareSigned = cmpSigned;
    node->lhs = std::move(lhsNode);
    node->rhs = std::move(rhsNode);
    std::unique_ptr<ResolvedNamedBoolExpr> current = std::move(node);
    if (logicalNot) {
      auto notNode = std::make_unique<ResolvedNamedBoolExpr>();
      notNode->kind = ResolvedNamedBoolExpr::Kind::Not;
      notNode->lhs = std::move(current);
      current = std::move(notNode);
    }
    resolvedExpr = maybeWrapGroup(std::move(current));
    return true;
  }

  auto nameAttr =
      dyn_cast_or_null<StringAttr>(detail.get(key("name")));
  if (!nameAttr || nameAttr.getValue().empty())
    return false;

  auto it = inputNameToArgIndex.find(nameAttr.getValue());
  if (it == inputNameToArgIndex.end() || it->second >= numNonStateArgs)
    return false;
  unsigned sourceIndex = it->second;

  auto lsbAttr =
      dyn_cast_or_null<IntegerAttr>(detail.get(key("lsb")));
  auto msbAttr =
      dyn_cast_or_null<IntegerAttr>(detail.get(key("msb")));
  bool hasSliceAttrs = static_cast<bool>(lsbAttr) || static_cast<bool>(msbAttr);
  if (hasSliceAttrs && (!lsbAttr || !msbAttr))
    return false;
  auto dynIndexNameAttr =
      dyn_cast_or_null<StringAttr>(detail.get(key("dyn_index_name")));
  auto dynSignAttr =
      dyn_cast_or_null<IntegerAttr>(detail.get(key("dyn_sign")));
  auto dynOffsetAttr =
      dyn_cast_or_null<IntegerAttr>(detail.get(key("dyn_offset")));
  auto dynWidthAttr =
      dyn_cast_or_null<IntegerAttr>(detail.get(key("dyn_width")));
  bool hasDynSliceAttrs = static_cast<bool>(dynIndexNameAttr) ||
                          static_cast<bool>(dynSignAttr) ||
                          static_cast<bool>(dynOffsetAttr) ||
                          static_cast<bool>(dynWidthAttr);
  if (hasDynSliceAttrs &&
      (!dynIndexNameAttr || !dynSignAttr || !dynOffsetAttr || !dynWidthAttr))
    return false;
  if (hasDynSliceAttrs && hasSliceAttrs)
    return false;

  std::optional<ResolvedNamedBoolExpr::ArgSlice> argSlice;
  if (lsbAttr && msbAttr) {
    int64_t lsb = lsbAttr.getInt();
    int64_t msb = msbAttr.getInt();
    if (lsb < 0 || msb < lsb)
      return false;
    argSlice = ResolvedNamedBoolExpr::ArgSlice{static_cast<unsigned>(lsb),
                                                static_cast<unsigned>(msb)};
  }
  if (hasDynSliceAttrs) {
    if (dynIndexNameAttr.getValue().empty())
      return false;
    auto dynIndexIt = inputNameToArgIndex.find(dynIndexNameAttr.getValue());
    if (dynIndexIt == inputNameToArgIndex.end() ||
        dynIndexIt->second >= numNonStateArgs)
      return false;
    int64_t sign = dynSignAttr.getInt();
    int64_t offset = dynOffsetAttr.getInt();
    int64_t width = dynWidthAttr.getInt();
    if ((sign != 1 && sign != -1) || width <= 0 || width > (1 << 20))
      return false;
    if (offset < std::numeric_limits<int32_t>::min() ||
        offset > std::numeric_limits<int32_t>::max())
      return false;

    ResolvedNamedBoolExpr::ArgSlice dynSlice;
    dynSlice.dynamicIndexArg = dynIndexIt->second;
    dynSlice.dynamicIndexSign = static_cast<int32_t>(sign);
    dynSlice.dynamicIndexOffset = static_cast<int32_t>(offset);
    dynSlice.dynamicWidth = static_cast<unsigned>(width);
    argSlice = dynSlice;
  }

  auto reductionAttr =
      dyn_cast_or_null<StringAttr>(detail.get(key("reduction")));
  auto bitwiseNotAttr = parseBoolLikeAttr("bitwise_not");
  if (!bitwiseNotAttr)
    return false;
  bool bitwiseNot = *bitwiseNotAttr;
  std::optional<ResolvedNamedBoolExpr::Kind> reductionKind;
  if (reductionAttr) {
    StringRef reduction = reductionAttr.getValue();
    if (reduction == "and")
      reductionKind = ResolvedNamedBoolExpr::Kind::ReduceAnd;
    else if (reduction == "or")
      reductionKind = ResolvedNamedBoolExpr::Kind::ReduceOr;
    else if (reduction == "xor")
      reductionKind = ResolvedNamedBoolExpr::Kind::ReduceXor;
    else if (reduction == "nand")
      reductionKind = ResolvedNamedBoolExpr::Kind::ReduceNand;
    else if (reduction == "nor")
      reductionKind = ResolvedNamedBoolExpr::Kind::ReduceNor;
    else if (reduction == "xnor")
      reductionKind = ResolvedNamedBoolExpr::Kind::ReduceXnor;
    else
      return false;
  }

  if (!argSlice && !reductionKind && !bitwiseNot && !logicalNot &&
      groupDepth == 0) {
    argIndex = sourceIndex;
    return true;
  }

  std::unique_ptr<ResolvedNamedBoolExpr> current =
      materializeArgNode(sourceIndex, argSlice);
  if (bitwiseNot) {
    auto notNode = std::make_unique<ResolvedNamedBoolExpr>();
    notNode->kind = ResolvedNamedBoolExpr::Kind::BitwiseNot;
    notNode->lhs = std::move(current);
    current = std::move(notNode);
  }
  if (reductionKind) {
    auto reduceNode = std::make_unique<ResolvedNamedBoolExpr>();
    reduceNode->kind = *reductionKind;
    reduceNode->lhs = std::move(current);
    current = std::move(reduceNode);
  }
  if (logicalNot) {
    auto notNode = std::make_unique<ResolvedNamedBoolExpr>();
    notNode->kind = ResolvedNamedBoolExpr::Kind::Not;
    notNode->lhs = std::move(current);
    current = std::move(notNode);
  }
  resolvedExpr = maybeWrapGroup(std::move(current));
  return true;
}

} // namespace

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// LTL Operation Conversion Patterns
//===----------------------------------------------------------------------===//

static Value materializeSMTBool(Value input, const TypeConverter &converter,
                                ConversionPatternRewriter &rewriter,
                                Location loc) {
  if (!input)
    return Value();
  if (isa<smt::BoolType>(input.getType()))
    return input;
  if (auto bvTy = dyn_cast<smt::BitVectorType>(input.getType());
      bvTy && bvTy.getWidth() == 1) {
    auto one = smt::BVConstantOp::create(rewriter, loc, 1, 1);
    return smt::EqOp::create(rewriter, loc, input, one);
  }
  if (auto intTy = dyn_cast<IntegerType>(input.getType());
      intTy && intTy.getWidth() == 1) {
    auto bvTy = smt::BitVectorType::get(rewriter.getContext(), 1);
    Value bv =
        converter.materializeTargetConversion(rewriter, loc, bvTy, input);
    if (!bv)
      return Value();
    auto one = smt::BVConstantOp::create(rewriter, loc, 1, 1);
    return smt::EqOp::create(rewriter, loc, bv, one);
  }
  return converter.materializeTargetConversion(
      rewriter, loc, smt::BoolType::get(rewriter.getContext()), input);
}

/// Convert ltl.and to smt.and
struct LTLAndOpConversion : OpConversionPattern<ltl::AndOp> {
  using OpConversionPattern<ltl::AndOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::AndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    SmallVector<Value, 4> smtOperands;
    for (Value input : adaptor.getInputs()) {
      Value converted = materializeSMTBool(input, *typeConverter, rewriter,
                                           op.getLoc());
      if (!converted)
        return failure();
      smtOperands.push_back(converted);
    }
    if (smtOperands.size() == 1) {
      rewriter.replaceOp(op, smtOperands[0]);
      return success();
    }
    rewriter.replaceOpWithNewOp<smt::AndOp>(op, smtOperands);
    return success();
  }
};

/// Convert ltl.or to smt.or
struct LTLOrOpConversion : OpConversionPattern<ltl::OrOp> {
  using OpConversionPattern<ltl::OrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::OrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    SmallVector<Value, 4> smtOperands;
    for (Value input : adaptor.getInputs()) {
      Value converted = materializeSMTBool(input, *typeConverter, rewriter,
                                           op.getLoc());
      if (!converted)
        return failure();
      smtOperands.push_back(converted);
    }
    if (smtOperands.size() == 1) {
      rewriter.replaceOp(op, smtOperands[0]);
      return success();
    }
    rewriter.replaceOpWithNewOp<smt::OrOp>(op, smtOperands);
    return success();
  }
};

/// Convert ltl.intersect to smt.and
struct LTLIntersectOpConversion : OpConversionPattern<ltl::IntersectOp> {
  using OpConversionPattern<ltl::IntersectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::IntersectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    SmallVector<Value, 4> smtOperands;
    for (Value input : adaptor.getInputs()) {
      Value converted = materializeSMTBool(input, *typeConverter, rewriter,
                                           op.getLoc());
      if (!converted)
        return failure();
      smtOperands.push_back(converted);
    }
    if (smtOperands.size() == 1) {
      rewriter.replaceOp(op, smtOperands[0]);
      return success();
    }
    rewriter.replaceOpWithNewOp<smt::AndOp>(op, smtOperands);
    return success();
  }
};

/// Convert ltl.not to smt.not
struct LTLNotOpConversion : OpConversionPattern<ltl::NotOp> {
  using OpConversionPattern<ltl::NotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value input =
        materializeSMTBool(adaptor.getInput(), *typeConverter, rewriter,
                           op.getLoc());
    if (!input)
      return failure();
    rewriter.replaceOpWithNewOp<smt::NotOp>(op, input);
    return success();
  }
};

/// Convert ltl.implication to smt.or(not(antecedent), consequent)
struct LTLImplicationOpConversion : OpConversionPattern<ltl::ImplicationOp> {
  using OpConversionPattern<ltl::ImplicationOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::ImplicationOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value antecedent =
        materializeSMTBool(adaptor.getAntecedent(), *typeConverter, rewriter,
                           op.getLoc());
    Value consequent =
        materializeSMTBool(adaptor.getConsequent(), *typeConverter, rewriter,
                           op.getLoc());
    if (!antecedent || !consequent)
      return failure();
    Value notAntecedent = smt::NotOp::create(rewriter, op.getLoc(), antecedent);
    rewriter.replaceOpWithNewOp<smt::OrOp>(op, notAntecedent, consequent);
    return success();
  }
};

/// Convert ltl.eventually to SMT boolean.
/// For bounded model checking, eventually(p) at the current time step
/// contributes p to an OR over all time steps. The BMC loop handles
/// the accumulation; here we just convert the inner property.
struct LTLEventuallyOpConversion : OpConversionPattern<ltl::EventuallyOp> {
  LTLEventuallyOpConversion(const TypeConverter &typeConverter,
                            MLIRContext *context, bool approxTemporalOps)
      : OpConversionPattern<ltl::EventuallyOp>(typeConverter, context),
        approxTemporalOps(approxTemporalOps) {}

  LogicalResult
  matchAndRewrite(ltl::EventuallyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    if (op->hasAttr(kWeakEventuallyAttr)) {
      rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, true);
      return success();
    }
    if (!approxTemporalOps) {
      op.emitError("ltl.eventually must be lowered by the BMC/LTLToCore "
                   "infrastructure; refusing UNSOUND approximation (rerun "
                   "with --convert-verif-to-smt=approx-temporal=true to "
                   "approximate as its input)");
      return failure();
    }
    // For BMC: eventually(p) means p should hold at some point.
    // At each time step, we check if p holds. The BMC loop accumulates
    // these checks with OR. Here we convert the inner property.
    Value input =
        materializeSMTBool(adaptor.getInput(), *typeConverter, rewriter,
                           op.getLoc());
    if (!input)
      return failure();
    // The eventually property at this step is just the inner property value
    // The liveness aspect (must eventually hold) is checked by the BMC framework
    rewriter.replaceOp(op, input);
    return success();
  }

private:
  bool approxTemporalOps;
};

/// Convert ltl.until to SMT boolean.
/// p until q: p holds continuously until q holds.
/// For bounded checking at a single step: q || (p && X(p U q))
/// Since X requires next-state which BMC handles, we encode:
/// weak until semantics: q || p (either q holds or p holds at this step)
struct LTLUntilOpConversion : OpConversionPattern<ltl::UntilOp> {
  LTLUntilOpConversion(const TypeConverter &typeConverter, MLIRContext *context,
                       bool approxTemporalOps)
      : OpConversionPattern<ltl::UntilOp>(typeConverter, context),
        approxTemporalOps(approxTemporalOps) {}

  LogicalResult
  matchAndRewrite(ltl::UntilOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    if (!approxTemporalOps) {
      op.emitError("ltl.until must be lowered by the BMC/LTLToCore "
                   "infrastructure; refusing UNSOUND approximation (rerun "
                   "with --convert-verif-to-smt=approx-temporal=true to "
                   "approximate as `q || p`)");
      return failure();
    }
    Value p = materializeSMTBool(adaptor.getInput(), *typeConverter, rewriter,
                                 op.getLoc());
    Value q = materializeSMTBool(adaptor.getCondition(), *typeConverter,
                                 rewriter, op.getLoc());
    if (!p || !q)
      return failure();
    // Weak until: the property q || p
    // Full until semantics requires tracking across time steps in BMC
    rewriter.replaceOpWithNewOp<smt::OrOp>(op, q, p);
    return success();
  }

private:
  bool approxTemporalOps;
};

/// Convert ltl.boolean_constant to smt.constant
struct LTLBooleanConstantOpConversion
    : OpConversionPattern<ltl::BooleanConstantOp> {
  using OpConversionPattern<ltl::BooleanConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::BooleanConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, op.getValue());
    return success();
  }
};

/// Convert ltl.clock to SMT boolean.
/// The clocking itself is handled by BMC-specific gating; for SMT lowering we
/// drop the clock annotation and lower the input.
struct LTLClockOpConversion : OpConversionPattern<ltl::ClockOp> {
  using OpConversionPattern<ltl::ClockOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::ClockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value input =
        materializeSMTBool(adaptor.getInput(), *typeConverter, rewriter,
                           op.getLoc());
    if (!input)
      return failure();
    rewriter.replaceOp(op, input);
    return success();
  }
};

/// Convert ltl.delay to SMT boolean.
///
/// NOTE:
/// For generic SMT lowering (outside the BMC pipeline), delay(seq, N>0) is
/// UNSOUND unless modeled with explicit multi-step tracking. By default we
/// therefore refuse to lower nonzero delays outside of the BMC infrastructure,
/// so temporal gaps fail loudly instead of silently producing incorrect
/// results. Use `--convert-verif-to-smt=approx-temporal=true` to opt into the
/// legacy approximation behavior (treat as `true`).
///
/// The BMC pipeline handles delay semantics separately:
/// - Delay/past buffers for standalone temporal ops.
/// - Sequence NFAs for multi-step sequence operators (concat/repeat/goto/etc.).
/// Any ltl.delay that survives BMC rewriting is therefore a fallback path.
///
/// For non-BMC uses that require temporal correctness, a dedicated delay
/// buffering scheme is still needed.
///
/// See BMC_MULTISTEP_DESIGN.md for the buffered BMC architecture.
struct LTLDelayOpConversion : OpConversionPattern<ltl::DelayOp> {
  LTLDelayOpConversion(const TypeConverter &typeConverter, MLIRContext *context,
                       bool approxTemporalOps)
      : OpConversionPattern<ltl::DelayOp>(typeConverter, context),
        approxTemporalOps(approxTemporalOps) {}

  LogicalResult
  matchAndRewrite(ltl::DelayOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    // Get the delay amount
    uint64_t delay = op.getDelay();

    if (delay == 0) {
      // No delay: just pass through the input sequence
      Value input =
          materializeSMTBool(adaptor.getInput(), *typeConverter, rewriter,
                             op.getLoc());
      if (!input)
        return failure();
      rewriter.replaceOp(op, input);
    } else {
      if (!approxTemporalOps) {
        op.emitError("ltl.delay with delay > 0 must be lowered by the BMC "
                     "multi-step infrastructure; refusing UNSOUND "
                     "approximation (rerun with "
                     "--convert-verif-to-smt=approx-temporal=true to "
                     "approximate as `true`)");
        return failure();
      }
      // UNSOUND fallback (opt-in): approximate as true.
      rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, true);
    }
    return success();
  }

private:
  bool approxTemporalOps;
};

/// Convert ltl.past to SMT boolean.
/// past(signal, N) returns the value of signal from N cycles ago.
/// For BMC, this is handled specially in the BMC conversion by creating
/// past buffers that track signal history. This fallback pattern handles
/// any remaining past ops that weren't processed by BMC (e.g., past(x, 0)).
///
/// NOTE: For past with N > 0 outside of BMC, we return false (conservative
/// approximation that the past value was unknown/false). The proper handling
/// is done in the BMC infrastructure.
struct LTLPastOpConversion : OpConversionPattern<ltl::PastOp> {
  LTLPastOpConversion(const TypeConverter &typeConverter, MLIRContext *context,
                      bool approxTemporalOps)
      : OpConversionPattern<ltl::PastOp>(typeConverter, context),
        approxTemporalOps(approxTemporalOps) {}

  LogicalResult
  matchAndRewrite(ltl::PastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    uint64_t delay = op.getDelay();

    if (delay == 0) {
      // past(x, 0) is just x - pass through the input
      Value input =
          materializeSMTBool(adaptor.getInput(), *typeConverter, rewriter,
                             op.getLoc());
      if (!input)
        return failure();
      rewriter.replaceOp(op, input);
    } else {
      if (!approxTemporalOps) {
        op.emitError("ltl.past with delay > 0 must be lowered by the BMC "
                     "multi-step infrastructure; refusing UNSOUND "
                     "approximation (rerun with "
                     "--convert-verif-to-smt=approx-temporal=true to "
                     "approximate as `false`)");
        return failure();
      }
      // For past with delay > 0 outside of BMC, return false (conservative).
      // This handles edge cases where ltl.past appears outside BMC context.
      // The proper handling with buffer tracking is done in VerifBoundedModelCheckingOpConversion.
      rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, false);
    }
    return success();
  }

private:
  bool approxTemporalOps;
};

/// Convert ltl.concat to SMT boolean.
/// Concatenation in LTL joins sequences end-to-end. In BMC, multi-step
/// semantics are handled by sequence NFAs and delay buffers; this pattern
/// is a fallback for cases that reach generic SMT lowering.
struct LTLConcatOpConversion : OpConversionPattern<ltl::ConcatOp> {
  LTLConcatOpConversion(const TypeConverter &typeConverter, MLIRContext *context,
                        bool approxTemporalOps)
      : OpConversionPattern<ltl::ConcatOp>(typeConverter, context),
        approxTemporalOps(approxTemporalOps) {}

  LogicalResult
  matchAndRewrite(ltl::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    SmallVector<Value, 4> smtOperands;
    for (Value input : adaptor.getInputs()) {
      Value converted = materializeSMTBool(input, *typeConverter, rewriter,
                                           op.getLoc());
      if (!converted)
        return failure();
      smtOperands.push_back(converted);
    }

    if (smtOperands.empty()) {
      // Empty concatenation is trivially true
      rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, true);
      return success();
    }

    if (smtOperands.size() == 1) {
      rewriter.replaceOp(op, smtOperands[0]);
      return success();
    }

    if (!approxTemporalOps) {
      op.emitError("ltl.concat with multiple inputs must be lowered by the "
                   "BMC/LTLToCore infrastructure; refusing UNSOUND "
                   "approximation (rerun with "
                   "--convert-verif-to-smt=approx-temporal=true to "
                   "approximate as `and`)");
      return failure();
    }

    // For BMC: concatenation of sequences at a single step is AND
    // (all parts of the sequence must hold in their respective positions)
    rewriter.replaceOpWithNewOp<smt::AndOp>(op, smtOperands);
    return success();
  }

private:
  bool approxTemporalOps;
};

/// Convert ltl.repeat to SMT boolean.
/// Repetition in LTL means the sequence must match N times consecutively.
/// In BMC, multi-step semantics are handled by sequence NFAs; this pattern
/// is a fallback for generic SMT lowering.
struct LTLRepeatOpConversion : OpConversionPattern<ltl::RepeatOp> {
  LTLRepeatOpConversion(const TypeConverter &typeConverter, MLIRContext *context,
                        bool approxTemporalOps)
      : OpConversionPattern<ltl::RepeatOp>(typeConverter, context),
        approxTemporalOps(approxTemporalOps) {}

  LogicalResult
  matchAndRewrite(ltl::RepeatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    uint64_t base = op.getBase();

    if (base == 0) {
      // Zero repetitions: empty sequence, trivially true
      rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, true);
      return success();
    }

    if (!approxTemporalOps) {
      op.emitError("ltl.repeat must be lowered by the BMC/LTLToCore "
                   "infrastructure; refusing UNSOUND approximation (rerun "
                   "with --convert-verif-to-smt=approx-temporal=true)");
      return failure();
    }

    // For base >= 1: the sequence must match at least once
    // In BMC single-step semantics, this is just the input sequence value
    Value input =
        materializeSMTBool(adaptor.getInput(), *typeConverter, rewriter,
                           op.getLoc());
    if (!input)
      return failure();

    // For exact repetition (base=N, more=0), at a single step this is
    // equivalent to the input holding (since a boolean sequence is
    // instantaneous and N repetitions of it all overlap at the same step).
    // The temporal tracking of multiple repetitions is handled by BMC.
    rewriter.replaceOp(op, input);
    return success();
  }

private:
  bool approxTemporalOps;
};

/// Convert ltl.goto_repeat to SMT boolean.
/// goto_repeat is a non-consecutive repetition where the final repetition
/// must hold at the end. For BMC single-step semantics:
/// - goto_repeat(seq, 0, N) with base=0 means the sequence can match 0 times (true)
/// - goto_repeat(seq, N, M) with N>0 means seq must hold at least once at this step
/// The full temporal semantics (non-consecutive with final match) requires
/// multi-step tracking which is handled by the BMC loop or LTLToCore pass.
struct LTLGoToRepeatOpConversion : OpConversionPattern<ltl::GoToRepeatOp> {
  LTLGoToRepeatOpConversion(const TypeConverter &typeConverter,
                            MLIRContext *context, bool approxTemporalOps)
      : OpConversionPattern<ltl::GoToRepeatOp>(typeConverter, context),
        approxTemporalOps(approxTemporalOps) {}

  LogicalResult
  matchAndRewrite(ltl::GoToRepeatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    uint64_t base = op.getBase();

    if (base == 0) {
      // Zero repetitions: empty sequence, trivially true
      rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, true);
      return success();
    }

    if (!approxTemporalOps) {
      op.emitError("ltl.goto_repeat must be lowered by the BMC/LTLToCore "
                   "infrastructure; refusing UNSOUND approximation (rerun "
                   "with --convert-verif-to-smt=approx-temporal=true)");
      return failure();
    }

    // For base >= 1: at a single step, the sequence must hold
    Value input =
        materializeSMTBool(adaptor.getInput(), *typeConverter, rewriter,
                           op.getLoc());
    if (!input)
      return failure();

    rewriter.replaceOp(op, input);
    return success();
  }

private:
  bool approxTemporalOps;
};

/// Convert ltl.non_consecutive_repeat to SMT boolean.
/// non_consecutive_repeat is like goto_repeat but the final match doesn't
/// need to be at the end. For BMC single-step semantics:
/// - non_consecutive_repeat(seq, 0, N) with base=0 means trivially true
/// - non_consecutive_repeat(seq, N, M) with N>0 means seq must hold at this step
struct LTLNonConsecutiveRepeatOpConversion
    : OpConversionPattern<ltl::NonConsecutiveRepeatOp> {
  LTLNonConsecutiveRepeatOpConversion(const TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      bool approxTemporalOps)
      : OpConversionPattern<ltl::NonConsecutiveRepeatOp>(typeConverter, context),
        approxTemporalOps(approxTemporalOps) {}

  LogicalResult
  matchAndRewrite(ltl::NonConsecutiveRepeatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    uint64_t base = op.getBase();

    if (base == 0) {
      // Zero repetitions: empty sequence, trivially true
      rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, true);
      return success();
    }

    if (!approxTemporalOps) {
      op.emitError("ltl.non_consecutive_repeat must be lowered by the "
                   "BMC/LTLToCore infrastructure; refusing UNSOUND "
                   "approximation (rerun with "
                   "--convert-verif-to-smt=approx-temporal=true)");
      return failure();
    }

    // For base >= 1: at a single step, the sequence must hold
    Value input =
        materializeSMTBool(adaptor.getInput(), *typeConverter, rewriter,
                           op.getLoc());
    if (!input)
      return failure();

    rewriter.replaceOp(op, input);
    return success();
  }

private:
  bool approxTemporalOps;
};

//===----------------------------------------------------------------------===//
// Verif Operation Conversion Patterns
//===----------------------------------------------------------------------===//

/// Lower a verif::AssertOp operation with an i1 operand to a smt::AssertOp,
/// negated to check for unsatisfiability.
struct VerifAssertOpConversion : OpConversionPattern<verif::AssertOp> {
  using OpConversionPattern<verif::AssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ltl::SequenceType, ltl::PropertyType>(
            adaptor.getProperty().getType())) {
      return op.emitError("LTL properties are not supported by VerifToSMT yet");
    }
    Value cond = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getProperty());
    Value enable = adaptor.getEnable();
    if (enable) {
      enable = typeConverter->materializeTargetConversion(
          rewriter, op.getLoc(), smt::BoolType::get(getContext()), enable);
      if (!enable)
        return failure();
      cond = gateSMTWithEnable(cond, enable, /*isCover=*/false, rewriter,
                               op.getLoc());
    }
    Value notCond = smt::NotOp::create(rewriter, op.getLoc(), cond);
    auto assertOp = rewriter.replaceOpWithNewOp<smt::AssertOp>(op, notCond);
    if (auto label = op.getLabelAttr(); label && !label.getValue().empty())
      assertOp->setAttr("smtlib.name", label);
    return success();
  }
};

/// Lower a verif::AssumeOp operation with an i1 operand to a smt::AssertOp
struct VerifAssumeOpConversion : OpConversionPattern<verif::AssumeOp> {
  using OpConversionPattern<verif::AssumeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::AssumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ltl::SequenceType, ltl::PropertyType>(
            adaptor.getProperty().getType())) {
      return op.emitError("LTL properties are not supported by VerifToSMT yet");
    }
    Value cond = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getProperty());
    Value enable = adaptor.getEnable();
    if (enable) {
      enable = typeConverter->materializeTargetConversion(
          rewriter, op.getLoc(), smt::BoolType::get(getContext()), enable);
      if (!enable)
        return failure();
      cond = gateSMTWithEnable(cond, enable, /*isCover=*/false, rewriter,
                               op.getLoc());
    }
    auto assertOp = rewriter.replaceOpWithNewOp<smt::AssertOp>(op, cond);
    if (auto label = op.getLabelAttr(); label && !label.getValue().empty())
      assertOp->setAttr("smtlib.name", label);
    return success();
  }
};

/// Drop verif::CoverOp for now; cover checking is not modeled in SMT.
struct VerifCoverOpConversion : OpConversionPattern<verif::CoverOp> {
  using OpConversionPattern<verif::CoverOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::CoverOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ltl::SequenceType, ltl::PropertyType>(
            adaptor.getProperty().getType())) {
      return op.emitError("LTL cover properties are not supported by "
                          "VerifToSMT yet");
    }
    Value cond = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getProperty());
    Value enable = adaptor.getEnable();
    if (enable) {
      enable = typeConverter->materializeTargetConversion(
          rewriter, op.getLoc(), smt::BoolType::get(getContext()), enable);
      if (!enable)
        return failure();
      cond = gateSMTWithEnable(cond, enable, /*isCover=*/true, rewriter,
                               op.getLoc());
    }
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, cond);
    return success();
  }
};

/// Lower unrealized casts between smt.bool and smt.bv<1> into explicit SMT ops.
struct BoolBVCastOpRewrite : OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    OpBuilder::InsertionGuard guard(rewriter);

    Value input = op.getInputs()[0];
    Type srcTy = input.getType();
    Type dstTy = op.getOutputs()[0].getType();
    Location loc = op.getLoc();

    Region *inputRegion = input.getParentRegion();
    Region *castRegion = op->getParentRegion();
    if (inputRegion && castRegion && inputRegion != castRegion) {
      auto allUsesInInputRegion = llvm::all_of(
          op->getUsers(), [&](Operation *user) {
            Region *userRegion = user->getParentRegion();
            return userRegion == inputRegion ||
                   inputRegion->isProperAncestor(userRegion);
          });
      if (!allUsesInInputRegion)
        return failure();
      if (auto *defOp = input.getDefiningOp())
        rewriter.setInsertionPointAfter(defOp);
      else if (auto *block = input.getParentBlock())
        rewriter.setInsertionPointToStart(block);
    } else {
      rewriter.setInsertionPoint(op);
    }

    auto dstBvTy = dyn_cast<smt::BitVectorType>(dstTy);
    if (dstBvTy && dstBvTy.getWidth() == 1) {
      Value boolVal;
      if (isa<smt::BoolType>(srcTy)) {
        boolVal = input;
      } else if (auto intTy = dyn_cast<IntegerType>(srcTy);
                 intTy && intTy.getWidth() == 1) {
        if (auto innerCast =
                op.getInputs()[0]
                    .getDefiningOp<UnrealizedConversionCastOp>()) {
          if (innerCast.getInputs().size() == 1 &&
              isa<smt::BoolType>(innerCast.getInputs()[0].getType()))
            boolVal = innerCast.getInputs()[0];
        }
      }
      if (boolVal) {
        auto one = smt::BVConstantOp::create(rewriter, loc, 1, 1);
        auto zero = smt::BVConstantOp::create(rewriter, loc, 0, 1);
        auto ite = smt::IteOp::create(rewriter, loc, boolVal, one, zero);
        rewriter.replaceOp(op, ite);
        return success();
      }
    }

    if (isa<smt::BoolType>(dstTy)) {
      if (auto bvTy = dyn_cast<smt::BitVectorType>(srcTy);
          bvTy && bvTy.getWidth() == 1) {
        auto one = smt::BVConstantOp::create(rewriter, loc, 1, 1);
        auto eq = smt::EqOp::create(rewriter, loc, input, one);
        rewriter.replaceOp(op, eq);
        return success();
      }
    }

    return failure();
  }
};

/// Move unrealized casts into the region that defines their input if the cast
/// is used exclusively there. This avoids cross-region value uses.
struct RelocateCastIntoInputRegion
    : OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    Value input = op.getInputs()[0];
    Region *inputRegion = input.getParentRegion();
    Region *castRegion = op->getParentRegion();
    if (!inputRegion || !castRegion || inputRegion == castRegion)
      return failure();

    auto allUsesInInputRegion = llvm::all_of(
        op->getUsers(), [&](Operation *user) {
          Region *userRegion = user->getParentRegion();
          return userRegion == inputRegion ||
                 inputRegion->isProperAncestor(userRegion);
        });
    if (!allUsesInInputRegion)
      return failure();

    OpBuilder::InsertionGuard guard(rewriter);
    if (auto *defOp = input.getDefiningOp()) {
      rewriter.setInsertionPointAfter(defOp);
    } else if (auto *block = input.getParentBlock()) {
      rewriter.setInsertionPointToStart(block);
    } else {
      return failure();
    }

    auto relocated = UnrealizedConversionCastOp::create(
        rewriter, op.getLoc(), op.getResultTypes(), input);
    rewriter.replaceOp(op, relocated.getResults());
    return success();
  }
};

/// Move SMT equality ops into the region that defines their operands to avoid
/// cross-region value uses. Constants are re-materialized in the target region.
struct RelocateSMTEqIntoOperandRegion : OpRewritePattern<smt::EqOp> {
  using OpRewritePattern<smt::EqOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(smt::EqOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    Region *opRegion = op->getParentRegion();
    if (!opRegion)
      return failure();

    auto *lhsRegion = lhs.getParentRegion();
    auto *rhsRegion = rhs.getParentRegion();

    Region *targetRegion = nullptr;
    if (lhsRegion && lhsRegion != opRegion)
      targetRegion = lhsRegion;
    else if (rhsRegion && rhsRegion != opRegion)
      targetRegion = rhsRegion;
    else
      return failure();

    auto allUsesInTarget = llvm::all_of(op->getUsers(), [&](Operation *user) {
      Region *userRegion = user->getParentRegion();
      return userRegion == targetRegion ||
             targetRegion->isProperAncestor(userRegion);
    });
    if (!allUsesInTarget)
      return failure();

    OpBuilder::InsertionGuard guard(rewriter);
    if (auto *defOp = lhs.getDefiningOp())
      rewriter.setInsertionPointAfter(defOp);
    else if (auto *block = lhs.getParentBlock())
      rewriter.setInsertionPointToStart(block);
    else
      return failure();

    auto materializeInTarget = [&](Value input) -> Value {
      if (input.getParentRegion() == targetRegion)
        return input;
      if (auto cst = input.getDefiningOp<smt::BVConstantOp>())
        return smt::BVConstantOp::create(rewriter, op.getLoc(),
                                         cst.getValue());
      if (auto cst = input.getDefiningOp<smt::BoolConstantOp>())
        return smt::BoolConstantOp::create(rewriter, op.getLoc(),
                                           cst.getValue());
      return Value();
    };

    Value newLhs = materializeInTarget(lhs);
    Value newRhs = materializeInTarget(rhs);
    if (!newLhs || !newRhs)
      return failure();

    auto relocated = smt::EqOp::create(rewriter, op.getLoc(), newLhs, newRhs);
    rewriter.replaceOp(op, relocated.getResult());
    return success();
  }
};

template <typename OpTy>
struct CircuitRelationCheckOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

protected:
  using ConversionPattern::typeConverter;
  void
  createOutputsDifferentOps(Operation *firstOutputs, Operation *secondOutputs,
                            Location &loc, ConversionPatternRewriter &rewriter,
                            SmallVectorImpl<Value> &outputsDifferent) const {
    // Convert the yielded values back to the source type system (since
    // the operations of the inlined blocks will be converted by other patterns
    // later on and we should make sure the IR is well-typed after each pattern
    // application), and compare the output values.
    for (auto [out1, out2] :
         llvm::zip(firstOutputs->getOperands(), secondOutputs->getOperands())) {
      Value o1 = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(out1.getType()), out1);
      Value o2 = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(out1.getType()), out2);
      outputsDifferent.emplace_back(
          smt::DistinctOp::create(rewriter, loc, o1, o2));
    }
  }

  void replaceOpWithSatCheck(OpTy &op, Location &loc,
                             ConversionPatternRewriter &rewriter,
                             smt::SolverOp &solver) const {
    // If no operation uses the result of this solver, we leave our check
    // operations empty. If the result is used, we create a check operation with
    // the result type of the operation and yield the result of the check
    // operation.
    if (op.getNumResults() == 0) {
      auto checkOp = smt::CheckOp::create(rewriter, loc, TypeRange{});
      rewriter.createBlock(&checkOp.getSatRegion());
      smt::YieldOp::create(rewriter, loc);
      rewriter.createBlock(&checkOp.getUnknownRegion());
      smt::YieldOp::create(rewriter, loc);
      rewriter.createBlock(&checkOp.getUnsatRegion());
      smt::YieldOp::create(rewriter, loc);
      rewriter.setInsertionPointAfter(checkOp);
      smt::YieldOp::create(rewriter, loc);

      // Erase as operation is replaced by an operator without a return value.
      rewriter.eraseOp(op);
    } else {
      Value falseVal =
          arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(false));
      Value trueVal =
          arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
      auto checkOp = smt::CheckOp::create(rewriter, loc, rewriter.getI1Type());
      rewriter.createBlock(&checkOp.getSatRegion());
      smt::YieldOp::create(rewriter, loc, falseVal);
      rewriter.createBlock(&checkOp.getUnknownRegion());
      smt::YieldOp::create(rewriter, loc, falseVal);
      rewriter.createBlock(&checkOp.getUnsatRegion());
      smt::YieldOp::create(rewriter, loc, trueVal);
      rewriter.setInsertionPointAfter(checkOp);
      smt::YieldOp::create(rewriter, loc, checkOp->getResults());

      rewriter.replaceOp(op, solver->getResults());
    }
  }
};

/// Lower a verif::LecOp operation to a miter circuit encoded in SMT.
/// More information on miter circuits can be found, e.g., in this paper:
/// Brand, D., 1993, November. Verification of large synthesized designs. In
/// Proceedings of 1993 International Conference on Computer Aided Design
/// (ICCAD) (pp. 534-537). IEEE.
struct LogicEquivalenceCheckingOpConversion
    : CircuitRelationCheckOpConversion<verif::LogicEquivalenceCheckingOp> {
  using CircuitRelationCheckOpConversion<
      verif::LogicEquivalenceCheckingOp>::CircuitRelationCheckOpConversion;
  LogicEquivalenceCheckingOpConversion(TypeConverter &converter,
                                       MLIRContext *context,
                                       bool assumeKnownInputs,
                                       bool xOptimisticOutputs)
      : CircuitRelationCheckOpConversion(converter, context),
        assumeKnownInputs(assumeKnownInputs),
        xOptimisticOutputs(xOptimisticOutputs) {}

  LogicalResult
  matchAndRewrite(verif::LogicEquivalenceCheckingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *firstOutputs = adaptor.getFirstCircuit().front().getTerminator();
    auto *secondOutputs = adaptor.getSecondCircuit().front().getTerminator();
    auto *origFirstOutputs = op.getFirstCircuit().front().getTerminator();
    SmallVector<Type> originalOutputTypes(origFirstOutputs->getOperandTypes());

    auto hasNoResult = op.getNumResults() == 0;

    // Solver will only return a result when it is used to check the returned
    // value.
    smt::SolverOp solver;
    if (hasNoResult)
      solver = smt::SolverOp::create(rewriter, loc, TypeRange{}, ValueRange{});
    else
      solver = smt::SolverOp::create(rewriter, loc, rewriter.getI1Type(),
                                     ValueRange{});
    rewriter.createBlock(&solver.getBodyRegion());

    // Capture the original argument types BEFORE conversion for assume-known
    // logic which needs to identify hw.struct types with value/unknown fields.
    ArrayAttr inputNames = op->getAttrOfType<ArrayAttr>("lec.input_names");
    ArrayAttr inputTypes = op->getAttrOfType<ArrayAttr>("lec.input_types");
    auto originalArgTypes = op.getFirstCircuit().getArgumentTypes();
    auto getOriginalType = [&](unsigned index) -> Type {
      if (inputTypes && index < inputTypes.size())
        if (auto typeAttr = dyn_cast<TypeAttr>(inputTypes[index]))
          return typeAttr.getValue();
      if (index < originalArgTypes.size())
        return originalArgTypes[index];
      return Type{};
    };
    if (!assumeKnownInputs) {
      bool hasFourStateInputs = false;
      for (unsigned i = 0, e = originalArgTypes.size(); i < e; ++i) {
        if (auto originalTy = getOriginalType(i);
            originalTy && isFourStateStruct(originalTy)) {
          hasFourStateInputs = true;
          break;
        }
      }
      if (hasFourStateInputs) {
        op.emitWarning(
            "4-state inputs are unconstrained; consider "
            "--assume-known-inputs or full X-propagation support");
      }
    }

    // First, convert the block arguments of the miter bodies.
    if (failed(rewriter.convertRegionTypes(&adaptor.getFirstCircuit(),
                                           *typeConverter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&adaptor.getSecondCircuit(),
                                           *typeConverter)))
      return failure();

    // Second, create the symbolic values we replace the block arguments with
    SmallVector<Value> inputs;
    for (auto arg : adaptor.getFirstCircuit().getArguments()) {
      unsigned index = arg.getArgNumber();
      StringAttr namePrefix;
      if (inputNames && index < inputNames.size())
        namePrefix = dyn_cast_or_null<StringAttr>(inputNames[index]);
      Value decl =
          smt::DeclareFunOp::create(rewriter, loc, arg.getType(), namePrefix);
      if (assumeKnownInputs) {
        if (auto originalTy = getOriginalType(index))
          maybeAssertKnownInput(originalTy, decl, loc, rewriter);
      }
      inputs.push_back(decl);
    }

    // Third, inline the blocks
    // Note: the argument value replacement does not happen immediately, but
    // only after all the operations are already legalized.
    // Also, it has to be ensured that the original argument type and the type
    // of the value with which is is to be replaced match. The value is looked
    // up (transitively) in the replacement map at the time the replacement
    // pattern is committed.
    rewriter.mergeBlocks(&adaptor.getFirstCircuit().front(), solver.getBody(),
                         inputs);
    rewriter.mergeBlocks(&adaptor.getSecondCircuit().front(), solver.getBody(),
                         inputs);
    rewriter.setInsertionPointToEnd(solver.getBody());

    ArrayAttr outputNames = op->getAttrOfType<ArrayAttr>("lec.output_names");
    auto getOutputBaseName = [&](unsigned index) -> std::string {
      if (outputNames && index < outputNames.size()) {
        if (auto strAttr = dyn_cast_or_null<StringAttr>(outputNames[index])) {
          if (!strAttr.getValue().empty())
            return strAttr.getValue().str();
        }
      }
      return (Twine("out") + Twine(index)).str();
    };

    auto materializeOutput = [&](Value value) -> Value {
      auto targetTy = typeConverter->convertType(value.getType());
      if (!targetTy)
        return Value();
      return typeConverter->materializeTargetConversion(rewriter, loc, targetTy,
                                                        value);
    };

    SmallVector<Value> outputSymbolsA;
    SmallVector<Value> outputSymbolsB;
    for (auto [index, outPair] :
         llvm::enumerate(llvm::zip(firstOutputs->getOperands(),
                                   secondOutputs->getOperands()))) {
      auto [out1, out2] = outPair;
      Value out1Conv = materializeOutput(out1);
      Value out2Conv = materializeOutput(out2);
      if (!out1Conv || !out2Conv)
        return failure();
      std::string base = getOutputBaseName(index);
      auto nameA = rewriter.getStringAttr((Twine("c1_") + base).str());
      auto nameB = rewriter.getStringAttr((Twine("c2_") + base).str());
      Value symA =
          smt::DeclareFunOp::create(rewriter, loc, out1Conv.getType(), nameA);
      Value symB =
          smt::DeclareFunOp::create(rewriter, loc, out2Conv.getType(), nameB);
      outputSymbolsA.push_back(symA);
      outputSymbolsB.push_back(symB);
      smt::AssertOp::create(
          rewriter, loc, smt::EqOp::create(rewriter, loc, symA, out1Conv));
      smt::AssertOp::create(
          rewriter, loc, smt::EqOp::create(rewriter, loc, symB, out2Conv));
    }

    // Fourth, build the assertion.
    SmallVector<Value> outputsDifferent;
    if (!outputSymbolsA.empty() &&
        outputSymbolsA.size() == outputSymbolsB.size()) {
      for (auto [index, outPair] :
           llvm::enumerate(llvm::zip(outputSymbolsA, outputSymbolsB))) {
        auto [symA, symB] = outPair;
        if (xOptimisticOutputs && index < originalOutputTypes.size()) {
          if (Value diff = buildXOptimisticDiff(
                  symA, symB, originalOutputTypes[index], loc, rewriter)) {
            outputsDifferent.emplace_back(diff);
            continue;
          }
        }
        outputsDifferent.emplace_back(
            smt::DistinctOp::create(rewriter, loc, symA, symB));
      }
    } else {
      createOutputsDifferentOps(firstOutputs, secondOutputs, loc, rewriter,
                                outputsDifferent);
    }

    rewriter.eraseOp(firstOutputs);
    rewriter.eraseOp(secondOutputs);

    Value toAssert;
    if (outputsDifferent.empty()) {
      toAssert = smt::BoolConstantOp::create(rewriter, loc, false);
    } else if (outputsDifferent.size() == 1) {
      toAssert = outputsDifferent[0];
    } else {
      toAssert = smt::OrOp::create(rewriter, loc, outputsDifferent);
    }

    smt::AssertOp::create(rewriter, loc, toAssert);

    // Fifth, check for satisfiablility and report the result back.
    replaceOpWithSatCheck(op, loc, rewriter, solver);
    return success();
  }

private:
  bool assumeKnownInputs = false;
  bool xOptimisticOutputs = false;
};

struct RefinementCheckingOpConversion
    : CircuitRelationCheckOpConversion<verif::RefinementCheckingOp> {
  using CircuitRelationCheckOpConversion<
      verif::RefinementCheckingOp>::CircuitRelationCheckOpConversion;
  RefinementCheckingOpConversion(TypeConverter &converter, MLIRContext *context,
                                 bool assumeKnownInputs,
                                 bool /*xOptimisticOutputs*/)
      : CircuitRelationCheckOpConversion(converter, context),
        assumeKnownInputs(assumeKnownInputs) {}

  LogicalResult
  matchAndRewrite(verif::RefinementCheckingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Find non-deterministic values (free variables) in the source circuit.
    // For now, only support quantification over 'primitive' types.
    SmallVector<Value> srcNonDetValues;
    bool canBind = true;
    for (auto ndOp : op.getFirstCircuit().getOps<smt::DeclareFunOp>()) {
      if (!isa<smt::IntType, smt::BoolType, smt::BitVectorType>(
              ndOp.getType())) {
        ndOp.emitError("Uninterpreted function of non-primitive type cannot be "
                       "converted.");
        canBind = false;
      }
      srcNonDetValues.push_back(ndOp.getResult());
    }
    if (!canBind)
      return failure();

    if (srcNonDetValues.empty()) {
      // If there is no non-determinism in the source circuit, the
      // refinement check becomes an equivalence check, which does not
      // need quantified expressions.
      auto eqOp = verif::LogicEquivalenceCheckingOp::create(
          rewriter, op.getLoc(), op.getNumResults() != 0);
      rewriter.moveBlockBefore(&op.getFirstCircuit().front(),
                               &eqOp.getFirstCircuit(),
                               eqOp.getFirstCircuit().end());
      rewriter.moveBlockBefore(&op.getSecondCircuit().front(),
                               &eqOp.getSecondCircuit(),
                               eqOp.getSecondCircuit().end());
      rewriter.replaceOp(op, eqOp);
      return success();
    }

    Location loc = op.getLoc();
    auto *firstOutputs = adaptor.getFirstCircuit().front().getTerminator();
    auto *secondOutputs = adaptor.getSecondCircuit().front().getTerminator();

    auto hasNoResult = op.getNumResults() == 0;

    if (firstOutputs->getNumOperands() == 0) {
      // Trivially equivalent
      if (hasNoResult) {
        rewriter.eraseOp(op);
      } else {
        Value trueVal = arith::ConstantOp::create(rewriter, loc,
                                                  rewriter.getBoolAttr(true));
        rewriter.replaceOp(op, trueVal);
      }
      return success();
    }

    // Solver will only return a result when it is used to check the returned
    // value.
    smt::SolverOp solver;
    if (hasNoResult)
      solver = smt::SolverOp::create(rewriter, loc, TypeRange{}, ValueRange{});
    else
      solver = smt::SolverOp::create(rewriter, loc, rewriter.getI1Type(),
                                     ValueRange{});
    rewriter.createBlock(&solver.getBodyRegion());

    // Convert the block arguments of the miter bodies.
    if (failed(rewriter.convertRegionTypes(&adaptor.getFirstCircuit(),
                                           *typeConverter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&adaptor.getSecondCircuit(),
                                           *typeConverter)))
      return failure();

    // Create the symbolic values we replace the block arguments with
    SmallVector<Value> inputs;
    auto originalArgTypes = op.getFirstCircuit().getArgumentTypes();
    for (auto arg : adaptor.getFirstCircuit().getArguments()) {
      Value decl =
          smt::DeclareFunOp::create(rewriter, loc, arg.getType(), StringAttr{});
      if (assumeKnownInputs && arg.getArgNumber() < originalArgTypes.size())
        maybeAssertKnownInput(originalArgTypes[arg.getArgNumber()], decl, loc,
                              rewriter);
      inputs.push_back(decl);
    }

    // Inline the target circuit. Free variables remain free variables.
    rewriter.mergeBlocks(&adaptor.getSecondCircuit().front(), solver.getBody(),
                         inputs);
    rewriter.setInsertionPointToEnd(solver.getBody());

    // Create the universally quantified expression containing the source
    // circuit. Free variables in the circuit's body become bound variables.
    auto forallOp = smt::ForallOp::create(
        rewriter, op.getLoc(), TypeRange(srcNonDetValues),
        [&](OpBuilder &builder, auto, ValueRange args) -> Value {
          // Inline the source circuit
          Block *body = builder.getBlock();
          rewriter.mergeBlocks(&adaptor.getFirstCircuit().front(), body,
                               inputs);

          // Replace non-deterministic values with the quantifier's bound
          // variables
          for (auto [freeVar, boundVar] : llvm::zip(srcNonDetValues, args))
            rewriter.replaceOp(freeVar.getDefiningOp(), boundVar);

          // Compare the output values
          rewriter.setInsertionPointToEnd(body);
          SmallVector<Value> outputsDifferent;
          createOutputsDifferentOps(firstOutputs, secondOutputs, loc, rewriter,
                                    outputsDifferent);
          if (outputsDifferent.size() == 1)
            return outputsDifferent[0];
          else
            return rewriter.createOrFold<smt::OrOp>(loc, outputsDifferent);
        });

    rewriter.eraseOp(firstOutputs);
    rewriter.eraseOp(secondOutputs);

    // Assert the quantified expression
    rewriter.setInsertionPointAfter(forallOp);
    smt::AssertOp::create(rewriter, op.getLoc(), forallOp.getResult());

    // Check for satisfiability and report the result back.
    replaceOpWithSatCheck(op, loc, rewriter, solver);
    return success();
  }

private:
  bool assumeKnownInputs = false;
};

/// Information about a ltl.delay operation that needs multi-step tracking.
/// For delay N, we need N slots in the delay buffer to track the signal history.
struct DelayInfo {
  ltl::DelayOp op;           // The original delay operation
  Value inputSignal;         // The signal being delayed
  uint64_t delay;            // The delay amount (N cycles)
  uint64_t length;           // Additional range length (0 for exact delay)
  uint64_t bufferSize;       // Total buffer slots (delay + length)
  size_t bufferStartIndex;   // Index into the delay buffer iter_args
  StringAttr clockName;      // Optional clock name for buffer updates
  std::optional<ltl::ClockEdge> edge; // Optional clock edge for buffer updates
  Value clockValue;          // Optional i1 clock value for buffer updates
};

/// Information about a ltl.past operation that needs multi-step tracking.
/// For past N, we need N slots in the past buffer to track the signal history.
/// This is used for $rose/$fell which look at the previous cycle's value.
struct PastInfo {
  ltl::PastOp op;            // The original past operation
  Value inputSignal;         // The signal being observed in the past
  uint64_t delay;            // How many cycles ago (N)
  uint64_t bufferSize;       // Buffer slots needed (same as delay)
  size_t bufferStartIndex;   // Index into the past buffer iter_args
  StringAttr clockName;      // Optional clock name for buffer updates
  std::optional<ltl::ClockEdge> edge; // Optional clock edge for buffer updates
  Value clockValue;          // Optional i1 clock value for buffer updates
};

/// Information about an NFA sequence lowered for BMC tracking.
struct SequenceNFAInfo {
  Value root;                // Root sequence value
  size_t stateStartIndex;    // Index into the NFA state slot vector
  size_t numStates;          // Number of NFA states
  unsigned tickArgIndex;     // Circuit argument index for the tick input
  StringAttr clockName;      // Optional clock name for tick gating
  std::optional<ltl::ClockEdge> edge; // Optional clock edge for tick gating
  Value clockValue;          // Optional i1 clock value for tick gating
};

struct NFATickGateInfo {
  unsigned tickArgIndex = 0;
  std::optional<unsigned> clockPos;
  bool invert = false;
  ltl::ClockEdge edge = ltl::ClockEdge::Pos;
  bool hasExplicitClock = false;
};

static void collectSequenceOpsForBMC(Value seq,
                                     llvm::DenseSet<Operation *> &ops,
                                     llvm::DenseSet<Value> &visited) {
  if (!seq || !isa<ltl::SequenceType>(seq.getType()))
    return;
  if (!visited.insert(seq).second)
    return;
  Operation *def = seq.getDefiningOp();
  if (!def || !isa_and_nonnull<ltl::LTLDialect>(def->getDialect()))
    return;
  // Past ops are lowered via dedicated past-buffer slots, not NFAs. Avoid
  // tracking them here to prevent double-erasure when the past ops are already
  // scheduled for deletion during past-buffer setup.
  if (isa<ltl::PastOp>(def))
    return;
  ops.insert(def);
  for (Value operand : def->getOperands())
    if (isa<ltl::SequenceType>(operand.getType()))
      collectSequenceOpsForBMC(operand, ops, visited);
}

/// Rewrite ltl.implication with an exact delayed consequent into a past-form
/// implication for BMC. This allows the BMC loop to use past buffers instead
/// of looking ahead into future time steps.
static void rewriteImplicationDelaysForBMC(Block &circuitBlock,
                                           RewriterBase &rewriter) {
  SmallVector<ltl::ImplicationOp> implicationOps;
  circuitBlock.walk(
      [&](ltl::ImplicationOp op) { implicationOps.push_back(op); });

  for (auto implOp : implicationOps) {
    auto delayOp = implOp.getConsequent().getDefiningOp<ltl::DelayOp>();
    if (!delayOp)
      continue;
    auto inputTy = delayOp.getInput().getType();
    auto intTy = dyn_cast<IntegerType>(inputTy);
    if (!intTy || intTy.getWidth() != 1)
      continue;
    uint64_t delay = delayOp.getDelay();
    if (delay == 0)
      continue;
    auto lengthAttr = delayOp.getLengthAttr();
    if (!lengthAttr || lengthAttr.getValue().getZExtValue() != 0)
      continue;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(implOp);
    auto shiftedAntecedent = ltl::DelayOp::create(
        rewriter, implOp.getLoc(), implOp.getAntecedent(), delay,
        rewriter.getI64IntegerAttr(0));
    if (auto clockAttr = delayOp->getAttr("bmc.clock"))
      shiftedAntecedent->setAttr("bmc.clock", clockAttr);
    if (auto edgeAttr = delayOp->getAttr("bmc.clock_edge"))
      shiftedAntecedent->setAttr("bmc.clock_edge", edgeAttr);
    implOp.setOperand(0, shiftedAntecedent.getResult());
    implOp.setOperand(1, delayOp.getInput());

    // The original delay is now bypassed. Erase it if it became dead so it
    // does not survive into the later LTL->SMT lowering phase.
    if (delayOp.use_empty())
      rewriter.eraseOp(delayOp);
  }
}

/// Expand ltl.repeat into explicit delay/and/or sequences inside a BMC circuit.
[[maybe_unused]] static void expandRepeatOpsInBMC(
    verif::BoundedModelCheckingOp bmcOp, RewriterBase &rewriter) {
  auto &circuitBlock = bmcOp.getCircuit().front();
  SmallVector<ltl::RepeatOp> repeatOps;
  circuitBlock.walk([&](ltl::RepeatOp repeatOp) { repeatOps.push_back(repeatOp); });
  if (repeatOps.empty())
    return;

  uint64_t boundValue = bmcOp.getBound();

  for (auto repeatOp : repeatOps) {
    uint64_t base = repeatOp.getBase();
    if (base == 0)
      continue;

    uint64_t maxRepeat = base;
    if (auto moreAttr = repeatOp.getMoreAttr()) {
      maxRepeat = base + moreAttr.getValue().getZExtValue();
    } else if (boundValue > base) {
      maxRepeat = boundValue;
    }

    if (maxRepeat < base)
      continue;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(repeatOp);

    auto buildExactRepeat = [&](uint64_t count) -> Value {
      SmallVector<Value> terms;
      terms.reserve(count);
      for (uint64_t i = 0; i < count; ++i) {
        if (i == 0) {
          terms.push_back(repeatOp.getInput());
          continue;
        }
        auto delayOp = ltl::DelayOp::create(rewriter, repeatOp.getLoc(),
                                            repeatOp.getInput(), i,
                                            rewriter.getI64IntegerAttr(0));
        terms.push_back(delayOp.getResult());
      }
      if (terms.size() == 1) {
        if (isa<ltl::SequenceType>(terms[0].getType()))
          return terms[0];
        auto delayOp = ltl::DelayOp::create(rewriter, repeatOp.getLoc(),
                                            terms[0], 0,
                                            rewriter.getI64IntegerAttr(0));
        return delayOp.getResult();
      }
      return ltl::AndOp::create(rewriter, repeatOp.getLoc(), terms)
          .getResult();
    };

    SmallVector<Value> choices;
    for (uint64_t count = base; count <= maxRepeat; ++count)
      choices.push_back(buildExactRepeat(count));

    Value replacement =
        choices.size() == 1
            ? choices.front()
            : ltl::OrOp::create(rewriter, repeatOp.getLoc(), choices)
                  .getResult();

    rewriter.replaceOp(repeatOp, replacement);
  }
}

struct SequenceLengthBounds {
  uint64_t min = 0;
  uint64_t max = 0;
};

static bool safeAdd(uint64_t a, uint64_t b, uint64_t &out) {
  if (a > std::numeric_limits<uint64_t>::max() - b)
    return false;
  out = a + b;
  return true;
}

static bool safeMul(uint64_t a, uint64_t b, uint64_t &out) {
  if (a == 0 || b == 0) {
    out = 0;
    return true;
  }
  if (a > std::numeric_limits<uint64_t>::max() / b)
    return false;
  out = a * b;
  return true;
}

static std::optional<SequenceLengthBounds>
getSequenceLengthBoundsForBMC(Value seq, uint64_t boundValue) {
  if (!seq)
    return SequenceLengthBounds{0, 0};
  if (!isa<ltl::SequenceType>(seq.getType()))
    return SequenceLengthBounds{1, 1};

  Operation *def = seq.getDefiningOp();
  if (!def)
    return std::nullopt;

  if (auto clockOp = dyn_cast<ltl::ClockOp>(def))
    return getSequenceLengthBoundsForBMC(clockOp.getInput(), boundValue);
  if (auto pastOp = dyn_cast<ltl::PastOp>(def))
    return getSequenceLengthBoundsForBMC(pastOp.getInput(), boundValue);
  if (auto notOp = dyn_cast<ltl::NotOp>(def))
    return getSequenceLengthBoundsForBMC(notOp.getInput(), boundValue);
  if (auto delayOp = dyn_cast<ltl::DelayOp>(def)) {
    auto inputBounds =
        getSequenceLengthBoundsForBMC(delayOp.getInput(), boundValue);
    if (!inputBounds)
      return std::nullopt;
    uint64_t minDelay = delayOp.getDelay();
    uint64_t maxDelay = minDelay;
    if (auto lengthAttr = delayOp.getLengthAttr()) {
      uint64_t length = lengthAttr.getValue().getZExtValue();
      if (!safeAdd(minDelay, length, maxDelay))
        return std::nullopt;
    } else if (boundValue > 0 && minDelay < boundValue) {
      uint64_t length = boundValue - 1 - minDelay;
      if (!safeAdd(minDelay, length, maxDelay))
        return std::nullopt;
    }
    uint64_t minLen = 0;
    uint64_t maxLen = 0;
    if (!safeAdd(inputBounds->min, minDelay, minLen))
      return std::nullopt;
    if (!safeAdd(inputBounds->max, maxDelay, maxLen))
      return std::nullopt;
    return SequenceLengthBounds{minLen, maxLen};
  }
  if (auto concatOp = dyn_cast<ltl::ConcatOp>(def)) {
    uint64_t minLen = 0;
    uint64_t maxLen = 0;
    for (Value input : concatOp.getInputs()) {
      auto bounds = getSequenceLengthBoundsForBMC(input, boundValue);
      if (!bounds)
        return std::nullopt;
      if (!safeAdd(minLen, bounds->min, minLen))
        return std::nullopt;
      if (!safeAdd(maxLen, bounds->max, maxLen))
        return std::nullopt;
    }
    return SequenceLengthBounds{minLen, maxLen};
  }
  if (auto repeatOp = dyn_cast<ltl::RepeatOp>(def)) {
    auto bounds =
        getSequenceLengthBoundsForBMC(repeatOp.getInput(), boundValue);
    if (!bounds)
      return std::nullopt;
    uint64_t base = repeatOp.getBase();
    uint64_t maxRepeat = base;
    if (auto moreAttr = repeatOp.getMoreAttr()) {
      uint64_t more = moreAttr.getValue().getZExtValue();
      if (!safeAdd(base, more, maxRepeat))
        return std::nullopt;
    } else if (boundValue > base) {
      maxRepeat = boundValue;
    }
    uint64_t minLen = 0;
    uint64_t maxLen = 0;
    if (!safeMul(bounds->min, base, minLen))
      return std::nullopt;
    if (!safeMul(bounds->max, maxRepeat, maxLen))
      return std::nullopt;
    return SequenceLengthBounds{minLen, maxLen};
  }
  if (auto orOp = dyn_cast<ltl::OrOp>(def)) {
    std::optional<uint64_t> minLen;
    std::optional<uint64_t> maxLen;
    for (Value input : orOp.getInputs()) {
      auto bounds = getSequenceLengthBoundsForBMC(input, boundValue);
      if (!bounds)
        return std::nullopt;
      if (!minLen) {
        minLen = bounds->min;
        maxLen = bounds->max;
      } else {
        minLen = std::min(*minLen, bounds->min);
        maxLen = std::max(*maxLen, bounds->max);
      }
    }
    if (!minLen || !maxLen)
      return std::nullopt;
    return SequenceLengthBounds{*minLen, *maxLen};
  }
  if (auto andOp = dyn_cast<ltl::AndOp>(def)) {
    std::optional<uint64_t> minLen;
    std::optional<uint64_t> maxLen;
    for (Value input : andOp.getInputs()) {
      auto bounds = getSequenceLengthBoundsForBMC(input, boundValue);
      if (!bounds)
        return std::nullopt;
      if (!minLen) {
        minLen = bounds->min;
        maxLen = bounds->max;
      } else {
        minLen = std::max(*minLen, bounds->min);
        maxLen = std::max(*maxLen, bounds->max);
      }
    }
    if (!minLen || !maxLen)
      return std::nullopt;
    return SequenceLengthBounds{*minLen, *maxLen};
  }
  if (auto intersectOp = dyn_cast<ltl::IntersectOp>(def)) {
    std::optional<uint64_t> minLen;
    std::optional<uint64_t> maxLen;
    for (Value input : intersectOp.getInputs()) {
      auto bounds = getSequenceLengthBoundsForBMC(input, boundValue);
      if (!bounds)
        return std::nullopt;
      if (!minLen) {
        minLen = bounds->min;
        maxLen = bounds->max;
      } else {
        minLen = std::max(*minLen, bounds->min);
        maxLen = std::min(*maxLen, bounds->max);
      }
    }
    if (!minLen || !maxLen || *minLen > *maxLen)
      return std::nullopt;
    return SequenceLengthBounds{*minLen, *maxLen};
  }
  if (auto firstMatch = dyn_cast<ltl::FirstMatchOp>(def))
    return getSequenceLengthBoundsForBMC(firstMatch.getInput(), boundValue);

  return std::nullopt;
}

static Value buildSequenceConstant(OpBuilder &builder, Location loc,
                                   bool value) {
  auto cst = hw::ConstantOp::create(builder, loc, builder.getI1Type(),
                                    value ? 1 : 0);
  auto zero = builder.getI64IntegerAttr(0);
  return ltl::DelayOp::create(builder, loc, cst, zero, zero).getResult();
}

static Value buildSequenceAnd(OpBuilder &builder, Location loc,
                              ArrayRef<Value> inputs) {
  if (inputs.empty())
    return buildSequenceConstant(builder, loc, true);
  if (inputs.size() == 1)
    return inputs.front();
  return ltl::AndOp::create(builder, loc, inputs).getResult();
}

static Value buildSequenceOr(OpBuilder &builder, Location loc,
                             ArrayRef<Value> inputs) {
  if (inputs.empty())
    return buildSequenceConstant(builder, loc, false);
  if (inputs.size() == 1)
    return inputs.front();
  return ltl::OrOp::create(builder, loc, inputs).getResult();
}

[[maybe_unused]] static LogicalResult expandConcatOpsInBMC(
    verif::BoundedModelCheckingOp bmcOp, RewriterBase &rewriter) {
  auto &circuitBlock = bmcOp.getCircuit().front();
  SmallVector<ltl::ConcatOp> concatOps;
  circuitBlock.walk([&](ltl::ConcatOp concatOp) {
    concatOps.push_back(concatOp);
  });
  if (concatOps.empty())
    return success();

  uint64_t boundValue = bmcOp.getBound();

  for (auto concatOp : concatOps) {
    auto inputs = concatOp.getInputs();
    if (inputs.size() < 2)
      continue;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(concatOp);
    Location loc = concatOp.getLoc();

    SmallVector<SequenceLengthBounds> bounds;
    bounds.reserve(inputs.size());
    bool hasBounds = true;
    for (Value input : inputs) {
      auto maybeBounds = getSequenceLengthBoundsForBMC(input, boundValue);
      if (!maybeBounds) {
        hasBounds = false;
        break;
      }
      bounds.push_back(*maybeBounds);
    }
    if (!hasBounds) {
      concatOp.emitError("concat expansion requires bounded sequence lengths");
      return failure();
    }

    constexpr uint64_t kMaxCombos = 16384;
    uint64_t comboCount = 1;
    for (size_t idx = 0; idx + 1 < bounds.size(); ++idx) {
      auto rangeSize = bounds[idx].max - bounds[idx].min + 1;
      if (rangeSize == 0)
        rangeSize = 1;
      if (comboCount > kMaxCombos / rangeSize) {
        concatOp.emitError("concat expansion too large; reduce the BMC bound "
                           "or simplify the sequence ranges");
        return failure();
      }
      comboCount *= rangeSize;
    }
    bool lastCanBeEmpty = bounds.back().min == 0;
    bool lastCanBeNonEmpty = bounds.back().max > 0;
    if (lastCanBeEmpty && lastCanBeNonEmpty &&
        comboCount > kMaxCombos / 2) {
      concatOp.emitError("concat expansion too large; reduce the BMC bound "
                         "or simplify the sequence ranges");
      return failure();
    }

    SmallVector<Value> choices;
    SmallVector<Value> terms;
    terms.reserve(inputs.size());

    std::function<void(size_t, uint64_t)> expandPrefixes =
        [&](size_t idx, uint64_t offset) {
          if (idx + 1 == inputs.size()) {
            auto addChoice = [&](bool includeLast) {
              if (includeLast) {
                Value term = inputs.back();
                if (offset > 0) {
                  auto delayAttr = rewriter.getI64IntegerAttr(offset);
                  auto zeroAttr = rewriter.getI64IntegerAttr(0);
                  term =
                      ltl::DelayOp::create(rewriter, loc, term, delayAttr,
                                           zeroAttr)
                          .getResult();
                }
                terms.push_back(term);
              }
              choices.push_back(buildSequenceAnd(rewriter, loc, terms));
              if (includeLast)
                terms.pop_back();
            };
            if (lastCanBeEmpty)
              addChoice(false);
            if (lastCanBeNonEmpty)
              addChoice(true);
            return;
          }

          auto range = bounds[idx];
          for (uint64_t len = range.min; len <= range.max; ++len) {
            bool includeTerm = len != 0;
            if (includeTerm) {
              Value term = inputs[idx];
              if (offset > 0) {
                auto delayAttr = rewriter.getI64IntegerAttr(offset);
                auto zeroAttr = rewriter.getI64IntegerAttr(0);
                term =
                    ltl::DelayOp::create(rewriter, loc, term, delayAttr,
                                         zeroAttr)
                        .getResult();
              }
              terms.push_back(term);
            }
            uint64_t increment = 0;
            if (len > 0)
              increment = len - 1;
            expandPrefixes(idx + 1, offset + increment);
            if (includeTerm)
              terms.pop_back();
            if (len == range.max)
              break;
          }
        };

    expandPrefixes(0, 0);
    Value replacement = buildSequenceOr(rewriter, loc, choices);
    rewriter.replaceOp(concatOp, replacement);
  }

  return success();
}

static bool exceedsCombinationLimit(uint64_t n, uint64_t k, uint64_t limit) {
  if (k > n)
    return false;
  if (k > n - k)
    k = n - k;
  uint64_t count = 1;
  for (uint64_t i = 1; i <= k; ++i) {
    uint64_t numerator = n - k + i;
    if (count > limit / numerator)
      return true;
    count *= numerator;
    count /= i;
    if (count > limit)
      return true;
  }
  return count > limit;
}

[[maybe_unused]] static LogicalResult
expandGotoRepeatOpsInBMC(verif::BoundedModelCheckingOp bmcOp,
                         RewriterBase &rewriter) {
  auto &circuitBlock = bmcOp.getCircuit().front();
  SmallVector<ltl::GoToRepeatOp> gotoOps;
  SmallVector<ltl::NonConsecutiveRepeatOp> nonConsecOps;
  circuitBlock.walk([&](ltl::GoToRepeatOp op) { gotoOps.push_back(op); });
  circuitBlock.walk(
      [&](ltl::NonConsecutiveRepeatOp op) { nonConsecOps.push_back(op); });
  if (gotoOps.empty() && nonConsecOps.empty())
    return success();

  uint64_t boundValue = bmcOp.getBound();
  uint64_t maxOffset = boundValue > 0 ? boundValue - 1 : 0;
  constexpr uint64_t kMaxCombos = 16384;

  auto expandOp = [&](auto op, bool requireCurrentMatch) -> LogicalResult {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    uint64_t base = op.getBase();
    uint64_t maxCount = base;
    if (auto more = op.getMore())
      maxCount = base + *more;
    else if (boundValue > base)
      maxCount = boundValue;

    auto delayZero = rewriter.getI64IntegerAttr(0);
    Value input = op.getInput();
    Value seqInput = input;
    if (!isa<ltl::SequenceType>(input.getType()))
      seqInput =
          ltl::DelayOp::create(rewriter, loc, input, delayZero, delayZero)
              .getResult();

    SmallVector<Value> offsetValues;
    offsetValues.reserve(maxOffset + 1);
    offsetValues.push_back(seqInput);
    for (uint64_t offset = 1; offset <= maxOffset; ++offset) {
      offsetValues.push_back(ltl::DelayOp::create(rewriter, loc, seqInput,
                                                 rewriter.getI64IntegerAttr(
                                                     offset),
                                                 delayZero)
                                  .getResult());
    }

    SmallVector<Value> choices;
    for (uint64_t count = base; count <= maxCount; ++count) {
      if (count == 0) {
        choices.push_back(buildSequenceConstant(rewriter, loc, true));
        continue;
      }
      if (requireCurrentMatch && count > 0 && maxOffset + 1 < count)
        continue;
      if (!requireCurrentMatch && maxOffset + 1 < count)
        continue;

      uint64_t remaining = requireCurrentMatch ? count - 1 : count;
      uint64_t available = requireCurrentMatch ? maxOffset : maxOffset + 1;
      if (exceedsCombinationLimit(available, remaining, kMaxCombos))
        return op.emitError("goto/non-consecutive repeat expansion too large; "
                            "reduce the BMC bound or repetition range");

      SmallVector<Value> current;
      if (requireCurrentMatch)
        current.push_back(offsetValues[0]);
      auto buildCombos = [&](uint64_t start, uint64_t need,
                             auto &&buildCombosRef) -> void {
        if (need == 0) {
          choices.push_back(buildSequenceAnd(rewriter, loc, current));
          return;
        }
        for (uint64_t idx = start; idx + need <= maxOffset + 1; ++idx) {
          current.push_back(offsetValues[idx]);
          buildCombosRef(idx + 1, need - 1, buildCombosRef);
          current.pop_back();
        }
      };
      buildCombos(requireCurrentMatch ? 1 : 0, remaining, buildCombos);
    }

    Value replacement = buildSequenceOr(rewriter, loc, choices);
    rewriter.replaceOp(op, replacement);
    return success();
  };

  for (auto op : gotoOps)
    if (failed(expandOp(op, /*requireCurrentMatch=*/true)))
      return failure();
  for (auto op : nonConsecOps)
    if (failed(expandOp(op, /*requireCurrentMatch=*/false)))
      return failure();
  return success();
}

/// Lower a verif::BMCOp operation to an MLIR program that performs the bounded
/// model check
struct VerifBoundedModelCheckingOpConversion
    : OpConversionPattern<verif::BoundedModelCheckingOp> {
  using OpConversionPattern<verif::BoundedModelCheckingOp>::OpConversionPattern;

  VerifBoundedModelCheckingOpConversion(
      TypeConverter &converter, MLIRContext *context, Namespace &names,
      bool risingClocksOnly, bool assumeKnownInputs, bool forSMTLIBExport,
      BMCCheckMode bmcMode,
      SmallVectorImpl<Operation *> &propertylessBMCOps,
      SmallVectorImpl<Operation *> &coverBMCOps)
      : OpConversionPattern(converter, context), names(names),
        risingClocksOnly(risingClocksOnly),
        assumeKnownInputs(assumeKnownInputs),
        forSMTLIBExport(forSMTLIBExport),
        bmcMode(bmcMode),
        propertylessBMCOps(propertylessBMCOps), coverBMCOps(coverBMCOps) {}
  LogicalResult
  matchAndRewrite(verif::BoundedModelCheckingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool emitSMTLIB = forSMTLIBExport;
    bool inductionStep = bmcMode == BMCCheckMode::InductionStep;
    bool livenessMode = bmcMode == BMCCheckMode::Liveness ||
                        bmcMode == BMCCheckMode::LivenessLasso;
    bool livenessLassoMode = bmcMode == BMCCheckMode::LivenessLasso;

    if (inductionStep && !emitSMTLIB) {
      op.emitError("k-induction currently requires SMT-LIB export "
                   "(use --run-smtlib)");
      return failure();
    }
    if (livenessLassoMode && !emitSMTLIB) {
      op.emitError("liveness-lasso mode currently requires SMT-LIB export "
                   "(use --emit-smtlib or --run-smtlib)");
      return failure();
    }

    bool isCoverCheck =
        std::find(coverBMCOps.begin(), coverBMCOps.end(), op) !=
        coverBMCOps.end();

    if (std::find(propertylessBMCOps.begin(), propertylessBMCOps.end(), op) !=
        propertylessBMCOps.end()) {
      // No properties to check, so we don't bother solving, we just return true
      // (without this we would incorrectly find violations, since the solver
      // will always return SAT)
      Value trueVal =
          arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
      rewriter.replaceOp(op, trueVal);
      return success();
    }

    auto &circuitBlock = op.getCircuit().front();
    struct ClockEquivalenceUF {
      DenseMap<Value, Value> parent;
      DenseMap<Value, bool> parity;
      bool conflict = false;

      bool has(Value value) const { return parent.contains(value); }

      Value find(Value value, bool &invert) {
        auto it = parent.find(value);
        if (it == parent.end()) {
          parent.try_emplace(value, value);
          parity.try_emplace(value, false);
          invert = false;
          return value;
        }
        Value root = it->second;
        if (root == value) {
          invert = parity.lookup(value);
          return root;
        }
        bool parentInvert = false;
        Value newRoot = find(root, parentInvert);
        bool newInvert = parity.lookup(value) ^ parentInvert;
        parent[value] = newRoot;
        parity[value] = newInvert;
        invert = newInvert;
        return newRoot;
      }

      void unite(Value lhs, Value rhs, bool invert) {
        if (conflict)
          return;
        bool lhsInvert = false;
        bool rhsInvert = false;
        Value lhsRoot = find(lhs, lhsInvert);
        Value rhsRoot = find(rhs, rhsInvert);
        if (lhsRoot == rhsRoot) {
          if ((lhsInvert ^ rhsInvert) != invert)
            conflict = true;
          return;
        }
        parent[rhsRoot] = lhsRoot;
        parity[rhsRoot] = lhsInvert ^ rhsInvert ^ invert;
      }
    };
    ClockEquivalenceUF clockEquivalenceUF;

    auto getConstI1Value = [&](Value val) -> std::optional<bool> {
      if (!val)
        return std::nullopt;
      if (auto cst = val.getDefiningOp<hw::ConstantOp>()) {
        if (auto intTy = dyn_cast<IntegerType>(cst.getType());
            intTy && intTy.getWidth() == 1)
          return cst.getValue().isAllOnes();
        return std::nullopt;
      }
      if (auto cst = val.getDefiningOp<arith::ConstantOp>()) {
        if (auto boolAttr = dyn_cast<BoolAttr>(cst.getValue()))
          return boolAttr.getValue();
        if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue())) {
          if (auto intTy = dyn_cast<IntegerType>(intAttr.getType());
              intTy && intTy.getWidth() == 1)
            return intAttr.getValue().isAllOnes();
        }
      }
      return std::nullopt;
    };

    auto isI1Value = [](Value val) -> bool {
      if (!val)
        return false;
      if (auto intTy = dyn_cast<IntegerType>(val.getType()))
        return intTy.getWidth() == 1;
      return false;
    };

    // The BMC multi-step infrastructure can only lower sequences that are fully
    // defined by LTL IR. In particular, ltl.sequence-typed block arguments are
    // not representable and will cause NFA construction to fail (and may emit
    // diagnostics without signaling pass failure). Detect this early so the
    // tool exits with a failure code.
    auto containsSequenceBlockArgument = [&](Value root) -> bool {
      SmallVector<Operation *> worklist;
      DenseSet<Operation *> visited;

      if (!root)
        return false;
      if (isa<ltl::SequenceType>(root.getType()) && isa<BlockArgument>(root))
        return true;

      if (Operation *def = root.getDefiningOp())
        worklist.push_back(def);

      while (!worklist.empty()) {
        Operation *cur = worklist.pop_back_val();
        if (!cur || !isa_and_nonnull<ltl::LTLDialect>(cur->getDialect()))
          continue;
        if (!visited.insert(cur).second)
          continue;
        for (Value operand : cur->getOperands()) {
          if (!isa<ltl::PropertyType, ltl::SequenceType>(operand.getType()))
            continue;
          if (isa<ltl::SequenceType>(operand.getType()) &&
              isa<BlockArgument>(operand))
            return true;
          if (Operation *def = operand.getDefiningOp())
            worklist.push_back(def);
        }
      }
      return false;
    };

    bool foundSequenceBlockArg = false;
    circuitBlock.walk([&](verif::AssertOp assertOp) {
      foundSequenceBlockArg |=
          containsSequenceBlockArgument(assertOp.getProperty());
    });
    circuitBlock.walk([&](verif::AssumeOp assumeOp) {
      foundSequenceBlockArg |=
          containsSequenceBlockArgument(assumeOp.getProperty());
    });
    circuitBlock.walk([&](verif::CoverOp coverOp) {
      foundSequenceBlockArg |=
          containsSequenceBlockArgument(coverOp.getProperty());
    });
    if (foundSequenceBlockArg) {
      op.emitError("unsupported sequence lowering for block argument");
      return failure();
    }

    bool derivedClockConflict = false;
    auto reportDerivedClockConflict = [&](Operation *context,
                                          StringRef message) {
      if (derivedClockConflict)
        return;
      if (context)
        context->emitError(message);
      else
        op.emitError(message);
      derivedClockConflict = true;
    };
    // Sequence NFAs provide exact multi-step semantics for repeat/concat/goto
    // constructs, so we avoid expanding them into delay buffers here.

    // Clone LTL subtrees per property when they are shared across assertions.
    // This avoids accidental clock domain sharing for delay/past buffers.
    auto cloneSharedLTLSubtrees = [&]() {
      struct PropRef {
        Operation *op;
        Value prop;
      };
      SmallVector<PropRef> props;
      circuitBlock.walk([&](verif::AssertOp assertOp) {
        props.push_back({assertOp.getOperation(), assertOp.getProperty()});
      });
      circuitBlock.walk([&](verif::AssumeOp assumeOp) {
        props.push_back({assumeOp.getOperation(), assumeOp.getProperty()});
      });
      circuitBlock.walk([&](verif::CoverOp coverOp) {
        props.push_back({coverOp.getOperation(), coverOp.getProperty()});
      });
      if (props.empty())
        return;

      DenseMap<Operation *, llvm::SmallDenseSet<unsigned, 4>> opUseMap;
      for (auto [idx, prop] : llvm::enumerate(props)) {
        DenseSet<Operation *> visited;
        SmallVector<Operation *> worklist;
        if (Operation *def = prop.prop.getDefiningOp())
          worklist.push_back(def);
        while (!worklist.empty()) {
          Operation *cur = worklist.pop_back_val();
          if (!cur || !isa_and_nonnull<ltl::LTLDialect>(cur->getDialect()))
            continue;
          if (!visited.insert(cur).second)
            continue;
          opUseMap[cur].insert(idx);
          for (Value operand : cur->getOperands()) {
            if (Operation *def = operand.getDefiningOp())
              worklist.push_back(def);
          }
        }
      }

      SmallVector<bool> needsClone(props.size(), false);
      for (auto &entry : opUseMap) {
        if (entry.second.size() <= 1)
          continue;
        for (unsigned idx : entry.second)
          needsClone[idx] = true;
      }

      auto cloneLTLSubtree = [&](Value root, Operation *insertBefore) -> Value {
        Operation *def = root.getDefiningOp();
        if (!def || !isa_and_nonnull<ltl::LTLDialect>(def->getDialect()))
          return root;
        DenseSet<Operation *> visited;
        SmallVector<Operation *> postorder;
        std::function<void(Operation *)> dfs = [&](Operation *cur) {
          if (!cur || !isa_and_nonnull<ltl::LTLDialect>(cur->getDialect()))
            return;
          if (!visited.insert(cur).second)
            return;
          for (Value operand : cur->getOperands()) {
            if (Operation *defOp = operand.getDefiningOp())
              dfs(defOp);
          }
          postorder.push_back(cur);
        };
        dfs(def);
        OpBuilder builder(insertBefore);
        IRMapping mapping;
        for (Operation *cur : postorder) {
          Operation *cloned = builder.clone(*cur, mapping);
          for (auto [oldResult, newResult] :
               llvm::zip(cur->getResults(), cloned->getResults()))
            mapping.map(oldResult, newResult);
        }
        return mapping.lookupOrDefault(root);
      };

      for (auto [idx, prop] : llvm::enumerate(props)) {
        if (!needsClone[idx])
          continue;
        Value cloned = cloneLTLSubtree(prop.prop, prop.op);
        if (cloned == prop.prop)
          continue;
        prop.op->setOperand(0, cloned);
      }
    };

    cloneSharedLTLSubtrees();

    // Cloning LTL subtrees can leave the original LTL ops dead. Since the
    // second conversion phase marks many multi-step LTL ops illegal (and strict
    // lowering may refuse approximations), proactively erase any now-dead LTL
    // ops to avoid spurious legalization failures.
    auto eraseDeadLTLOps = [&]() {
      SmallVector<Operation *> ordered;
      circuitBlock.walk([&](Operation *cur) {
        if (isa_and_nonnull<ltl::LTLDialect>(cur->getDialect()))
          ordered.push_back(cur);
      });
      bool changed = true;
      while (changed) {
        changed = false;
        for (int64_t i = static_cast<int64_t>(ordered.size()) - 1; i >= 0;
             --i) {
          Operation *cur = ordered[static_cast<size_t>(i)];
          if (!cur)
            continue;
          if (!cur->use_empty())
            continue;
          rewriter.eraseOp(cur);
          ordered[static_cast<size_t>(i)] = nullptr;
          changed = true;
        }
      }
    };
    eraseDeadLTLOps();

    rewriteImplicationDelaysForBMC(circuitBlock, rewriter);

    struct ClockInfo {
      StringAttr clockName;
      std::optional<ltl::ClockEdge> edge;
      Value clockValue;
      bool seen = false;
    };

    DenseMap<Operation *, ClockInfo> ltlClockInfo;

    auto mergeClockInfo = [&](ClockInfo &dst, const ClockInfo &src) {
      dst.seen |= src.seen;
      if (src.clockName) {
        if (!dst.clockName)
          dst.clockName = src.clockName;
        else if (dst.clockName != src.clockName)
          dst.clockName = StringAttr{};
      }
      if (src.edge) {
        if (!dst.edge)
          dst.edge = src.edge;
        else if (dst.edge != src.edge)
          dst.edge.reset();
      }
      if (src.clockValue) {
        if (!dst.clockValue)
          dst.clockValue = src.clockValue;
        else if (dst.clockValue != src.clockValue)
          dst.clockValue = Value{};
      }
    };

    auto hasClockContext = [&](const ClockInfo &info) {
      return info.clockName || info.clockValue;
    };

    auto effectiveEdge =
        [&](const ClockInfo &info) -> std::optional<ltl::ClockEdge> {
      if (!hasClockContext(info))
        return std::nullopt;
      // Treat an unspecified edge as "unknown" here. Defaults (posedge) are
      // applied later, once all clock information has been merged.
      return info.edge;
    };

    auto clockInfoConflict = [&](const ClockInfo &a, const ClockInfo &b) {
      if (!a.seen || !b.seen)
        return false;
      if (a.clockName && b.clockName && a.clockName != b.clockName)
        return true;
      if (a.clockValue && b.clockValue && a.clockValue != b.clockValue)
        return true;
      auto edgeA = effectiveEdge(a);
      auto edgeB = effectiveEdge(b);
      if (edgeA && edgeB && edgeA != edgeB)
        return true;
      return false;
    };

    bool clockConflict = false;
    auto reportClockConflict = [&](Operation *op) {
      if (clockConflict)
        return;
      op->emitError("ltl.delay/ltl.past used with conflicting clock "
                    "information; ensure each property uses a single "
                    "clock/edge");
      clockConflict = true;
    };

    auto recordClockForProperty = [&](Value prop, const ClockInfo &info) {
      if (clockConflict)
        return;
      if (!prop)
        return;
      SmallVector<std::pair<Operation *, ClockInfo>> worklist;
      if (auto *def = prop.getDefiningOp())
        worklist.push_back({def, info});
      DenseMap<Operation *, ClockInfo> seenInfo;
      while (!worklist.empty()) {
        auto [cur, curInfo] = worklist.pop_back_val();
        if (!cur || !isa_and_nonnull<ltl::LTLDialect>(cur->getDialect()))
          continue;
        auto it = seenInfo.find(cur);
        if (it != seenInfo.end()) {
          ClockInfo merged = it->second;
          mergeClockInfo(merged, curInfo);
          bool changed = (merged.clockName != it->second.clockName) ||
                         (merged.edge != it->second.edge) ||
                         (merged.clockValue != it->second.clockValue);
          if (!changed)
            continue;
          it->second = merged;
          curInfo = merged;
        } else {
          seenInfo.insert({cur, curInfo});
        }

        if (auto clockOp = dyn_cast<ltl::ClockOp>(cur)) {
          ClockInfo childInfo;
          childInfo.edge = clockOp.getEdge();
          childInfo.clockValue = clockOp.getClock();
          childInfo.seen = true;
          if (auto *def = clockOp.getInput().getDefiningOp())
            worklist.push_back({def, childInfo});
          continue;
        }

        if (isa<ltl::DelayOp, ltl::PastOp>(cur)) {
          ClockInfo &slot = ltlClockInfo[cur];
          ClockInfo opInfo;
          opInfo.clockName = cur->getAttrOfType<StringAttr>("bmc.clock");
          if (auto edgeAttr =
                  cur->getAttrOfType<ltl::ClockEdgeAttr>("bmc.clock_edge"))
            opInfo.edge = edgeAttr.getValue();
          opInfo.seen = opInfo.clockName || opInfo.edge;
          if (opInfo.seen) {
            if (clockInfoConflict(slot, opInfo)) {
              reportClockConflict(cur);
              return;
            }
            mergeClockInfo(slot, opInfo);
          }
          if (clockInfoConflict(slot, curInfo)) {
            reportClockConflict(cur);
            return;
          }
          mergeClockInfo(slot, curInfo);
        }

        for (Value operand : cur->getOperands()) {
          if (!isa<ltl::PropertyType, ltl::SequenceType>(operand.getType()))
            continue;
          if (Operation *def = operand.getDefiningOp())
            worklist.push_back({def, curInfo});
        }
      }
    };

    auto getClockInfoFromOp = [&](Operation *op) {
      ClockInfo info;
      info.clockName = op->getAttrOfType<StringAttr>("bmc.clock");
      if (auto edgeAttr =
              op->getAttrOfType<ltl::ClockEdgeAttr>("bmc.clock_edge"))
        info.edge = edgeAttr.getValue();
      info.seen = info.clockName || info.edge;
      return info;
    };

    circuitBlock.walk([&](verif::AssertOp assertOp) {
      recordClockForProperty(assertOp.getProperty(),
                             getClockInfoFromOp(assertOp));
    });
    circuitBlock.walk([&](verif::AssumeOp assumeOp) {
      recordClockForProperty(assumeOp.getProperty(),
                             getClockInfoFromOp(assumeOp));
    });
    circuitBlock.walk([&](verif::CoverOp coverOp) {
      recordClockForProperty(coverOp.getProperty(),
                             getClockInfoFromOp(coverOp));
    });

    if (clockConflict)
      return failure();

    DenseSet<Value> sequenceRootSet;
    DenseMap<Value, ClockInfo> sequenceRootClocks;
    // Only sequences that use multi-step operators require the NFA-based BMC
    // infrastructure. Purely single-step sequences are correctly lowered by the
    // generic LTL-to-SMT conversion after delay/past buffers have been set up,
    // and routing them through the NFA path would introduce unnecessary state
    // (extra tick/state arguments and iter_args) and potential blow-ups.
    auto sequenceNeedsNFA = [&](Value seq) -> bool {
      if (!seq || !isa<ltl::SequenceType>(seq.getType()))
        return false;
      Operation *def = seq.getDefiningOp();
      if (!def || !isa_and_nonnull<ltl::LTLDialect>(def->getDialect()))
        return false;

      SmallVector<Operation *> worklist;
      DenseSet<Operation *> visited;
      worklist.push_back(def);
      while (!worklist.empty()) {
        Operation *cur = worklist.pop_back_val();
        if (!cur || !isa_and_nonnull<ltl::LTLDialect>(cur->getDialect()))
          continue;
        if (!visited.insert(cur).second)
          continue;
        if (isa<ltl::ConcatOp, ltl::RepeatOp, ltl::GoToRepeatOp,
                ltl::NonConsecutiveRepeatOp>(cur))
          return true;
        for (Value operand : cur->getOperands()) {
          if (!isa<ltl::SequenceType>(operand.getType()))
            continue;
          if (Operation *opDef = operand.getDefiningOp())
            worklist.push_back(opDef);
        }
      }
      return false;
    };
    auto addSequenceRoot = [&](Value seq, const ClockInfo &info) {
      if (!seq || !isa<ltl::SequenceType>(seq.getType()))
        return;
      if (!sequenceNeedsNFA(seq))
        return;
      auto &slot = sequenceRootClocks[seq];
      if (!slot.seen) {
        slot = info;
        slot.seen = info.seen;
      } else {
        if (clockInfoConflict(slot, info)) {
          if (Operation *def = seq.getDefiningOp())
            reportClockConflict(def);
          else
            clockConflict = true;
          return;
        }
        mergeClockInfo(slot, info);
      }
      sequenceRootSet.insert(seq);
    };
    auto collectSequenceRoots = [&](Value prop, const ClockInfo &info) {
      if (!prop || clockConflict)
        return;
      SmallVector<std::pair<Value, ClockInfo>> worklist;
      DenseMap<Value, ClockInfo> seenInfo;
      worklist.push_back({prop, info});
      while (!worklist.empty()) {
        auto [cur, curInfo] = worklist.pop_back_val();
        auto it = seenInfo.find(cur);
        if (it != seenInfo.end()) {
          if (clockInfoConflict(it->second, curInfo)) {
            if (Operation *def = cur.getDefiningOp())
              reportClockConflict(def);
            else
              clockConflict = true;
            return;
          }
          ClockInfo merged = it->second;
          mergeClockInfo(merged, curInfo);
          bool changed = (merged.clockName != it->second.clockName) ||
                         (merged.edge != it->second.edge) ||
                         (merged.clockValue != it->second.clockValue);
          if (!changed)
            continue;
          it->second = merged;
          curInfo = merged;
        } else {
          seenInfo.insert({cur, curInfo});
        }
        if (isa<ltl::SequenceType>(cur.getType())) {
          if (auto clockOp = cur.getDefiningOp<ltl::ClockOp>()) {
            ClockInfo clockInfo;
            clockInfo.edge = clockOp.getEdge();
            clockInfo.clockValue = clockOp.getClock();
            clockInfo.seen = clockInfo.clockValue || clockInfo.edge;
            if (clockInfoConflict(curInfo, clockInfo)) {
              reportClockConflict(clockOp);
              return;
            }
            mergeClockInfo(curInfo, clockInfo);
            worklist.push_back({clockOp.getInput(), curInfo});
            continue;
          }
          if (auto delayOp = cur.getDefiningOp<ltl::DelayOp>()) {
            if (delayOp.getDelay() == 0) {
              auto lengthAttr = delayOp.getLengthAttr();
              if (lengthAttr &&
                  lengthAttr.getValue().getZExtValue() == 0) {
                worklist.push_back({delayOp.getInput(), curInfo});
                continue;
              }
            }
          }
          addSequenceRoot(cur, curInfo);
          continue;
        }
        Operation *def = cur.getDefiningOp();
        if (!def || !isa_and_nonnull<ltl::LTLDialect>(def->getDialect()))
          continue;
        if (auto clockOp = dyn_cast<ltl::ClockOp>(def)) {
          ClockInfo clockInfo;
          clockInfo.edge = clockOp.getEdge();
          clockInfo.clockValue = clockOp.getClock();
          clockInfo.seen = clockInfo.clockValue || clockInfo.edge;
          if (clockInfoConflict(curInfo, clockInfo)) {
            reportClockConflict(def);
            return;
          }
          mergeClockInfo(curInfo, clockInfo);
          worklist.push_back({clockOp.getInput(), curInfo});
          continue;
        }
        for (Value operand : def->getOperands()) {
          if (!isa<ltl::PropertyType, ltl::SequenceType>(operand.getType()))
            continue;
          worklist.push_back({operand, curInfo});
        }
      }
    };

    circuitBlock.walk([&](verif::AssertOp assertOp) {
      collectSequenceRoots(assertOp.getProperty(),
                           getClockInfoFromOp(assertOp));
    });
    circuitBlock.walk([&](verif::AssumeOp assumeOp) {
      collectSequenceRoots(assumeOp.getProperty(),
                           getClockInfoFromOp(assumeOp));
    });
    circuitBlock.walk([&](verif::CoverOp coverOp) {
      collectSequenceRoots(coverOp.getProperty(), getClockInfoFromOp(coverOp));
    });

    if (clockConflict)
      return failure();

    // Capture derived clock equivalences from assume constraints before any
    // verification ops are erased.
    auto isTriviallyTrueEnable = [&](Value enable) -> bool {
      if (!enable)
        return true;
      if (auto literal = getConstI1Value(enable))
        return *literal;
      if (auto xorOp = enable.getDefiningOp<comb::XorOp>()) {
        if (xorOp.getNumOperands() != 2)
          return false;
        auto lhs = getConstI1Value(xorOp.getOperand(0));
        auto rhs = getConstI1Value(xorOp.getOperand(1));
        if (lhs && rhs)
          return (*lhs) ^ (*rhs);
      }
      return false;
    };

    auto registerEquivalence = [&](Value lhs, Value rhs, bool invert,
                                   Operation *context) {
      if (!lhs || !rhs)
        return;
      if (lhs.getType() != rhs.getType())
        return;
      if (!isI1Value(lhs))
        return;
      if (getConstI1Value(lhs) || getConstI1Value(rhs))
        return;
      clockEquivalenceUF.unite(lhs, rhs, invert);
      if (clockEquivalenceUF.conflict)
        reportDerivedClockConflict(
            context, "derived clock assumptions are inconsistent");
    };

    // Record definitional equivalences for XOR with constants, e.g. (x ^ 0) == x
    // and (x ^ 1) == !x, so assume-based mapping can reach the base clock.
    circuitBlock.walk([&](comb::XorOp xorOp) {
      if (derivedClockConflict)
        return;
      if (!isI1Value(xorOp.getResult()))
        return;
      bool parity = false;
      SmallVector<Value> nonConst;
      nonConst.reserve(xorOp.getNumOperands());
      for (Value operand : xorOp.getOperands()) {
        if (auto literal = getConstI1Value(operand))
          parity ^= *literal;
        else
          nonConst.push_back(operand);
      }
      if (nonConst.size() != 1)
        return;
      registerEquivalence(xorOp.getResult(), nonConst[0], parity,
                          xorOp.getOperation());
    });

    circuitBlock.walk([&](verif::AssumeOp assumeOp) {
      if (derivedClockConflict)
        return;
      if (auto enable = assumeOp.getEnable()) {
        if (!isTriviallyTrueEnable(enable))
          return;
      }
      auto cmpOp = assumeOp.getProperty().getDefiningOp<comb::ICmpOp>();
      if (cmpOp) {
        bool invertOther = false;
        switch (cmpOp.getPredicate()) {
        case comb::ICmpPredicate::eq:
          invertOther = false;
          break;
        case comb::ICmpPredicate::ne:
          invertOther = true;
          break;
        case comb::ICmpPredicate::ceq:
          invertOther = false;
          break;
        case comb::ICmpPredicate::cne:
          invertOther = true;
          break;
        case comb::ICmpPredicate::weq:
          invertOther = false;
          break;
        case comb::ICmpPredicate::wne:
          invertOther = true;
          break;
        default:
          return;
        }
        registerEquivalence(cmpOp.getLhs(), cmpOp.getRhs(), invertOther,
                            assumeOp.getOperation());
        return;
      }
      if (auto xorOp = assumeOp.getProperty().getDefiningOp<comb::XorOp>()) {
        bool parity = false;
        SmallVector<Value> nonConst;
        nonConst.reserve(xorOp.getNumOperands());
        for (Value operand : xorOp.getOperands()) {
          if (auto literal = getConstI1Value(operand))
            parity ^= *literal;
          else
            nonConst.push_back(operand);
        }
        if (nonConst.size() != 2)
          return;
        bool invert = !parity;
        registerEquivalence(nonConst[0], nonConst[1], invert,
                            assumeOp.getOperation());
      }
    });

    if (derivedClockConflict)
      return failure();

    if (clockEquivalenceUF.conflict) {
      reportDerivedClockConflict(op.getOperation(),
                                 "derived clock assumptions are inconsistent");
      return failure();
    }

    DenseSet<Operation *> nfaSequenceOps;
    DenseSet<Value> visitedSequenceOps;
    for (Value seqRoot : sequenceRootSet)
      collectSequenceOpsForBMC(seqRoot, nfaSequenceOps, visitedSequenceOps);

    struct NonFinalCheckInfo {
      explicit NonFinalCheckInfo(Location loc) : loc(loc) {}

      Location loc;
      StringAttr clockName;
      std::optional<ltl::ClockEdge> edge;
    };

    // Collect non-final properties so BMC can detect any violating property.
    SmallVector<Operation *> nonFinalOps;
    SmallVector<NonFinalCheckInfo> nonFinalCheckInfos;
    if (isCoverCheck) {
      circuitBlock.walk([&](verif::CoverOp coverOp) {
        if (coverOp->hasAttr("bmc.final"))
          return;
        nonFinalOps.push_back(coverOp);
        NonFinalCheckInfo info(coverOp.getLoc());
        info.clockName = coverOp->getAttrOfType<StringAttr>("bmc.clock");
        if (auto edgeAttr = coverOp->getAttrOfType<ltl::ClockEdgeAttr>(
                "bmc.clock_edge"))
          info.edge = edgeAttr.getValue();
        nonFinalCheckInfos.push_back(info);
      });
    } else {
      circuitBlock.walk([&](verif::AssertOp assertOp) {
        if (assertOp->hasAttr("bmc.final"))
          return;
        nonFinalOps.push_back(assertOp);
        NonFinalCheckInfo info(assertOp.getLoc());
        info.clockName = assertOp->getAttrOfType<StringAttr>("bmc.clock");
        if (auto edgeAttr = assertOp->getAttrOfType<ltl::ClockEdgeAttr>(
                "bmc.clock_edge"))
          info.edge = edgeAttr.getValue();
        nonFinalCheckInfos.push_back(info);
      });
    }
    // Materialize non-final check values after NFA/delay rewriting to ensure
    // the yielded check expressions reflect any sequence-root rewrites.
    SmallVector<Value> nonFinalCheckValues;
    SmallVector<Value> nonFinalCheckProps;

    // Hoist any final-only checks into circuit outputs so we can check them
    // only at the final step.
    // Materialize final check values after NFA/delay rewriting for the same
    // reason as non-final checks.
    SmallVector<Value> finalCheckValues;
    SmallVector<Value> finalCheckProps;
    SmallVector<bool> finalCheckIsCover;
    SmallVector<NonFinalCheckInfo> finalCheckInfos;
    SmallVector<Operation *> opsToErase;
    circuitBlock.walk([&](Operation *curOp) {
      if (!curOp->hasAttr("bmc.final"))
        return;
      if (auto assertOp = dyn_cast<verif::AssertOp>(curOp)) {
        finalCheckIsCover.push_back(false);
        NonFinalCheckInfo info(assertOp.getLoc());
        info.clockName = assertOp->getAttrOfType<StringAttr>("bmc.clock");
        if (auto edgeAttr = assertOp->getAttrOfType<ltl::ClockEdgeAttr>(
                "bmc.clock_edge"))
          info.edge = edgeAttr.getValue();
        finalCheckInfos.push_back(info);
        opsToErase.push_back(curOp);
        return;
      }
      if (auto assumeOp = dyn_cast<verif::AssumeOp>(curOp)) {
        finalCheckIsCover.push_back(false);
        NonFinalCheckInfo info(assumeOp.getLoc());
        info.clockName = assumeOp->getAttrOfType<StringAttr>("bmc.clock");
        if (auto edgeAttr = assumeOp->getAttrOfType<ltl::ClockEdgeAttr>(
                "bmc.clock_edge"))
          info.edge = edgeAttr.getValue();
        finalCheckInfos.push_back(info);
        opsToErase.push_back(curOp);
        return;
      }
      if (auto coverOp = dyn_cast<verif::CoverOp>(curOp)) {
        finalCheckIsCover.push_back(true);
        NonFinalCheckInfo info(coverOp.getLoc());
        info.clockName = coverOp->getAttrOfType<StringAttr>("bmc.clock");
        if (auto edgeAttr = coverOp->getAttrOfType<ltl::ClockEdgeAttr>(
                "bmc.clock_edge"))
          info.edge = edgeAttr.getValue();
        finalCheckInfos.push_back(info);
        opsToErase.push_back(curOp);
        return;
      }
    });
    // Erase the bmc.final ops using the rewriter to properly notify the
    // conversion framework
    if (livenessMode) {
      if (opsToErase.empty()) {
        op.emitError(
            "liveness mode requires at least one bmc.final property");
        return failure();
      }
      // Liveness mode focuses on final-only obligations. Non-final checks
      // remain lowered into the circuit but are not treated as violations.
      nonFinalOps.clear();
      nonFinalCheckInfos.clear();
    }
    size_t numNonFinalChecks = nonFinalOps.size();
    size_t numFinalChecks = opsToErase.size();
    uint64_t boundValue = op.getBound();

    if (inductionStep) {
      if (isCoverCheck) {
        op.emitError("k-induction does not support cover properties yet");
        return failure();
      }
      if (numNonFinalChecks == 0 && numFinalChecks == 0) {
        op.emitError("k-induction requires at least one assertion");
        return failure();
      }
    }

    SmallVector<Type> oldLoopInputTy(op.getLoop().getArgumentTypes());
    SmallVector<Type> oldCircuitInputTy(op.getCircuit().getArgumentTypes());
    unsigned numRegs = op.getNumRegs();
    DenseSet<unsigned> knownClockArgIndices;
    auto collectKnownClockArgIndices = [&](ArrayAttr sourcesAttr) {
      if (!sourcesAttr)
        return;
      for (auto attr : sourcesAttr) {
        auto dict = dyn_cast<DictionaryAttr>(attr);
        if (!dict)
          continue;
        auto argAttr = dyn_cast_or_null<IntegerAttr>(dict.get("arg_index"));
        if (!argAttr)
          continue;
        knownClockArgIndices.insert(argAttr.getValue().getZExtValue());
      }
    };
    collectKnownClockArgIndices(
        op->getAttrOfType<ArrayAttr>("bmc_clock_sources"));
    collectKnownClockArgIndices(
        op->getAttrOfType<ArrayAttr>("bmc_reg_clock_sources"));
    if (!assumeKnownInputs) {
      bool hasFourStateInputs = false;
      size_t numNonStateInputs = oldCircuitInputTy.size();
      if (numRegs <= numNonStateInputs)
        numNonStateInputs -= numRegs;
      else
        numNonStateInputs = 0;
      for (auto [idx, ty] :
           llvm::enumerate(TypeRange(oldCircuitInputTy).take_front(
               numNonStateInputs))) {
        if (isFourStateStruct(ty) && !knownClockArgIndices.contains(idx)) {
          hasFourStateInputs = true;
          break;
        }
      }
      if (hasFourStateInputs) {
        op.emitWarning(
            "4-state inputs are unconstrained; consider "
            "--assume-known-inputs or full X-propagation support");
      }
    }
    // TODO: the init and loop regions should be able to be concrete instead of
    // symbolic which is probably preferable - just need to convert back and
    // forth
    SmallVector<Type> loopInputTy, circuitInputTy, initOutputTy,
        circuitOutputTy;
    if (failed(typeConverter->convertTypes(oldLoopInputTy, loopInputTy))) {
      op.emitError("failed to convert verif.bmc loop input types");
      return failure();
    }
    if (failed(typeConverter->convertTypes(oldCircuitInputTy, circuitInputTy))) {
      op.emitError("failed to convert verif.bmc circuit input types");
      return failure();
    }
    if (failed(typeConverter->convertTypes(
            op.getInit().front().back().getOperandTypes(), initOutputTy))) {
      op.emitError("failed to convert verif.bmc init yield types");
      return failure();
    }
    if (failed(typeConverter->convertTypes(
            op.getCircuit().front().back().getOperandTypes(), circuitOutputTy))) {
      op.emitError("failed to convert verif.bmc circuit yield types");
      return failure();
    }

    auto initialValues = op.getInitialValues();

    // Count clocks from the original init yield types (BEFORE region conversion)
    // This handles both explicit seq::ClockType and i1 clocks converted via
    // ToClockOp
    size_t numInitClocks = 0;
    for (auto ty : op.getInit().front().back().getOperandTypes())
      if (isa<seq::ClockType>(ty))
        numInitClocks++;

    // Collect circuit argument names for debug-friendly SMT declarations.
    size_t originalArgCount = oldCircuitInputTy.size();
    SmallVector<StringAttr> inputNamePrefixes;
    if (auto nameAttr =
            op->getAttrOfType<ArrayAttr>("bmc_input_names")) {
      inputNamePrefixes.reserve(nameAttr.size());
      for (auto attr : nameAttr) {
        if (auto strAttr = dyn_cast_or_null<StringAttr>(attr))
          inputNamePrefixes.push_back(strAttr);
        else
          inputNamePrefixes.push_back(StringAttr{});
      }
    } else {
      inputNamePrefixes.assign(originalArgCount, StringAttr{});
    }
    DenseMap<StringRef, unsigned> inputNameToArgIndex;
    for (auto [idx, nameAttr] : llvm::enumerate(inputNamePrefixes)) {
      if (!nameAttr || nameAttr.getValue().empty())
        continue;
      inputNameToArgIndex.try_emplace(nameAttr.getValue(), idx);
    }

    auto maybeAssertKnown = [&](size_t argIndex, Type originalTy, Value smtVal,
                                OpBuilder &builder) {
      if (!assumeKnownInputs && !knownClockArgIndices.contains(argIndex))
        return;
      maybeAssertKnownInput(originalTy, smtVal, loc, builder);
    };

    SmallVector<SequenceNFAInfo> nfaInfos;
    SmallVector<bool> nfaStartStates;
    size_t totalNFAStateSlots = 0;

    // =========================================================================
    // Multi-step delay tracking:
    // Scan the circuit for ltl.delay operations with delay > 0 and set up
    // the infrastructure to track delayed signals across time steps.
    //
    // For each ltl.delay(signal, N) with N > 0:
    // 1. Add N new block arguments to the circuit (for the delay buffer)
    // 2. Add N new yield operands (shifted buffer + current signal)
    // 3. Replace the delay op with the oldest buffer entry (what was N steps ago)
    //
    // The delay buffer is a shift register: each iteration, values shift down
    // and the current signal value enters at position N-1.
    // buffer[0] contains the value from N steps ago (the delayed value to use).
    // =========================================================================
    SmallVector<DelayInfo> delayInfos;
    size_t totalDelaySlots = 0;
    bool delaySetupFailed = false;
    SmallVector<Value> delayRootOverrides;
    // First pass: collect all delay ops with meaningful temporal ranges.
    circuitBlock.walk([&](ltl::DelayOp delayOp) {
      // Delay ops can become temporarily unused while we erase the original
      // verif.assert/assume/cover ops and only later append the check values to
      // the circuit yield. Still collect any delay reachable from a property,
      // even if it currently has no SSA uses.
      if (delayOp.use_empty() &&
          ltlClockInfo.find(delayOp.getOperation()) == ltlClockInfo.end())
        return;
      bool isSequenceRoot = sequenceRootSet.contains(delayOp.getResult());
      if (nfaSequenceOps.contains(delayOp.getOperation()) && !isSequenceRoot)
        return;
      uint64_t delay = delayOp.getDelay();

      uint64_t length = 0;
      if (auto lengthAttr = delayOp.getLengthAttr()) {
        length = lengthAttr.getValue().getZExtValue();
      } else {
        // Unbounded delay (##[N:$]) is approximated within the BMC bound as
        // a bounded range [N : bound-1]. When the delay exceeds the bound,
        // there is no match within the window.
        if (boundValue > 0 && delay < boundValue)
          length = boundValue - 1 - delay;
      }

      // NOTE: Unbounded delay (missing length) is approximated by the BMC
      // bound, bounded range is supported by widening the delay buffer.
      if (delay == 0 && length == 0)
        return;
      uint64_t bufferSize = delay + length;
      if (bufferSize == 0)
        return;

      DelayInfo info;
      info.op = delayOp;
      info.inputSignal = delayOp.getInput();
      info.delay = delay;
      info.length = length;
      info.bufferSize = bufferSize;
      info.bufferStartIndex = totalDelaySlots;
      if (auto it = ltlClockInfo.find(delayOp.getOperation());
          it != ltlClockInfo.end()) {
        info.clockName = it->second.clockName;
        info.edge = it->second.edge;
        info.clockValue = it->second.clockValue;
      }
      delayInfos.push_back(info);
      totalDelaySlots += bufferSize;
      if (isSequenceRoot)
        delayRootOverrides.push_back(delayOp.getResult());
    });

    // Track delay buffer names (added immediately after original inputs).
    size_t delaySlots = totalDelaySlots;
    for (Value root : delayRootOverrides) {
      sequenceRootSet.erase(root);
      sequenceRootClocks.erase(root);
    }

    // =========================================================================
    // Past operation tracking for $rose/$fell:
    // ltl.past operations look at signal values from previous cycles.
    // For past(signal, N), we need N buffer slots to track signal history.
    // The buffer works identically to delay buffers - buffer[0] holds the
    // oldest value (from N cycles ago).
    // =========================================================================
    SmallVector<PastInfo> pastInfos;
    size_t totalPastSlots = 0;
    SmallVector<Value> pastRootOverrides;

    circuitBlock.walk([&](ltl::PastOp pastOp) {
      // Past ops may become temporarily unused for the same reason as delay ops
      // above (assert/assume/cover erased before check values are yielded).
      if (pastOp.use_empty() &&
          ltlClockInfo.find(pastOp.getOperation()) == ltlClockInfo.end())
        return;
      uint64_t delay = pastOp.getDelay();
      if (delay == 0)
        return;

      PastInfo info;
      info.op = pastOp;
      info.inputSignal = pastOp.getInput();
      info.delay = delay;
      info.bufferSize = delay;
      info.bufferStartIndex = totalPastSlots;
      if (auto it = ltlClockInfo.find(pastOp.getOperation());
          it != ltlClockInfo.end()) {
        info.clockName = it->second.clockName;
        info.edge = it->second.edge;
        info.clockValue = it->second.clockValue;
      }
      pastInfos.push_back(info);
      totalPastSlots += delay;
      if (sequenceRootSet.contains(pastOp.getResult()))
        pastRootOverrides.push_back(pastOp.getResult());
    });
    for (Value root : pastRootOverrides) {
      sequenceRootSet.erase(root);
      sequenceRootClocks.erase(root);
    }

    // Second pass: modify the circuit block to add delay buffer infrastructure
    // We need to do this BEFORE region type conversion.
    //
    // Output order after modification:
    // [original outputs (registers)] [delay buffer outputs]
    SmallVector<ltl::DelayOp> delayOpsToErase;
    if (!delayInfos.empty()) {
      // For each delay op, add buffer arguments and modify the yield
      auto yieldOp = cast<verif::YieldOp>(circuitBlock.getTerminator());

      // Get current operands (no check outputs appended yet)
      SmallVector<Value> origOperands(yieldOp.getOperands().begin(),
                                      yieldOp.getOperands().end());
      SmallVector<Value> newYieldOperands(origOperands.begin(),
                                          origOperands.end());
      SmallVector<Value> delayBufferOutputs;

      for (auto &info : delayInfos) {
        // The input signal type - used for delay buffer slots.
        Type bufferElementType = info.inputSignal.getType();

        // Add buffer arguments for the delay buffer (oldest to newest)
        SmallVector<Value> bufferArgs;
        for (uint64_t i = 0; i < info.bufferSize; ++i) {
          auto arg = circuitBlock.addArgument(bufferElementType, loc);
          bufferArgs.push_back(arg);
        }

        // Replace all uses of the delay op with the delayed value.
        // For exact delay: use the oldest buffer entry (value from N steps ago).
        // For bounded range: OR over the window [delay, delay+length].
        Value inputSig = info.inputSignal;
        Value delayedValue;
        auto toSequenceValue = [&](Value val) -> Value {
          if (!val)
            return Value{};
          if (isa<ltl::SequenceType>(val.getType()))
            return val;
          if (auto intTy = dyn_cast<IntegerType>(val.getType());
              intTy && intTy.getWidth() == 1) {
            // Wrap an i1 "matches now" value as a sequence value.
            auto zeroAttr = rewriter.getI64IntegerAttr(0);
            return ltl::DelayOp::create(rewriter, loc, val, zeroAttr, zeroAttr)
                .getResult();
          }
          return Value{};
        };

        if (info.delay == 0) {
          // Range includes the current cycle, so OR the input with the buffer.
          bool isI1 = false;
          if (auto intTy = dyn_cast<IntegerType>(bufferElementType))
            isI1 = intTy.getWidth() == 1;
          if (!isI1 && !isa<ltl::SequenceType>(bufferElementType)) {
            info.op.emitError(
                "bounded delay in BMC requires i1 or ltl.sequence input");
            delaySetupFailed = true;
            continue;
          }
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(info.op);
          delayedValue = toSequenceValue(inputSig);
          if (!delayedValue) {
            info.op.emitError(
                "failed to build sequence value for bounded delay input");
            delaySetupFailed = true;
            continue;
          }
          for (auto arg : bufferArgs) {
            Value argSeq = toSequenceValue(arg);
            if (!argSeq) {
              info.op.emitError(
                  "failed to build sequence value for bounded delay input");
              delaySetupFailed = true;
              continue;
            }
            delayedValue =
                ltl::OrOp::create(rewriter, loc,
                                  ValueRange{delayedValue, argSeq})
                    .getResult();
          }
        } else {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(info.op);
          delayedValue = toSequenceValue(bufferArgs[0]);
          if (!delayedValue) {
            info.op.emitError(
                "failed to build sequence value for bounded delay input");
            delaySetupFailed = true;
            continue;
          }
          if (info.length > 0) {
            bool isI1 = false;
            if (auto intTy = dyn_cast<IntegerType>(bufferElementType))
              isI1 = intTy.getWidth() == 1;
            if (!isI1 && !isa<ltl::SequenceType>(bufferElementType)) {
              info.op.emitError(
                  "bounded delay in BMC requires i1 or ltl.sequence input");
              delaySetupFailed = true;
              continue;
            }
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(info.op);
            for (uint64_t i = 1; i <= info.length; ++i) {
              Value argSeq = toSequenceValue(bufferArgs[i]);
              if (!argSeq) {
                info.op.emitError(
                    "failed to build sequence value for bounded delay input");
                delaySetupFailed = true;
                break;
              }
              delayedValue =
                  ltl::OrOp::create(rewriter, loc,
                                    ValueRange{delayedValue, argSeq})
                      .getResult();
            }
            if (delaySetupFailed)
              continue;
          }
        }
        info.op.replaceAllUsesWith(delayedValue);
        delayOpsToErase.push_back(info.op);

        // Add yield operands: shifted buffer (drop oldest, add current)
        // new_buffer = [buffer[1], buffer[2], ..., buffer[bufferSize-1], current_signal]
        for (uint64_t i = 1; i < info.bufferSize; ++i)
          delayBufferOutputs.push_back(bufferArgs[i]);
        delayBufferOutputs.push_back(inputSig);
      }

      // Construct final yield: [orig outputs] [delay buffers]
      newYieldOperands.append(delayBufferOutputs);

      // Update the yield with new operands
      yieldOp->setOperands(newYieldOperands);

      if (delaySetupFailed)
        return failure();

      // Erase the delay ops (after all replacements are done)
      for (auto delayOp : delayOpsToErase)
        rewriter.eraseOp(delayOp);

      // Update the type vectors to include the new arguments and outputs
      oldCircuitInputTy.clear();
      oldCircuitInputTy.append(op.getCircuit().getArgumentTypes().begin(),
                               op.getCircuit().getArgumentTypes().end());
    }

    // =========================================================================
    // Process ltl.past operations for $rose/$fell support.
    // Past operations look at signal values from previous cycles.
    // The buffer mechanism is the same as delay ops: buffer[0] is the oldest
    // value (from N cycles ago), which is exactly what past(signal, N) needs.
    // =========================================================================
    SmallVector<ltl::PastOp> pastOpsToErase;
    if (!pastInfos.empty()) {
      auto yieldOp = cast<verif::YieldOp>(circuitBlock.getTerminator());

      // Get current operands (no check outputs appended yet)
      SmallVector<Value> origOperands(yieldOp.getOperands().begin(),
                                      yieldOp.getOperands().end());
      SmallVector<Value> newYieldOperands(origOperands.begin(),
                                          origOperands.end());
      SmallVector<Value> pastBufferOutputs;

      for (auto &info : pastInfos) {
        // The input signal type - past ops work on i1 signals
        Type bufferElementType = info.inputSignal.getType();

        // Add buffer arguments for the past buffer (oldest to newest)
        SmallVector<Value> bufferArgs;
        for (uint64_t i = 0; i < info.bufferSize; ++i) {
          auto arg = circuitBlock.addArgument(bufferElementType, loc);
          bufferArgs.push_back(arg);
        }

        // Replace all uses of the past op with the past value.
        // ltl.past returns an ltl.sequence, so wrap the i1 buffer value as an
        // instantaneous sequence.
        Value pastValue;
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(info.op);
          auto zeroAttr = rewriter.getI64IntegerAttr(0);
          pastValue = ltl::DelayOp::create(rewriter, loc, bufferArgs[0],
                                           zeroAttr, zeroAttr)
                          .getResult();
        }
        info.op.replaceAllUsesWith(pastValue);
        pastOpsToErase.push_back(info.op);

        // Add yield operands: shifted buffer (drop oldest, add current)
        // new_buffer = [buffer[1], buffer[2], ..., buffer[N-1], current_signal]
        for (uint64_t i = 1; i < info.bufferSize; ++i)
          pastBufferOutputs.push_back(bufferArgs[i]);
        pastBufferOutputs.push_back(info.inputSignal);
      }

      // Construct final yield: [orig outputs] [past buffers]
      newYieldOperands.append(pastBufferOutputs);

      yieldOp->setOperands(newYieldOperands);

      // Erase the past ops
      for (auto pastOp : pastOpsToErase)
        rewriter.eraseOp(pastOp);

      // Update the type vectors
      oldCircuitInputTy.clear();
      oldCircuitInputTy.append(op.getCircuit().getArgumentTypes().begin(),
                               op.getCircuit().getArgumentTypes().end());

      // Update totalDelaySlots to include past slots for buffer initialization
      totalDelaySlots += totalPastSlots;
    }

    // =========================================================================
    // NFA-based sequence tracking for multi-step semantics.
    // Build a per-sequence NFA and carry its state through the BMC loop.
    // =========================================================================
    SmallVector<Value> nfaStateOutputs;
    if (!sequenceRootSet.empty()) {
      auto i1Type = rewriter.getI1Type();
      size_t registerStartIndex =
          oldCircuitInputTy.size() - totalDelaySlots - numRegs;

      auto computeSequenceClockInfo = [&](Value seq, ClockInfo info)
          -> std::optional<ClockInfo> {
        SmallVector<Operation *> worklist;
        DenseSet<Operation *> visited;
        if (Operation *def = seq.getDefiningOp())
          worklist.push_back(def);
        while (!worklist.empty()) {
          Operation *cur = worklist.pop_back_val();
          if (!cur || !isa_and_nonnull<ltl::LTLDialect>(cur->getDialect()))
            continue;
          if (!visited.insert(cur).second)
            continue;

          if (auto clockOp = dyn_cast<ltl::ClockOp>(cur)) {
            ClockInfo clockInfo;
            clockInfo.edge = clockOp.getEdge();
            clockInfo.clockValue = clockOp.getClock();
            clockInfo.seen = clockInfo.clockValue || clockInfo.edge;
            if (clockInfoConflict(info, clockInfo))
              return std::nullopt;
            mergeClockInfo(info, clockInfo);
            if (Operation *def = clockOp.getInput().getDefiningOp())
              worklist.push_back(def);
            continue;
          }

          for (Value operand : cur->getOperands()) {
            if (!isa<ltl::SequenceType>(operand.getType()))
              continue;
            if (Operation *def = operand.getDefiningOp())
              worklist.push_back(def);
          }
        }
        return info;
      };

      // Collect sequence roots in a deterministic order.
      SmallVector<Value> sequenceRoots;
      circuitBlock.walk([&](Operation *op) {
        for (Value result : op->getResults())
          if (sequenceRootSet.contains(result))
            sequenceRoots.push_back(result);
      });

      for (Value seqRoot : sequenceRoots) {
        if (!seqRoot || !isa<ltl::SequenceType>(seqRoot.getType()))
          continue;
        Operation *rootDef = seqRoot.getDefiningOp();
        if (!rootDef ||
            !isa_and_nonnull<ltl::LTLDialect>(rootDef->getDialect()))
          continue;

        ClockInfo baseInfo;
        if (auto it = sequenceRootClocks.find(seqRoot);
            it != sequenceRootClocks.end())
          baseInfo = it->second;
        auto maybeInfo = computeSequenceClockInfo(seqRoot, baseInfo);
        if (!maybeInfo) {
          rootDef->emitError(
              "ltl.sequence uses conflicting clock annotations");
          return failure();
        }

        // Build the NFA for this sequence.
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(rootDef);
        auto trueVal = hw::ConstantOp::create(rewriter, loc, i1Type, 1);
        auto falseVal = hw::ConstantOp::create(rewriter, loc, i1Type, 0);
        auto [nfa, fragment] = ltl::NFABuilder::buildEpsilonFreeNFA(
            seqRoot, loc, trueVal, rewriter);
        size_t numStates = nfa.states.size();
        if (numStates == 0) {
          auto zeroAttr = rewriter.getI64IntegerAttr(0);
          auto seqFalse =
              ltl::DelayOp::create(rewriter, loc, falseVal, zeroAttr, zeroAttr)
                  .getResult();
          seqRoot.replaceAllUsesWith(seqFalse);
          continue;
        }

        // Insert a tick argument before the register block arguments.
        unsigned tickIndex = registerStartIndex;
        auto tickArg = circuitBlock.insertArgument(
            registerStartIndex, i1Type, loc);
        oldCircuitInputTy.insert(oldCircuitInputTy.begin() + registerStartIndex,
                                 i1Type);
        if (inputNamePrefixes.size() < registerStartIndex)
          inputNamePrefixes.resize(registerStartIndex, StringAttr{});
        inputNamePrefixes.insert(
            inputNamePrefixes.begin() + registerStartIndex,
            rewriter.getStringAttr("nfa_tick_" +
                                   Twine(nfaInfos.size()).str()));
        registerStartIndex++;

        SmallVector<Value> stateArgs;
        stateArgs.reserve(numStates);
        for (size_t i = 0; i < numStates; ++i) {
          auto arg = circuitBlock.addArgument(i1Type, loc);
          stateArgs.push_back(arg);
          oldCircuitInputTy.push_back(i1Type);
        }

        SmallVector<SmallVector<SmallVector<Value, 4>, 4>, 8> incoming;
        incoming.resize(numStates);
        for (size_t from = 0; from < numStates; ++from) {
          for (auto &tr : nfa.states[from].transitions) {
            if (tr.isEpsilon)
              continue;
            Value cond = nfa.conditions[tr.condIndex];
            auto intTy = dyn_cast<IntegerType>(cond.getType());
            if (!intTy || intTy.getWidth() != 1) {
              rootDef->emitError(
                  "sequence NFA conditions must be i1 values");
              return failure();
            }
            incoming[tr.to].push_back(
                SmallVector<Value, 4>{stateArgs[from], cond});
          }
        }

        Value oneVal = hw::ConstantOp::create(rewriter, loc, i1Type, 1).getResult();
        Value notTick = comb::XorOp::create(rewriter, loc, tickArg, oneVal).getResult();
        Value trueResult = trueVal.getResult();
        Value falseResult = falseVal.getResult();

        SmallVector<Value> nextVals;
        nextVals.resize(numStates, falseResult);
        for (size_t i = 0; i < numStates; ++i) {
          SmallVector<Value, 8> orInputs;
          if (static_cast<int>(i) == fragment.start)
            orInputs.push_back(trueResult);

          Value hold = comb::AndOp::create(
              rewriter, loc, SmallVector<Value, 2>{stateArgs[i], notTick},
              true).getResult();
          orInputs.push_back(hold);

          for (auto &edgeVals : incoming[i]) {
            SmallVector<Value, 4> andInputs(edgeVals.begin(),
                                            edgeVals.end());
            andInputs.push_back(tickArg);
            Value andVal =
                comb::AndOp::create(rewriter, loc, andInputs, true).getResult();
            orInputs.push_back(andVal);
          }

          if (orInputs.empty())
            nextVals[i] = falseResult;
          else
            nextVals[i] = comb::OrOp::create(rewriter, loc, orInputs, true).getResult();
        }

        SmallVector<Value, 8> accepting;
        for (size_t i = 0; i < numStates; ++i)
          if (nfa.states[i].accepting)
            accepting.push_back(nextVals[i]);
        Value match =
            accepting.empty()
                ? falseVal.getResult()
                : comb::OrOp::create(rewriter, loc, accepting, true)
                      .getResult();

        auto zeroAttr = rewriter.getI64IntegerAttr(0);
        auto seqMatch =
            ltl::DelayOp::create(rewriter, loc, match, zeroAttr, zeroAttr)
                .getResult();
        seqRoot.replaceAllUsesWith(seqMatch);

        size_t stateStartIndex = totalNFAStateSlots;
        totalNFAStateSlots += numStates;
        for (size_t i = 0; i < numStates; ++i)
          nfaStartStates.push_back(static_cast<int>(i) == fragment.start);
        nfaStateOutputs.append(nextVals.begin(), nextVals.end());
        SequenceNFAInfo info;
        info.root = seqRoot;
        info.stateStartIndex = stateStartIndex;
        info.numStates = numStates;
        info.tickArgIndex = tickIndex;
        info.clockName = maybeInfo->clockName;
        info.edge = maybeInfo->edge;
        info.clockValue = maybeInfo->clockValue;
        nfaInfos.push_back(info);
      }

      // NFA lowering replaces all uses of the sequence roots with a new
      // instantaneous "match now" sequence. This typically leaves the original
      // sequence DAG (concat/repeat/delay/goto/etc.) dead. Erase any now-dead
      // LTL sequence ops so the later generic LTL->SMT lowering does not see
      // multi-step operators that were already handled by the NFA path.
      if (!nfaSequenceOps.empty()) {
        SmallVector<Operation *> ordered;
        ordered.reserve(nfaSequenceOps.size());
        circuitBlock.walk([&](Operation *op) {
          if (nfaSequenceOps.contains(op))
            ordered.push_back(op);
        });

        bool changed = true;
        while (changed) {
          changed = false;
          for (int64_t i = static_cast<int64_t>(ordered.size()) - 1; i >= 0;
               --i) {
            Operation *op = ordered[static_cast<size_t>(i)];
            if (!op)
              continue;
            if (!isa_and_nonnull<ltl::LTLDialect>(op->getDialect()))
              continue;
            if (!op->use_empty())
              continue;
            rewriter.eraseOp(op);
            ordered[static_cast<size_t>(i)] = nullptr;
            changed = true;
          }
        }
      }
    }

    if (!nfaStateOutputs.empty()) {
      auto yieldOp = cast<verif::YieldOp>(circuitBlock.getTerminator());
      SmallVector<Value> newYieldOperands(yieldOp.getOperands().begin(),
                                          yieldOp.getOperands().end());
      newYieldOperands.append(nfaStateOutputs.begin(), nfaStateOutputs.end());
      yieldOp->setOperands(newYieldOperands);
    }

    // Materialize check expressions late, after any delay-buffer and sequence
    // NFA rewriting has updated the property/sequence DAGs used by the original
    // verif operations.
    if (!nonFinalOps.empty() && nonFinalCheckValues.empty()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(circuitBlock.getTerminator());
      nonFinalCheckValues.reserve(nonFinalOps.size());
      nonFinalCheckProps.reserve(nonFinalOps.size());
      for (Operation *opToCheck : nonFinalOps) {
        if (isCoverCheck) {
          auto coverOp = cast<verif::CoverOp>(opToCheck);
          nonFinalCheckProps.push_back(coverOp.getProperty());
          nonFinalCheckValues.push_back(gatePropertyWithEnable(
              coverOp.getProperty(), coverOp.getEnable(), /*isCover=*/true,
              rewriter, coverOp.getLoc()));
        } else {
          auto assertOp = cast<verif::AssertOp>(opToCheck);
          nonFinalCheckProps.push_back(assertOp.getProperty());
          nonFinalCheckValues.push_back(gatePropertyWithEnable(
              assertOp.getProperty(), assertOp.getEnable(), /*isCover=*/false,
              rewriter, assertOp.getLoc()));
        }
      }
    }
    if (!opsToErase.empty() && finalCheckValues.empty()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(circuitBlock.getTerminator());
      finalCheckValues.reserve(opsToErase.size());
      finalCheckProps.reserve(opsToErase.size());
      for (Operation *opToCheck : opsToErase) {
        if (auto assertOp = dyn_cast<verif::AssertOp>(opToCheck)) {
          finalCheckValues.push_back(gatePropertyWithEnable(
              assertOp.getProperty(), assertOp.getEnable(), /*isCover=*/false,
              rewriter, assertOp.getLoc()));
          finalCheckProps.push_back(assertOp.getProperty());
        } else if (auto assumeOp = dyn_cast<verif::AssumeOp>(opToCheck)) {
          finalCheckValues.push_back(gatePropertyWithEnable(
              assumeOp.getProperty(), assumeOp.getEnable(), /*isCover=*/false,
              rewriter, assumeOp.getLoc()));
          finalCheckProps.push_back(assumeOp.getProperty());
        } else if (auto coverOp = dyn_cast<verif::CoverOp>(opToCheck)) {
          finalCheckValues.push_back(gatePropertyWithEnable(
              coverOp.getProperty(), coverOp.getEnable(), /*isCover=*/true,
              rewriter, coverOp.getLoc()));
          finalCheckProps.push_back(coverOp.getProperty());
        }
      }
    }

    // Append non-final and final check outputs after delay/past buffers so
    // circuit outputs are ordered as:
    // [original outputs] [delay/past buffers] [nfa states]
    // [non-final checks] [final checks]
    if (!nonFinalCheckValues.empty() || !finalCheckValues.empty()) {
      auto yieldOp = cast<verif::YieldOp>(circuitBlock.getTerminator());
      SmallVector<Value> newYieldOperands(yieldOp.getOperands());
      newYieldOperands.append(nonFinalCheckValues.begin(),
                              nonFinalCheckValues.end());
      newYieldOperands.append(finalCheckValues.begin(),
                              finalCheckValues.end());
      yieldOp->setOperands(newYieldOperands);
    }

    // Now that the check values are routed through the circuit yield and any
    // multi-step sequence rewriting has had a chance to update their uses, we
    // can erase the original verification ops from the circuit.
    for (auto *opToErase : nonFinalOps)
      rewriter.eraseOp(opToErase);
    for (auto *opToErase : opsToErase)
      rewriter.eraseOp(opToErase);

    // Extend name list with delay/past buffer slots appended to the circuit
    // arguments. These slots are appended after the original inputs.
    if (inputNamePrefixes.size() < originalArgCount)
      inputNamePrefixes.resize(originalArgCount, StringAttr{});
    if (delaySlots > 0) {
      for (size_t i = 0; i < delaySlots; ++i) {
        auto name =
            rewriter.getStringAttr("delay_buf_" + Twine(i).str());
        inputNamePrefixes.push_back(name);
      }
    }
    if (totalPastSlots > 0) {
      for (size_t i = 0; i < totalPastSlots; ++i) {
        auto name =
            rewriter.getStringAttr("past_buf_" + Twine(i).str());
        inputNamePrefixes.push_back(name);
      }
    }
    if (totalNFAStateSlots > 0) {
      for (size_t i = 0; i < totalNFAStateSlots; ++i) {
        auto name =
            rewriter.getStringAttr("nfa_state_" + Twine(i).str());
        inputNamePrefixes.push_back(name);
      }
    }

    // Re-compute circuit types after potential modification
    circuitInputTy.clear();
    circuitOutputTy.clear();
    if (failed(typeConverter->convertTypes(oldCircuitInputTy, circuitInputTy)))
      return failure();
    if (failed(typeConverter->convertTypes(
            op.getCircuit().front().back().getOperandTypes(), circuitOutputTy)))
      return failure();

    auto initFuncTy = rewriter.getFunctionType({}, initOutputTy);
    // Loop and init output types are necessarily the same, so just use init
    // output types
    auto loopFuncTy = rewriter.getFunctionType(loopInputTy, initOutputTy);
    auto circuitFuncTy =
        rewriter.getFunctionType(circuitInputTy, circuitOutputTy);

    func::FuncOp initFuncOp, loopFuncOp, circuitFuncOp;

    if (!emitSMTLIB) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(
          op->getParentOfType<ModuleOp>().getBody());
      initFuncOp = func::FuncOp::create(rewriter, loc,
                                        names.newName("bmc_init"), initFuncTy);
      rewriter.inlineRegionBefore(op.getInit(), initFuncOp.getFunctionBody(),
                                  initFuncOp.end());
      if (failed(rewriter.convertRegionTypes(&initFuncOp.getFunctionBody(),
                                             *typeConverter))) {
        op.emitError("failed to convert verif.bmc init region types");
        return failure();
      }
      loopFuncOp = func::FuncOp::create(rewriter, loc,
                                        names.newName("bmc_loop"), loopFuncTy);
      rewriter.inlineRegionBefore(op.getLoop(), loopFuncOp.getFunctionBody(),
                                  loopFuncOp.end());
      if (failed(rewriter.convertRegionTypes(&loopFuncOp.getFunctionBody(),
                                             *typeConverter))) {
        op.emitError("failed to convert verif.bmc loop region types");
        return failure();
      }
      circuitFuncOp = func::FuncOp::create(
          rewriter, loc, names.newName("bmc_circuit"), circuitFuncTy);
      rewriter.inlineRegionBefore(op.getCircuit(),
                                  circuitFuncOp.getFunctionBody(),
                                  circuitFuncOp.end());
      if (failed(rewriter.convertRegionTypes(&circuitFuncOp.getFunctionBody(),
                                             *typeConverter))) {
        op.emitError("failed to convert verif.bmc circuit region types");
        return failure();
      }
      auto funcOps = {&initFuncOp, &loopFuncOp, &circuitFuncOp};
      // initOutputTy is the same as loop output types
      auto outputTys = {initOutputTy, initOutputTy, circuitOutputTy};
      for (auto [funcOp, outputTy] : llvm::zip(funcOps, outputTys)) {
        auto operands = funcOp->getBody().front().back().getOperands();
        rewriter.eraseOp(&funcOp->getFunctionBody().front().back());
        rewriter.setInsertionPointToEnd(&funcOp->getBody().front());
        SmallVector<Value> toReturn;
        for (unsigned i = 0; i < outputTy.size(); ++i)
          toReturn.push_back(typeConverter->materializeTargetConversion(
              rewriter, loc, outputTy[i], operands[i]));
        func::ReturnOp::create(rewriter, loc, toReturn);
      }
    }

    auto solver =
        emitSMTLIB
            ? smt::SolverOp::create(rewriter, loc, TypeRange{}, ValueRange{})
            : smt::SolverOp::create(rewriter, loc, rewriter.getI1Type(),
                                    ValueRange{});
    if (auto eventSources =
            op->getAttrOfType<ArrayAttr>("bmc_event_sources")) {
      solver->setAttr("bmc_event_sources", eventSources);
      solver->setAttr("bmc_mixed_event_sources", eventSources);
    } else if (auto mixedEventSources =
                   op->getAttrOfType<ArrayAttr>(
                       "bmc_mixed_event_sources")) {
      solver->setAttr("bmc_event_sources", mixedEventSources);
      solver->setAttr("bmc_mixed_event_sources", mixedEventSources);
    }
    if (auto eventSourceDetails =
            op->getAttrOfType<ArrayAttr>("bmc_event_source_details"))
      solver->setAttr("bmc_event_source_details", eventSourceDetails);
    rewriter.createBlock(&solver.getBodyRegion());

    auto inlineBMCBlock =
        [&](Block &block, ValueRange inputs,
            ArrayRef<Type> resultTypes) -> FailureOr<SmallVector<Value>> {
      if (block.getNumArguments() != inputs.size()) {
        op.emitError("verif.bmc region has unexpected argument count when "
                     "exporting SMT-LIB");
        return failure();
      }
      auto yieldOp = dyn_cast<verif::YieldOp>(block.getTerminator());
      if (!yieldOp) {
        op.emitError(
            "expected verif.yield terminator when exporting SMT-LIB");
        return failure();
      }
      if (yieldOp.getNumOperands() != resultTypes.size()) {
        op.emitError("verif.yield operand count does not match expected output "
                     "arity when exporting SMT-LIB");
        return failure();
      }

      IRMapping mapping;
      for (auto [arg, input] : llvm::zip(block.getArguments(), inputs)) {
        Value mappedInput = input;
        if (mappedInput.getType() != arg.getType()) {
          auto cast = UnrealizedConversionCastOp::create(
              rewriter, loc, TypeRange{arg.getType()}, mappedInput);
          mappedInput = cast.getResult(0);
        }
        mapping.map(arg, mappedInput);
      }

      for (Operation &nestedOp : block.without_terminator()) {
        // Assert/assume/cover are handled by the BMC conversion, and are erased
        // using the conversion rewriter (which can be deferred). Skip them so
        // we don't clone ops slated for deletion.
        if (isa<verif::AssertOp, verif::AssumeOp, verif::CoverOp>(nestedOp))
          continue;
        Operation *cloned = rewriter.clone(nestedOp, mapping);
        for (auto [oldRes, newRes] :
             llvm::zip(nestedOp.getResults(), cloned->getResults()))
          mapping.map(oldRes, newRes);
      }

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (auto [operand, resultTy] :
           llvm::zip(yieldOp.getOperands(), resultTypes))
        results.push_back(typeConverter->materializeTargetConversion(
            rewriter, loc, resultTy, mapping.lookup(operand)));
      return results;
    };

    // Call/init or inline the init region to get initial clock values.
    SmallVector<Value> initVals;
    if (emitSMTLIB) {
      auto initValsOr = inlineBMCBlock(op.getInit().front(), ValueRange{},
                                       initOutputTy);
      if (failed(initValsOr))
        return failure();
      initVals = std::move(*initValsOr);
    } else {
      auto call = func::CallOp::create(rewriter, loc, initFuncOp, ValueRange{});
      initVals.append(call.getResults().begin(), call.getResults().end());
    }

    // InputDecls order should be <circuit arguments> <state arguments>
    // <finalChecks> <wasViolated>
    // Get list of clock indexes in circuit args
    //
    // Circuit arguments layout (after delay buffer & NFA modification):
    // [original args (clocks, inputs, regs, ticks)] [delay/past slots] [nfa states]
    //
    // Original args have size: oldCircuitInputTy.size() - totalDelaySlots - totalNFAStateSlots
    // Delay/past slots have size: totalDelaySlots
    // NFA state slots have size: totalNFAStateSlots
    size_t origCircuitArgsSize =
        oldCircuitInputTy.size() - totalDelaySlots - totalNFAStateSlots;

    bool useInitValues = !inductionStep;
    size_t initIndex = 0;
    SmallVector<Value> inputDecls;
    SmallVector<int> clockIndexes;
    size_t nonRegIndex = 0; // Track position among non-register inputs
    DenseSet<unsigned> tickArgIndices;
    for (const auto &info : nfaInfos)
      tickArgIndices.insert(info.tickArgIndex);
    size_t delayStartIndex = origCircuitArgsSize;
    size_t nfaStartIndex = origCircuitArgsSize + totalDelaySlots;
    auto makeBoolConstant = [&](Type ty, bool value) -> Value {
      if (isa<smt::BoolType>(ty))
        return smt::BoolConstantOp::create(rewriter, loc, value);
      if (auto bvTy = dyn_cast<smt::BitVectorType>(ty))
        return smt::BVConstantOp::create(rewriter, loc, value ? 1 : 0,
                                         bvTy.getWidth());
      op.emitError("unsupported boolean type in BMC conversion");
      return Value();
    };
    auto declareInput = [&](Type ty, size_t index) -> Value {
      StringAttr namePrefix;
      if (index < inputNamePrefixes.size())
        namePrefix = inputNamePrefixes[index];
      return smt::DeclareFunOp::create(rewriter, loc, ty, namePrefix);
    };
    for (auto [curIndex, oldTy, newTy] :
         llvm::enumerate(oldCircuitInputTy, circuitInputTy)) {
      // Check if this is a delay buffer or NFA state slot (added at the end)
      bool isNFAState = curIndex >= nfaStartIndex;
      bool isDelayBuffer =
          curIndex >= delayStartIndex && curIndex < nfaStartIndex;
      if (isDelayBuffer) {
        if (useInitValues) {
          // Initialize delay buffers to false (no prior history at step 0)
          Value initVal = makeBoolConstant(newTy, false);
          if (!initVal)
            return failure();
          inputDecls.push_back(initVal);
        } else {
          auto decl = declareInput(newTy, curIndex);
          inputDecls.push_back(decl);
        }
        continue;
      }
      if (isNFAState) {
        size_t stateIndex = curIndex - nfaStartIndex;
        if (useInitValues) {
          bool isStart =
              stateIndex < nfaStartStates.size() && nfaStartStates[stateIndex];
          Value initVal = makeBoolConstant(newTy, isStart);
          if (!initVal)
            return failure();
          inputDecls.push_back(initVal);
        } else {
          auto decl = declareInput(newTy, curIndex);
          inputDecls.push_back(decl);
        }
        continue;
      }

      // Check if this is a register input (registers are at the end of original args)
      bool isRegister = curIndex >= origCircuitArgsSize - numRegs;

      // Check if this is a clock - either explicit seq::ClockType or
      // an i1 that corresponds to an init clock (for i1 clocks converted via
      // ToClockOp inside the circuit)
      bool isI1Type = isa<IntegerType>(oldTy) &&
                      cast<IntegerType>(oldTy).getWidth() == 1;
      bool isClock = isa<seq::ClockType>(oldTy) ||
                     (!isRegister && isI1Type && nonRegIndex < numInitClocks);

      if (tickArgIndices.contains(curIndex)) {
        Value initVal = makeBoolConstant(newTy, false);
        if (!initVal)
          return failure();
        inputDecls.push_back(initVal);
        nonRegIndex++;
        continue;
      }
      if (isClock) {
        if (initIndex >= initVals.size()) {
          op.emitError("verif.bmc init region does not yield enough clock/state "
                       "values for SMT-LIB export");
          return failure();
        }
        if (useInitValues) {
          inputDecls.push_back(initVals[initIndex++]);
        } else {
          auto decl = declareInput(newTy, curIndex);
          inputDecls.push_back(decl);
          initIndex++;
        }
        clockIndexes.push_back(curIndex);
        nonRegIndex++;
        continue;
      }
      if (!isRegister)
        nonRegIndex++;
      if (isRegister) {
        if (!useInitValues) {
          auto decl = declareInput(newTy, curIndex);
          inputDecls.push_back(decl);
          maybeAssertKnown(curIndex, oldTy, decl, rewriter);
          continue;
        }
        auto initVal =
            initialValues[curIndex - origCircuitArgsSize + numRegs];
        if (auto initIntAttr = dyn_cast<IntegerAttr>(initVal)) {
          const auto &cstInt = initIntAttr.getValue();
          if (auto bvTy = dyn_cast<smt::BitVectorType>(newTy)) {
            assert(cstInt.getBitWidth() == bvTy.getWidth() &&
                   "Width mismatch between initial value and target type");
            auto initVal = smt::BVConstantOp::create(rewriter, loc, cstInt);
            inputDecls.push_back(initVal);
            maybeAssertKnown(curIndex, oldTy, initVal, rewriter);
            continue;
          }
          if (isa<smt::BoolType>(newTy)) {
            auto initVal =
                smt::BoolConstantOp::create(rewriter, loc, !cstInt.isZero());
            inputDecls.push_back(initVal);
            maybeAssertKnown(curIndex, oldTy, initVal, rewriter);
            continue;
          }
          op.emitError("unsupported integer initial value in BMC conversion");
          return failure();
        }
        if (auto initBoolAttr = dyn_cast<BoolAttr>(initVal)) {
          if (auto bvTy = dyn_cast<smt::BitVectorType>(newTy)) {
            auto initVal = smt::BVConstantOp::create(
                rewriter, loc, initBoolAttr.getValue() ? 1 : 0,
                bvTy.getWidth());
            inputDecls.push_back(initVal);
            maybeAssertKnown(curIndex, oldTy, initVal, rewriter);
            continue;
          }
          if (isa<smt::BoolType>(newTy)) {
            auto initVal = smt::BoolConstantOp::create(
                rewriter, loc, initBoolAttr.getValue());
            inputDecls.push_back(initVal);
            maybeAssertKnown(curIndex, oldTy, initVal, rewriter);
            continue;
          }
          op.emitError("unsupported bool initial value in BMC conversion");
          return failure();
        }
      }
      StringAttr namePrefix;
      if (curIndex < inputNamePrefixes.size())
        namePrefix = inputNamePrefixes[curIndex];
      auto decl =
          smt::DeclareFunOp::create(rewriter, loc, newTy, namePrefix);
      inputDecls.push_back(decl);
      maybeAssertKnown(curIndex, oldTy, decl, rewriter);
    }

    auto numStateArgs = initVals.size() - initIndex;
    // Add the rest of the init vals (state args)
    if (useInitValues) {
      for (; initIndex < initVals.size(); ++initIndex)
        inputDecls.push_back(initVals[initIndex]);
    } else {
      for (; initIndex < initOutputTy.size(); ++initIndex) {
        auto decl = smt::DeclareFunOp::create(rewriter, loc,
                                              initOutputTy[initIndex],
                                              StringAttr{});
        inputDecls.push_back(decl);
      }
    }

    struct ClockSourceInfo {
      unsigned pos = 0;
      bool invert = false;
    };
    DenseMap<unsigned, ClockSourceInfo> clockSourceInputs;
    if (auto sources = op->getAttrOfType<ArrayAttr>("bmc_clock_sources")) {
      for (auto attr : sources) {
        auto dict = dyn_cast<DictionaryAttr>(attr);
        if (!dict) {
          op.emitError("invalid bmc_clock_sources attribute");
          return failure();
        }
        auto argAttr = dyn_cast<IntegerAttr>(dict.get("arg_index"));
        auto posAttr = dyn_cast<IntegerAttr>(dict.get("clock_pos"));
        auto invertAttr = dyn_cast<BoolAttr>(dict.get("invert"));
        if (!argAttr || !posAttr || !invertAttr) {
          op.emitError("invalid bmc_clock_sources entry");
          return failure();
        }
        unsigned argIndex = argAttr.getValue().getZExtValue();
        unsigned clockPos = posAttr.getValue().getZExtValue();
        if (clockPos >= clockIndexes.size()) {
          op.emitError("bmc_clock_sources clock_pos out of range");
          return failure();
        }
        ClockSourceInfo info{clockPos, invertAttr.getValue()};
        auto insert = clockSourceInputs.try_emplace(argIndex, info);
        if (!insert.second &&
            (insert.first->second.pos != info.pos ||
             insert.first->second.invert != info.invert)) {
          op.emitError("clock source maps to multiple BMC clock inputs");
          return failure();
        }
      }
    }

    DenseMap<StringRef, unsigned> clockNameToPos;
    for (auto [pos, clockIdx] : llvm::enumerate(clockIndexes)) {
      if (clockIdx < static_cast<int>(inputNamePrefixes.size())) {
        auto nameAttr = inputNamePrefixes[clockIdx];
        if (nameAttr && !nameAttr.getValue().empty())
          clockNameToPos[nameAttr.getValue()] = pos;
      }
    }

    struct ClockPosInfo {
      unsigned pos = 0;
      bool invert = false;
    };
    DenseMap<Value, ClockPosInfo> clockValueToPos;
    DenseMap<Value, ClockPosInfo> clockRootToPos;
    DenseMap<StringRef, ClockPosInfo> clockKeyToPos;
    CommutativeValueEquivalence clockEquivalence;
    if (auto keys = op->getAttrOfType<ArrayAttr>("bmc_clock_keys")) {
      for (auto [idx, attr] : llvm::enumerate(keys)) {
        if (idx >= clockIndexes.size())
          break;
        auto keyAttr = dyn_cast_or_null<StringAttr>(attr);
        if (!keyAttr || keyAttr.getValue().empty())
          continue;
        ClockPosInfo info{static_cast<unsigned>(idx), false};
        auto insert = clockKeyToPos.try_emplace(keyAttr.getValue(), info);
        if (!insert.second &&
            (insert.first->second.pos != info.pos ||
             insert.first->second.invert != info.invert)) {
          op.emitError("bmc_clock_keys maps to multiple BMC clock inputs");
          return failure();
        }
      }
    }
    DenseMap<Value, StringAttr> clockValueKeys;
    op->walk([&](ltl::ClockOp clockOp) {
      if (auto keyAttr = clockOp->getAttrOfType<StringAttr>("bmc.clock_key")) {
        if (!keyAttr.getValue().empty())
          clockValueKeys.try_emplace(clockOp.getClock(), keyAttr);
      }
    });

    for (auto [pos, clockIdx] : llvm::enumerate(clockIndexes)) {
      if (clockIdx >= static_cast<int>(circuitBlock.getNumArguments()))
        continue;
      Value clockArg = circuitBlock.getArgument(clockIdx);
      SmallVector<Value> worklist{clockArg};
      DenseSet<Value> visited;
      while (!worklist.empty()) {
        Value cur = worklist.pop_back_val();
        if (!visited.insert(cur).second)
          continue;
        for (Operation *user : cur.getUsers()) {
          if (auto fromClock = dyn_cast<seq::FromClockOp>(user)) {
            clockValueToPos[fromClock.getResult()] =
                ClockPosInfo{static_cast<unsigned>(pos), false};
            continue;
          }
          if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user)) {
            if (cast->getNumResults() == 1)
              worklist.push_back(cast->getResult(0));
          }
        }
      }
    }
    // Also map direct i1 clock arguments so ltl.clock on i1 clocks can
    // resolve a clock position without requiring seq.from_clock.
    for (auto [pos, clockIdx] : llvm::enumerate(clockIndexes)) {
      if (clockIdx >= static_cast<int>(circuitBlock.getNumArguments()))
        continue;
      Value clockArg = circuitBlock.getArgument(clockIdx);
      if (auto intTy = dyn_cast<IntegerType>(clockArg.getType());
          intTy && intTy.getWidth() == 1) {
        clockValueToPos.try_emplace(
            clockArg, ClockPosInfo{static_cast<unsigned>(pos), false});
      }
    }

    auto simplifyClockValue = [&](Value val, bool &invert) -> Value {
      auto simplified = simplifyI1Value(val);
      invert = simplified.invert;
      return simplified.value ? simplified.value : val;
    };

    auto invertClockEdge = [](ltl::ClockEdge edge) -> ltl::ClockEdge {
      switch (edge) {
      case ltl::ClockEdge::Pos:
        return ltl::ClockEdge::Neg;
      case ltl::ClockEdge::Neg:
        return ltl::ClockEdge::Pos;
      case ltl::ClockEdge::Both:
        return ltl::ClockEdge::Both;
      }
      return edge;
    };

    auto traceClockRoot = [&](Value value, BlockArgument &root) -> bool {
      return traceI1ValueRoot(value, root);
    };

    auto resolveClockSourcePos = [&](Value clockValue)
        -> std::optional<ClockPosInfo> {
      BlockArgument root;
      if (!traceClockRoot(clockValue, root))
        return std::nullopt;
      auto it = clockSourceInputs.find(root.getArgNumber());
      if (it == clockSourceInputs.end())
        return std::nullopt;
      return ClockPosInfo{it->second.pos, it->second.invert};
    };

    std::function<std::optional<ClockPosInfo>(Value)>
        resolveClockPosInfoSimple =
            [&](Value clockValue) -> std::optional<ClockPosInfo> {
      if (!clockValue)
        return std::nullopt;
      if (!clockKeyToPos.empty()) {
        if (auto itKey = clockValueKeys.find(clockValue);
            itKey != clockValueKeys.end()) {
          auto it = clockKeyToPos.find(itKey->second.getValue());
          if (it != clockKeyToPos.end())
            return it->second;
        }
        if (auto key = getI1ValueKey(clockValue)) {
          auto it = clockKeyToPos.find(*key);
          if (it != clockKeyToPos.end())
            return it->second;
        }
      }
      bool invert = false;
      Value simplified = simplifyClockValue(clockValue, invert);
      if (!simplified)
        return std::nullopt;
      if (auto it = clockValueToPos.find(simplified);
          it != clockValueToPos.end())
        return it->second;
      for (auto &entry : clockValueToPos) {
        if (clockEquivalence.isEquivalent(simplified, entry.first)) {
          ClockPosInfo info = entry.second;
          if (invert)
            info.invert = !info.invert;
          return info;
        }
      }
      if (auto info = resolveClockSourcePos(simplified)) {
        if (invert)
          info->invert = !info->invert;
        return info;
      }
      return std::nullopt;
    };

    std::function<std::optional<ClockPosInfo>(Value)> resolveClockPosInfo =
        [&](Value clockValue) -> std::optional<ClockPosInfo> {
      if (auto info = resolveClockPosInfoSimple(clockValue))
        return info;
      if (clockEquivalenceUF.has(clockValue)) {
        bool invert = false;
        Value root = clockEquivalenceUF.find(clockValue, invert);
        auto it = clockRootToPos.find(root);
        if (it != clockRootToPos.end()) {
          ClockPosInfo info = it->second;
          if (invert)
            info.invert = !info.invert;
          return info;
        }
      }
      return std::nullopt;
    };

    for (auto &entry : clockValueToPos) {
      bool invert = false;
      Value root = clockEquivalenceUF.find(entry.first, invert);
      ClockPosInfo info = entry.second;
      if (invert)
        info.invert = !info.invert;
      auto insert = clockRootToPos.try_emplace(root, info);
      if (!insert.second &&
          (insert.first->second.pos != info.pos ||
           insert.first->second.invert != info.invert)) {
        reportDerivedClockConflict(
            op.getOperation(),
            "derived clock maps to multiple BMC clock inputs");
      }
    }

    if (!clockEquivalenceUF.parent.empty()) {
      SmallVector<Value> ufValues;
      ufValues.reserve(clockEquivalenceUF.parent.size());
      for (auto &entry : clockEquivalenceUF.parent)
        ufValues.push_back(entry.first);
      for (Value value : ufValues) {
        bool invert = false;
        Value root = clockEquivalenceUF.find(value, invert);
        if (clockRootToPos.contains(root))
          continue;
        auto info = resolveClockPosInfoSimple(value);
        if (!info)
          continue;
        if (invert)
          info->invert = !info->invert;
        clockRootToPos.try_emplace(root, *info);
      }
    }

    if (derivedClockConflict)
      return failure();

    SmallVector<NFATickGateInfo> nfaTickGateInfos;
    nfaTickGateInfos.reserve(nfaInfos.size());
    for (const auto &info : nfaInfos) {
      NFATickGateInfo gateInfo;
      gateInfo.tickArgIndex = info.tickArgIndex;
      gateInfo.edge = info.edge.value_or(ltl::ClockEdge::Pos);
      gateInfo.hasExplicitClock = info.clockName || info.clockValue;
      if (info.clockName && !info.clockName.getValue().empty()) {
        auto it = clockNameToPos.find(info.clockName.getValue());
        if (it != clockNameToPos.end())
          gateInfo.clockPos = it->second;
      } else if (info.clockValue) {
        if (auto posInfo = resolveClockPosInfo(info.clockValue)) {
          gateInfo.clockPos = posInfo->pos;
          gateInfo.invert = posInfo->invert;
        }
      } else if (clockIndexes.size() == 1) {
        gateInfo.clockPos = 0;
      }
      nfaTickGateInfos.push_back(gateInfo);
    }

    struct InferredClockInfo {
      std::optional<unsigned> pos;
      std::optional<ltl::ClockEdge> edge;
      bool sawClock = false;
      bool conflict = false;
    };

    auto inferClockFromProperty = [&](Value prop) -> InferredClockInfo {
      InferredClockInfo result;
      if (!prop)
        return result;
      SmallVector<Operation *> worklist;
      if (auto *def = prop.getDefiningOp())
        worklist.push_back(def);
      llvm::DenseSet<Operation *> visited;
      std::optional<unsigned> clockPos;
      std::optional<ltl::ClockEdge> edge;
      bool conflict = false;
      auto mergeEdge = [&](ltl::ClockEdge newEdge) {
        if (!edge) {
          edge = newEdge;
          return;
        }
        if (*edge == newEdge)
          return;
        if (*edge == ltl::ClockEdge::Both ||
            newEdge == ltl::ClockEdge::Both) {
          edge = ltl::ClockEdge::Both;
          return;
        }
        if ((*edge == ltl::ClockEdge::Pos &&
             newEdge == ltl::ClockEdge::Neg) ||
            (*edge == ltl::ClockEdge::Neg &&
             newEdge == ltl::ClockEdge::Pos)) {
          edge = ltl::ClockEdge::Both;
          return;
        }
        edge.reset();
      };
      while (!worklist.empty()) {
        Operation *cur = worklist.pop_back_val();
        if (!visited.insert(cur).second)
          continue;
        if (!isa_and_nonnull<ltl::LTLDialect>(cur->getDialect()))
          continue;
        if (auto clockOp = dyn_cast<ltl::ClockOp>(cur)) {
          result.sawClock = true;
          if (auto info = resolveClockPosInfo(clockOp.getClock())) {
            if (!clockPos)
              clockPos = info->pos;
            else if (*clockPos != info->pos)
              conflict = true;
            mergeEdge(info->invert ? invertClockEdge(clockOp.getEdge())
                                   : clockOp.getEdge());
          } else {
            mergeEdge(clockOp.getEdge());
          }
        }
        for (Value operand : cur->getOperands()) {
          if (Operation *def = operand.getDefiningOp())
            worklist.push_back(def);
        }
      }
      if (conflict)
        return result;
      result.pos = clockPos;
      result.edge = edge;
      return result;
    };

    bool unmappedClockError = false;
    auto reportUnmappedClock = [&](Location loc, StringRef detail) {
      if (unmappedClockError)
        return;
      emitError(loc) << detail;
      unmappedClockError = true;
    };
    auto reportUnmappedClockForOp = [&](Operation *op, StringRef detail) {
      reportUnmappedClock(op->getLoc(), detail);
    };

    auto resolveClockPos = [&](Operation *op, StringAttr clockName,
                               Value clockValue) -> std::optional<unsigned> {
      if (risingClocksOnly)
        return std::nullopt;
      if (clockName && !clockName.getValue().empty()) {
        if (clockIndexes.empty()) {
          reportUnmappedClockForOp(
              op, "clock name does not match any BMC clock input");
          return std::nullopt;
        }
        auto it = clockNameToPos.find(clockName.getValue());
        if (it != clockNameToPos.end())
          return it->second;
        reportUnmappedClockForOp(op,
                                 "clock name does not match any BMC clock input");
        return std::nullopt;
      }
      if (clockValue) {
        if (clockIndexes.empty()) {
          reportUnmappedClockForOp(
              op, "clocked property uses a clock that is not a BMC "
                  "clock input");
          return std::nullopt;
        }
        if (auto info = resolveClockPosInfo(clockValue))
          return info->pos;
        reportUnmappedClockForOp(
            op, "clocked property uses a clock that is not a BMC "
                "clock input");
        return std::nullopt;
      }
      return std::nullopt;
    };

    auto resolveCheckClockPos =
        [&](SmallVectorImpl<std::optional<unsigned>> &out,
            MutableArrayRef<NonFinalCheckInfo> infos,
            ArrayRef<Value> props) {
          out.reserve(infos.size());
          for (size_t idx = 0; idx < infos.size(); ++idx) {
            auto &info = infos[idx];
            std::optional<unsigned> pos;
            auto inferred = inferClockFromProperty(
                idx < props.size() ? props[idx] : Value{});
            if (!info.edge && inferred.edge)
              info.edge = inferred.edge;
            if (inferred.pos)
              pos = inferred.pos;
            if (info.clockName && !info.clockName.getValue().empty()) {
              auto it = clockNameToPos.find(info.clockName.getValue());
              if (it != clockNameToPos.end()) {
                if (inferred.pos && *inferred.pos != it->second) {
                  reportUnmappedClock(
                      info.loc,
                      "clocked property uses conflicting clock information; "
                      "ensure each property uses a single clock/edge");
                }
                pos = it->second;
              }
            } else if (!risingClocksOnly && clockIndexes.size() == 1) {
              pos = 0;
            }
            if (!pos && !risingClocksOnly) {
              bool explicitClock =
                  (info.clockName && !info.clockName.getValue().empty()) ||
                  inferred.sawClock;
              if (explicitClock) {
                reportUnmappedClock(info.loc, "clocked property uses a clock that is "
                                             "not a BMC clock input");
              }
            }
            out.push_back(pos);
          }
        };

    SmallVector<std::optional<unsigned>> nonFinalCheckClockPos;
    resolveCheckClockPos(nonFinalCheckClockPos, nonFinalCheckInfos,
                         nonFinalCheckProps);
    SmallVector<std::optional<unsigned>> finalCheckClockPos;
    resolveCheckClockPos(finalCheckClockPos, finalCheckInfos, finalCheckProps);
    size_t numNonStateArgs =
        oldCircuitInputTy.size() - numRegs - totalDelaySlots -
        totalNFAStateSlots;

    struct EventArmWitnessInfo {
      enum class Kind { Signal, Sequence };
      Kind kind = Kind::Signal;
      StringAttr witnessName;
      std::optional<unsigned> sourceArgIndex;
      std::unique_ptr<ResolvedNamedBoolExpr> sourceExpr;
      std::optional<unsigned> iffArgIndex;
      std::unique_ptr<ResolvedNamedBoolExpr> iffExpr;
      ltl::ClockEdge edge = ltl::ClockEdge::Both;
    };
    SmallVector<EventArmWitnessInfo> armWitnesses;
    if (auto eventSourceDetails =
            op->getAttrOfType<ArrayAttr>("bmc_event_source_details")) {
      auto parseEdge = [&](StringRef edgeText)
          -> std::optional<ltl::ClockEdge> {
        if (edgeText == "posedge")
          return ltl::ClockEdge::Pos;
        if (edgeText == "negedge")
          return ltl::ClockEdge::Neg;
        if (edgeText == "both")
          return ltl::ClockEdge::Both;
        return std::nullopt;
      };

      DenseMap<uint64_t, StringAttr> witnessByArm;
      for (auto [setIdx, detailSetAttr] : llvm::enumerate(eventSourceDetails)) {
        auto detailSet = dyn_cast<ArrayAttr>(detailSetAttr);
        if (!detailSet)
          continue;
        for (auto [armIdx, detailAttr] : llvm::enumerate(detailSet)) {
          auto detail = dyn_cast<DictionaryAttr>(detailAttr);
          if (!detail)
            continue;
          auto kindAttr = dyn_cast_or_null<StringAttr>(detail.get("kind"));
          if (!kindAttr)
            continue;
          StringRef kind = kindAttr.getValue();

          EventArmWitnessInfo witnessInfo;
          witnessInfo.kind = EventArmWitnessInfo::Kind::Signal;
          bool sourceResolved = false;
          if (kind == "signal") {
            auto edgeAttr = dyn_cast_or_null<StringAttr>(detail.get("edge"));
            if (!edgeAttr)
              continue;
            auto edge = parseEdge(edgeAttr.getValue());
            if (!edge)
              continue;
            witnessInfo.edge = *edge;
            sourceResolved = resolveStructuredExprFromDetail(
                detail, "signal", inputNameToArgIndex, numNonStateArgs,
                witnessInfo.sourceArgIndex, witnessInfo.sourceExpr);
            if (!sourceResolved)
              if (auto signalExprAttr =
                      dyn_cast_or_null<StringAttr>(detail.get("signal_expr")))
                if (auto parsedExpr =
                        parseNamedBoolExpr(signalExprAttr.getValue()))
                  if (auto resolvedExpr = resolveNamedBoolExpr(
                          *parsedExpr, inputNameToArgIndex, numNonStateArgs)) {
                    witnessInfo.sourceExpr = std::move(resolvedExpr);
                    sourceResolved = true;
                  }
          } else if (kind == "sequence") {
            witnessInfo.kind = EventArmWitnessInfo::Kind::Sequence;
            sourceResolved = resolveStructuredExprFromDetail(
                detail, "sequence", inputNameToArgIndex, numNonStateArgs,
                witnessInfo.sourceArgIndex, witnessInfo.sourceExpr);
            if (!sourceResolved)
              if (auto sequenceExprAttr =
                      dyn_cast_or_null<StringAttr>(detail.get("sequence_expr")))
                if (auto parsedExpr =
                        parseNamedBoolExpr(sequenceExprAttr.getValue()))
                  if (auto resolvedExpr = resolveNamedBoolExpr(
                          *parsedExpr, inputNameToArgIndex, numNonStateArgs)) {
                    witnessInfo.sourceExpr = std::move(resolvedExpr);
                    sourceResolved = true;
                  }
          } else {
            continue;
          }
          if (!sourceResolved)
            continue;

          bool hasIffConstraint =
              detail.get("iff_name") || detail.get("iff_expr") ||
              detail.get("iff_lsb") || detail.get("iff_msb") ||
              detail.get("iff_reduction") || detail.get("iff_bitwise_not") ||
              detail.get("iff_logical_not") ||
              detail.get("iff_group") ||
              detail.get("iff_group_depth") ||
              detail.get("iff_bin_op") || detail.get("iff_unary_op") ||
              detail.get("iff_dyn_index_name") || detail.get("iff_dyn_sign") ||
              detail.get("iff_dyn_offset") || detail.get("iff_dyn_width");
          (void)resolveStructuredExprFromDetail(
              detail, "iff", inputNameToArgIndex, numNonStateArgs,
              witnessInfo.iffArgIndex, witnessInfo.iffExpr);
          if (!witnessInfo.iffArgIndex && !witnessInfo.iffExpr)
            if (auto iffExprAttr =
                    dyn_cast_or_null<StringAttr>(detail.get("iff_expr")))
              if (auto parsedExpr = parseNamedBoolExpr(iffExprAttr.getValue()))
                if (auto resolvedExpr = resolveNamedBoolExpr(
                        *parsedExpr, inputNameToArgIndex, numNonStateArgs))
                  witnessInfo.iffExpr = std::move(resolvedExpr);

          if (hasIffConstraint && !witnessInfo.iffArgIndex &&
              !witnessInfo.iffExpr)
            continue;

          StringAttr witnessName =
              dyn_cast_or_null<StringAttr>(detail.get("witness_name"));
          if (!witnessName || witnessName.getValue().empty()) {
            witnessName = rewriter.getStringAttr(
                (Twine("event_arm_witness_") + Twine(setIdx) + "_" +
                 Twine(armIdx))
                    .str());
          }
          witnessInfo.witnessName = witnessName;
          armWitnesses.push_back(std::move(witnessInfo));
          uint64_t key = (static_cast<uint64_t>(setIdx) << 32) | armIdx;
          witnessByArm[key] = witnessName;
        }
      }

      if (!witnessByArm.empty()) {
        SmallVector<Attribute> rewrittenDetailSets;
        rewrittenDetailSets.reserve(eventSourceDetails.size());
        for (auto [setIdx, detailSetAttr] :
             llvm::enumerate(eventSourceDetails)) {
          auto detailSet = dyn_cast<ArrayAttr>(detailSetAttr);
          if (!detailSet) {
            rewrittenDetailSets.push_back(detailSetAttr);
            continue;
          }
          SmallVector<Attribute> rewrittenDetails;
          rewrittenDetails.reserve(detailSet.size());
          for (auto [armIdx, detailAttr] : llvm::enumerate(detailSet)) {
            auto detail = dyn_cast<DictionaryAttr>(detailAttr);
            if (!detail) {
              rewrittenDetails.push_back(detailAttr);
              continue;
            }
            uint64_t key = (static_cast<uint64_t>(setIdx) << 32) | armIdx;
            auto witnessIt = witnessByArm.find(key);
            if (witnessIt == witnessByArm.end()) {
              rewrittenDetails.push_back(detailAttr);
              continue;
            }
            SmallVector<NamedAttribute> attrs(detail.begin(), detail.end());
            bool replaced = false;
            for (auto &namedAttr : attrs) {
              if (namedAttr.getName().getValue() == "witness_name") {
                namedAttr = rewriter.getNamedAttr("witness_name",
                                                  witnessIt->second);
                replaced = true;
                break;
              }
            }
            if (!replaced)
              attrs.push_back(
                  rewriter.getNamedAttr("witness_name", witnessIt->second));
            rewrittenDetails.push_back(rewriter.getDictionaryAttr(attrs));
          }
          rewrittenDetailSets.push_back(rewriter.getArrayAttr(rewrittenDetails));
        }
        solver->setAttr("bmc_event_source_details",
                        rewriter.getArrayAttr(rewrittenDetailSets));
      }
    }

    if (!risingClocksOnly) {
      for (auto &info : delayInfos) {
        if (info.clockName || info.clockValue) {
          (void)resolveClockPos(info.op.getOperation(), info.clockName,
                                info.clockValue);
        }
      }
      for (auto &info : pastInfos) {
        if (info.clockName || info.clockValue) {
          (void)resolveClockPos(info.op.getOperation(), info.clockName,
                                info.clockValue);
        }
      }
    }

    if (unmappedClockError)
      return failure();

    SmallVector<unsigned> regClockToLoopIndex;
    SmallVector<bool> regClockInverts;
    bool usePerRegClocks =
        !risingClocksOnly && clockIndexes.size() > 1 && numRegs > 0;
    if (usePerRegClocks) {
      auto regClocksAttr = op->getAttrOfType<ArrayAttr>("bmc_reg_clocks");
      auto regClockSourcesAttr =
          op->getAttrOfType<ArrayAttr>("bmc_reg_clock_sources");
      bool regClocksValid =
          regClocksAttr && regClocksAttr.size() == numRegs;
      bool regSourcesValid =
          regClockSourcesAttr && regClockSourcesAttr.size() == numRegs;
      if (!regClocksValid && !regSourcesValid) {
        op.emitError(
            "multi-clock BMC requires bmc_reg_clocks or bmc_reg_clock_sources "
            "with one entry per register");
        return failure();
      }
      DenseMap<StringRef, unsigned> inputNameToIndex;
      for (auto [idx, nameAttr] : llvm::enumerate(inputNamePrefixes)) {
        if (nameAttr && !nameAttr.getValue().empty())
          inputNameToIndex[nameAttr.getValue()] = idx;
      }
      auto mapArgIndexToClockPos = [&](unsigned argIndex)
          -> std::optional<unsigned> {
        if (auto it = clockSourceInputs.find(argIndex);
            it != clockSourceInputs.end())
          return it->second.pos;
        auto clockIt =
            llvm::find(clockIndexes, static_cast<int>(argIndex));
        if (clockIt == clockIndexes.end())
          return std::nullopt;
        return static_cast<unsigned>(clockIt - clockIndexes.begin());
      };
      regClockToLoopIndex.reserve(numRegs);
      regClockInverts.reserve(numRegs);
      for (unsigned regIndex = 0; regIndex < numRegs; ++regIndex) {
        bool mapped = false;
        bool invert = false;
        if (regSourcesValid) {
          auto dict = dyn_cast<DictionaryAttr>(
              regClockSourcesAttr[regIndex]);
          if (dict) {
            auto invertAttr = dyn_cast_or_null<BoolAttr>(dict.get("invert"));
            bool dictInvert = invertAttr ? invertAttr.getValue() : false;

            // Support clock-key-based mapping for derived clocks. This is used
            // when a register clock cannot be traced to a single module input
            // (e.g. clocks derived from complex i1 expressions). LowerToBMC
            // supplies bmc_clock_keys to map expression keys to inserted BMC
            // clock inputs.
            auto keyAttr = dyn_cast_or_null<StringAttr>(dict.get("clock_key"));
            if (keyAttr && !keyAttr.getValue().empty()) {
              auto it = clockKeyToPos.find(keyAttr.getValue());
              if (it != clockKeyToPos.end()) {
                regClockToLoopIndex.push_back(it->second.pos);
                invert = dictInvert ^ it->second.invert;
                mapped = true;
              }
            }

            auto argAttr =
                dyn_cast_or_null<IntegerAttr>(dict.get("arg_index"));
            if (!mapped && argAttr) {
              unsigned argIndex = argAttr.getValue().getZExtValue();
              if (auto pos = mapArgIndexToClockPos(argIndex)) {
                regClockToLoopIndex.push_back(*pos);
                invert = dictInvert;
                mapped = true;
              }
            }
          }
        }
        if (!mapped && regClocksValid) {
          auto nameAttr =
              dyn_cast_or_null<StringAttr>(regClocksAttr[regIndex]);
          if (nameAttr && !nameAttr.getValue().empty()) {
            auto nameIt = inputNameToIndex.find(nameAttr.getValue());
            if (nameIt == inputNameToIndex.end()) {
              op.emitError("bmc_reg_clocks entry does not match any input name");
              return failure();
            }
            unsigned clockInputIndex = nameIt->second;
            auto clockIt =
                llvm::find(clockIndexes, static_cast<int>(clockInputIndex));
            if (clockIt == clockIndexes.end()) {
              op.emitError("bmc_reg_clocks entry does not name a clock input");
              return failure();
            }
            regClockToLoopIndex.push_back(
                static_cast<unsigned>(clockIt - clockIndexes.begin()));
            mapped = true;
          }
        }
        if (!mapped) {
          op.emitError("multi-clock BMC requires named clock entries in "
                       "bmc_reg_clocks or valid entries in "
                       "bmc_reg_clock_sources");
          return failure();
        }
        regClockInverts.push_back(invert);
      }
    }

    bool checkFinalAtEnd = true;
    if (!risingClocksOnly && clockIndexes.size() == 1 &&
        (boundValue % 2 != 0)) {
      // In non-rising mode, odd bounds end on a negedge; skip final-only checks.
      checkFinalAtEnd = false;
    }

    Value lowerBound;
    Value step;
    Value upperBound;
    if (!emitSMTLIB) {
      lowerBound = arith::ConstantOp::create(rewriter, loc,
                                             rewriter.getI32IntegerAttr(0));
      step = arith::ConstantOp::create(rewriter, loc,
                                       rewriter.getI32IntegerAttr(1));
      upperBound =
          arith::ConstantOp::create(rewriter, loc, adaptor.getBoundAttr());
    }
    Value constFalse =
        emitSMTLIB
            ? smt::BoolConstantOp::create(rewriter, loc, false).getResult()
            : arith::ConstantOp::create(rewriter, loc,
                                        rewriter.getBoolAttr(false))
                  .getResult();
    Value constTrue =
        emitSMTLIB
            ? smt::BoolConstantOp::create(rewriter, loc, true).getResult()
            : arith::ConstantOp::create(rewriter, loc,
                                        rewriter.getBoolAttr(true))
                  .getResult();
    // Initialize final check iter_args with false values matching their types
    // (these will be updated with circuit outputs). Types may be !smt.bv<1>
    // for i1 properties or !smt.bool for LTL properties.
    for (size_t i = 0; i < numFinalChecks; ++i) {
      Type checkTy = circuitOutputTy[circuitOutputTy.size() - numFinalChecks + i];
      if (isa<smt::BoolType>(checkTy))
        inputDecls.push_back(smt::BoolConstantOp::create(rewriter, loc, false));
      else
        inputDecls.push_back(smt::BVConstantOp::create(rewriter, loc, 0, 1));
    }
    inputDecls.push_back(constFalse); // wasViolated?

    auto toSMTBool = [&](OpBuilder &builder, Value value) -> FailureOr<Value> {
      if (isa<smt::BoolType>(value.getType()))
        return value;
      if (auto bvTy = dyn_cast<smt::BitVectorType>(value.getType())) {
        if (bvTy.getWidth() != 1) {
          op.emitError("event-arm witness lowering expects i1-compatible "
                       "signal values");
          return failure();
        }
        auto one = smt::BVConstantOp::create(builder, loc, 1, 1);
        return Value(smt::EqOp::create(builder, loc, value, one));
      }
      op.emitError("event-arm witness lowering expects bool or bv<1> "
                   "signal values");
      return failure();
    };

    auto ignoreAssertionsUntilAttr =
        op->getAttrOfType<IntegerAttr>("ignore_asserts_until");
    std::optional<uint64_t> ignoreAssertionsUntilValue;
    if (ignoreAssertionsUntilAttr)
      ignoreAssertionsUntilValue =
          ignoreAssertionsUntilAttr.getValue().getZExtValue();

    auto boolTy = smt::BoolType::get(rewriter.getContext());
    auto witnessFalse = smt::BoolConstantOp::create(rewriter, loc, false);
    auto evalResolvedExpr = [&](auto &self, OpBuilder &builder, ValueRange values,
                                const ResolvedNamedBoolExpr &expr)
        -> FailureOr<Value> {
      auto evalArgValue =
          [&](unsigned argIndex,
              std::optional<ResolvedNamedBoolExpr::ArgSlice> slice)
          -> FailureOr<Value> {
        if (argIndex >= values.size())
          return failure();
        Value value = values[argIndex];
        if (!slice)
          return value;
        auto materializeAsSMTBV = [&](Value input, unsigned width)
            -> FailureOr<Value> {
          auto targetTy = smt::BitVectorType::get(builder.getContext(), width);
          Value bv = input;
          if (!isa<smt::BitVectorType>(bv.getType())) {
            bv = typeConverter->materializeTargetConversion(builder, loc, targetTy,
                                                            input);
            if (!bv)
              return failure();
          }
          auto bvTy = dyn_cast<smt::BitVectorType>(bv.getType());
          if (!bvTy)
            return failure();
          if (bvTy.getWidth() == width)
            return bv;
          if (bvTy.getWidth() > width)
            return Value(smt::ExtractOp::create(builder, loc, targetTy, 0, bv));
          unsigned padWidth = width - bvTy.getWidth();
          auto zeroPad =
              smt::BVConstantOp::create(builder, loc, APInt(padWidth, 0));
          return Value(smt::ConcatOp::create(builder, loc, zeroPad, bv));
        };
        if (slice->dynamicIndexArg) {
          if (*slice->dynamicIndexArg >= values.size())
            return failure();
          auto baseTy = dyn_cast<smt::BitVectorType>(value.getType());
          if (!baseTy || slice->dynamicWidth == 0 ||
              slice->dynamicWidth > baseTy.getWidth())
            return failure();
          auto indexBvOr =
              materializeAsSMTBV(values[*slice->dynamicIndexArg], baseTy.getWidth());
          if (failed(indexBvOr))
            return failure();
          Value shift = *indexBvOr;
          if (slice->dynamicIndexSign == -1)
            shift = smt::BVNegOp::create(builder, loc, shift);
          if (slice->dynamicIndexOffset != 0) {
            auto offset = smt::BVConstantOp::create(
                builder, loc,
                APInt(baseTy.getWidth(), slice->dynamicIndexOffset, true));
            shift = smt::BVAddOp::create(builder, loc, shift, offset);
          }
          Value shifted = smt::BVLShrOp::create(builder, loc, value, shift);
          auto extractTy =
              smt::BitVectorType::get(builder.getContext(), slice->dynamicWidth);
          return Value(
              smt::ExtractOp::create(builder, loc, extractTy, 0, shifted));
        }
        unsigned lsb = slice->lsb;
        unsigned msb = slice->msb;
        if (lsb > msb)
          return failure();
        if (isa<smt::BoolType>(value.getType())) {
          if (lsb != 0 || msb != 0)
            return failure();
          return value;
        }
        if (auto bvTy = dyn_cast<smt::BitVectorType>(value.getType())) {
          if (msb >= bvTy.getWidth())
            return failure();
          unsigned width = msb - lsb + 1;
          auto extractTy = smt::BitVectorType::get(builder.getContext(), width);
          return Value(
              smt::ExtractOp::create(builder, loc, extractTy, lsb, value));
        }
        if (auto intTy = dyn_cast<IntegerType>(value.getType())) {
          if (intTy.getWidth() == 1 && lsb == 0 && msb == 0)
            return value;
        }
        return failure();
      };
      auto evalReduce = [&](Value input, ResolvedNamedBoolExpr::Kind kind)
          -> FailureOr<Value> {
        bool invertResult = kind == ResolvedNamedBoolExpr::Kind::ReduceNand ||
                            kind == ResolvedNamedBoolExpr::Kind::ReduceNor ||
                            kind == ResolvedNamedBoolExpr::Kind::ReduceXnor;
        ResolvedNamedBoolExpr::Kind baseKind = kind;
        if (kind == ResolvedNamedBoolExpr::Kind::ReduceNand)
          baseKind = ResolvedNamedBoolExpr::Kind::ReduceAnd;
        else if (kind == ResolvedNamedBoolExpr::Kind::ReduceNor)
          baseKind = ResolvedNamedBoolExpr::Kind::ReduceOr;
        else if (kind == ResolvedNamedBoolExpr::Kind::ReduceXnor)
          baseKind = ResolvedNamedBoolExpr::Kind::ReduceXor;

        auto maybeInvert = [&](Value value) -> Value {
          if (!invertResult)
            return value;
          return smt::NotOp::create(builder, loc, value);
        };

        if (isa<smt::BoolType>(input.getType()))
          return maybeInvert(input);
        auto bvTy = dyn_cast<smt::BitVectorType>(input.getType());
        if (!bvTy)
          return failure();
        unsigned width = bvTy.getWidth();
        if (width == 0)
          return failure();
        if (baseKind == ResolvedNamedBoolExpr::Kind::ReduceOr) {
          auto zero = smt::BVConstantOp::create(builder, loc, APInt(width, 0));
          return maybeInvert(
              Value(smt::DistinctOp::create(builder, loc, input, zero)));
        }
        if (baseKind == ResolvedNamedBoolExpr::Kind::ReduceAnd) {
          auto allOnes =
              smt::BVConstantOp::create(builder, loc, APInt::getAllOnes(width));
          return maybeInvert(
              Value(smt::EqOp::create(builder, loc, input, allOnes)));
        }
        if (baseKind == ResolvedNamedBoolExpr::Kind::ReduceXor) {
          auto bitTy = smt::BitVectorType::get(builder.getContext(), 1);
          auto one = smt::BVConstantOp::create(builder, loc, APInt(1, 1));
          Value parity = smt::BoolConstantOp::create(builder, loc, false);
          for (unsigned bit = 0; bit < width; ++bit) {
            auto bitValue =
                smt::ExtractOp::create(builder, loc, bitTy, bit, input);
            Value bitBool = smt::EqOp::create(builder, loc, bitValue, one);
            parity = smt::DistinctOp::create(builder, loc, parity, bitBool);
          }
          return maybeInvert(parity);
        }
        return failure();
      };
      auto evalBitwiseNotAsBool = [&](Value input) -> FailureOr<Value> {
        if (isa<smt::BoolType>(input.getType()))
          return Value(smt::NotOp::create(builder, loc, input));
        if (auto bvTy = dyn_cast<smt::BitVectorType>(input.getType())) {
          unsigned width = bvTy.getWidth();
          if (width == 0)
            return failure();
          auto allOnes =
              smt::BVConstantOp::create(builder, loc, APInt::getAllOnes(width));
          return Value(smt::DistinctOp::create(builder, loc, input, allOnes));
        }
        if (auto intTy = dyn_cast<IntegerType>(input.getType())) {
          if (intTy.getWidth() == 1) {
            auto boolOr = toSMTBool(builder, input);
            if (failed(boolOr))
              return failure();
            return Value(smt::NotOp::create(builder, loc, *boolOr));
          }
        }
        return failure();
      };
      auto getIntegralWidth = [&](Type ty) -> std::optional<unsigned> {
        if (isa<smt::BoolType>(ty))
          return 1;
        if (auto bvTy = dyn_cast<smt::BitVectorType>(ty))
          return bvTy.getWidth();
        if (auto intTy = dyn_cast<IntegerType>(ty); intTy &&
                                                    intTy.getWidth() > 0)
          return static_cast<unsigned>(intTy.getWidth());
        return std::nullopt;
      };
      auto materializeAsSMTBVForCmp = [&](Value input, unsigned width,
                                          bool signExtend)
          -> FailureOr<Value> {
        auto targetTy = smt::BitVectorType::get(builder.getContext(), width);
        Value bv = input;
        if (isa<smt::BoolType>(input.getType())) {
          auto one = smt::BVConstantOp::create(builder, loc, APInt(1, 1));
          auto zero = smt::BVConstantOp::create(builder, loc, APInt(1, 0));
          bv = smt::IteOp::create(builder, loc, input, one, zero);
        } else if (!isa<smt::BitVectorType>(input.getType())) {
          bv = typeConverter->materializeTargetConversion(builder, loc, targetTy,
                                                          input);
          if (!bv)
            return failure();
        }
        auto bvTy = dyn_cast<smt::BitVectorType>(bv.getType());
        if (!bvTy)
          return failure();
        if (bvTy.getWidth() == width)
          return bv;
        if (bvTy.getWidth() > width)
          return Value(smt::ExtractOp::create(builder, loc, targetTy, 0, bv));
        unsigned padWidth = width - bvTy.getWidth();
        Value pad;
        if (!signExtend || bvTy.getWidth() == 0) {
          pad = smt::BVConstantOp::create(builder, loc, APInt(padWidth, 0));
        } else {
          auto oneBitTy = smt::BitVectorType::get(builder.getContext(), 1);
          Value signBit = smt::ExtractOp::create(builder, loc, oneBitTy,
                                                 bvTy.getWidth() - 1, bv);
          pad = smt::RepeatOp::create(builder, loc, padWidth, signBit);
        }
        return Value(smt::ConcatOp::create(builder, loc, pad, bv));
      };
      auto evalRelCompare = [&](Value lhs, Value rhs,
                                ResolvedNamedBoolExpr::Kind kind,
                                bool compareSigned) -> FailureOr<Value> {
        auto lhsWidth = getIntegralWidth(lhs.getType());
        auto rhsWidth = getIntegralWidth(rhs.getType());
        if (lhsWidth && rhsWidth) {
          unsigned commonWidth = std::max(*lhsWidth, *rhsWidth);
          auto lhsBvOr =
              materializeAsSMTBVForCmp(lhs, commonWidth, compareSigned);
          auto rhsBvOr =
              materializeAsSMTBVForCmp(rhs, commonWidth, compareSigned);
          if (succeeded(lhsBvOr) && succeeded(rhsBvOr)) {
            smt::BVCmpPredicate pred;
            switch (kind) {
            case ResolvedNamedBoolExpr::Kind::Lt:
              pred = compareSigned ? smt::BVCmpPredicate::slt
                                   : smt::BVCmpPredicate::ult;
              break;
            case ResolvedNamedBoolExpr::Kind::Le:
              pred = compareSigned ? smt::BVCmpPredicate::sle
                                   : smt::BVCmpPredicate::ule;
              break;
            case ResolvedNamedBoolExpr::Kind::Gt:
              pred = compareSigned ? smt::BVCmpPredicate::sgt
                                   : smt::BVCmpPredicate::ugt;
              break;
            case ResolvedNamedBoolExpr::Kind::Ge:
              pred = compareSigned ? smt::BVCmpPredicate::sge
                                   : smt::BVCmpPredicate::uge;
              break;
            default:
              return failure();
            }
            return Value(
                smt::BVCmpOp::create(builder, loc, pred, *lhsBvOr, *rhsBvOr));
          }
        }
        auto lhsBoolOr = toSMTBool(builder, lhs);
        auto rhsBoolOr = toSMTBool(builder, rhs);
        if (failed(lhsBoolOr) || failed(rhsBoolOr))
          return failure();
        Value lhsBool = *lhsBoolOr;
        Value rhsBool = *rhsBoolOr;
        switch (kind) {
        case ResolvedNamedBoolExpr::Kind::Lt:
          return Value(smt::AndOp::create(
              builder, loc, Value(smt::NotOp::create(builder, loc, lhsBool)),
              rhsBool));
        case ResolvedNamedBoolExpr::Kind::Le:
          return Value(smt::OrOp::create(
              builder, loc, Value(smt::NotOp::create(builder, loc, lhsBool)),
              rhsBool));
        case ResolvedNamedBoolExpr::Kind::Gt:
          return Value(smt::AndOp::create(
              builder, loc, lhsBool,
              Value(smt::NotOp::create(builder, loc, rhsBool))));
        case ResolvedNamedBoolExpr::Kind::Ge:
          return Value(smt::OrOp::create(
              builder, loc, lhsBool,
              Value(smt::NotOp::create(builder, loc, rhsBool))));
        default:
          return failure();
        }
      };
      auto evalSubExprAsValue = [&](auto &selfValue,
                                    const ResolvedNamedBoolExpr &subexpr)
          -> FailureOr<Value> {
        switch (subexpr.kind) {
        case ResolvedNamedBoolExpr::Kind::Arg:
          return evalArgValue(subexpr.argIndex, subexpr.argSlice);
        case ResolvedNamedBoolExpr::Kind::Const:
          return Value(
              smt::BoolConstantOp::create(builder, loc, subexpr.constValue));
        case ResolvedNamedBoolExpr::Kind::Group:
          if (!subexpr.lhs)
            return failure();
          return selfValue(selfValue, *subexpr.lhs);
        case ResolvedNamedBoolExpr::Kind::Not:
        case ResolvedNamedBoolExpr::Kind::ReduceAnd:
        case ResolvedNamedBoolExpr::Kind::ReduceOr:
        case ResolvedNamedBoolExpr::Kind::ReduceXor:
        case ResolvedNamedBoolExpr::Kind::ReduceNand:
        case ResolvedNamedBoolExpr::Kind::ReduceNor:
        case ResolvedNamedBoolExpr::Kind::ReduceXnor:
        case ResolvedNamedBoolExpr::Kind::Implies:
        case ResolvedNamedBoolExpr::Kind::Iff:
        case ResolvedNamedBoolExpr::Kind::Lt:
        case ResolvedNamedBoolExpr::Kind::Le:
        case ResolvedNamedBoolExpr::Kind::Gt:
        case ResolvedNamedBoolExpr::Kind::Ge: {
          auto boolOr = self(self, builder, values, subexpr);
          if (failed(boolOr))
            return failure();
          return *boolOr;
        }
        case ResolvedNamedBoolExpr::Kind::BitwiseNot: {
          auto operandOr = selfValue(selfValue, *subexpr.lhs);
          if (failed(operandOr))
            return failure();
          Value operand = *operandOr;
          if (auto bvTy = dyn_cast<smt::BitVectorType>(operand.getType())) {
            if (bvTy.getWidth() == 0)
              return failure();
            return Value(smt::BVNotOp::create(builder, loc, operand));
          }
          if (isa<smt::BoolType>(operand.getType()))
            return Value(smt::NotOp::create(builder, loc, operand));
          return failure();
        }
        case ResolvedNamedBoolExpr::Kind::And:
        case ResolvedNamedBoolExpr::Kind::Or:
        case ResolvedNamedBoolExpr::Kind::Xor: {
          auto lhsOr = selfValue(selfValue, *subexpr.lhs);
          auto rhsOr = selfValue(selfValue, *subexpr.rhs);
          if (failed(lhsOr) || failed(rhsOr))
            return failure();
          Value lhs = *lhsOr;
          Value rhs = *rhsOr;
          if (lhs.getType() != rhs.getType())
            return failure();
          if (isa<smt::BoolType>(lhs.getType())) {
            if (subexpr.kind == ResolvedNamedBoolExpr::Kind::And)
              return Value(smt::AndOp::create(builder, loc, lhs, rhs));
            if (subexpr.kind == ResolvedNamedBoolExpr::Kind::Or)
              return Value(smt::OrOp::create(builder, loc, lhs, rhs));
            return Value(smt::DistinctOp::create(builder, loc, lhs, rhs));
          }
          if (isa<smt::BitVectorType>(lhs.getType())) {
            if (subexpr.kind == ResolvedNamedBoolExpr::Kind::And)
              return Value(smt::BVAndOp::create(builder, loc, lhs, rhs));
            if (subexpr.kind == ResolvedNamedBoolExpr::Kind::Or)
              return Value(smt::BVOrOp::create(builder, loc, lhs, rhs));
            return Value(smt::BVXOrOp::create(builder, loc, lhs, rhs));
          }
          return failure();
        }
        case ResolvedNamedBoolExpr::Kind::Eq:
        case ResolvedNamedBoolExpr::Kind::Ne:
          return failure();
        }
        return failure();
      };
      switch (expr.kind) {
      case ResolvedNamedBoolExpr::Kind::Arg:
        if (expr.argIndex >= values.size())
          return failure();
        if (!expr.argSlice)
          return toSMTBool(builder, values[expr.argIndex]);
        {
          auto argValueOr = evalArgValue(expr.argIndex, expr.argSlice);
          if (failed(argValueOr))
            return failure();
          Value value = *argValueOr;
          if (isa<smt::BoolType>(value.getType()))
            return value;
          if (auto bvTy = dyn_cast<smt::BitVectorType>(value.getType())) {
            if (bvTy.getWidth() == 1) {
              auto one = smt::BVConstantOp::create(builder, loc, APInt(1, 1));
              return Value(smt::EqOp::create(builder, loc, value, one));
            }
            auto zero =
                smt::BVConstantOp::create(builder, loc, APInt(bvTy.getWidth(), 0));
            return Value(smt::DistinctOp::create(builder, loc, value, zero));
          }
          if (auto intTy = dyn_cast<IntegerType>(value.getType())) {
            if (intTy.getWidth() == 1)
              return toSMTBool(builder, value);
          }
          return failure();
        }
      case ResolvedNamedBoolExpr::Kind::Const:
        return Value(smt::BoolConstantOp::create(builder, loc, expr.constValue));
      case ResolvedNamedBoolExpr::Kind::Group:
        if (!expr.lhs)
          return failure();
        return self(self, builder, values, *expr.lhs);
      case ResolvedNamedBoolExpr::Kind::Not: {
        auto operandOr = self(self, builder, values, *expr.lhs);
        if (failed(operandOr))
          return failure();
        return Value(smt::NotOp::create(builder, loc, *operandOr));
      }
      case ResolvedNamedBoolExpr::Kind::BitwiseNot: {
        if (expr.lhs && expr.lhs->kind == ResolvedNamedBoolExpr::Kind::Arg) {
          auto argValueOr = evalArgValue(expr.lhs->argIndex, expr.lhs->argSlice);
          if (failed(argValueOr))
            return failure();
          return evalBitwiseNotAsBool(*argValueOr);
        }
        auto operandOr = self(self, builder, values, *expr.lhs);
        if (failed(operandOr))
          return failure();
        return evalBitwiseNotAsBool(*operandOr);
      }
      case ResolvedNamedBoolExpr::Kind::ReduceAnd:
      case ResolvedNamedBoolExpr::Kind::ReduceOr:
      case ResolvedNamedBoolExpr::Kind::ReduceXor:
      case ResolvedNamedBoolExpr::Kind::ReduceNand:
      case ResolvedNamedBoolExpr::Kind::ReduceNor:
      case ResolvedNamedBoolExpr::Kind::ReduceXnor: {
        if (expr.lhs && expr.lhs->kind == ResolvedNamedBoolExpr::Kind::Arg) {
          auto argValueOr = evalArgValue(expr.lhs->argIndex, expr.lhs->argSlice);
          if (failed(argValueOr))
            return failure();
          return evalReduce(*argValueOr, expr.kind);
        }
        auto operandOr = self(self, builder, values, *expr.lhs);
        if (failed(operandOr))
          return failure();
        return evalReduce(*operandOr, expr.kind);
      }
      case ResolvedNamedBoolExpr::Kind::And:
      case ResolvedNamedBoolExpr::Kind::Or:
      case ResolvedNamedBoolExpr::Kind::Xor:
      case ResolvedNamedBoolExpr::Kind::Implies:
      case ResolvedNamedBoolExpr::Kind::Iff:
      {
        auto lhsOr = self(self, builder, values, *expr.lhs);
        auto rhsOr = self(self, builder, values, *expr.rhs);
        if (failed(lhsOr) || failed(rhsOr))
          return failure();
        switch (expr.kind) {
        case ResolvedNamedBoolExpr::Kind::And:
          return Value(smt::AndOp::create(builder, loc, *lhsOr, *rhsOr));
        case ResolvedNamedBoolExpr::Kind::Or:
          return Value(smt::OrOp::create(builder, loc, *lhsOr, *rhsOr));
        case ResolvedNamedBoolExpr::Kind::Xor:
          return Value(smt::DistinctOp::create(builder, loc, *lhsOr, *rhsOr));
        case ResolvedNamedBoolExpr::Kind::Implies: {
          auto notLhs = smt::NotOp::create(builder, loc, *lhsOr);
          return Value(smt::OrOp::create(builder, loc, notLhs, *rhsOr));
        }
        case ResolvedNamedBoolExpr::Kind::Iff:
          return Value(smt::EqOp::create(builder, loc, *lhsOr, *rhsOr));
        default:
          return failure();
        }
      }
      case ResolvedNamedBoolExpr::Kind::Lt:
      case ResolvedNamedBoolExpr::Kind::Le:
      case ResolvedNamedBoolExpr::Kind::Gt:
      case ResolvedNamedBoolExpr::Kind::Ge: {
        auto lhsValueOr = evalSubExprAsValue(evalSubExprAsValue, *expr.lhs);
        auto rhsValueOr = evalSubExprAsValue(evalSubExprAsValue, *expr.rhs);
        if (failed(lhsValueOr) || failed(rhsValueOr))
          return failure();
        return evalRelCompare(*lhsValueOr, *rhsValueOr, expr.kind,
                              expr.compareSigned);
      }
      case ResolvedNamedBoolExpr::Kind::Eq:
      case ResolvedNamedBoolExpr::Kind::Ne: {
        auto lhsValueOr = evalSubExprAsValue(evalSubExprAsValue, *expr.lhs);
        auto rhsValueOr = evalSubExprAsValue(evalSubExprAsValue, *expr.rhs);
        if (succeeded(lhsValueOr) && succeeded(rhsValueOr)) {
          Value lhsValue = *lhsValueOr;
          Value rhsValue = *rhsValueOr;
          if (lhsValue.getType() == rhsValue.getType()) {
            if (isa<smt::BoolType>(lhsValue.getType()) ||
                isa<smt::BitVectorType>(lhsValue.getType())) {
              if (expr.kind == ResolvedNamedBoolExpr::Kind::Eq)
                return Value(
                    smt::EqOp::create(builder, loc, lhsValue, rhsValue));
              return Value(
                  smt::DistinctOp::create(builder, loc, lhsValue, rhsValue));
            }
          } else {
            auto lhsBoolOr = toSMTBool(builder, lhsValue);
            auto rhsBoolOr = toSMTBool(builder, rhsValue);
            if (succeeded(lhsBoolOr) && succeeded(rhsBoolOr)) {
              if (expr.kind == ResolvedNamedBoolExpr::Kind::Eq)
                return Value(
                    smt::EqOp::create(builder, loc, *lhsBoolOr, *rhsBoolOr));
              return Value(smt::DistinctOp::create(builder, loc, *lhsBoolOr,
                                                   *rhsBoolOr));
            }
          }
        }
        auto lhsOr = self(self, builder, values, *expr.lhs);
        auto rhsOr = self(self, builder, values, *expr.rhs);
        if (failed(lhsOr) || failed(rhsOr))
          return failure();
        if (expr.kind == ResolvedNamedBoolExpr::Kind::Eq)
          return Value(smt::EqOp::create(builder, loc, *lhsOr, *rhsOr));
        return Value(smt::DistinctOp::create(builder, loc, *lhsOr, *rhsOr));
      }
      }
      return failure();
    };

    auto getBoolValue = [&](OpBuilder &builder, ValueRange values,
                            std::optional<unsigned> argIndex,
                            const std::unique_ptr<ResolvedNamedBoolExpr> &expr)
        -> FailureOr<Value> {
      if (argIndex) {
        if (*argIndex >= values.size())
          return failure();
        return toSMTBool(builder, values[*argIndex]);
      }
      if (expr)
        return evalResolvedExpr(evalResolvedExpr, builder, values, *expr);
      return failure();
    };

    for (const auto &witness : armWitnesses) {
      if (!witness.sourceArgIndex && !witness.sourceExpr)
        continue;
      Value initWitness = witnessFalse;
      if (witness.kind == EventArmWitnessInfo::Kind::Sequence) {
        auto initBoolOr = getBoolValue(
            rewriter, ValueRange(inputDecls).take_front(numNonStateArgs),
            witness.sourceArgIndex, witness.sourceExpr);
        if (failed(initBoolOr))
          return failure();
        initWitness = *initBoolOr;
        if (witness.iffArgIndex || witness.iffExpr) {
          auto iffBoolOr = getBoolValue(
              rewriter, ValueRange(inputDecls).take_front(numNonStateArgs),
              witness.iffArgIndex, witness.iffExpr);
          if (failed(iffBoolOr))
            return failure();
          initWitness =
              smt::AndOp::create(rewriter, loc, *iffBoolOr, initWitness);
        }
      }
      auto initWitnessDecl =
          smt::DeclareFunOp::create(rewriter, loc, boolTy, witness.witnessName);
      auto initWitnessEq =
          smt::EqOp::create(rewriter, loc, initWitnessDecl, initWitness);
      smt::AssertOp::create(rewriter, loc, initWitnessEq);
    }

    auto buildWitnessFired = [&](OpBuilder &builder, const auto &prevValues,
                                 const auto &currValues,
                                 const EventArmWitnessInfo &witness)
        -> FailureOr<Value> {
      auto currBoolOr =
          getBoolValue(builder, ValueRange(currValues), witness.sourceArgIndex,
                       witness.sourceExpr);
      if (failed(currBoolOr))
        return failure();
      Value currBool = *currBoolOr;

      Value fired = currBool;
      if (witness.kind == EventArmWitnessInfo::Kind::Signal) {
        auto prevBoolOr =
            getBoolValue(builder, ValueRange(prevValues), witness.sourceArgIndex,
                         witness.sourceExpr);
        if (failed(prevBoolOr))
          return failure();
        Value prevBool = *prevBoolOr;
        switch (witness.edge) {
        case ltl::ClockEdge::Pos: {
          auto notPrev = smt::NotOp::create(builder, loc, prevBool);
          fired = smt::AndOp::create(builder, loc, notPrev, currBool);
          break;
        }
        case ltl::ClockEdge::Neg: {
          auto notCurr = smt::NotOp::create(builder, loc, currBool);
          fired = smt::AndOp::create(builder, loc, prevBool, notCurr);
          break;
        }
        case ltl::ClockEdge::Both:
          fired = smt::DistinctOp::create(builder, loc, prevBool, currBool);
          break;
        }
      }

      if (witness.iffArgIndex || witness.iffExpr) {
        auto iffBoolOr =
            getBoolValue(builder, ValueRange(currValues), witness.iffArgIndex,
                         witness.iffExpr);
        if (failed(iffBoolOr))
          return failure();
        fired = smt::AndOp::create(builder, loc, *iffBoolOr, fired);
      }
      return fired;
    };

    if (emitSMTLIB) {
      auto combineOr = [&](Value lhs, Value rhs) -> Value {
        if (!lhs)
          return rhs;
        if (!rhs)
          return lhs;
        return smt::OrOp::create(rewriter, loc, lhs, rhs);
      };
      auto combineAnd = [&](Value lhs, Value rhs) -> Value {
        if (!lhs)
          return rhs;
        if (!rhs)
          return lhs;
        return smt::AndOp::create(rewriter, loc, lhs, rhs);
      };

      size_t numCircuitArgs = circuitInputTy.size();
      size_t numBMCStateArgs = numRegs + totalDelaySlots + totalNFAStateSlots;
      SmallVector<Value> iterArgs = inputDecls;
      SmallVector<SmallVector<Value>> lassoStateHistory;
      SmallVector<SmallVector<Value>> lassoFinalCheckSampleHistory;
      SmallVector<SmallVector<Value>> lassoFinalCheckSampledTrueHistory;
      auto captureStateSnapshot = [&](ValueRange values) {
        if (!livenessLassoMode)
          return;
        SmallVector<Value> snapshot;
        if (numBMCStateArgs > 0) {
          ValueRange stateValues =
              values.take_front(numCircuitArgs).take_back(numBMCStateArgs);
          snapshot.append(stateValues.begin(), stateValues.end());
        }
        lassoStateHistory.push_back(std::move(snapshot));
      };
      captureStateSnapshot(ValueRange(iterArgs));
      for (uint64_t iter = 0; iter < boundValue; ++iter) {
        ValueRange iterRange(iterArgs);

        // Assert 2-state constraints on the current iteration inputs.
        for (auto [index, pair] : llvm::enumerate(llvm::zip(
                 TypeRange(oldCircuitInputTy).take_front(numCircuitArgs),
                 iterRange.take_front(numCircuitArgs)))) {
          auto [oldTy, arg] = pair;
          maybeAssertKnown(index, oldTy, arg, rewriter);
        }

        // Call loop func to update clock & state arg values.
        SmallVector<Value> loopCallInputs;
        for (auto index : clockIndexes)
          loopCallInputs.push_back(iterRange[index]);
        for (auto stateArg :
             iterRange.drop_back(1 + numFinalChecks).take_back(numStateArgs))
          loopCallInputs.push_back(stateArg);
        auto loopValsVecOr =
            inlineBMCBlock(op.getLoop().front(), loopCallInputs, initOutputTy);
        if (failed(loopValsVecOr))
          return failure();
        SmallVector<Value> loopValsVec = std::move(*loopValsVecOr);
        ValueRange loopVals(loopValsVec);

        // Compute clock edges for this iteration.
        Value isPosedge;
        Value isNegedge;
        SmallVector<Value> posedges;
        SmallVector<Value> negedges;
        Value anyPosedge;
        Value anyNegedge;
        bool usePosedge = !risingClocksOnly && clockIndexes.size() == 1;
        bool usePerRegPosedge =
            !risingClocksOnly && clockIndexes.size() > 1 && numRegs > 0;
        bool needClockEdges =
            !risingClocksOnly && !clockIndexes.empty() &&
            (usePerRegPosedge || totalDelaySlots > 0 ||
             totalNFAStateSlots > 0 || !nonFinalCheckInfos.empty() ||
             !finalCheckInfos.empty());
        if (usePosedge) {
          auto clockIndex = clockIndexes[0];
          auto oldClock = iterRange[clockIndex];
          auto newClock = loopVals[0];
          auto oldClockLow = smt::BVNotOp::create(rewriter, loc, oldClock);
          auto newClockLow = smt::BVNotOp::create(rewriter, loc, newClock);
          auto isPosedgeBV =
              smt::BVAndOp::create(rewriter, loc, oldClockLow, newClock);
          auto isNegedgeBV =
              smt::BVAndOp::create(rewriter, loc, oldClock, newClockLow);
          auto trueBV = smt::BVConstantOp::create(rewriter, loc, 1, 1);
          isPosedge = smt::EqOp::create(rewriter, loc, isPosedgeBV, trueBV);
          isNegedge = smt::EqOp::create(rewriter, loc, isNegedgeBV, trueBV);
          if (needClockEdges) {
            posedges.push_back(isPosedge);
            negedges.push_back(isNegedge);
            anyPosedge = isPosedge;
            anyNegedge = isNegedge;
          }
        } else if (needClockEdges) {
          posedges.reserve(clockIndexes.size());
          negedges.reserve(clockIndexes.size());
          auto trueBV = smt::BVConstantOp::create(rewriter, loc, 1, 1);
          for (auto [idx, clockIndex] : llvm::enumerate(clockIndexes)) {
            auto oldClock = iterRange[clockIndex];
            auto newClock = loopVals[idx];
            auto oldClockLow = smt::BVNotOp::create(rewriter, loc, oldClock);
            auto newClockLow = smt::BVNotOp::create(rewriter, loc, newClock);
            auto isPosedgeBV =
                smt::BVAndOp::create(rewriter, loc, oldClockLow, newClock);
            auto isNegedgeBV =
                smt::BVAndOp::create(rewriter, loc, oldClock, newClockLow);
            posedges.push_back(
                smt::EqOp::create(rewriter, loc, isPosedgeBV, trueBV));
            negedges.push_back(
                smt::EqOp::create(rewriter, loc, isNegedgeBV, trueBV));
          }
          if (!posedges.empty()) {
            anyPosedge = posedges.front();
            anyNegedge = negedges.front();
            for (auto [pos, neg] :
                 llvm::zip(ArrayRef<Value>(posedges).drop_front(),
                           ArrayRef<Value>(negedges).drop_front())) {
              anyPosedge =
                  smt::OrOp::create(rewriter, loc, anyPosedge, pos);
              anyNegedge =
                  smt::OrOp::create(rewriter, loc, anyNegedge, neg);
            }
          }
        }

        Value defaultDelayGate;
        if (!risingClocksOnly && !clockIndexes.empty()) {
          if (usePosedge)
            defaultDelayGate = isPosedge;
          else if (anyPosedge)
            defaultDelayGate = anyPosedge;
        }

        auto getDelayGate =
            [&](StringAttr clockName, Value clockValue,
                std::optional<ltl::ClockEdge> edge) -> Value {
          if (risingClocksOnly || clockIndexes.empty())
            return Value();
          std::optional<unsigned> pos;
          if (clockName && !clockName.getValue().empty()) {
            auto it = clockNameToPos.find(clockName.getValue());
            if (it != clockNameToPos.end())
              pos = it->second;
          } else if (clockValue) {
            if (auto info = resolveClockPosInfo(clockValue))
              pos = info->pos;
          } else if (clockIndexes.size() == 1) {
            pos = 0;
          }
          if (!pos || posedges.empty() || negedges.empty())
            return defaultDelayGate;
          Value posedge = posedges[*pos];
          Value negedge = negedges[*pos];
          ltl::ClockEdge edgeKind = edge.value_or(ltl::ClockEdge::Pos);
          if (clockValue) {
            if (auto info = resolveClockPosInfo(clockValue);
                info && info->invert)
              edgeKind = invertClockEdge(edgeKind);
          }
          switch (edgeKind) {
          case ltl::ClockEdge::Pos:
            return posedge;
          case ltl::ClockEdge::Neg:
            return negedge;
          case ltl::ClockEdge::Both:
            return smt::OrOp::create(rewriter, loc, posedge, negedge);
          }
          return defaultDelayGate;
        };

        auto materializeTickValue = [&](Type argTy, Value tickBool) -> Value {
          if (!tickBool)
            return tickBool;
          if (isa<smt::BoolType>(argTy))
            return tickBool;
          if (auto bvTy = dyn_cast<smt::BitVectorType>(argTy)) {
            if (bvTy.getWidth() == 1) {
              auto one = smt::BVConstantOp::create(rewriter, loc, 1, 1);
              auto zero = smt::BVConstantOp::create(rewriter, loc, 0, 1);
              return smt::IteOp::create(rewriter, loc, tickBool, one, zero);
            }
          }
          return tickBool;
        };

        auto getNFATickGate = [&](const NFATickGateInfo &info) -> Value {
          if (risingClocksOnly || clockIndexes.empty())
            return Value();
          if (info.clockPos && *info.clockPos < posedges.size() &&
              *info.clockPos < negedges.size()) {
            Value posedge = posedges[*info.clockPos];
            Value negedge = negedges[*info.clockPos];
            ltl::ClockEdge edgeKind = info.edge;
            if (info.invert)
              edgeKind = invertClockEdge(edgeKind);
            switch (edgeKind) {
            case ltl::ClockEdge::Pos:
              return posedge;
            case ltl::ClockEdge::Neg:
              return negedge;
            case ltl::ClockEdge::Both:
              return smt::OrOp::create(rewriter, loc, posedge, negedge);
            }
          }
          if (info.hasExplicitClock)
            return defaultDelayGate;
          return defaultDelayGate;
        };

        SmallVector<Value> circuitInputs(
            iterRange.take_front(numCircuitArgs).begin(),
            iterRange.take_front(numCircuitArgs).end());
        // Use post-edge clock values for property/circuit evaluation so
        // clocked expressions observe the sampled clock value.
        if (!clockIndexes.empty()) {
          for (auto [idx, clockIndex] : llvm::enumerate(clockIndexes)) {
            if (static_cast<size_t>(clockIndex) < circuitInputs.size() &&
                static_cast<size_t>(idx) < loopVals.size())
              circuitInputs[clockIndex] = loopVals[idx];
          }
        }
        if (!clockSourceInputs.empty()) {
          for (auto [argIndex, info] : clockSourceInputs) {
            if (argIndex >= circuitInputs.size() ||
                info.pos >= loopVals.size())
              continue;
            int64_t valueWidth = 0;
            int64_t unknownWidth = 0;
            if (!isFourStateStruct(oldCircuitInputTy[argIndex], valueWidth,
                                   unknownWidth))
              continue;
            if (valueWidth != 1)
              continue;
            auto bvTy =
                dyn_cast<smt::BitVectorType>(circuitInputs[argIndex].getType());
            if (!bvTy || bvTy.getWidth() !=
                             static_cast<unsigned>(valueWidth + unknownWidth))
              continue;
            Value valueBit = loopVals[info.pos];
            if (info.invert)
              valueBit = smt::BVNotOp::create(rewriter, loc, valueBit);
            auto unkTy =
                smt::BitVectorType::get(rewriter.getContext(), unknownWidth);
            Value unknownBits =
                smt::ExtractOp::create(rewriter, loc, unkTy, 0,
                                       circuitInputs[argIndex]);
            circuitInputs[argIndex] =
                smt::ConcatOp::create(rewriter, loc, valueBit, unknownBits);
          }
        }
        if (!nfaTickGateInfos.empty()) {
          for (const auto &tickInfo : nfaTickGateInfos) {
            if (tickInfo.tickArgIndex >= circuitInputs.size())
              continue;
            Value gate = getNFATickGate(tickInfo);
            Value tick = gate ? gate : constTrue;
            Type argTy = circuitInputTy[tickInfo.tickArgIndex];
            circuitInputs[tickInfo.tickArgIndex] =
                materializeTickValue(argTy, tick);
          }
        }

        // Execute the circuit.
        auto circuitCallOutsVecOr = inlineBMCBlock(
            op.getCircuit().front(), circuitInputs, circuitOutputTy);
        if (failed(circuitCallOutsVecOr))
          return failure();
        SmallVector<Value> circuitCallOutsVec = std::move(*circuitCallOutsVecOr);
        ValueRange circuitCallOuts(circuitCallOutsVec);

        ValueRange finalCheckOutputs =
            numFinalChecks == 0 ? ValueRange{}
                                : circuitCallOuts.take_back(numFinalChecks);
        ValueRange beforeFinal =
            numFinalChecks == 0 ? circuitCallOuts
                                : circuitCallOuts.drop_back(numFinalChecks);
        ValueRange nonFinalCheckOutputs =
            numNonFinalChecks == 0
                ? ValueRange{}
                : beforeFinal.take_back(numNonFinalChecks);
        ValueRange nonFinalOutputs =
            numNonFinalChecks == 0 ? beforeFinal
                                   : beforeFinal.drop_back(numNonFinalChecks);
        ValueRange nfaStateOutputs =
            totalNFAStateSlots == 0
                ? ValueRange{}
                : nonFinalOutputs.take_back(totalNFAStateSlots);
        ValueRange beforeNFA =
            totalNFAStateSlots == 0
                ? nonFinalOutputs
                : nonFinalOutputs.drop_back(totalNFAStateSlots);
        ValueRange delayBufferOutputs =
            totalDelaySlots == 0 ? ValueRange{}
                                 : beforeNFA.take_back(totalDelaySlots);
        ValueRange circuitOutputs =
            totalDelaySlots == 0 ? beforeNFA
                                 : beforeNFA.drop_back(totalDelaySlots);

        Value violated = iterRange.back();
        if (numNonFinalChecks > 0) {
          bool skipChecks = ignoreAssertionsUntilValue &&
                            iter < *ignoreAssertionsUntilValue;
          if (!skipChecks) {
            Value combinedCheckCond;
            for (auto [checkIdx, checkVal] :
                 llvm::enumerate(nonFinalCheckOutputs)) {
              Value isTrue;
              if (isa<smt::BoolType>(checkVal.getType())) {
                isTrue = checkVal;
              } else {
                auto trueBV = smt::BVConstantOp::create(rewriter, loc, 1, 1);
                isTrue = smt::EqOp::create(rewriter, loc, checkVal, trueBV);
              }

              Value term = isCoverCheck
                               ? isTrue
                               : smt::NotOp::create(rewriter, loc, isTrue);
              Value gate;
              if (!risingClocksOnly && !clockIndexes.empty() &&
                  checkIdx < nonFinalCheckInfos.size()) {
                auto edge = nonFinalCheckInfos[checkIdx].edge.value_or(
                    ltl::ClockEdge::Pos);
                auto clockPos =
                    checkIdx < nonFinalCheckClockPos.size()
                        ? nonFinalCheckClockPos[checkIdx]
                        : std::optional<unsigned>{};
                if (clockPos) {
                  if (edge == ltl::ClockEdge::Pos) {
                    if (clockIndexes.size() == 1)
                      gate = isPosedge;
                    else if (*clockPos < posedges.size())
                      gate = posedges[*clockPos];
                  } else if (edge == ltl::ClockEdge::Neg) {
                    if (clockIndexes.size() == 1)
                      gate = isNegedge;
                    else if (*clockPos < negedges.size())
                      gate = negedges[*clockPos];
                  } else if (edge == ltl::ClockEdge::Both) {
                    if (clockIndexes.size() == 1)
                      gate = smt::OrOp::create(rewriter, loc, isPosedge,
                                               isNegedge);
                    else if (*clockPos < posedges.size())
                      gate = smt::OrOp::create(rewriter, loc,
                                               posedges[*clockPos],
                                               negedges[*clockPos]);
                  }
                }
                if (!gate && anyPosedge) {
                  if (edge == ltl::ClockEdge::Pos) {
                    gate = anyPosedge;
                  } else if (edge == ltl::ClockEdge::Neg && anyNegedge) {
                    gate = anyNegedge;
                  } else if (edge == ltl::ClockEdge::Both && anyNegedge) {
                    gate =
                        smt::OrOp::create(rewriter, loc, anyPosedge, anyNegedge);
                  }
                }
              }

              if (gate)
                term = smt::AndOp::create(rewriter, loc, gate, term);
              if (!combinedCheckCond)
                combinedCheckCond = term;
              else
                combinedCheckCond =
                    smt::OrOp::create(rewriter, loc, combinedCheckCond, term);
            }
            if (combinedCheckCond) {
              if (inductionStep && iter + 1 < boundValue) {
                Value mustHold =
                    smt::NotOp::create(rewriter, loc, combinedCheckCond);
                smt::AssertOp::create(rewriter, loc, mustHold);
              } else {
                violated = smt::OrOp::create(rewriter, loc, violated,
                                             combinedCheckCond);
              }
            }
          }
        }

        if (inductionStep && numFinalChecks > 0 && iter + 1 < boundValue) {
          bool skipChecks = ignoreAssertionsUntilValue &&
                            iter < *ignoreAssertionsUntilValue;
          if (!skipChecks) {
            Value combinedFinalCond;
            for (auto [checkIdx, checkVal] : llvm::enumerate(finalCheckOutputs)) {
              Value isTrue;
              if (isa<smt::BoolType>(checkVal.getType())) {
                isTrue = checkVal;
              } else {
                auto trueBV = smt::BVConstantOp::create(rewriter, loc, 1, 1);
                isTrue = smt::EqOp::create(rewriter, loc, checkVal, trueBV);
              }

              bool isFinalCover = checkIdx < finalCheckIsCover.size() &&
                                  finalCheckIsCover[checkIdx];
              Value term = isFinalCover
                               ? isTrue
                               : smt::NotOp::create(rewriter, loc, isTrue);
              Value gate;
              if (!risingClocksOnly && !clockIndexes.empty() &&
                  checkIdx < finalCheckInfos.size()) {
                auto edge =
                    finalCheckInfos[checkIdx].edge.value_or(ltl::ClockEdge::Pos);
                auto clockPos =
                    checkIdx < finalCheckClockPos.size()
                        ? finalCheckClockPos[checkIdx]
                        : std::optional<unsigned>{};
                if (clockPos) {
                  if (edge == ltl::ClockEdge::Pos) {
                    if (clockIndexes.size() == 1)
                      gate = isPosedge;
                    else if (*clockPos < posedges.size())
                      gate = posedges[*clockPos];
                  } else if (edge == ltl::ClockEdge::Neg) {
                    if (clockIndexes.size() == 1)
                      gate = isNegedge;
                    else if (*clockPos < negedges.size())
                      gate = negedges[*clockPos];
                  } else if (edge == ltl::ClockEdge::Both) {
                    if (clockIndexes.size() == 1)
                      gate = smt::OrOp::create(rewriter, loc, isPosedge,
                                               isNegedge);
                    else if (*clockPos < posedges.size())
                      gate = smt::OrOp::create(rewriter, loc, posedges[*clockPos],
                                               negedges[*clockPos]);
                  }
                }
                if (!gate && anyPosedge) {
                  if (edge == ltl::ClockEdge::Pos) {
                    gate = anyPosedge;
                  } else if (edge == ltl::ClockEdge::Neg && anyNegedge) {
                    gate = anyNegedge;
                  } else if (edge == ltl::ClockEdge::Both && anyNegedge) {
                    gate =
                        smt::OrOp::create(rewriter, loc, anyPosedge, anyNegedge);
                  }
                }
              }

              if (gate)
                term = smt::AndOp::create(rewriter, loc, gate, term);
              if (!combinedFinalCond)
                combinedFinalCond = term;
              else
                combinedFinalCond =
                    smt::OrOp::create(rewriter, loc, combinedFinalCond, term);
            }
            if (combinedFinalCond) {
              Value mustHold =
                  smt::NotOp::create(rewriter, loc, combinedFinalCond);
              smt::AssertOp::create(rewriter, loc, mustHold);
            }
          }
        }

        size_t loopIndex = 0;
        SmallVector<Value> newDecls;
        size_t nonRegIdx = 0;
        size_t argIndex = 0;
        size_t numNonStateArgs = oldCircuitInputTy.size() - numRegs -
                                 totalDelaySlots - totalNFAStateSlots;
        for (auto [oldTy, newTy] :
             llvm::zip(TypeRange(oldCircuitInputTy).take_front(numNonStateArgs),
                       TypeRange(circuitInputTy).take_front(numNonStateArgs))) {
          bool isI1Type =
              isa<IntegerType>(oldTy) &&
              cast<IntegerType>(oldTy).getWidth() == 1;
          bool isClock =
              isa<seq::ClockType>(oldTy) ||
              (isI1Type && nonRegIdx < numInitClocks);
          if (tickArgIndices.contains(argIndex)) {
            Value initVal = makeBoolConstant(newTy, false);
            if (!initVal)
              return failure();
            newDecls.push_back(initVal);
          } else if (isClock) {
            if (loopIndex >= loopVals.size()) {
              op.emitError("verif.bmc loop region did not produce expected "
                           "clock/state values for SMT-LIB export");
              return failure();
            }
            newDecls.push_back(loopVals[loopIndex++]);
          } else {
            auto decl = smt::DeclareFunOp::create(
                rewriter, loc, newTy,
                argIndex < inputNamePrefixes.size()
                    ? inputNamePrefixes[argIndex]
                    : StringAttr{});
            newDecls.push_back(decl);
            maybeAssertKnown(argIndex, oldTy, decl, rewriter);
          }
          nonRegIdx++;
          argIndex++;
        }

        for (const auto &witness : armWitnesses) {
          if (witness.sourceArgIndex >= numNonStateArgs ||
              witness.sourceArgIndex >= newDecls.size() ||
              witness.sourceArgIndex >= iterRange.size())
            continue;

          auto firedOr = buildWitnessFired(
              rewriter, iterRange.take_front(numNonStateArgs),
              ValueRange(newDecls).take_front(numNonStateArgs), witness);
          if (failed(firedOr))
            return failure();

          auto witnessDecl = smt::DeclareFunOp::create(rewriter, loc, boolTy,
                                                       witness.witnessName);
          auto witnessEq =
              smt::EqOp::create(rewriter, loc, witnessDecl, *firedOr);
          smt::AssertOp::create(rewriter, loc, witnessEq);
        }

        if (clockIndexes.size() >= 1) {
          SmallVector<Value> regInputs = circuitOutputs.take_back(numRegs);
          if (risingClocksOnly || clockIndexes.size() == 1) {
            if (risingClocksOnly) {
              newDecls.append(regInputs);
            } else {
              auto regStates =
                  iterRange.take_front(numCircuitArgs)
                      .take_back(numRegs + totalDelaySlots + totalNFAStateSlots)
                      .drop_back(totalDelaySlots + totalNFAStateSlots);
              SmallVector<Value> nextRegStates;
              for (auto [regState, regInput] :
                   llvm::zip(regStates, regInputs)) {
                nextRegStates.push_back(smt::IteOp::create(
                    rewriter, loc, isPosedge, regInput, regState));
              }
              newDecls.append(nextRegStates);
            }
          } else if (usePerRegPosedge) {
            auto regStates =
                iterRange.take_front(numCircuitArgs)
                    .take_back(numRegs + totalDelaySlots + totalNFAStateSlots)
                    .drop_back(totalDelaySlots + totalNFAStateSlots);
            SmallVector<Value> nextRegStates;
            nextRegStates.reserve(numRegs);
            for (auto [idx, pair] :
                 llvm::enumerate(llvm::zip(regStates, regInputs))) {
              auto [regState, regInput] = pair;
              Value regPosedge =
                  regClockInverts[idx] ? negedges[regClockToLoopIndex[idx]]
                                       : posedges[regClockToLoopIndex[idx]];
              nextRegStates.push_back(smt::IteOp::create(
                  rewriter, loc, regPosedge, regInput, regState));
            }
            newDecls.append(nextRegStates);
          }
        }

        if (totalDelaySlots > 0) {

          auto delayStates =
              iterRange.take_front(numCircuitArgs)
                  .take_back(totalDelaySlots + totalNFAStateSlots)
                  .drop_back(totalNFAStateSlots);
          size_t delayIndex = 0;
          auto appendDelayUpdates = [&](Value gate, size_t count) {
            for (size_t i = 0; i < count; ++i) {
              Value delayState = delayStates[delayIndex + i];
              Value delayVal = delayBufferOutputs[delayIndex + i];
              if (gate) {
                newDecls.push_back(smt::IteOp::create(rewriter, loc, gate,
                                                     delayVal, delayState));
              } else {
                newDecls.push_back(delayVal);
              }
            }
            delayIndex += count;
          };

          for (const auto &info : delayInfos) {
            Value gate = getDelayGate(info.clockName, info.clockValue, info.edge);
            appendDelayUpdates(gate, info.bufferSize);
          }
          for (const auto &info : pastInfos) {
            Value gate = getDelayGate(info.clockName, info.clockValue, info.edge);
            appendDelayUpdates(gate, info.bufferSize);
          }
          for (; delayIndex < totalDelaySlots; ++delayIndex)
            newDecls.push_back(delayBufferOutputs[delayIndex]);
        }

        if (totalNFAStateSlots > 0)
          newDecls.append(nfaStateOutputs.begin(), nfaStateOutputs.end());

        for (; loopIndex < loopVals.size(); ++loopIndex)
          newDecls.push_back(loopVals[loopIndex]);

        ValueRange finalCheckStates =
            numFinalChecks == 0 ? ValueRange{}
                                : iterRange.drop_back(1).take_back(numFinalChecks);
        SmallVector<Value> finalCheckSamplesThisIter;
        SmallVector<Value> finalCheckSampledTrueThisIter;
        if (livenessLassoMode && numFinalChecks > 0)
          finalCheckSamplesThisIter.reserve(numFinalChecks);
        if (livenessLassoMode && numFinalChecks > 0)
          finalCheckSampledTrueThisIter.reserve(numFinalChecks);
        auto getCheckGate =
            [&](size_t checkIdx, ArrayRef<std::optional<unsigned>> clockPoses,
                ArrayRef<NonFinalCheckInfo> infos) -> Value {
          if (risingClocksOnly || clockIndexes.empty() ||
              checkIdx >= infos.size())
            return Value();
          auto edge = infos[checkIdx].edge.value_or(ltl::ClockEdge::Pos);
          auto clockPos = checkIdx < clockPoses.size()
                              ? clockPoses[checkIdx]
                              : std::optional<unsigned>{};
          if (!clockPos) {
            if (!anyPosedge)
              return Value();
            if (edge == ltl::ClockEdge::Pos)
              return anyPosedge;
            if (edge == ltl::ClockEdge::Neg && anyNegedge)
              return anyNegedge;
            if (edge == ltl::ClockEdge::Both && anyNegedge)
              return smt::OrOp::create(rewriter, loc, anyPosedge, anyNegedge);
            return Value();
          }
          if (edge == ltl::ClockEdge::Pos) {
            if (clockIndexes.size() == 1)
              return isPosedge;
            if (*clockPos < posedges.size())
              return posedges[*clockPos];
          } else if (edge == ltl::ClockEdge::Neg) {
            if (clockIndexes.size() == 1)
              return isNegedge;
            if (*clockPos < negedges.size())
              return negedges[*clockPos];
          } else if (edge == ltl::ClockEdge::Both) {
            if (clockIndexes.size() == 1)
              return smt::OrOp::create(rewriter, loc, isPosedge, isNegedge);
            if (*clockPos < posedges.size())
              return smt::OrOp::create(rewriter, loc, posedges[*clockPos],
                                       negedges[*clockPos]);
          }
          return Value();
        };
        for (auto [idx, finalVal] : llvm::enumerate(finalCheckOutputs)) {
          Value gate = getCheckGate(idx, finalCheckClockPos, finalCheckInfos);
          Value sampled = gate ? gate : constTrue;
          if (livenessLassoMode) {
            finalCheckSamplesThisIter.push_back(sampled);
            Value isTrue;
            if (isa<smt::BoolType>(finalVal.getType())) {
              isTrue = finalVal;
            } else {
              auto trueBV = smt::BVConstantOp::create(rewriter, loc, 1, 1);
              isTrue = smt::EqOp::create(rewriter, loc, finalVal, trueBV);
            }
            finalCheckSampledTrueThisIter.push_back(
                smt::AndOp::create(rewriter, loc, sampled, isTrue));
          }
          if (gate && idx < finalCheckStates.size()) {
            newDecls.push_back(smt::IteOp::create(rewriter, loc, gate, finalVal,
                                                 finalCheckStates[idx]));
          } else {
            newDecls.push_back(finalVal);
          }
        }
        newDecls.push_back(violated);
        iterArgs = std::move(newDecls);
        captureStateSnapshot(ValueRange(iterArgs));
        if (livenessLassoMode) {
          lassoFinalCheckSampleHistory.push_back(std::move(finalCheckSamplesThisIter));
          lassoFinalCheckSampledTrueHistory.push_back(
              std::move(finalCheckSampledTrueThisIter));
        }
      }

      if (livenessLassoMode) {
        if (boundValue == 0) {
          op.emitError("liveness-lasso mode requires bound >= 1");
          return failure();
        }
        if (lassoFinalCheckSampleHistory.size() != boundValue) {
          op.emitError("internal error while building liveness-lasso "
                       "fairness history");
          return failure();
        }
        if (lassoFinalCheckSampledTrueHistory.size() != boundValue) {
          op.emitError("internal error while building liveness-lasso "
                       "sampled-true fairness history");
          return failure();
        }
        Value hasLassoLoop;
        if (lassoStateHistory.size() >= 2) {
          ArrayRef<Value> finalState = lassoStateHistory.back();
          for (auto [candidateIndex, candidate] :
               llvm::enumerate(
                   ArrayRef<SmallVector<Value>>(lassoStateHistory).drop_back(1))) {
            Value candidateLoop = constTrue;
            if (candidate.size() != finalState.size()) {
              op.emitError("internal error while building liveness-lasso "
                           "state snapshots");
              return failure();
            }
            if (numBMCStateArgs > 0) {
              Value thisLoop;
              for (auto [lhs, rhs] : llvm::zip(candidate, finalState)) {
                Value eq = smt::EqOp::create(rewriter, loc, lhs, rhs);
                thisLoop = combineAnd(thisLoop, eq);
              }
              candidateLoop = combineAnd(candidateLoop, thisLoop);
            }

            // Fairness-style acceptance: for each final check, require at least
            // one sampling point on the selected loop segment.
            Value samplingFairness;
            for (size_t checkIdx = 0; checkIdx < numFinalChecks; ++checkIdx) {
              Value sampledOnLoop;
              Value sampledTrueOnLoop;
              for (size_t iterIdx = candidateIndex; iterIdx < boundValue;
                   ++iterIdx) {
                ArrayRef<Value> iterSamples = lassoFinalCheckSampleHistory[iterIdx];
                ArrayRef<Value> iterSampledTrue =
                    lassoFinalCheckSampledTrueHistory[iterIdx];
                if (checkIdx >= iterSamples.size()) {
                  op.emitError("internal error while building liveness-lasso "
                               "sampling constraints");
                  return failure();
                }
                if (checkIdx >= iterSampledTrue.size()) {
                  op.emitError("internal error while building liveness-lasso "
                               "sampled-true constraints");
                  return failure();
                }
                sampledOnLoop = combineOr(sampledOnLoop, iterSamples[checkIdx]);
                sampledTrueOnLoop =
                    combineOr(sampledTrueOnLoop, iterSampledTrue[checkIdx]);
              }
              if (!sampledOnLoop)
                sampledOnLoop = constFalse;
              samplingFairness = combineAnd(samplingFairness, sampledOnLoop);
              if (!finalCheckIsCover[checkIdx]) {
                if (!sampledTrueOnLoop)
                  sampledTrueOnLoop = constFalse;
                Value noSampledTrue =
                    smt::NotOp::create(rewriter, loc, sampledTrueOnLoop);
                samplingFairness = combineAnd(samplingFairness, noSampledTrue);
              }
            }
            if (samplingFairness)
              candidateLoop = combineAnd(candidateLoop, samplingFairness);

            hasLassoLoop = combineOr(hasLassoLoop, candidateLoop);
          }
        }
        if (!hasLassoLoop)
          hasLassoLoop = constFalse;
        smt::AssertOp::create(rewriter, loc, hasLassoLoop);
      }

      Value violated = iterArgs.back();
      Value finalCheckViolated;
      Value finalCoverHit;
      if (numFinalChecks > 0 && checkFinalAtEnd) {
        ValueRange results(iterArgs);
        size_t finalStart = results.size() - 1 - numFinalChecks;
        SmallVector<Value> finalAssertOutputs;
        SmallVector<Value> finalCoverOutputs;
        finalAssertOutputs.reserve(numFinalChecks);
        finalCoverOutputs.reserve(numFinalChecks);
        for (size_t i = 0; i < numFinalChecks; ++i) {
          Value finalVal = results[finalStart + i];
          if (finalCheckIsCover[i])
            finalCoverOutputs.push_back(finalVal);
          else
            finalAssertOutputs.push_back(finalVal);
        }

        if (!finalAssertOutputs.empty()) {
          for (Value finalVal : finalAssertOutputs) {
            Value isTrue;
            if (isa<smt::BoolType>(finalVal.getType())) {
              isTrue = finalVal;
            } else {
              auto trueBV = smt::BVConstantOp::create(rewriter, loc, 1, 1);
              isTrue = smt::EqOp::create(rewriter, loc, finalVal, trueBV);
            }
            Value isFalse = smt::NotOp::create(rewriter, loc, isTrue);
            finalCheckViolated = combineOr(finalCheckViolated, isFalse);
          }
        }

        if (!finalCoverOutputs.empty()) {
          SmallVector<Value> coverTerms;
          coverTerms.reserve(finalCoverOutputs.size());
          for (Value finalVal : finalCoverOutputs) {
            if (isa<smt::BoolType>(finalVal.getType())) {
              coverTerms.push_back(finalVal);
            } else {
              auto trueBV = smt::BVConstantOp::create(rewriter, loc, 1, 1);
              coverTerms.push_back(
                  smt::EqOp::create(rewriter, loc, finalVal, trueBV));
            }
          }
          finalCoverHit =
              coverTerms.size() == 1
                  ? coverTerms.front()
                  : smt::OrOp::create(rewriter, loc, coverTerms).getResult();
        }
      }

      Value overallCond;
      if (isCoverCheck) {
        overallCond = combineOr(violated, finalCoverHit);
      } else {
        overallCond = combineOr(violated, finalCheckViolated);
      }
      if (overallCond)
        smt::AssertOp::create(rewriter, loc, overallCond);

      auto checkOp = smt::CheckOp::create(rewriter, loc, TypeRange{});
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.createBlock(&checkOp.getSatRegion());
        smt::YieldOp::create(rewriter, loc);
        rewriter.createBlock(&checkOp.getUnknownRegion());
        smt::YieldOp::create(rewriter, loc);
        rewriter.createBlock(&checkOp.getUnsatRegion());
        smt::YieldOp::create(rewriter, loc);
      }
      rewriter.setInsertionPointAfter(checkOp);
      smt::YieldOp::create(rewriter, loc);

      rewriter.setInsertionPointAfter(solver);
      if (op->getNumResults() == 0) {
        rewriter.eraseOp(op);
      } else {
        auto dummy =
            arith::ConstantOp::create(rewriter, loc,
                                      rewriter.getBoolAttr(false));
        rewriter.replaceOp(op, dummy.getResult());
      }
      return success();
    }

    // TODO: swapping to a whileOp here would allow early exit once the property
    // is violated
    // Perform model check up to the provided bound
    auto forOp = scf::ForOp::create(
        rewriter, loc, lowerBound, upperBound, step, inputDecls,
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          // Assert 2-state constraints on the current iteration inputs.
          size_t numCircuitArgs = circuitFuncOp.getNumArguments();
          for (auto [index, pair] : llvm::enumerate(llvm::zip(
                   TypeRange(oldCircuitInputTy).take_front(numCircuitArgs),
                   iterArgs.take_front(numCircuitArgs)))) {
            auto [oldTy, arg] = pair;
            maybeAssertKnown(index, oldTy, arg, builder);
          }

          // Call loop func to update clock & state arg values
          SmallVector<Value> loopCallInputs;
          // Fetch clock values to feed to loop
          for (auto index : clockIndexes)
            loopCallInputs.push_back(iterArgs[index]);
          // Fetch state args to feed to loop
          for (auto stateArg :
               iterArgs.drop_back(1 + numFinalChecks).take_back(numStateArgs))
            loopCallInputs.push_back(stateArg);
          ValueRange loopVals =
              func::CallOp::create(builder, loc, loopFuncOp, loopCallInputs)
                  ->getResults();

          // Compute clock edges for this iteration.
          Value isPosedge;
          Value isNegedge;
          SmallVector<Value> posedges;
          SmallVector<Value> negedges;
          Value anyPosedge;
          Value anyNegedge;
          bool usePosedge = !risingClocksOnly && clockIndexes.size() == 1;
          bool usePerRegPosedge =
              !risingClocksOnly && clockIndexes.size() > 1 && numRegs > 0;
          bool needClockEdges =
              !risingClocksOnly && !clockIndexes.empty() &&
              (usePerRegPosedge || totalDelaySlots > 0 ||
               totalNFAStateSlots > 0 || !nonFinalCheckInfos.empty() ||
               !finalCheckInfos.empty());
          if (usePosedge) {
            auto clockIndex = clockIndexes[0];
            auto oldClock = iterArgs[clockIndex];
            // The clock is necessarily the first value returned by the loop
            // region
            auto newClock = loopVals[0];
            auto oldClockLow = smt::BVNotOp::create(builder, loc, oldClock);
            auto newClockLow = smt::BVNotOp::create(builder, loc, newClock);
            auto isPosedgeBV =
                smt::BVAndOp::create(builder, loc, oldClockLow, newClock);
            auto isNegedgeBV =
                smt::BVAndOp::create(builder, loc, oldClock, newClockLow);
            // Convert edge bv<1> to bool
            auto trueBV = smt::BVConstantOp::create(builder, loc, 1, 1);
            isPosedge = smt::EqOp::create(builder, loc, isPosedgeBV, trueBV);
            isNegedge = smt::EqOp::create(builder, loc, isNegedgeBV, trueBV);
            if (needClockEdges) {
              posedges.push_back(isPosedge);
              negedges.push_back(isNegedge);
              anyPosedge = isPosedge;
              anyNegedge = isNegedge;
            }
          } else if (needClockEdges) {
            posedges.reserve(clockIndexes.size());
            negedges.reserve(clockIndexes.size());
            auto trueBV = smt::BVConstantOp::create(builder, loc, 1, 1);
            for (auto [idx, clockIndex] : llvm::enumerate(clockIndexes)) {
              auto oldClock = iterArgs[clockIndex];
              auto newClock = loopVals[idx];
              auto oldClockLow =
                  smt::BVNotOp::create(builder, loc, oldClock);
              auto newClockLow =
                  smt::BVNotOp::create(builder, loc, newClock);
              auto isPosedgeBV =
                  smt::BVAndOp::create(builder, loc, oldClockLow, newClock);
              auto isNegedgeBV =
                  smt::BVAndOp::create(builder, loc, oldClock, newClockLow);
              posedges.push_back(
                  smt::EqOp::create(builder, loc, isPosedgeBV, trueBV));
              negedges.push_back(
                  smt::EqOp::create(builder, loc, isNegedgeBV, trueBV));
            }
            if (!posedges.empty()) {
              anyPosedge = posedges.front();
              anyNegedge = negedges.front();
              for (auto [pos, neg] :
                   llvm::zip(ArrayRef<Value>(posedges).drop_front(),
                             ArrayRef<Value>(negedges).drop_front())) {
                anyPosedge = smt::OrOp::create(builder, loc, anyPosedge, pos);
                anyNegedge = smt::OrOp::create(builder, loc, anyNegedge, neg);
              }
            }
          }

          Value defaultDelayGate;
          if (!risingClocksOnly && !clockIndexes.empty()) {
            if (usePosedge)
              defaultDelayGate = isPosedge;
            else if (anyPosedge)
              defaultDelayGate = anyPosedge;
          }

          auto getDelayGate = [&](StringAttr clockName, Value clockValue,
                                  std::optional<ltl::ClockEdge> edge) -> Value {
            if (risingClocksOnly || clockIndexes.empty())
              return Value();
            std::optional<unsigned> pos;
            if (clockName && !clockName.getValue().empty()) {
              auto it = clockNameToPos.find(clockName.getValue());
              if (it != clockNameToPos.end())
                pos = it->second;
            } else if (clockValue) {
              if (auto info = resolveClockPosInfo(clockValue))
                pos = info->pos;
            } else if (clockIndexes.size() == 1) {
              pos = 0;
            }
            if (!pos || posedges.empty() || negedges.empty())
              return defaultDelayGate;
            Value posedge = posedges[*pos];
            Value negedge = negedges[*pos];
            ltl::ClockEdge edgeKind = edge.value_or(ltl::ClockEdge::Pos);
            if (clockValue) {
              if (auto info = resolveClockPosInfo(clockValue);
                  info && info->invert)
                edgeKind = invertClockEdge(edgeKind);
            }
            switch (edgeKind) {
            case ltl::ClockEdge::Pos:
              return posedge;
            case ltl::ClockEdge::Neg:
              return negedge;
            case ltl::ClockEdge::Both:
              return smt::OrOp::create(builder, loc, posedge, negedge);
            }
            return defaultDelayGate;
          };

          auto materializeTickValue = [&](Type argTy, Value tickBool) -> Value {
            if (!tickBool)
              return tickBool;
            if (isa<smt::BoolType>(argTy))
              return tickBool;
            if (auto bvTy = dyn_cast<smt::BitVectorType>(argTy)) {
              if (bvTy.getWidth() == 1) {
                auto one = smt::BVConstantOp::create(builder, loc, 1, 1);
                auto zero = smt::BVConstantOp::create(builder, loc, 0, 1);
                return smt::IteOp::create(builder, loc, tickBool, one, zero);
              }
            }
            return tickBool;
          };

          auto getNFATickGate = [&](const NFATickGateInfo &info) -> Value {
            if (risingClocksOnly || clockIndexes.empty())
              return Value();
            if (info.clockPos && *info.clockPos < posedges.size() &&
                *info.clockPos < negedges.size()) {
              Value posedge = posedges[*info.clockPos];
              Value negedge = negedges[*info.clockPos];
              ltl::ClockEdge edgeKind = info.edge;
              if (info.invert)
                edgeKind = invertClockEdge(edgeKind);
              switch (edgeKind) {
              case ltl::ClockEdge::Pos:
                return posedge;
              case ltl::ClockEdge::Neg:
                return negedge;
              case ltl::ClockEdge::Both:
                return smt::OrOp::create(builder, loc, posedge, negedge);
              }
            }
            if (info.hasExplicitClock)
              return defaultDelayGate;
            return defaultDelayGate;
          };

          SmallVector<Value> circuitInputs(
              iterArgs.take_front(numCircuitArgs).begin(),
              iterArgs.take_front(numCircuitArgs).end());
          // Use post-edge clock values for property/circuit evaluation so
          // clocked expressions observe the sampled clock value.
          if (!clockIndexes.empty()) {
            for (auto [idx, clockIndex] : llvm::enumerate(clockIndexes)) {
              if (static_cast<size_t>(clockIndex) < circuitInputs.size() &&
                  static_cast<size_t>(idx) < loopVals.size())
                circuitInputs[clockIndex] = loopVals[idx];
            }
          }
          if (!clockSourceInputs.empty()) {
            for (auto [argIndex, info] : clockSourceInputs) {
              if (argIndex >= circuitInputs.size() ||
                  info.pos >= loopVals.size())
                continue;
              int64_t valueWidth = 0;
              int64_t unknownWidth = 0;
              if (!isFourStateStruct(oldCircuitInputTy[argIndex], valueWidth,
                                     unknownWidth))
                continue;
              if (valueWidth != 1)
                continue;
              auto bvTy = dyn_cast<smt::BitVectorType>(
                  circuitInputs[argIndex].getType());
              if (!bvTy || bvTy.getWidth() !=
                               static_cast<unsigned>(valueWidth +
                                                     unknownWidth))
                continue;
              Value valueBit = loopVals[info.pos];
              if (info.invert)
                valueBit = smt::BVNotOp::create(builder, loc, valueBit);
              auto unkTy =
                  smt::BitVectorType::get(builder.getContext(), unknownWidth);
              Value unknownBits =
                  smt::ExtractOp::create(builder, loc, unkTy, 0,
                                         circuitInputs[argIndex]);
              circuitInputs[argIndex] =
                  smt::ConcatOp::create(builder, loc, valueBit, unknownBits);
            }
          }
          if (!nfaTickGateInfos.empty()) {
            auto trueTick =
                smt::BoolConstantOp::create(builder, loc, true).getResult();
            for (const auto &tickInfo : nfaTickGateInfos) {
              if (tickInfo.tickArgIndex >= circuitInputs.size())
                continue;
              Value gate = getNFATickGate(tickInfo);
              Value tick = gate ? gate : trueTick;
              Type argTy = circuitInputTy[tickInfo.tickArgIndex];
              circuitInputs[tickInfo.tickArgIndex] =
                  materializeTickValue(argTy, tick);
            }
          }

          // Execute the circuit
          ValueRange circuitCallOuts =
              func::CallOp::create(builder, loc, circuitFuncOp, circuitInputs)
                  ->getResults();

          // Circuit outputs are ordered as:
          // [original outputs (registers)] [delay buffer outputs] [nfa states]
          // [non-final checks] [final checks]
          //
          // Note: totalDelaySlots is captured from the outer scope
          ValueRange finalCheckOutputs =
              numFinalChecks == 0 ? ValueRange{}
                                  : circuitCallOuts.take_back(numFinalChecks);
          ValueRange beforeFinal =
              numFinalChecks == 0 ? circuitCallOuts
                                  : circuitCallOuts.drop_back(numFinalChecks);
          ValueRange nonFinalCheckOutputs =
              numNonFinalChecks == 0
                  ? ValueRange{}
                  : beforeFinal.take_back(numNonFinalChecks);
          ValueRange nonFinalOutputs =
              numNonFinalChecks == 0
                  ? beforeFinal
                  : beforeFinal.drop_back(numNonFinalChecks);
          // Split non-final outputs into register outputs, delay buffers, and NFA states.
          ValueRange nfaStateOutputs =
              totalNFAStateSlots == 0
                  ? ValueRange{}
                  : nonFinalOutputs.take_back(totalNFAStateSlots);
          ValueRange beforeNFA =
              totalNFAStateSlots == 0
                  ? nonFinalOutputs
                  : nonFinalOutputs.drop_back(totalNFAStateSlots);
          ValueRange delayBufferOutputs =
              totalDelaySlots == 0 ? ValueRange{}
                                   : beforeNFA.take_back(totalDelaySlots);
          ValueRange circuitOutputs =
              totalDelaySlots == 0 ? beforeNFA
                                   : beforeNFA.drop_back(totalDelaySlots);

          Value violated = iterArgs.back();
          if (numNonFinalChecks > 0) {
            // If we have a cycle up to which we ignore assertions, we need an
            // IfOp to track this.
            auto insideForPoint = builder.saveInsertionPoint();
            // We need to still have the yielded result of the op in scope after
            // we've built the check.
            Value yieldedValue;
            auto ignoreAssertionsUntil =
                op->getAttrOfType<IntegerAttr>("ignore_asserts_until");
            if (ignoreAssertionsUntil) {
              auto ignoreUntilConstant = arith::ConstantOp::create(
                  builder, loc,
                  rewriter.getI32IntegerAttr(
                      ignoreAssertionsUntil.getValue().getZExtValue()));
              auto shouldSkip = arith::CmpIOp::create(
                  builder, loc, arith::CmpIPredicate::ult, i,
                  ignoreUntilConstant);
              auto ifShouldSkip = scf::IfOp::create(
                  builder, loc, builder.getI1Type(), shouldSkip, true);
              // If we should skip, yield the existing value.
              builder.setInsertionPointToEnd(
                  &ifShouldSkip.getThenRegion().front());
              scf::YieldOp::create(builder, loc, ValueRange(iterArgs.back()));
              builder.setInsertionPointToEnd(
                  &ifShouldSkip.getElseRegion().front());
              yieldedValue = ifShouldSkip.getResult(0);
            }

            Value combinedCheckCond;
            for (auto [checkIdx, checkVal] :
                 llvm::enumerate(nonFinalCheckOutputs)) {
              Value isTrue;
              if (isa<smt::BoolType>(checkVal.getType())) {
                // LTL properties are converted to !smt.bool, use directly
                isTrue = checkVal;
              } else {
                // i1 properties are converted to !smt.bv<1>, compare with 1
                auto trueBV = smt::BVConstantOp::create(builder, loc, 1, 1);
                isTrue = smt::EqOp::create(builder, loc, checkVal, trueBV);
              }

              Value term = isCoverCheck
                               ? isTrue
                               : smt::NotOp::create(builder, loc, isTrue);
              Value gate;
              if (!risingClocksOnly && !clockIndexes.empty() &&
                  checkIdx < nonFinalCheckInfos.size()) {
                auto edge = nonFinalCheckInfos[checkIdx].edge.value_or(
                    ltl::ClockEdge::Pos);
                auto clockPos =
                    checkIdx < nonFinalCheckClockPos.size()
                        ? nonFinalCheckClockPos[checkIdx]
                        : std::optional<unsigned>{};
                if (clockPos) {
                  if (edge == ltl::ClockEdge::Pos) {
                    if (clockIndexes.size() == 1)
                      gate = isPosedge;
                    else if (*clockPos < posedges.size())
                      gate = posedges[*clockPos];
                  } else if (edge == ltl::ClockEdge::Neg) {
                    if (clockIndexes.size() == 1)
                      gate = isNegedge;
                    else if (*clockPos < negedges.size())
                      gate = negedges[*clockPos];
                  } else if (edge == ltl::ClockEdge::Both) {
                    if (clockIndexes.size() == 1)
                      gate = smt::OrOp::create(builder, loc, isPosedge,
                                               isNegedge);
                    else if (*clockPos < posedges.size())
                      gate = smt::OrOp::create(builder, loc,
                                               posedges[*clockPos],
                                               negedges[*clockPos]);
                  }
                }
                if (!gate && anyPosedge) {
                  if (edge == ltl::ClockEdge::Pos) {
                    gate = anyPosedge;
                  } else if (edge == ltl::ClockEdge::Neg && anyNegedge) {
                    gate = anyNegedge;
                  } else if (edge == ltl::ClockEdge::Both && anyNegedge) {
                    gate = smt::OrOp::create(builder, loc, anyPosedge,
                                             anyNegedge);
                  }
                }
              }

              if (gate)
                term = smt::AndOp::create(builder, loc, gate, term);
              if (!combinedCheckCond)
                combinedCheckCond = term;
              else
                combinedCheckCond =
                    smt::OrOp::create(builder, loc, combinedCheckCond, term);
            }
            Value checkCond = combinedCheckCond;

            smt::PushOp::create(builder, loc, 1);
            smt::AssertOp::create(builder, loc, checkCond);
            auto checkOp =
                smt::CheckOp::create(rewriter, loc, builder.getI1Type());
            {
              OpBuilder::InsertionGuard guard(builder);
              builder.createBlock(&checkOp.getSatRegion());
              smt::YieldOp::create(builder, loc, constTrue);
              builder.createBlock(&checkOp.getUnknownRegion());
              smt::YieldOp::create(builder, loc, constTrue);
              builder.createBlock(&checkOp.getUnsatRegion());
              smt::YieldOp::create(builder, loc, constFalse);
            }
            smt::PopOp::create(builder, loc, 1);

            Value newViolated = arith::OrIOp::create(
                builder, loc, checkOp.getResult(0), iterArgs.back());

            // If we've packaged everything in an IfOp, we need to yield the
            // new violated value.
            if (ignoreAssertionsUntil) {
              scf::YieldOp::create(builder, loc, newViolated);
              // Replace the variable with the yielded value.
              violated = yieldedValue;
            } else {
              violated = newViolated;
            }

            // If we created an IfOp, make sure we start inserting after it
            // again.
            builder.restoreInsertionPoint(insideForPoint);
          }

          size_t loopIndex = 0;
          // Collect decls to yield at end of iteration
          SmallVector<Value> newDecls;
          size_t nonRegIdx = 0;
          size_t argIndex = 0;
          // Circuit args are: [clocks, inputs, ticks] [registers]
          // [delay buffers] [nfa states]
          // Drop registers/buffers/nfa to get just clocks and inputs
          size_t numNonStateArgs = oldCircuitInputTy.size() - numRegs -
                                   totalDelaySlots - totalNFAStateSlots;
          for (auto [oldTy, newTy] :
               llvm::zip(TypeRange(oldCircuitInputTy).take_front(numNonStateArgs),
                         TypeRange(circuitInputTy).take_front(numNonStateArgs))) {
            // Check if this is a clock - either explicit seq::ClockType or
            // an i1 that corresponds to an init clock
            bool isI1Type = isa<IntegerType>(oldTy) &&
                            cast<IntegerType>(oldTy).getWidth() == 1;
            bool isClock = isa<seq::ClockType>(oldTy) ||
                           (isI1Type && nonRegIdx < numInitClocks);
            if (tickArgIndices.contains(argIndex)) {
              Value initVal = makeBoolConstant(newTy, false);
              assert(initVal && "expected makeBoolConstant to succeed for tick arg");
              newDecls.push_back(initVal);
            } else if (isClock) {
              newDecls.push_back(loopVals[loopIndex++]);
            } else {
              auto decl = smt::DeclareFunOp::create(
                  builder, loc, newTy,
                  argIndex < inputNamePrefixes.size()
                      ? inputNamePrefixes[argIndex]
                      : StringAttr{});
              newDecls.push_back(decl);
              maybeAssertKnown(argIndex, oldTy, decl, builder);
            }
            nonRegIdx++;
            argIndex++;
          }

          for (const auto &witness : armWitnesses) {
            if (witness.sourceArgIndex >= numNonStateArgs ||
                witness.sourceArgIndex >= newDecls.size() ||
                witness.sourceArgIndex >= iterArgs.size())
              continue;
            auto firedOr = buildWitnessFired(
                builder, iterArgs.take_front(numNonStateArgs),
                ValueRange(newDecls).take_front(numNonStateArgs), witness);
            if (failed(firedOr))
              continue;
            auto witnessDecl = smt::DeclareFunOp::create(builder, loc, boolTy,
                                                         witness.witnessName);
            auto witnessEq =
                smt::EqOp::create(builder, loc, witnessDecl, *firedOr);
            smt::AssertOp::create(builder, loc, witnessEq);
          }

          // Update registers using the computed clock edge signals.
          ValueRange regInputs =
              numRegs == 0 ? ValueRange{} : circuitOutputs.take_back(numRegs);
          if (clockIndexes.size() >= 1) {
            if (risingClocksOnly || clockIndexes.size() == 1) {
              if (risingClocksOnly) {
                // In rising clocks only mode we don't need to worry about
                // whether there was a posedge.
                newDecls.append(regInputs.begin(), regInputs.end());
              } else {
                auto regStates =
                    iterArgs.take_front(circuitFuncOp.getNumArguments())
                        .take_back(numRegs + totalDelaySlots + totalNFAStateSlots)
                        .drop_back(totalDelaySlots + totalNFAStateSlots);
                SmallVector<Value> nextRegStates;
                for (auto [regState, regInput] :
                     llvm::zip(regStates, regInputs)) {
                  // Create an ITE to calculate the next reg state
                  // TODO: we create a lot of ITEs here that will slow things down
                  // - these could be avoided by making init/loop regions concrete
                  nextRegStates.push_back(smt::IteOp::create(
                      builder, loc, isPosedge, regInput, regState));
                }
                newDecls.append(nextRegStates);
              }
            } else if (usePerRegPosedge) {
              auto regStates =
                  iterArgs.take_front(circuitFuncOp.getNumArguments())
                      .take_back(numRegs + totalDelaySlots + totalNFAStateSlots)
                      .drop_back(totalDelaySlots + totalNFAStateSlots);
              SmallVector<Value> nextRegStates;
              nextRegStates.reserve(numRegs);
            for (auto [idx, pair] :
                 llvm::enumerate(llvm::zip(regStates, regInputs))) {
              auto [regState, regInput] = pair;
              Value regPosedge =
                  regClockInverts[idx] ? negedges[regClockToLoopIndex[idx]]
                                       : posedges[regClockToLoopIndex[idx]];
              nextRegStates.push_back(smt::IteOp::create(
                  builder, loc, regPosedge, regInput, regState));
            }
              newDecls.append(nextRegStates);
            }
          } else if (numRegs > 0) {
            // If no clock inputs were detected, keep register state flowing so
            // the BMC loop arity remains consistent.
            newDecls.append(regInputs.begin(), regInputs.end());
          }

          // Add delay buffer outputs for the next iteration
          // These are the shifted buffer values from the circuit
          if (totalDelaySlots > 0) {
            auto delayStates =
                iterArgs.take_front(circuitFuncOp.getNumArguments())
                    .take_back(totalDelaySlots + totalNFAStateSlots)
                    .drop_back(totalNFAStateSlots);
            size_t delayIndex = 0;
            auto appendDelayUpdates = [&](Value gate, size_t count) {
              for (size_t i = 0; i < count; ++i) {
                Value delayState = delayStates[delayIndex + i];
                Value delayVal = delayBufferOutputs[delayIndex + i];
                if (gate) {
                  newDecls.push_back(smt::IteOp::create(builder, loc, gate,
                                                       delayVal, delayState));
                } else {
                  newDecls.push_back(delayVal);
                }
              }
              delayIndex += count;
            };

            for (const auto &info : delayInfos) {
              Value gate =
                  getDelayGate(info.clockName, info.clockValue, info.edge);
              appendDelayUpdates(gate, info.bufferSize);
            }
            for (const auto &info : pastInfos) {
              Value gate =
                  getDelayGate(info.clockName, info.clockValue, info.edge);
              appendDelayUpdates(gate, info.bufferSize);
            }
            // Defensive fallback if buffers were not fully covered.
            for (; delayIndex < totalDelaySlots; ++delayIndex)
              newDecls.push_back(delayBufferOutputs[delayIndex]);
          }

          if (totalNFAStateSlots > 0)
            newDecls.append(nfaStateOutputs.begin(), nfaStateOutputs.end());

          // Add the rest of the loop state args
          for (; loopIndex < loopVals.size(); ++loopIndex)
            newDecls.push_back(loopVals[loopIndex]);

          // Pass through finalCheckOutputs (already !smt.bv<1> or !smt.bool)
          // for next iteration, gated by their clock edge if available.
          ValueRange finalCheckStates =
              numFinalChecks == 0
                  ? ValueRange{}
                  : iterArgs.drop_back(1).take_back(numFinalChecks);
          auto getCheckGate =
              [&](size_t checkIdx,
                  ArrayRef<std::optional<unsigned>> clockPoses,
                  ArrayRef<NonFinalCheckInfo> infos) -> Value {
            if (risingClocksOnly || clockIndexes.empty() ||
                checkIdx >= infos.size())
              return Value();
            auto edge = infos[checkIdx].edge.value_or(ltl::ClockEdge::Pos);
            auto clockPos = checkIdx < clockPoses.size()
                                ? clockPoses[checkIdx]
                                : std::optional<unsigned>{};
            if (!clockPos) {
              if (!anyPosedge)
                return Value();
              if (edge == ltl::ClockEdge::Pos)
                return anyPosedge;
              if (edge == ltl::ClockEdge::Neg && anyNegedge)
                return anyNegedge;
              if (edge == ltl::ClockEdge::Both && anyNegedge)
                return smt::OrOp::create(builder, loc, anyPosedge,
                                         anyNegedge);
              return Value();
            }
            if (edge == ltl::ClockEdge::Pos) {
              if (clockIndexes.size() == 1)
                return isPosedge;
              if (*clockPos < posedges.size())
                return posedges[*clockPos];
            } else if (edge == ltl::ClockEdge::Neg) {
              if (clockIndexes.size() == 1)
                return isNegedge;
              if (*clockPos < negedges.size())
                return negedges[*clockPos];
            } else if (edge == ltl::ClockEdge::Both) {
              if (clockIndexes.size() == 1)
                return smt::OrOp::create(builder, loc, isPosedge, isNegedge);
              if (*clockPos < posedges.size())
                return smt::OrOp::create(builder, loc, posedges[*clockPos],
                                         negedges[*clockPos]);
            }
            return Value();
          };
          for (auto [idx, finalVal] : llvm::enumerate(finalCheckOutputs)) {
            Value gate =
                getCheckGate(idx, finalCheckClockPos, finalCheckInfos);
            if (gate && idx < finalCheckStates.size()) {
              newDecls.push_back(smt::IteOp::create(
                  builder, loc, gate, finalVal, finalCheckStates[idx]));
            } else {
              newDecls.push_back(finalVal);
            }
          }
          newDecls.push_back(violated);

          scf::YieldOp::create(builder, loc, newDecls);
        });

    // Get the violation flag from the loop
    Value violated = forOp->getResults().back();

    // If there are final checks, compute any final assertion violation and
    // any final cover success.
    Value finalCheckViolated = constFalse;
    Value finalCoverHit = constFalse;
    if (numFinalChecks > 0 && checkFinalAtEnd) {
      auto results = forOp->getResults();
      size_t finalStart = results.size() - 1 - numFinalChecks;
      SmallVector<Value> finalAssertOutputs;
      SmallVector<Value> finalCoverOutputs;
      finalAssertOutputs.reserve(numFinalChecks);
      finalCoverOutputs.reserve(numFinalChecks);
      for (size_t i = 0; i < numFinalChecks; ++i) {
        Value finalVal = results[finalStart + i];
        if (finalCheckIsCover[i])
          finalCoverOutputs.push_back(finalVal);
        else
          finalAssertOutputs.push_back(finalVal);
      }

      if (!finalAssertOutputs.empty()) {
        smt::PushOp::create(rewriter, loc, 1);
        Value anyFinalAssertViolation;
        for (Value finalVal : finalAssertOutputs) {
          Value isTrue;
          if (isa<smt::BoolType>(finalVal.getType())) {
            // LTL properties are converted to !smt.bool, use directly
            isTrue = finalVal;
          } else {
            // i1 properties are converted to !smt.bv<1>, compare with 1
            auto trueBV = smt::BVConstantOp::create(rewriter, loc, 1, 1);
            isTrue = smt::EqOp::create(rewriter, loc, finalVal, trueBV);
          }
          // Assert the negation: we're looking for cases where the check FAILS
          Value isFalse = smt::NotOp::create(rewriter, loc, isTrue);
          if (!anyFinalAssertViolation)
            anyFinalAssertViolation = isFalse;
          else
            anyFinalAssertViolation =
                smt::OrOp::create(rewriter, loc, anyFinalAssertViolation,
                                  isFalse);
        }
        smt::AssertOp::create(rewriter, loc, anyFinalAssertViolation);
        // Now check if there's a satisfying assignment (i.e., a violation)
        auto finalCheckOp =
            smt::CheckOp::create(rewriter, loc, rewriter.getI1Type());
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.createBlock(&finalCheckOp.getSatRegion());
          smt::YieldOp::create(rewriter, loc, constTrue);
          rewriter.createBlock(&finalCheckOp.getUnknownRegion());
          smt::YieldOp::create(rewriter, loc, constTrue);
          rewriter.createBlock(&finalCheckOp.getUnsatRegion());
          smt::YieldOp::create(rewriter, loc, constFalse);
        }
        smt::PopOp::create(rewriter, loc, 1);
        finalCheckViolated = finalCheckOp.getResult(0);
      }

      if (!finalCoverOutputs.empty()) {
        smt::PushOp::create(rewriter, loc, 1);
        SmallVector<Value> coverTerms;
        coverTerms.reserve(finalCoverOutputs.size());
        for (Value finalVal : finalCoverOutputs) {
          if (isa<smt::BoolType>(finalVal.getType())) {
            coverTerms.push_back(finalVal);
          } else {
            auto trueBV = smt::BVConstantOp::create(rewriter, loc, 1, 1);
            coverTerms.push_back(
                smt::EqOp::create(rewriter, loc, finalVal, trueBV));
          }
        }
        Value anyCover =
            coverTerms.size() == 1
                ? coverTerms.front()
                : smt::OrOp::create(rewriter, loc, coverTerms).getResult();
        smt::AssertOp::create(rewriter, loc, anyCover);
        auto finalCoverOp =
            smt::CheckOp::create(rewriter, loc, rewriter.getI1Type());
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.createBlock(&finalCoverOp.getSatRegion());
          smt::YieldOp::create(rewriter, loc, constTrue);
          rewriter.createBlock(&finalCoverOp.getUnknownRegion());
          smt::YieldOp::create(rewriter, loc, constTrue);
          rewriter.createBlock(&finalCoverOp.getUnsatRegion());
          smt::YieldOp::create(rewriter, loc, constFalse);
        }
        smt::PopOp::create(rewriter, loc, 1);
        finalCoverHit = finalCoverOp.getResult(0);
      }
    }

    // Combine results: return true if no "interesting" condition was found.
    // This matches the tool convention of printing UNSAT when nothing was
    // found (assertions hold; covers not hit), and SAT when something was
    // found (assertion violation; cover witness).
    Value res;
    if (isCoverCheck) {
      // Covers are "interesting" when they are hit; invert so `true` means
      // "no cover witness found".
      Value anyCoverHit =
          arith::OrIOp::create(rewriter, loc, violated, finalCoverHit);
      res = arith::XOrIOp::create(rewriter, loc, anyCoverHit, constTrue);
    } else {
      Value anyViolation =
          arith::OrIOp::create(rewriter, loc, violated, finalCheckViolated);
      res = arith::XOrIOp::create(rewriter, loc, anyViolation, constTrue);
    }
    smt::YieldOp::create(rewriter, loc, res);
    rewriter.replaceOp(op, solver.getResults());
    return success();
  }

  Namespace &names;
  bool risingClocksOnly;
  bool assumeKnownInputs;
  bool forSMTLIBExport;
  BMCCheckMode bmcMode;
  SmallVectorImpl<Operation *> &propertylessBMCOps;
  SmallVectorImpl<Operation *> &coverBMCOps;
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Verif to SMT pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertVerifToSMTPass
    : public circt::impl::ConvertVerifToSMTBase<ConvertVerifToSMTPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

static void populateLTLToSMTConversionPatterns(TypeConverter &converter,
                                               RewritePatternSet &patterns,
                                               bool approxTemporalOps) {
  patterns.add<LTLAndOpConversion, LTLOrOpConversion, LTLIntersectOpConversion,
               LTLNotOpConversion, LTLImplicationOpConversion,
               LTLBooleanConstantOpConversion, LTLClockOpConversion>(
      converter, patterns.getContext());
  patterns.add<LTLEventuallyOpConversion, LTLUntilOpConversion, LTLPastOpConversion,
               LTLConcatOpConversion, LTLRepeatOpConversion,
               LTLGoToRepeatOpConversion, LTLNonConsecutiveRepeatOpConversion>(
      converter, patterns.getContext(), approxTemporalOps);
  patterns.add<LTLDelayOpConversion>(converter, patterns.getContext(),
                                     approxTemporalOps);
}

static FailureOr<BMCCheckMode> parseBMCMode(StringRef mode) {
  if (mode.empty() || mode == "bounded")
    return BMCCheckMode::Bounded;
  if (mode == "liveness")
    return BMCCheckMode::Liveness;
  if (mode == "liveness-lasso")
    return BMCCheckMode::LivenessLasso;
  if (mode == "induction-base")
    return BMCCheckMode::InductionBase;
  if (mode == "induction-step")
    return BMCCheckMode::InductionStep;
  return failure();
}

void circt::populateVerifToSMTConversionPatterns(
    TypeConverter &converter, RewritePatternSet &patterns, Namespace &names,
    bool risingClocksOnly, bool assumeKnownInputs, bool xOptimisticOutputs,
    bool forSMTLIBExport, BMCCheckMode bmcMode, bool approxTemporalOps,
    SmallVectorImpl<Operation *> &propertylessBMCOps,
    SmallVectorImpl<Operation *> &coverBMCOps) {
  // Add LTL operation conversion patterns
  populateLTLToSMTConversionPatterns(converter, patterns, approxTemporalOps);

  // Add Verif operation conversion patterns
  patterns.add<VerifAssertOpConversion, VerifAssumeOpConversion,
               VerifCoverOpConversion>(converter, patterns.getContext());
  patterns.add<LogicEquivalenceCheckingOpConversion,
               RefinementCheckingOpConversion>(
      converter, patterns.getContext(), assumeKnownInputs, xOptimisticOutputs);
  patterns.add<VerifBoundedModelCheckingOpConversion>(
      converter, patterns.getContext(), names, risingClocksOnly,
      assumeKnownInputs,
      forSMTLIBExport, bmcMode,
      propertylessBMCOps, coverBMCOps);
}

void ConvertVerifToSMTPass::runOnOperation() {
  ConversionTarget verifTarget(getContext());
  verifTarget.addIllegalDialect<verif::VerifDialect>();
  verifTarget.addLegalDialect<smt::SMTDialect, arith::ArithDialect,
                              scf::SCFDialect, func::FuncDialect,
                              ltl::LTLDialect, comb::CombDialect,
                              hw::HWDialect, seq::SeqDialect>();
  verifTarget.addLegalOp<UnrealizedConversionCastOp>();

  // Check BMC ops contain only one assertion (done outside pattern to avoid
  // issues with whether assertions are/aren't lowered yet)
  SymbolTable symbolTable(getOperation());
  SmallVector<Operation *> propertylessBMCOps;
  SmallVector<Operation *> coverBMCOps;
  auto bmcModeOr = parseBMCMode(bmcMode);
  if (failed(bmcModeOr)) {
    getOperation().emitError()
        << "invalid bmc-mode: '" << bmcMode
        << "' (expected bounded, liveness, liveness-lasso, "
           "induction-base, or induction-step)";
    signalPassFailure();
    return;
  }
  WalkResult assertionCheck = getOperation().walk(
      [&](Operation *op) { // Check there is exactly one assertion and clock
        if (auto bmcOp = dyn_cast<verif::BoundedModelCheckingOp>(op)) {
          auto regTypes = TypeRange(bmcOp.getCircuit().getArgumentTypes())
                              .take_back(bmcOp.getNumRegs());
          for (auto [regType, initVal] :
               llvm::zip(regTypes, bmcOp.getInitialValues())) {
            if (!isa<UnitAttr>(initVal)) {
              int64_t regWidth = hw::getBitWidth(regType);
              if (regWidth <= 0) {
                op->emitError(
                    "initial values require registers with known bit widths");
                return WalkResult::interrupt();
              }
              if (auto initIntAttr = dyn_cast<IntegerAttr>(initVal)) {
                if (initIntAttr.getValue().getBitWidth() !=
                    static_cast<unsigned>(regWidth)) {
                  op->emitError(
                      "bit width of initial value does not match register");
                  return WalkResult::interrupt();
                }
                continue;
              }
              if (auto initBoolAttr = dyn_cast<BoolAttr>(initVal)) {
                if (regWidth != 1) {
                  op->emitError(
                      "bool initial value requires 1-bit register width");
                  return WalkResult::interrupt();
                }
                continue;
              }
              auto tyAttr = dyn_cast<TypedAttr>(initVal);
              if (!tyAttr || tyAttr.getType() != regType) {
                op->emitError("unsupported initial value for register");
                return WalkResult::interrupt();
              }
            }
          }
          // Check only one clock is present in the circuit inputs
          auto numClockArgs = 0;
          for (auto argType : bmcOp.getCircuit().getArgumentTypes())
            if (isa<seq::ClockType>(argType))
              numClockArgs++;
          // TODO: this can be removed once we have a way to associate reg
          // ins/outs with clocks
          if (numClockArgs > 1) {
            if (risingClocksOnly) {
              op->emitError("multi-clock BMC is not supported with "
                            "--rising-clocks-only");
              return WalkResult::interrupt();
            }
            unsigned numRegs = bmcOp.getNumRegs();
            auto regClocks =
                bmcOp->getAttrOfType<ArrayAttr>("bmc_reg_clocks");
            auto regClockSources =
                bmcOp->getAttrOfType<ArrayAttr>("bmc_reg_clock_sources");
            bool regClocksValid =
                regClocks && regClocks.size() == numRegs;
            bool regSourcesValid =
                regClockSources && regClockSources.size() == numRegs;
            if (numRegs > 0 && !regClocksValid && !regSourcesValid) {
              op->emitError("multi-clock BMC requires bmc_reg_clocks or "
                            "bmc_reg_clock_sources with one entry per "
                            "register");
              return WalkResult::interrupt();
            }
            if (numRegs > 0 && !regSourcesValid &&
                !bmcOp->getAttrOfType<ArrayAttr>("bmc_input_names")) {
              op->emitError(
                  "multi-clock BMC requires bmc_input_names for clock mapping");
              return WalkResult::interrupt();
            }
          }
          SmallVector<mlir::Operation *> worklist;
          int numAssertions = 0;
          int numCovers = 0;
          op->walk([&](Operation *curOp) {
            if (auto assertOp = dyn_cast<verif::AssertOp>(curOp)) {
              numAssertions++;
            }
            if (auto coverOp = dyn_cast<verif::CoverOp>(curOp)) {
              numCovers++;
            }
            if (auto inst = dyn_cast<InstanceOp>(curOp))
              worklist.push_back(symbolTable.lookup(inst.getModuleName()));
          });
          // TODO: probably negligible compared to actual model checking time
          // but cacheing the assertion count of modules would speed this up
          while (!worklist.empty()) {
            auto *module = worklist.pop_back_val();
            module->walk([&](Operation *curOp) {
              if (auto assertOp = dyn_cast<verif::AssertOp>(curOp)) {
                numAssertions++;
              }
              if (auto coverOp = dyn_cast<verif::CoverOp>(curOp)) {
                numCovers++;
              }
              if (auto inst = dyn_cast<InstanceOp>(curOp))
                worklist.push_back(symbolTable.lookup(inst.getModuleName()));
            });
            if (numAssertions > 1 || numCovers > 1)
              break;
          }
          if (numAssertions == 0 && numCovers == 0) {
            op->emitWarning("no property provided to check in module - will "
                            "trivially find no violations.");
            propertylessBMCOps.push_back(bmcOp);
          }
          if (numAssertions > 0 && numCovers > 0) {
            op->emitError(
                "bounded model checking problems with mixed assert/cover "
                "properties are not yet correctly handled - instead, check one "
                "kind at a time");
            return WalkResult::interrupt();
          }
          if (numCovers > 0)
            coverBMCOps.push_back(bmcOp);
        }
        return WalkResult::advance();
      });
  if (assertionCheck.wasInterrupted())
    return signalPassFailure();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  populateHWToSMTTypeConverter(converter);

  // Add LTL type conversions to SMT boolean
  // LTL sequences and properties are converted to SMT booleans for BMC
  converter.addConversion([](ltl::SequenceType type) -> Type {
    return smt::BoolType::get(type.getContext());
  });
  converter.addConversion([](ltl::PropertyType type) -> Type {
    return smt::BoolType::get(type.getContext());
  });

  // Keep assert/assume/cover legal inside BMC so the BMC conversion can handle
  // them, and illegal elsewhere so they get lowered normally.
  auto isInsideBMC = [](Operation *op) {
    return op->getParentOfType<verif::BoundedModelCheckingOp>() != nullptr;
  };
  verifTarget.addDynamicallyLegalOp<verif::AssertOp>(
      [&](verif::AssertOp op) { return isInsideBMC(op); });
  verifTarget.addDynamicallyLegalOp<verif::AssumeOp>(
      [&](verif::AssumeOp op) { return isInsideBMC(op); });
  verifTarget.addDynamicallyLegalOp<verif::CoverOp>(
      [&](verif::CoverOp op) { return isInsideBMC(op); });

  SymbolCache symCache;
  symCache.addDefinitions(getOperation());
  Namespace names;
  names.add(symCache);

  // First phase: lower Verif operations, leaving LTL ops untouched.
  patterns.add<VerifAssertOpConversion, VerifAssumeOpConversion,
               VerifCoverOpConversion>(converter, patterns.getContext());
  patterns.add<LogicEquivalenceCheckingOpConversion,
               RefinementCheckingOpConversion>(
      converter, patterns.getContext(), assumeKnownInputs,
      xOptimisticOutputs);
  patterns.add<VerifBoundedModelCheckingOpConversion>(
      converter, patterns.getContext(), names, (bool)risingClocksOnly,
      (bool)this->assumeKnownInputs,
      (bool)this->forSMTLIBExport, *bmcModeOr,
      propertylessBMCOps, coverBMCOps);

  if (failed(mlir::applyPartialConversion(getOperation(), verifTarget,
                                          std::move(patterns))))
    return signalPassFailure();

  // The first phase (notably BMC lowering) may rewrite sequence roots and leave
  // behind now-dead LTL ops (e.g. original sequence DAGs after NFA lowering).
  // Since the second phase marks many temporal ops illegal (and strict lowering
  // may refuse approximations), erase dead LTL ops before attempting to
  // legalize the remainder.
  {
    SmallVector<Operation *> ordered;
    getOperation()->walk([&](Operation *op) {
      if (isa_and_nonnull<ltl::LTLDialect>(op->getDialect()))
        ordered.push_back(op);
    });

    bool changed = true;
    while (changed) {
      changed = false;
      for (int64_t i = static_cast<int64_t>(ordered.size()) - 1; i >= 0; --i) {
        Operation *op = ordered[static_cast<size_t>(i)];
        if (!op)
          continue;
        if (!op->use_empty())
          continue;
        op->erase();
        ordered[static_cast<size_t>(i)] = nullptr;
        changed = true;
      }
    }
  }

  // Second phase: lower remaining LTL operations to SMT.
  ConversionTarget ltlTarget(getContext());
  ltlTarget.addLegalDialect<smt::SMTDialect, arith::ArithDialect,
                            scf::SCFDialect, func::FuncDialect,
                            comb::CombDialect, hw::HWDialect, seq::SeqDialect,
                            verif::VerifDialect>();
  ltlTarget.addLegalOp<UnrealizedConversionCastOp>();
  ltlTarget.addIllegalOp<ltl::AndOp, ltl::OrOp, ltl::IntersectOp, ltl::NotOp,
                         ltl::ImplicationOp, ltl::EventuallyOp, ltl::UntilOp,
                         ltl::BooleanConstantOp, ltl::DelayOp, ltl::ConcatOp,
                         ltl::RepeatOp, ltl::GoToRepeatOp,
                         ltl::NonConsecutiveRepeatOp, ltl::PastOp>();

  RewritePatternSet ltlPatterns(&getContext());
  populateLTLToSMTConversionPatterns(converter, ltlPatterns, approxTemporalOps);
  if (failed(mlir::applyPartialConversion(getOperation(), ltlTarget,
                                          std::move(ltlPatterns))))
    return signalPassFailure();

  RewritePatternSet relocatePatterns(&getContext());
  relocatePatterns.add<RelocateCastIntoInputRegion>(&getContext());
  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(relocatePatterns))))
    return signalPassFailure();

  RewritePatternSet postPatterns(&getContext());
  postPatterns.add<BoolBVCastOpRewrite>(&getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(postPatterns))))
    return signalPassFailure();

  RewritePatternSet regionFixups(&getContext());
  regionFixups.add<RelocateSMTEqIntoOperandRegion>(&getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(regionFixups))))
    return signalPassFailure();
}
