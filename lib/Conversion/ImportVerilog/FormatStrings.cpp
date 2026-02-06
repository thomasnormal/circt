//===- FormatStrings.cpp - Verilog format string conversion ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/SFormat.h"
#include "slang/ast/Symbol.h"

using namespace mlir;
using namespace circt;
using namespace ImportVerilog;
using moore::IntAlign;
using moore::IntFormat;
using moore::IntPadding;
using moore::RealFormat;
using slang::ast::SFormat::FormatOptions;

/// Build the hierarchical path string for a given scope, using "." as separator.
/// This produces paths like "top.sub.block" for use with the %m format specifier.
static void buildHierarchicalPath(const slang::ast::Scope *scope,
                                  SmallVectorImpl<char> &path) {
  if (!scope)
    return;

  // Get the symbol for this scope
  const auto &sym = scope->asSymbol();

  // Stop at root/compilation unit
  if (sym.kind == slang::ast::SymbolKind::Root ||
      sym.kind == slang::ast::SymbolKind::CompilationUnit)
    return;

  // Recursively build parent path first
  buildHierarchicalPath(sym.getParentScope(), path);

  // Add separator if we already have content
  if (!path.empty())
    path.push_back('.');

  // Add this scope's name
  StringRef name = sym.name;
  if (!name.empty()) {
    path.append(name.begin(), name.end());
  } else {
    // For anonymous scopes, use a placeholder or the symbol kind
    path.append("$unit", "$unit" + 5);
  }
}

namespace {
struct FormatStringParser {
  Context &context;
  OpBuilder &builder;
  /// The remaining arguments to be parsed.
  ArrayRef<const slang::ast::Expression *> arguments;
  /// The current location to use for ops and diagnostics.
  Location loc;
  /// The default format for integer arguments not covered by a format string
  /// literal.
  IntFormat defaultFormat;
  /// The interpolated string fragments that will be concatenated using a
  /// `moore.fmt.concat` op.
  SmallVector<Value> fragments;
  /// The scope where the format string is being evaluated (for %m).
  const slang::ast::Scope *scope;

  FormatStringParser(Context &context,
                     ArrayRef<const slang::ast::Expression *> arguments,
                     Location loc, IntFormat defaultFormat,
                     const slang::ast::Scope *scope = nullptr)
      : context(context), builder(context.builder), arguments(arguments),
        loc(loc), defaultFormat(defaultFormat), scope(scope) {
    // If no scope was explicitly provided, use the context's current scope.
    // This is set when entering a module body and is used for %m.
    if (!this->scope)
      this->scope = context.currentScope;
  }

  /// Entry point to the format string parser.
  FailureOr<Value> parse(bool appendNewline) {
    while (!arguments.empty()) {
      const auto &arg = *arguments[0];
      arguments = arguments.drop_front();
      if (arg.kind == slang::ast::ExpressionKind::EmptyArgument)
        continue;
      loc = context.convertLocation(arg.sourceRange);
      if (auto *lit = arg.as_if<slang::ast::StringLiteral>()) {
        if (failed(parseFormat(lit->getValue())))
          return failure();
      } else {
        if (failed(emitDefault(arg)))
          return failure();
      }
    }

    // Append the optional newline.
    if (appendNewline)
      emitLiteral("\n");

    // Concatenate all string fragments into one formatted string, or return an
    // empty literal if no fragments were generated.
    if (fragments.empty())
      return Value{};
    if (fragments.size() == 1)
      return fragments[0];
    return moore::FormatConcatOp::create(builder, loc, fragments).getResult();
  }

  /// Parse a format string literal and consume and format the arguments
  /// corresponding to the format specifiers it contains.
  LogicalResult parseFormat(StringRef format) {
    bool anyFailure = false;
    auto onText = [&](auto text) {
      if (anyFailure)
        return;
      emitLiteral(text);
    };
    auto onArg = [&](auto specifier, auto offset, auto len,
                     const auto &options) {
      if (anyFailure)
        return;
      if (failed(emitArgument(specifier, format.substr(offset, len), options)))
        anyFailure = true;
    };
    auto onError = [&](auto, auto, auto, auto) {
      assert(false && "Slang should have already reported all errors");
    };
    slang::ast::SFormat::parse(format, onText, onArg, onError);
    return failure(anyFailure);
  }

  /// Emit a string literal that requires no additional formatting.
  void emitLiteral(StringRef literal) {
    fragments.push_back(moore::FormatLiteralOp::create(builder, loc, literal));
  }

  /// Consume the next argument from the list and emit it according to the given
  /// format specifier.
  LogicalResult emitArgument(char specifier, StringRef fullSpecifier,
                             const FormatOptions &options) {
    auto specifierLower = std::tolower(specifier);

    // Special handling for %m - hierarchical name (consumes no argument).
    // See IEEE 1800-2017 Section 21.2.1.3 "Hierarchical name format".
    if (specifierLower == 'm') {
      return emitHierarchicalName(options);
    }

    // Special handling for %l - library binding (unsupported).
    if (specifierLower == 'l')
      return mlir::emitError(loc)
             << "unsupported format specifier `" << fullSpecifier << "`";

    // Consume the next argument, which will provide the value to be
    // formatted.
    assert(!arguments.empty() && "Slang guarantees correct arg count");
    const auto &arg = *arguments[0];
    arguments = arguments.drop_front();

    // Handle the different formatting options.
    // See IEEE 1800-2017 § 21.2.1.2 "Format specifications".
    switch (specifierLower) {
    case 'b':
      return emitInteger(arg, options, IntFormat::Binary);
    case 'o':
      return emitInteger(arg, options, IntFormat::Octal);
    case 'd':
      return emitInteger(arg, options, IntFormat::Decimal);
    case 'h':
    case 'x':
      return emitInteger(arg, options,
                         std::isupper(specifier) ? IntFormat::HexUpper
                                                 : IntFormat::HexLower);

    case 'e':
      return emitReal(arg, options, RealFormat::Exponential);
    case 'g':
      return emitReal(arg, options, RealFormat::General);
    case 'f':
      return emitReal(arg, options, RealFormat::Float);

    case 't':
      return emitTime(arg, options);

    case 'c':
      return emitChar(arg, options);

    case 's':
      return emitString(arg, options);

    case 'p':
      return emitPointer(arg, options);

    default:
      return mlir::emitError(loc)
             << "unsupported format specifier `" << fullSpecifier << "`";
    }
  }

  /// Emit an integer value with the given format.
  LogicalResult emitInteger(const slang::ast::Expression &arg,
                            const FormatOptions &options, IntFormat format) {

    Type intTy = {};
    Value val;
    auto rVal = context.convertRvalueExpression(arg);
    // To infer whether or not the value is signed while printing as a decimal
    // Since it only matters if it's a decimal, we add `format ==
    // IntFormat::Decimal`
    bool isSigned = arg.type->isSigned() && format == IntFormat::Decimal;
    if (!rVal)
      return failure();

    // An IEEE 754 float number is represented using a sign bit s, n mantissa,
    // and m exponent bits, representing (-1)**s * 1.fraction * 2**(E-bias).
    // This means that the largest finite value is (2-2**(-n) * 2**(2**m-1)),
    // just slightly less than ((2**(2**(m)))-1).
    // Since we need signed value representation, we need integers that can
    // represent values between [-(2**(2**(m))) ... (2**(2**(m)))-1], which
    // requires an m+1 bit signed integer.
    if (auto realTy = dyn_cast<moore::RealType>(rVal.getType())) {
      if (realTy.getWidth() == moore::RealWidth::f32) {
        // A 32 Bit IEEE 754 float number needs at most 129 integer bits
        // (signed).
        intTy = moore::IntType::getInt(context.getContext(), 129);
      } else if (realTy.getWidth() == moore::RealWidth::f64) {
        // A 64 Bit IEEE 754 float number needs at most 1025 integer bits
        // (signed).
        intTy = moore::IntType::getInt(context.getContext(), 1025);
      } else
        return failure();

      val = moore::RealToIntOp::create(builder, loc, intTy, rVal);
    } else {
      val = rVal;
    }

    auto value = context.convertToSimpleBitVector(val);
    if (!value)
      return failure();

    // Determine the alignment and padding.
    auto alignment = options.leftJustify ? IntAlign::Left : IntAlign::Right;
    auto padding =
        format == IntFormat::Decimal ? IntPadding::Space : IntPadding::Zero;
    IntegerAttr widthAttr = nullptr;
    if (options.width) {
      widthAttr = builder.getI32IntegerAttr(*options.width);
    }

    fragments.push_back(moore::FormatIntOp::create(
        builder, loc, value, format, alignment, padding, widthAttr, isSigned));
    return success();
  }

  /// Emit a character value with the %c format specifier.
  /// See IEEE 1800-2017 § 21.2.1.2 "Format specifications".
  LogicalResult emitChar(const slang::ast::Expression &arg,
                         const FormatOptions &options) {
    auto rVal = context.convertRvalueExpression(arg);
    if (!rVal)
      return failure();

    auto value = context.convertToSimpleBitVector(rVal);
    if (!value)
      return failure();

    fragments.push_back(moore::FormatCharOp::create(builder, loc, value));
    return success();
  }

  LogicalResult emitReal(const slang::ast::Expression &arg,
                         const FormatOptions &options, RealFormat format) {

    // Ensures that the given value is moore.real
    // i.e. $display("%f", 4) -> 4.000000, but 4 is not necessarily of real type
    auto value = context.convertRvalueExpression(
        arg, moore::RealType::get(context.getContext(), moore::RealWidth::f64));

    IntegerAttr widthAttr = nullptr;
    if (options.width) {
      widthAttr = builder.getI32IntegerAttr(*options.width);
    }

    IntegerAttr precisionAttr = nullptr;
    if (options.precision) {
      if (*options.precision)
        precisionAttr = builder.getI32IntegerAttr(*options.precision);
      else
        // If precision is 0, we set it to 1 instead
        precisionAttr = builder.getI32IntegerAttr(1);
    }

    auto alignment = options.leftJustify ? IntAlign::Left : IntAlign::Right;
    if (!value)
      return failure();

    fragments.push_back(moore::FormatRealOp::create(
        builder, loc, value, format, alignment, widthAttr, precisionAttr));

    return success();
  }

  // Format an integer with the %t specifier according to IEEE 1800-2023
  // § 20.4.3 "$timeformat". We currently don't support user-defined time
  // formats. Instead, we just convert the time to an integer and print it. This
  // applies the local timeunit/timescale and seem to be inline with what
  // Verilator does.
  LogicalResult emitTime(const slang::ast::Expression &arg,
                         const FormatOptions &options) {
    // Handle the time argument and convert it to a 64 bit integer.
    auto value = context.convertRvalueExpression(
        arg, moore::IntType::getInt(context.getContext(), 64));
    if (!value)
      return failure();

    // Create an integer formatting fragment.
    uint32_t width = 20; // default $timeformat field width
    if (options.width)
      width = *options.width;
    auto alignment = options.leftJustify ? IntAlign::Left : IntAlign::Right;
    auto padding = options.zeroPad ? IntPadding::Zero : IntPadding::Space;
    fragments.push_back(moore::FormatIntOp::create(
        builder, loc, value, IntFormat::Decimal, alignment, padding,
        builder.getI32IntegerAttr(width)));
    return success();
  }

  LogicalResult emitString(const slang::ast::Expression &arg,
                           const FormatOptions &options) {
    // Determine alignment and padding for string formatting with width
    auto alignment = options.leftJustify ? IntAlign::Left : IntAlign::Right;
    auto padding = IntPadding::Space;

    // Simplified handling for literals without width specifier.
    if (!options.width) {
      if (auto *lit = arg.as_if<slang::ast::StringLiteral>()) {
        emitLiteral(lit->getValue());
        return success();
      }
    }

    // Handle expressions - convert to string type
    auto value = context.convertRvalueExpression(
        arg, builder.getType<moore::StringType>());
    if (!value) {
      // Try converting without target type to see if it's a class handle
      value = context.convertRvalueExpression(arg);
      if (value && isa<moore::ClassHandleType>(value.getType())) {
        fragments.push_back(moore::FormatClassOp::create(builder, loc, value));
        return success();
      }
      return mlir::emitError(context.convertLocation(arg.sourceRange))
             << "expression cannot be formatted as string";
    }

    // Create FormatStringOp with optional width
    IntegerAttr widthAttr = options.width
                                ? builder.getI32IntegerAttr(*options.width)
                                : IntegerAttr();
    moore::IntAlignAttr alignAttr =
        options.width ? moore::IntAlignAttr::get(context.getContext(), alignment)
                      : moore::IntAlignAttr();
    moore::IntPaddingAttr padAttr =
        options.width
            ? moore::IntPaddingAttr::get(context.getContext(), padding)
            : moore::IntPaddingAttr();

    fragments.push_back(moore::FormatStringOp::create(builder, loc, value,
                                                      widthAttr, alignAttr,
                                                      padAttr));
    return success();
  }

  /// Emit a pointer/handle value with the %p format specifier.
  /// Per IEEE 1800-2017 section 21.2.1.2, %p prints class handles in a
  /// pointer format. Since actual pointer values are simulator-specific,
  /// we emit a placeholder string representation.
  LogicalResult emitPointer(const slang::ast::Expression &arg,
                            const FormatOptions &options) {
    // Consume the argument but emit a placeholder since actual pointer
    // values are runtime/simulator-specific. This allows code using %p
    // to be parsed and lowered without error.
    auto value = context.convertRvalueExpression(arg);
    if (!value)
      return failure();

    // For class handles, use FormatClassOp to represent the formatting.
    if (isa<moore::ClassHandleType>(value.getType())) {
      fragments.push_back(moore::FormatClassOp::create(builder, loc, value));
      return success();
    }

    // For other pointer types, emit a placeholder literal representing the
    // pointer format. The actual pointer value would be determined at
    // simulation runtime.
    emitLiteral("<ptr>");
    return success();
  }

  /// Emit the hierarchical name with the %m format specifier.
  /// Per IEEE 1800-2017 section 21.2.1.3, %m displays the hierarchical name
  /// of the scope where the format string is being evaluated. This does not
  /// consume an argument.
  LogicalResult emitHierarchicalName(const FormatOptions &options) {
    SmallString<64> path;
    buildHierarchicalPath(scope, path);

    // If we couldn't determine the path, emit a placeholder
    if (path.empty())
      path = "<unknown>";

    emitLiteral(path);
    return success();
  }

  /// Emit an expression argument with the appropriate default formatting.
  LogicalResult emitDefault(const slang::ast::Expression &expr) {
    // String types should be displayed as strings by default per IEEE 1800-2017.
    if (expr.type->isString()) {
      FormatOptions options;
      return emitString(expr, options);
    }

    // First try converting to get the value's type
    auto value = context.convertRvalueExpression(expr);
    if (!value)
      return failure();

    // Class handles get formatted using FormatClassOp (produces placeholder)
    if (isa<moore::ClassHandleType>(value.getType())) {
      fragments.push_back(moore::FormatClassOp::create(builder, loc, value));
      return success();
    }

    // For other types, use default integer formatting
    FormatOptions options;
    return emitInteger(expr, options, defaultFormat);
  }
};
} // namespace

FailureOr<Value> Context::convertFormatString(
    std::span<const slang::ast::Expression *const> arguments, Location loc,
    IntFormat defaultFormat, bool appendNewline,
    const slang::ast::Scope *scope) {
  FormatStringParser parser(*this, ArrayRef(arguments.data(), arguments.size()),
                            loc, defaultFormat, scope);
  return parser.parse(appendNewline);
}
