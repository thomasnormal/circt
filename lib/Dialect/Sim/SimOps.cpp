//===- SimOps.cpp - Implement the Sim operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements `sim` dialect ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/CustomDirectiveImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace circt;
using namespace sim;

static StringAttr formatIntegersByRadix(MLIRContext *ctx, unsigned radix,
                                        const Attribute &value,
                                        bool isUpperCase, bool isLeftAligned,
                                        char paddingChar,
                                        std::optional<unsigned> specifierWidth,
                                        bool isSigned = false) {

  if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(value)) {
    SmallVector<char, 32> strBuf;
    intAttr.getValue().toString(strBuf, radix, isSigned, false, isUpperCase);
    unsigned width = intAttr.getType().getIntOrFloatBitWidth();
    unsigned padWidth;
    switch (radix) {
    case 2:
      padWidth = width;
      break;
    case 8:
      padWidth = (width + 2) / 3;
      break;
    case 16:
      padWidth = (width + 3) / 4;
      break;
    default:
      padWidth = width;
      break;
    }

    unsigned numSpaces = 0;
    if (specifierWidth.has_value() &&
        (specifierWidth.value() >
         std::max(padWidth, static_cast<unsigned>(strBuf.size())))) {
      numSpaces = std::max(
          0U, specifierWidth.value() -
                  std::max(padWidth, static_cast<unsigned>(strBuf.size())));
    }

    SmallVector<char, 1> spacePadding(numSpaces, ' ');

    padWidth = padWidth > strBuf.size() ? padWidth - strBuf.size() : 0;

    SmallVector<char, 32> padding(padWidth, paddingChar);
    if (isLeftAligned) {
      return StringAttr::get(ctx, Twine(padding) + Twine(strBuf) +
                                      Twine(spacePadding));
    }
    return StringAttr::get(ctx, Twine(spacePadding) + Twine(padding) +
                                    Twine(strBuf));
  }
  return {};
}

static StringAttr formatFloatsBySpecifier(MLIRContext *ctx, Attribute value,
                                          bool isLeftAligned,
                                          std::optional<unsigned> fieldWidth,
                                          std::optional<unsigned> fracDigits,
                                          std::string formatSpecifier) {
  if (auto floatAttr = llvm::dyn_cast_or_null<FloatAttr>(value)) {
    std::string widthString = isLeftAligned ? "-" : "";
    if (fieldWidth.has_value()) {
      widthString += std::to_string(fieldWidth.value());
    }
    std::string fmtSpecifier = "%" + widthString + "." +
                               std::to_string(fracDigits.value()) +
                               formatSpecifier;

    // Calculates number of bytes needed to store the format string
    // excluding the null terminator
    int bufferSize = std::snprintf(nullptr, 0, fmtSpecifier.c_str(),
                                   floatAttr.getValue().convertToDouble());
    std::string floatFmtBuffer(bufferSize, '\0');
    snprintf(floatFmtBuffer.data(), bufferSize + 1, fmtSpecifier.c_str(),
             floatAttr.getValue().convertToDouble());
    return StringAttr::get(ctx, floatFmtBuffer);
  }
  return {};
}

ParseResult DPIFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto builder = parser.getBuilder();
  // Parse visibility.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<hw::module_like_impl::PortParse> ports;
  TypeAttr modType;
  if (failed(
          hw::module_like_impl::parseModuleSignature(parser, ports, modType)))
    return failure();

  result.addAttribute(DPIFuncOp::getModuleTypeAttrName(result.name), modType);

  // Convert the specified array of dictionary attrs (which may have null
  // entries) to an ArrayAttr of dictionaries.
  auto unknownLoc = builder.getUnknownLoc();
  SmallVector<Attribute> attrs, locs;
  auto nonEmptyLocsFn = [unknownLoc](Attribute attr) {
    return attr && cast<Location>(attr) != unknownLoc;
  };

  for (auto &port : ports) {
    attrs.push_back(port.attrs ? port.attrs : builder.getDictionaryAttr({}));
    locs.push_back(port.sourceLoc ? Location(*port.sourceLoc) : unknownLoc);
  }

  result.addAttribute(DPIFuncOp::getPerArgumentAttrsAttrName(result.name),
                      builder.getArrayAttr(attrs));
  result.addRegion();

  if (llvm::any_of(locs, nonEmptyLocsFn))
    result.addAttribute(DPIFuncOp::getArgumentLocsAttrName(result.name),
                        builder.getArrayAttr(locs));

  // Parse the attribute dict.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  return success();
}

LogicalResult
sim::DPICallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto referencedOp =
      symbolTable.lookupNearestSymbolFrom(*this, getCalleeAttr());
  if (!referencedOp)
    return emitError("cannot find function declaration '")
           << getCallee() << "'";
  if (isa<func::FuncOp, sim::DPIFuncOp>(referencedOp))
    return success();
  return emitError("callee must be 'sim.dpi.func' or 'func.func' but got '")
         << referencedOp->getName() << "'";
}

void DPIFuncOp::print(OpAsmPrinter &p) {
  DPIFuncOp op = *this;
  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p << ' ';

  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = op->getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';
  p.printSymbolName(funcName);
  hw::module_like_impl::printModuleSignatureNew(
      p, op->getRegion(0), op.getModuleType(),
      getPerArgumentAttrsAttr()
          ? ArrayRef<Attribute>(getPerArgumentAttrsAttr().getValue())
          : ArrayRef<Attribute>{},
      getArgumentLocs() ? SmallVector<Location>(
                              getArgumentLocs().value().getAsRange<Location>())
                        : ArrayRef<Location>{});

  mlir::function_interface_impl::printFunctionAttributes(
      p, op,
      {visibilityAttrName, getModuleTypeAttrName(),
       getPerArgumentAttrsAttrName(), getArgumentLocsAttrName()});
}

OpFoldResult FormatLiteralOp::fold(FoldAdaptor adaptor) {
  return getLiteralAttr();
}

OpFoldResult FormatDecOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType() == IntegerType::get(getContext(), 0U))
    return StringAttr::get(getContext(), "0");

  if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getValue())) {
    SmallVector<char, 16> strBuf;
    intAttr.getValue().toString(strBuf, 10U, adaptor.getIsSigned());
    unsigned padWidth;
    if (adaptor.getSpecifierWidth().has_value()) {
      padWidth = adaptor.getSpecifierWidth().value();
    } else {
      unsigned width = intAttr.getType().getIntOrFloatBitWidth();
      padWidth = FormatDecOp::getDecimalWidth(width, adaptor.getIsSigned());
    }

    padWidth = padWidth > strBuf.size() ? padWidth - strBuf.size() : 0;

    SmallVector<char, 10> padding(padWidth, adaptor.getPaddingChar());
    if (adaptor.getIsLeftAligned()) {
      return StringAttr::get(getContext(), Twine(strBuf) + Twine(padding));
    }
    return StringAttr::get(getContext(), Twine(padding) + Twine(strBuf));
  }
  return {};
}

OpFoldResult FormatHexOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType() == IntegerType::get(getContext(), 0U))
    return StringAttr::get(getContext(), "");

  return formatIntegersByRadix(
      getContext(), 16U, adaptor.getValue(), adaptor.getIsHexUppercase(),
      adaptor.getIsLeftAligned(), adaptor.getPaddingChar(),
      adaptor.getSpecifierWidth());
}

OpFoldResult FormatOctOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType() == IntegerType::get(getContext(), 0U))
    return StringAttr::get(getContext(), "");

  return formatIntegersByRadix(
      getContext(), 8U, adaptor.getValue(), false, adaptor.getIsLeftAligned(),
      adaptor.getPaddingChar(), adaptor.getSpecifierWidth());
}

OpFoldResult FormatBinOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType() == IntegerType::get(getContext(), 0U))
    return StringAttr::get(getContext(), "");

  return formatIntegersByRadix(
      getContext(), 2U, adaptor.getValue(), false, adaptor.getIsLeftAligned(),
      adaptor.getPaddingChar(), adaptor.getSpecifierWidth());
}

OpFoldResult FormatScientificOp::fold(FoldAdaptor adaptor) {
  return formatFloatsBySpecifier(
      getContext(), adaptor.getValue(), adaptor.getIsLeftAligned(),
      adaptor.getFieldWidth(), adaptor.getFracDigits(), "e");
}

OpFoldResult FormatFloatOp::fold(FoldAdaptor adaptor) {
  return formatFloatsBySpecifier(
      getContext(), adaptor.getValue(), adaptor.getIsLeftAligned(),
      adaptor.getFieldWidth(), adaptor.getFracDigits(), "f");
}

OpFoldResult FormatGeneralOp::fold(FoldAdaptor adaptor) {
  return formatFloatsBySpecifier(
      getContext(), adaptor.getValue(), adaptor.getIsLeftAligned(),
      adaptor.getFieldWidth(), adaptor.getFracDigits(), "g");
}

OpFoldResult FormatCharOp::fold(FoldAdaptor adaptor) {
  auto width = getValue().getType().getIntOrFloatBitWidth();
  if (width > 8)
    return {};
  if (width == 0)
    return StringAttr::get(getContext(), Twine(static_cast<char>(0)));

  if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(adaptor.getValue())) {
    auto intValue = intAttr.getValue().getZExtValue();
    return StringAttr::get(getContext(), Twine(static_cast<char>(intValue)));
  }

  return {};
}

static StringAttr concatLiterals(MLIRContext *ctxt, ArrayRef<StringRef> lits) {
  assert(!lits.empty() && "No literals to concatenate");
  if (lits.size() == 1)
    return StringAttr::get(ctxt, lits.front());
  SmallString<64> newLit;
  for (auto lit : lits)
    newLit += lit;
  return StringAttr::get(ctxt, newLit);
}

OpFoldResult FormatStringConcatOp::fold(FoldAdaptor adaptor) {
  if (getNumOperands() == 0)
    return StringAttr::get(getContext(), "");
  if (getNumOperands() == 1) {
    // Don't fold to our own result to avoid an infinte loop.
    if (getResult() == getOperand(0))
      return {};
    return getOperand(0);
  }

  // Fold if all operands are literals.
  SmallVector<StringRef> lits;
  for (auto attr : adaptor.getInputs()) {
    auto lit = dyn_cast_or_null<StringAttr>(attr);
    if (!lit)
      return {};
    lits.push_back(lit);
  }
  return concatLiterals(getContext(), lits);
}

LogicalResult FormatStringConcatOp::getFlattenedInputs(
    llvm::SmallVectorImpl<Value> &flatOperands) {
  llvm::SmallMapVector<FormatStringConcatOp, unsigned, 4> concatStack;
  bool isCyclic = false;

  // Perform a DFS on this operation's concatenated operands,
  // collect the leaf format string fragments.
  concatStack.insert({*this, 0});
  while (!concatStack.empty()) {
    auto &top = concatStack.back();
    auto currentConcat = top.first;
    unsigned operandIndex = top.second;

    // Iterate over concatenated operands
    while (operandIndex < currentConcat.getNumOperands()) {
      auto currentOperand = currentConcat.getOperand(operandIndex);

      if (auto nextConcat =
              currentOperand.getDefiningOp<FormatStringConcatOp>()) {
        // Concat of a concat
        if (!concatStack.contains(nextConcat)) {
          // Save the next operand index to visit on the
          // stack and put the new concat on top.
          top.second = operandIndex + 1;
          concatStack.insert({nextConcat, 0});
          break;
        }
        // Cyclic concatenation encountered. Don't recurse.
        isCyclic = true;
      }

      flatOperands.push_back(currentOperand);
      operandIndex++;
    }

    // Pop the concat off of the stack if we have visited all operands.
    if (operandIndex >= currentConcat.getNumOperands())
      concatStack.pop_back();
  }

  return success(!isCyclic);
}

LogicalResult FormatStringConcatOp::verify() {
  if (llvm::any_of(getOperands(),
                   [&](Value operand) { return operand == getResult(); }))
    return emitOpError("is infinitely recursive.");
  return success();
}

LogicalResult FormatStringConcatOp::canonicalize(FormatStringConcatOp op,
                                                 PatternRewriter &rewriter) {

  auto fmtStrType = FormatStringType::get(op.getContext());

  // Check if we can flatten concats of concats
  bool hasBeenFlattened = false;
  SmallVector<Value, 0> flatOperands;
  if (!op.isFlat()) {
    // Get a new, flattened list of operands
    flatOperands.reserve(op.getNumOperands() + 4);
    auto isAcyclic = op.getFlattenedInputs(flatOperands);

    if (failed(isAcyclic)) {
      // Infinite recursion, but we cannot fail compilation right here (can we?)
      // so just emit a warning and bail out.
      op.emitWarning("Cyclic concatenation detected.");
      return failure();
    }

    hasBeenFlattened = true;
  }

  if (!hasBeenFlattened && op.getNumOperands() < 2)
    return failure(); // Should be handled by the folder

  // Check if there are adjacent literals we can merge or empty literals to
  // remove
  SmallVector<StringRef> litSequence;
  SmallVector<Value> newOperands;
  newOperands.reserve(op.getNumOperands());
  FormatLiteralOp prevLitOp;

  auto oldOperands = hasBeenFlattened ? flatOperands : op.getOperands();
  for (auto operand : oldOperands) {
    if (auto litOp = operand.getDefiningOp<FormatLiteralOp>()) {
      if (!litOp.getLiteral().empty()) {
        prevLitOp = litOp;
        litSequence.push_back(litOp.getLiteral());
      }
    } else {
      if (!litSequence.empty()) {
        if (litSequence.size() > 1) {
          // Create a fused literal.
          auto newLit = rewriter.createOrFold<FormatLiteralOp>(
              op.getLoc(), fmtStrType,
              concatLiterals(op.getContext(), litSequence));
          newOperands.push_back(newLit);
        } else {
          // Reuse the existing literal.
          newOperands.push_back(prevLitOp.getResult());
        }
        litSequence.clear();
      }
      newOperands.push_back(operand);
    }
  }

  // Push trailing literals into the new operand list
  if (!litSequence.empty()) {
    if (litSequence.size() > 1) {
      // Create a fused literal.
      auto newLit = rewriter.createOrFold<FormatLiteralOp>(
          op.getLoc(), fmtStrType,
          concatLiterals(op.getContext(), litSequence));
      newOperands.push_back(newLit);
    } else {
      // Reuse the existing literal.
      newOperands.push_back(prevLitOp.getResult());
    }
  }

  if (!hasBeenFlattened && newOperands.size() == op.getNumOperands())
    return failure(); // Nothing changed

  if (newOperands.empty())
    rewriter.replaceOpWithNewOp<FormatLiteralOp>(op, fmtStrType,
                                                 rewriter.getStringAttr(""));
  else if (newOperands.size() == 1)
    rewriter.replaceOp(op, newOperands);
  else
    rewriter.modifyOpInPlace(op, [&]() { op->setOperands(newOperands); });

  return success();
}

LogicalResult PrintFormattedOp::canonicalize(PrintFormattedOp op,
                                             PatternRewriter &rewriter) {
  // Remove ops with constant false condition.
  if (auto cstCond = op.getCondition().getDefiningOp<hw::ConstantOp>()) {
    if (cstCond.getValue().isZero()) {
      rewriter.eraseOp(op);
      return success();
    }
  }
  return failure();
}

LogicalResult PrintFormattedProcOp::verify() {
  // Check if we know for sure that the parent is not procedural.
  auto *parentOp = getOperation()->getParentOp();

  if (!parentOp)
    return emitOpError("must be within a procedural region.");

  if (isa_and_nonnull<hw::HWDialect>(parentOp->getDialect())) {
    if (!isa<hw::TriggeredOp>(parentOp))
      return emitOpError("must be within a procedural region.");
    return success();
  }

  if (isa_and_nonnull<sv::SVDialect>(parentOp->getDialect())) {
    if (!parentOp->hasTrait<sv::ProceduralRegion>())
      return emitOpError("must be within a procedural region.");
    return success();
  }

  // Don't fail for dialects that are not explicitly handled.
  return success();
}

LogicalResult PrintFormattedProcOp::canonicalize(PrintFormattedProcOp op,
                                                 PatternRewriter &rewriter) {
  // Remove empty prints.
  if (auto litInput = op.getInput().getDefiningOp<FormatLiteralOp>()) {
    if (litInput.getLiteral().empty()) {
      rewriter.eraseOp(op);
      return success();
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// Event-Driven Simulation Operations (SimIR)
//===----------------------------------------------------------------------===//

LogicalResult SimSuspendOp::verify() {
  // Verify that we have either observed values or a delay (or both)
  if (getObserved().empty() && !getDelayFemtoseconds().has_value()) {
    // Allow region-only suspends for scheduling region changes
    if (!getRegion().has_value())
      return emitOpError(
          "must specify either observed values, a delay, or a region");
  }

  // Verify region attribute if present
  if (auto region = getRegion()) {
    StringRef regionStr = region.value();
    if (regionStr != "active" && regionStr != "inactive" && regionStr != "nba" &&
        regionStr != "reactive" && regionStr != "preponed" &&
        regionStr != "observed" && regionStr != "postponed") {
      return emitOpError("invalid scheduling region '")
             << regionStr
             << "', expected one of: active, inactive, nba, reactive, "
                "preponed, observed, postponed";
    }
  }

  return success();
}

LogicalResult SimYieldOp::verify() {
  // Verify that the parent is a SimCombProcessOp
  auto parent = (*this)->getParentOp();
  if (!isa<SimCombProcessOp>(parent)) {
    return emitOpError("must be directly nested within a 'sim.comb_process'");
  }

  // Verify result types match the parent process
  auto combProcess = cast<SimCombProcessOp>(parent);
  if (getNumOperands() != combProcess.getNumResults()) {
    return emitOpError("has ")
           << getNumOperands() << " operands but parent process expects "
           << combProcess.getNumResults() << " results";
  }

  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip(getOperandTypes(), combProcess.getResultTypes()))) {
    auto [yieldType, processType] = pair;
    if (yieldType != processType) {
      return emitOpError("operand type mismatch at index ")
             << idx << ": expected " << processType << " but got " << yieldType;
    }
  }

  return success();
}

LogicalResult SimDriveOp::verify() {
  // Verify mode attribute
  StringRef modeStr = getMode();
  if (modeStr != "blocking" && modeStr != "nonblocking" &&
      modeStr != "continuous") {
    return emitOpError("invalid drive mode '")
           << modeStr
           << "', expected one of: blocking, nonblocking, continuous";
  }

  // Verify strength attribute if present
  if (auto strength = getStrength()) {
    StringRef strengthStr = strength.value();
    if (strengthStr != "strong" && strengthStr != "pull" &&
        strengthStr != "weak" && strengthStr != "highz") {
      return emitOpError("invalid signal strength '")
             << strengthStr << "', expected one of: strong, pull, weak, highz";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Sensitivity List and Edge Detection Operations
//===----------------------------------------------------------------------===//

/// Helper function to verify edge type string.
static LogicalResult verifyEdgeType(Operation *op, StringRef edge) {
  if (edge != "posedge" && edge != "negedge" && edge != "anyedge" &&
      edge != "level") {
    return op->emitOpError("invalid edge type '")
           << edge << "', expected one of: posedge, negedge, anyedge, level";
  }
  return success();
}

LogicalResult SimSensitivityListOp::verify() {
  // Verify that the number of edges matches the number of signals
  auto edges = getEdges();
  auto signals = getSignals();

  if (edges.size() != signals.size()) {
    return emitOpError("number of edge types (")
           << edges.size() << ") must match number of signals ("
           << signals.size() << ")";
  }

  // Verify each edge type
  for (auto edgeAttr : edges) {
    auto edgeStr = dyn_cast<StringAttr>(edgeAttr);
    if (!edgeStr)
      return emitOpError("edge type must be a string attribute");
    if (failed(verifyEdgeType(*this, edgeStr.getValue())))
      return failure();
  }

  return success();
}

void SimSensitivityListOp::print(OpAsmPrinter &p) {
  auto edges = getEdges();
  auto signals = getSignals();

  p << " ";
  llvm::interleaveComma(llvm::zip(edges, signals), p, [&](auto pair) {
    auto [edgeAttr, signal] = pair;
    auto edge = cast<StringAttr>(edgeAttr).getValue();
    p << edge << " " << signal;
  });

  p << " : ";
  llvm::interleaveComma(signals.getTypes(), p);
  p.printOptionalAttrDict((*this)->getAttrs(), {"edges"});
}

ParseResult SimSensitivityListOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> signals;
  SmallVector<Type, 4> signalTypes;
  SmallVector<Attribute, 4> edges;

  // Parse edge-signal pairs
  do {
    StringRef edge;
    OpAsmParser::UnresolvedOperand signal;

    if (parser.parseKeyword(&edge) || parser.parseOperand(signal))
      return failure();

    edges.push_back(StringAttr::get(parser.getContext(), edge));
    signals.push_back(signal);
  } while (succeeded(parser.parseOptionalComma()));

  // Parse types
  if (parser.parseColon())
    return failure();

  if (parser.parseTypeList(signalTypes))
    return failure();

  if (signals.size() != signalTypes.size())
    return parser.emitError(parser.getCurrentLocation(),
                            "number of types doesn't match number of signals");

  // Resolve operands
  if (parser.resolveOperands(signals, signalTypes, parser.getCurrentLocation(),
                             result.operands))
    return failure();

  // Add edges attribute
  result.addAttribute("edges",
                      ArrayAttr::get(parser.getContext(), edges));

  // Add result type
  result.addTypes(IntegerType::get(parser.getContext(), 1));

  // Parse optional attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

LogicalResult SimEdgeDetectOp::verify() {
  // Verify edge type
  if (failed(verifyEdgeType(*this, getEdge())))
    return failure();

  // Verify that current and previous have the same type
  if (getCurrent().getType() != getPrevious().getType()) {
    return emitOpError("current and previous must have the same type, got ")
           << getCurrent().getType() << " and " << getPrevious().getType();
  }

  return success();
}

OpFoldResult SimEdgeDetectOp::fold(FoldAdaptor adaptor) {
  auto current = adaptor.getCurrent();
  auto previous = adaptor.getPrevious();

  // Can only fold if both are constants
  auto currInt = dyn_cast_or_null<IntegerAttr>(current);
  auto prevInt = dyn_cast_or_null<IntegerAttr>(previous);
  if (!currInt || !prevInt)
    return {};

  StringRef edge = getEdge();
  bool currBit = currInt.getValue().getLoBits(1) != 0;
  bool prevBit = prevInt.getValue().getLoBits(1) != 0;

  bool detected = false;
  if (edge == "posedge") {
    detected = !prevBit && currBit;
  } else if (edge == "negedge") {
    detected = prevBit && !currBit;
  } else if (edge == "anyedge" || edge == "level") {
    detected = currInt.getValue() != prevInt.getValue();
  }

  return IntegerAttr::get(IntegerType::get(getContext(), 1), detected ? 1 : 0);
}

LogicalResult SimTriggeredProcessOp::verify() {
  // Verify that the number of edges matches the number of signals
  auto edges = getSensitivityEdges();
  auto signals = getSensitivitySignals();

  if (edges.size() != signals.size()) {
    return emitOpError("number of edge types (")
           << edges.size() << ") must match number of sensitivity signals ("
           << signals.size() << ")";
  }

  // Verify each edge type
  for (auto edgeAttr : edges) {
    auto edgeStr = dyn_cast<StringAttr>(edgeAttr);
    if (!edgeStr)
      return emitOpError("edge type must be a string attribute");
    if (failed(verifyEdgeType(*this, edgeStr.getValue())))
      return failure();
  }

  return success();
}

void SimTriggeredProcessOp::print(OpAsmPrinter &p) {
  auto edges = getSensitivityEdges();
  auto signals = getSensitivitySignals();

  p << " @(";
  llvm::interleaveComma(llvm::zip(edges, signals), p, [&](auto pair) {
    auto [edgeAttr, signal] = pair;
    auto edge = cast<StringAttr>(edgeAttr).getValue();
    p << edge << " " << signal;
  });
  p << ")";

  if (!getResults().empty()) {
    p << " -> ";
    llvm::interleaveComma(getResultTypes(), p);
  }

  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), {"sensitivityEdges"});
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

ParseResult SimTriggeredProcessOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> signals;
  SmallVector<Type, 4> signalTypes;
  SmallVector<Attribute, 4> edges;

  // Parse @(...) sensitivity list
  if (parser.parseLParen())
    return failure();

  // Check for empty sensitivity list
  if (succeeded(parser.parseOptionalRParen())) {
    // Empty sensitivity list
  } else {
    // Parse edge-signal pairs
    do {
      StringRef edge;
      OpAsmParser::UnresolvedOperand signal;

      if (parser.parseKeyword(&edge) || parser.parseOperand(signal))
        return failure();

      edges.push_back(StringAttr::get(parser.getContext(), edge));
      signals.push_back(signal);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRParen())
      return failure();
  }

  // Parse optional result types
  SmallVector<Type, 4> resultTypes;
  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseTypeList(resultTypes))
      return failure();
  }

  // Parse optional attributes
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse the body region
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  // Infer signal types from the region's operands if not specified
  // For now, we'll require explicit parsing or use i1 for all
  for (size_t i = 0; i < signals.size(); ++i) {
    signalTypes.push_back(IntegerType::get(parser.getContext(), 1));
  }

  // Resolve operands
  if (parser.resolveOperands(signals, signalTypes, parser.getCurrentLocation(),
                             result.operands))
    return failure();

  // Add edges attribute
  result.addAttribute("sensitivityEdges",
                      ArrayAttr::get(parser.getContext(), edges));

  // Add result types
  result.addTypes(resultTypes);

  return success();
}

//===----------------------------------------------------------------------===//
// Process Control: Fork/Join Operations
//===----------------------------------------------------------------------===//

LogicalResult SimForkOp::verify() {
  // Verify join_type attribute
  StringRef joinTypeStr = getJoinType();
  if (joinTypeStr != "join" && joinTypeStr != "join_any" &&
      joinTypeStr != "join_none") {
    return emitOpError("invalid join type '")
           << joinTypeStr << "', expected one of: join, join_any, join_none";
  }

  // Verify we have at least one branch
  if (getBranches().empty()) {
    return emitOpError("must have at least one branch");
  }

  return success();
}

void SimForkOp::print(OpAsmPrinter &p) {
  p << " ";
  if (getJoinType() != "join") {
    p << "join_type \"" << getJoinType() << "\" ";
  }
  if (auto name = getName()) {
    p << "name \"" << name.value() << "\" ";
  }

  // Print branches
  bool first = true;
  for (auto &region : getBranches()) {
    if (!first)
      p << ", ";
    p.printRegion(region, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
    first = false;
  }

  p.printOptionalAttrDict((*this)->getAttrs(), {"joinType", "name"});
}

ParseResult SimForkOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  // Parse optional join_type
  StringRef joinType = "join";
  if (succeeded(parser.parseOptionalKeyword("join_type"))) {
    StringAttr joinTypeAttr;
    if (parser.parseAttribute(joinTypeAttr))
      return failure();
    joinType = joinTypeAttr.getValue();
  }
  result.addAttribute("joinType", builder.getStringAttr(joinType));

  // Parse optional name
  if (succeeded(parser.parseOptionalKeyword("name"))) {
    StringAttr nameAttr;
    if (parser.parseAttribute(nameAttr))
      return failure();
    result.addAttribute("name", nameAttr);
  }

  // Parse branches (regions separated by commas)
  do {
    Region *branch = result.addRegion();
    if (parser.parseRegion(*branch, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
    // Ensure the region has a terminator
    if (branch->empty())
      branch->emplaceBlock();
  } while (succeeded(parser.parseOptionalComma()));

  // Parse optional attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Add result type (i64 handle)
  result.addTypes(builder.getI64Type());

  return success();
}

//===----------------------------------------------------------------------===//
// Process Control: Wait Operations
//===----------------------------------------------------------------------===//

LogicalResult SimWaitOp::verify() {
  // If we have a timeout, we should have a timedOut result
  if (getTimeoutFemtoseconds().has_value() && !getTimedOut()) {
    return emitOpError(
        "wait with timeout must have a timedOut result to capture timeout status");
  }

  // If we have a timedOut result, we should have a timeout
  if (getTimedOut() && !getTimeoutFemtoseconds().has_value()) {
    return emitOpError(
        "wait without timeout should not have a timedOut result");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Sim/Sim.cpp.inc"
