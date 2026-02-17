//===- ConstructSeqLEC.cpp - Sequential LEC via BMC miter -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builds a verif.bmc-based miter for sequential trace equivalence checking.
// Given two hw.module ops processed by ExternalizeRegisters, this pass creates
// a verif.bmc op that shares primary inputs and clocks between the two circuits,
// keeps separate register state for each, and asserts output equivalence at
// every unrolling step.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Tools/circt-lec/Passes.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace circt {
#define GEN_PASS_DEF_CONSTRUCTSEQLEC
#include "circt/Tools/circt-lec/Passes.h.inc"
} // namespace circt

namespace {
struct ConstructSeqLECPass
    : public circt::impl::ConstructSeqLECBase<ConstructSeqLECPass> {
  using ConstructSeqLECBase::ConstructSeqLECBase;
  void runOnOperation() override;
};
} // namespace

/// Classify module ports into clocks, primary inputs, and register states.
/// ExternalizeRegisters appends register state ports at the end of the
/// module's inputs and outputs.
namespace {
struct PortClassification {
  /// Indices of clock-typed input ports.
  SmallVector<unsigned> clockIndices;
  /// Indices of non-clock, non-register primary input ports.
  SmallVector<unsigned> primaryInputIndices;
  /// Indices of register state input ports (last num_regs inputs).
  SmallVector<unsigned> regStateIndices;
  /// Number of externalized registers.
  unsigned numRegs = 0;
  /// Initial values for registers.
  ArrayAttr initialValues;
};
} // namespace

static PortClassification classifyPorts(hw::HWModuleOp mod) {
  PortClassification result;

  auto numRegsAttr = mod->getAttrOfType<IntegerAttr>("num_regs");
  result.numRegs = numRegsAttr ? numRegsAttr.getValue().getZExtValue() : 0;
  result.initialValues = mod->getAttrOfType<ArrayAttr>("initial_values");

  unsigned numInputs = mod.getNumInputPorts();
  unsigned regStart = numInputs - result.numRegs;

  auto inputTypes = mod.getInputTypes();
  for (unsigned i = 0; i < numInputs; ++i) {
    if (i >= regStart) {
      result.regStateIndices.push_back(i);
    } else if (isa<seq::ClockType>(inputTypes[i])) {
      result.clockIndices.push_back(i);
    } else {
      result.primaryInputIndices.push_back(i);
    }
  }
  return result;
}

void ConstructSeqLECPass::runOnOperation() {
  auto moduleOp = getOperation();
  auto ctx = &getContext();
  OpBuilder builder = OpBuilder::atBlockEnd(moduleOp.getBody());
  Location loc = moduleOp->getLoc();

  // Look up both modules.
  auto lookupModule = [&](StringRef name) -> hw::HWModuleOp {
    auto *op = SymbolTable::lookupNearestSymbolFrom(
        moduleOp, StringAttr::get(ctx, name));
    if (!op || !isa<hw::HWModuleOp>(op)) {
      moduleOp.emitError("module named '") << name << "' not found";
      return {};
    }
    return cast<hw::HWModuleOp>(op);
  };

  auto moduleA = lookupModule(firstModule);
  if (!moduleA)
    return signalPassFailure();
  auto moduleB = lookupModule(secondModule);
  if (!moduleB)
    return signalPassFailure();

  // Classify ports for both modules.
  auto portsA = classifyPorts(moduleA);
  auto portsB = classifyPorts(moduleB);

  // Verify that primary inputs (non-clock, non-register) match in count
  // and type. Clock inputs must also match.
  auto inputTypesA = moduleA.getInputTypes();
  auto inputTypesB = moduleB.getInputTypes();

  if (portsA.clockIndices.size() != portsB.clockIndices.size()) {
    moduleA.emitError("clock port count mismatch: ")
        << portsA.clockIndices.size() << " vs " << portsB.clockIndices.size();
    return signalPassFailure();
  }
  if (portsA.primaryInputIndices.size() != portsB.primaryInputIndices.size()) {
    moduleA.emitError("primary input count mismatch: ")
        << portsA.primaryInputIndices.size() << " vs "
        << portsB.primaryInputIndices.size();
    return signalPassFailure();
  }

  // Verify primary input types match.
  for (auto [ia, ib] : llvm::zip(portsA.primaryInputIndices,
                                  portsB.primaryInputIndices)) {
    if (inputTypesA[ia] != inputTypesB[ib]) {
      moduleA.emitError("primary input type mismatch at index ")
          << ia << ": " << inputTypesA[ia] << " vs " << inputTypesB[ib];
      return signalPassFailure();
    }
  }

  // Verify output counts and types match.
  auto outputTypesA = moduleA.getOutputTypes();
  auto outputTypesB = moduleB.getOutputTypes();

  // Outputs: after ExternalizeRegisters, each module has
  // (original_outputs..., reg_next_states...). We need to compare the
  // original outputs. The number of register-next outputs equals num_regs.
  unsigned numOrigOutputsA = outputTypesA.size() - portsA.numRegs;
  unsigned numOrigOutputsB = outputTypesB.size() - portsB.numRegs;
  if (numOrigOutputsA != numOrigOutputsB) {
    moduleA.emitError("output count mismatch: ")
        << numOrigOutputsA << " vs " << numOrigOutputsB;
    return signalPassFailure();
  }
  for (unsigned i = 0; i < numOrigOutputsA; ++i) {
    if (outputTypesA[i] != outputTypesB[i]) {
      moduleA.emitError("output type mismatch at index ")
          << i << ": " << outputTypesA[i] << " vs " << outputTypesB[i];
      return signalPassFailure();
    }
  }

  // Build combined initial_values: A's registers then B's registers.
  SmallVector<Attribute> combinedInitialValues;
  auto unitAttr = UnitAttr::get(ctx);
  auto addInitialValues = [&](const PortClassification &ports) {
    if (ports.initialValues) {
      for (auto attr : ports.initialValues)
        combinedInitialValues.push_back(attr);
    } else {
      for (unsigned i = 0; i < ports.numRegs; ++i)
        combinedInitialValues.push_back(unitAttr);
    }
  };
  addInitialValues(portsA);
  addInitialValues(portsB);

  unsigned totalRegs = portsA.numRegs + portsB.numRegs;
  bool hasClk = !portsA.clockIndices.empty();
  unsigned clockCount = portsA.clockIndices.size();

  // Scale bound: 2 * bound for toggle-based clocking (low→high per cycle).
  unsigned effectiveBound = 2 * bound;

  // Create the verif.bmc op.
  auto bmcOp = verif::BoundedModelCheckingOp::create(
      builder, loc, effectiveBound, totalRegs,
      ArrayAttr::get(ctx, combinedInitialValues));

  // --- Init region ---
  {
    OpBuilder::InsertionGuard guard(builder);
    auto *initBlock = builder.createBlock(&bmcOp.getInit());
    builder.setInsertionPointToStart(initBlock);
    if (hasClk) {
      auto initVal =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
      SmallVector<Value> yields;
      yields.reserve(clockCount);
      for (unsigned i = 0; i < clockCount; ++i)
        yields.push_back(seq::ToClockOp::create(builder, loc, initVal));
      verif::YieldOp::create(builder, loc, yields);
    } else {
      verif::YieldOp::create(builder, loc, ValueRange{});
    }
  }

  // --- Loop region ---
  {
    OpBuilder::InsertionGuard guard(builder);
    auto *loopBlock = builder.createBlock(&bmcOp.getLoop());
    builder.setInsertionPointToStart(loopBlock);
    if (hasClk) {
      for (unsigned i = 0; i < clockCount; ++i)
        loopBlock->addArgument(seq::ClockType::get(ctx), loc);

      auto cNeg1 =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), -1);
      SmallVector<Value> yields;
      yields.reserve(clockCount);
      for (unsigned i = 0; i < clockCount; ++i) {
        auto fromClk =
            seq::FromClockOp::create(builder, loc, loopBlock->getArgument(i));
        auto toggled = comb::XorOp::create(builder, loc, fromClk, cNeg1);
        yields.push_back(seq::ToClockOp::create(builder, loc, toggled));
      }
      verif::YieldOp::create(builder, loc, yields);
    } else {
      verif::YieldOp::create(builder, loc, ValueRange{});
    }
  }

  // --- Circuit region ---
  // Block args: (clocks..., primary_inputs..., reg_states_A..., reg_states_B...)
  {
    OpBuilder::InsertionGuard guard(builder);
    auto *circuitBlock = builder.createBlock(&bmcOp.getCircuit());
    builder.setInsertionPointToStart(circuitBlock);

    // Add clock arguments.
    SmallVector<Value> clockArgs;
    for (unsigned i = 0; i < clockCount; ++i)
      clockArgs.push_back(
          circuitBlock->addArgument(seq::ClockType::get(ctx), loc));

    // Add primary input arguments (fresh symbolic values each step).
    SmallVector<Value> primaryInputArgs;
    for (unsigned idx : portsA.primaryInputIndices)
      primaryInputArgs.push_back(
          circuitBlock->addArgument(inputTypesA[idx], loc));

    // Add register state arguments for circuit A.
    SmallVector<Value> regArgsA;
    for (unsigned idx : portsA.regStateIndices)
      regArgsA.push_back(
          circuitBlock->addArgument(inputTypesA[idx], loc));

    // Add register state arguments for circuit B.
    SmallVector<Value> regArgsB;
    for (unsigned idx : portsB.regStateIndices)
      regArgsB.push_back(
          circuitBlock->addArgument(inputTypesB[idx], loc));

    // Clone circuit A's body. Map its block arguments to our shared/A-specific
    // args.
    auto cloneCircuit = [&](hw::HWModuleOp mod,
                            const PortClassification &ports,
                            ArrayRef<Value> regArgs)
        -> std::pair<SmallVector<Value>, SmallVector<Value>> {
      IRMapping mapping;
      auto modInputTypes = mod.getInputTypes();
      Block &modBody = mod.getBody().front();

      // Map clock args.
      for (auto [i, clockIdx] : llvm::enumerate(ports.clockIndices))
        mapping.map(modBody.getArgument(clockIdx), clockArgs[i]);

      // Map primary inputs.
      for (auto [i, piIdx] : llvm::enumerate(ports.primaryInputIndices))
        mapping.map(modBody.getArgument(piIdx), primaryInputArgs[i]);

      // Map register state inputs.
      for (auto [i, regIdx] : llvm::enumerate(ports.regStateIndices))
        mapping.map(modBody.getArgument(regIdx), regArgs[i]);

      // Clone all operations except the terminator.
      for (auto &op : modBody.without_terminator())
        builder.clone(op, mapping);

      // Collect outputs and next register states from the terminator.
      auto *terminator = modBody.getTerminator();
      unsigned numOrigOutputs = mod.getOutputTypes().size() - ports.numRegs;

      SmallVector<Value> outputs;
      SmallVector<Value> regNexts;
      for (unsigned i = 0; i < terminator->getNumOperands(); ++i) {
        Value mapped = mapping.lookupOrDefault(terminator->getOperand(i));
        if (i < numOrigOutputs)
          outputs.push_back(mapped);
        else
          regNexts.push_back(mapped);
      }
      return {outputs, regNexts};
    };

    auto [outputsA, regNextsA] = cloneCircuit(moduleA, portsA, regArgsA);
    auto [outputsB, regNextsB] = cloneCircuit(moduleB, portsB, regArgsB);

    // Assert output equivalence.
    for (auto [outA, outB] : llvm::zip(outputsA, outputsB)) {
      Value eq = comb::ICmpOp::create(builder, loc, comb::ICmpPredicate::eq,
                                      outA, outB);
      verif::AssertOp::create(builder, loc, eq, /*enable=*/Value{},
                              /*label=*/StringAttr{});
    }

    // Yield next register states: A's then B's.
    SmallVector<Value> yieldOperands;
    yieldOperands.append(regNextsA.begin(), regNextsA.end());
    yieldOperands.append(regNextsB.begin(), regNextsB.end());
    verif::YieldOp::create(builder, loc, yieldOperands);

    sortTopologically(circuitBlock);
  }

  // Erase the original modules.
  moduleA->erase();
  if (moduleA != moduleB)
    moduleB->erase();
}
