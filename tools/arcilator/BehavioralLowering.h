//===- BehavioralLowering.h - Lower behavioral LLHD to LLVM -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers behavioral LLHD processes, Moore events, and Sim ops
// to LLVM IR with runtime scheduler calls. This enables arcilator to
// natively compile UVM/testbench designs that use LLHD process-based
// simulation.
//
//===----------------------------------------------------------------------===//

#ifndef ARCILATOR_BEHAVIORALLOWERING_H
#define ARCILATOR_BEHAVIORALLOWERING_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;
} // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLowerBehavioralToLLVMPass();

#endif // ARCILATOR_BEHAVIORALLOWERING_H
