//===- LLHDProcessInterpreterTest.cpp - circt-sim tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLHDProcessInterpreter.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace circt;
using namespace circt::sim;
using namespace mlir;

namespace circt {
namespace sim {
class LLHDProcessInterpreterTest {
public:
  static bool evaluateCombinational(LLHDProcessInterpreter &interpreter,
                                    llhd::CombinationalOp combOp,
                                    llvm::SmallVectorImpl<InterpretedValue> &results) {
    return interpreter.evaluateCombinationalOp(combOp, results);
  }
  static ProcessId getProcessId(LLHDProcessInterpreter &interpreter,
                                llhd::ProcessOp processOp) {
    auto it = interpreter.opToProcessId.find(processOp.getOperation());
    if (it == interpreter.opToProcessId.end())
      return InvalidProcessId;
    return it->second;
  }
  static void executeProcess(LLHDProcessInterpreter &interpreter,
                             ProcessId procId) {
    interpreter.executeProcess(procId);
  }
  static std::map<ProcessId, ProcessExecutionState> &
  getProcessStates(LLHDProcessInterpreter &interpreter) {
    return interpreter.processStates;
  }
  static Type getSignalValueType(LLHDProcessInterpreter &interpreter,
                                 SignalId sigId) {
    return interpreter.getSignalValueType(sigId);
  }
  static llvm::SmallVector<std::pair<InstanceId, SignalId>, 4>
  getInstanceSignalIds(LLHDProcessInterpreter &interpreter, Value value) {
    llvm::SmallVector<std::pair<InstanceId, SignalId>, 4> result;
    for (const auto &ctx : interpreter.instanceValueToSignal) {
      auto it = ctx.second.find(value);
      if (it != ctx.second.end())
        result.push_back({ctx.first, it->second});
    }
    return result;
  }
  static InterpretedValue evaluateContinuousValue(
      LLHDProcessInterpreter &interpreter, Value value) {
    return interpreter.evaluateContinuousValue(value);
  }
  static llvm::APInt convertLLVMToHWLayout(
      LLHDProcessInterpreter &interpreter, llvm::APInt value,
      Type llvmType, Type hwType) {
    return interpreter.convertLLVMToHWLayout(std::move(value), llvmType,
                                             hwType);
  }
  static llvm::APInt convertHWToLLVMLayout(
      LLHDProcessInterpreter &interpreter, llvm::APInt value,
      Type hwType, Type llvmType) {
    return interpreter.convertHWToLLVMLayout(std::move(value), hwType,
                                             llvmType);
  }
};
} // namespace sim
} // namespace circt

namespace {

static std::string buildDeepChainIR(unsigned depth) {
  std::string ir;
  llvm::raw_string_ostream os(ir);
  os << "module {\n";
  os << "  hw.module @test(out out: i1) {\n";
  os << "    %c0 = hw.constant false\n";
  os << "    %c1 = hw.constant true\n";
  os << "    %v0 = comb.xor %c1, %c0 : i1\n";
  for (unsigned i = 1; i < depth; ++i)
    os << "    %v" << i << " = comb.xor %v" << (i - 1)
       << ", %c0 : i1\n";
  os << "    hw.output %v" << (depth - 1) << " : i1\n";
  os << "  }\n";
  os << "}\n";
  return os.str();
}

static constexpr llvm::StringLiteral kMuxedRefIR = R"MLIR(
module {
  hw.module @test() {
    %false = hw.constant false
    %s0 = hw.aggregate_constant [false] : !hw.struct<value: i1>
    %s1 = hw.aggregate_constant [true] : !hw.struct<value: i1>

    %sig0 = llhd.sig %s0 : !hw.struct<value: i1>
    %sig1 = llhd.sig %s1 : !hw.struct<value: i1>

    %comb:1 = llhd.combinational -> !hw.struct<value: i1> {
      %ref = comb.mux %false, %sig1, %sig0 : !llhd.ref<!hw.struct<value: i1>>
      %val = llhd.prb %ref : !hw.struct<value: i1>
      llhd.yield %val : !hw.struct<value: i1>
    }

    hw.output
  }
}
)MLIR";

static constexpr llvm::StringLiteral kRefBlockArgIR = R"MLIR(
module {
  hw.module @test() {
    %false = hw.constant false
    %true = hw.constant true
    %s0 = hw.aggregate_constant [false] : !hw.struct<value: i1>
    %s1 = hw.aggregate_constant [true] : !hw.struct<value: i1>
    %eps = llhd.constant_time <0ns, 0d, 1e>

    %sig0 = llhd.sig %s0 : !hw.struct<value: i1>
    %sig1 = llhd.sig %s1 : !hw.struct<value: i1>
    %out = llhd.sig %s0 : !hw.struct<value: i1>

    llhd.process {
      cf.cond_br %false, ^bb1(%sig0 : !llhd.ref<!hw.struct<value: i1>>),
                         ^bb1(%sig1 : !llhd.ref<!hw.struct<value: i1>>)
    ^bb1(%ref: !llhd.ref<!hw.struct<value: i1>>):
      %val = llhd.prb %ref : !hw.struct<value: i1>
      llhd.drv %out, %val after %eps : !hw.struct<value: i1>
      llhd.halt
    }

    hw.output
  }
}
)MLIR";

static constexpr llvm::StringLiteral kLLVMSignalLoadStoreIR = R"MLIR(
module {
  hw.module @test() {
    %zero = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
    %one = hw.aggregate_constant [true, false] : !hw.struct<value: i1, unknown: i1>

    %sig_in = llhd.sig name "sig_in" %one : !hw.struct<value: i1, unknown: i1>
    %sig_out = llhd.sig name "sig_out" %zero : !hw.struct<value: i1, unknown: i1>

    llhd.process {
      %ptr_in = builtin.unrealized_conversion_cast %sig_in : !llhd.ref<!hw.struct<value: i1, unknown: i1>> to !llvm.ptr
      %loaded = llvm.load %ptr_in : !llvm.ptr -> !llvm.struct<(i1, i1)>
      %ptr_out = builtin.unrealized_conversion_cast %sig_out : !llhd.ref<!hw.struct<value: i1, unknown: i1>> to !llvm.ptr
      llvm.store %loaded, %ptr_out : !llvm.struct<(i1, i1)>, !llvm.ptr
      llhd.halt
    }

    hw.output
  }
}
)MLIR";

static constexpr llvm::StringLiteral kChildFirRegResetIR = R"MLIR(
module {
  hw.module @child(in %clk: !hw.struct<value: i1, unknown: i1>,
                   in %rst_ni: !hw.struct<value: i1, unknown: i1>,
                   out q: !hw.struct<value: i1, unknown: i1>) {
    %true = hw.constant true
    %zero = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
    %one = hw.aggregate_constant [true, false] : !hw.struct<value: i1, unknown: i1>
    %clk_val = hw.struct_extract %clk["value"] : !hw.struct<value: i1, unknown: i1>
    %clk_unknown = hw.struct_extract %clk["unknown"] : !hw.struct<value: i1, unknown: i1>
    %clk_known = comb.xor %clk_unknown, %true : i1
    %clk_clean = comb.and %clk_val, %clk_known : i1
    %clk_clock = seq.to_clock %clk_clean
    %rst_val = hw.struct_extract %rst_ni["value"] : !hw.struct<value: i1, unknown: i1>
    %rst_unknown = hw.struct_extract %rst_ni["unknown"] : !hw.struct<value: i1, unknown: i1>
    %rst_known = comb.xor %rst_unknown, %true : i1
    %rst_clean = comb.and %rst_val, %rst_known : i1
    %rst = comb.xor %rst_clean, %true : i1
    %q = seq.firreg %one clock %clk_clock reset async %rst, %zero preset 0 : !hw.struct<value: i1, unknown: i1>
    hw.output %q : !hw.struct<value: i1, unknown: i1>
  }

  hw.module @top() {
    %zero = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
    %clk = llhd.sig %zero : !hw.struct<value: i1, unknown: i1>
    %rst_ni = llhd.sig %zero : !hw.struct<value: i1, unknown: i1>
    %clk_val = llhd.prb %clk : !hw.struct<value: i1, unknown: i1>
    %rst_val = llhd.prb %rst_ni : !hw.struct<value: i1, unknown: i1>
    %inst_q = hw.instance "inst" @child(clk: %clk_val : !hw.struct<value: i1, unknown: i1>, rst_ni: %rst_val : !hw.struct<value: i1, unknown: i1>) -> (q: !hw.struct<value: i1, unknown: i1>)
    hw.output
  }
}
)MLIR";

static constexpr llvm::StringLiteral kProbeEncodedUnknownIR = R"MLIR(
module {
  hw.module @test() {
    %zero = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
    %sig = llhd.sig name "sig" %zero : !hw.struct<value: i1, unknown: i1>

    %comb:1 = llhd.combinational -> i1 {
      %val = llhd.prb %sig : !hw.struct<value: i1, unknown: i1>
      %unk = hw.struct_extract %val["unknown"] : !hw.struct<value: i1, unknown: i1>
      llhd.yield %unk : i1
    }

    hw.output
  }
}
)MLIR";

static constexpr llvm::StringLiteral kInstanceInputMappingIR = R"MLIR(
module {
  hw.module @child(in %in: i1, out out: i1) {
    %true = hw.constant true
    %out = comb.xor %in, %true : i1
    hw.output %out : i1
  }

  hw.module @top() {
    %true = hw.constant true
    %false = hw.constant false
    %sig0 = llhd.sig name "sig0" %false : i1
    %sig1 = llhd.sig name "sig1" %true : i1
    %val0 = llhd.prb %sig0 : i1
    %val1 = llhd.prb %sig1 : i1
    %inst0.out = hw.instance "i0" @child(in: %val0 : i1) -> (out: i1)
    %inst1.out = hw.instance "i1" @child(in: %val1 : i1) -> (out: i1)
    %eps = llhd.constant_time <0ns, 0d, 1e>
    %out0 = llhd.sig name "out0" %false : i1
    %out1 = llhd.sig name "out1" %false : i1
    llhd.drv %out0, %inst0.out after %eps : i1
    llhd.drv %out1, %inst1.out after %eps : i1
    hw.output
  }
}
)MLIR";

static constexpr llvm::StringLiteral kInstanceProcessModuleDriveIR = R"MLIR(
module {
  hw.module @child(in %in: !llhd.ref<i1>, out out: i1) {
    %false = hw.constant false
    %eps = llhd.constant_time <0ns, 0d, 1e>
    %out_sig = llhd.sig name "out" %false : i1
    llhd.process {
      %val = llhd.prb %in : i1
      llhd.drv %out_sig, %val after %eps : i1
      llhd.wait delay %eps, ^bb1
    ^bb1:
      llhd.halt
    }
    %out_val = llhd.prb %out_sig : i1
    hw.output %out_val : i1
  }

  hw.module @top() {
    %true = hw.constant true
    %false = hw.constant false
    %sig0 = llhd.sig name "sig0" %false : i1
    %sig1 = llhd.sig name "sig1" %true : i1
    %inst0.out = hw.instance "i0" @child(in: %sig0 : !llhd.ref<i1>) -> (out: i1)
    %inst1.out = hw.instance "i1" @child(in: %sig1 : !llhd.ref<i1>) -> (out: i1)
    %eps = llhd.constant_time <0ns, 0d, 1e>
    %out0 = llhd.sig name "out0" %false : i1
    %out1 = llhd.sig name "out1" %false : i1
    llhd.drv %out0, %inst0.out after %eps : i1
    llhd.drv %out1, %inst1.out after %eps : i1
    hw.output
  }
}
)MLIR";

static constexpr llvm::StringLiteral kInstanceScopedSignalLookupIR = R"MLIR(
module {
  hw.module @child(out out: i1) {
    %false = hw.constant false
    %local = llhd.sig name "local" %false : i1
    %val = llhd.prb %local : i1
    hw.output %val : i1
  }

  hw.module @top() {
    %inst0.out = hw.instance "i0" @child() -> (out: i1)
    %inst1.out = hw.instance "i1" @child() -> (out: i1)
    hw.output
  }
}
)MLIR";

static constexpr llvm::StringLiteral kInstanceChainedClockIR = R"MLIR(
module {
  hw.module @leaf(in %clk: i1, in %d: i1, out q: i1) {
    %false = hw.constant false
    %clk_clock = seq.to_clock %clk
    %q = seq.firreg %d clock %clk_clock reset async %false, %false preset 0 : i1
    hw.output %q : i1
  }

  hw.module @mid(in %clk: i1, in %d: i1, out q: i1) {
    %inst.q = hw.instance "leaf" @leaf(clk: %clk : i1, d: %d : i1) -> (q: i1)
    hw.output %inst.q : i1
  }

  hw.module @top() {
    %false = hw.constant false
    %true = hw.constant true
    %clk = llhd.sig name "clk" %false : i1
    %data = llhd.sig name "data" %true : i1
    %clk_val = llhd.prb %clk : i1
    %data_val = llhd.prb %data : i1
    %inst.q = hw.instance "mid" @mid(clk: %clk_val : i1, d: %data_val : i1) -> (q: i1)
    %eps = llhd.constant_time <0ns, 0d, 1e>
    %q_sig = llhd.sig name "q_sig" %false : i1
    llhd.drv %q_sig, %inst.q after %eps : i1
    llhd.process {
      llhd.drv %clk, %true after %eps : i1
      llhd.wait delay %eps, ^bb1
    ^bb1:
      llhd.halt
    }
    hw.output
  }
}
)MLIR";

static constexpr llvm::StringLiteral kLLVMSignalAggregateLayoutIR = R"MLIR(
module {
  hw.module @test() {
    %false = hw.constant false
    %zero = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
    %one = hw.aggregate_constant [true, false] : !hw.struct<value: i1, unknown: i1>

    %sig_in = llhd.sig name "sig_in" %one : !hw.struct<value: i1, unknown: i1>
    %sig_out = llhd.sig name "sig_out" %zero : !hw.struct<value: i1, unknown: i1>
    %sig_value = llhd.sig name "sig_value" %false : i1
    %sig_unknown = llhd.sig name "sig_unknown" %false : i1

    llhd.process {
      %ptr_in = builtin.unrealized_conversion_cast %sig_in : !llhd.ref<!hw.struct<value: i1, unknown: i1>> to !llvm.ptr
      %loaded = llvm.load %ptr_in : !llvm.ptr -> !llvm.struct<(i1, i1)>
      %val = llvm.extractvalue %loaded[0] : !llvm.struct<(i1, i1)>
      %unk = llvm.extractvalue %loaded[1] : !llvm.struct<(i1, i1)>
      %ptr_val = builtin.unrealized_conversion_cast %sig_value : !llhd.ref<i1> to !llvm.ptr
      %ptr_unk = builtin.unrealized_conversion_cast %sig_unknown : !llhd.ref<i1> to !llvm.ptr
      llvm.store %val, %ptr_val : i1, !llvm.ptr
      llvm.store %unk, %ptr_unk : i1, !llvm.ptr

      %ptr_out = builtin.unrealized_conversion_cast %sig_out : !llhd.ref<!hw.struct<value: i1, unknown: i1>> to !llvm.ptr
      %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
      %one_i1 = llvm.mlir.constant(1 : i1) : i1
      %zero_i1 = llvm.mlir.constant(0 : i1) : i1
      %v0 = llvm.insertvalue %one_i1, %undef[0] : !llvm.struct<(i1, i1)>
      %v1 = llvm.insertvalue %zero_i1, %v0[1] : !llvm.struct<(i1, i1)>
      llvm.store %v1, %ptr_out : !llvm.struct<(i1, i1)>, !llvm.ptr
      llhd.halt
    }

    hw.output
  }
}
)MLIR";

static constexpr llvm::StringLiteral kLLVMWideLoadPaddingIR = R"MLIR(
module {
  hw.module @test() {
    %zero_i1 = hw.constant 0 : i1
    %one_i64 = hw.constant 1 : i64

    %sig = llhd.sig name "sig" %zero_i1 : i1

    llhd.process {
      %ptr = llvm.alloca %one_i64 x !llvm.array<65 x i1> : (i64) -> !llvm.ptr
      %undef = llvm.mlir.undef : !llvm.array<65 x i1>
      %one_i1 = llvm.mlir.constant(1 : i1) : i1
      %val = llvm.insertvalue %one_i1, %undef[0] : !llvm.array<65 x i1>
      llvm.store %val, %ptr : !llvm.array<65 x i1>, !llvm.ptr
      %loaded = llvm.load %ptr : !llvm.ptr -> !llvm.array<65 x i1>
      %elem0 = llvm.extractvalue %loaded[0] : !llvm.array<65 x i1>
      %ptr_sig = builtin.unrealized_conversion_cast %sig : !llhd.ref<i1> to !llvm.ptr
      llvm.store %elem0, %ptr_sig : i1, !llvm.ptr
      llhd.halt
    }

    hw.output
  }
}
)MLIR";

static constexpr llvm::StringLiteral kCoverageDbStubIR = R"MLIR(
module {
  llvm.func @__moore_coverage_set_test_name(!llvm.ptr)
  llvm.func @__moore_coverage_load_db(!llvm.ptr) -> !llvm.ptr

  hw.module @test() {
    %zero_i64 = hw.constant 0 : i64
    %eps = llhd.constant_time <0ns, 0d, 1e>
    %sig = llhd.sig name "sig" %zero_i64 : i64

    llhd.process {
      %zero_i64_llvm = llvm.mlir.constant(0 : i64) : i64
      %null = llvm.inttoptr %zero_i64_llvm : i64 to !llvm.ptr
      llvm.call @__moore_coverage_set_test_name(%null) : (!llvm.ptr) -> ()
      %handle = llvm.call @__moore_coverage_load_db(%null) : (!llvm.ptr) -> !llvm.ptr
      %val = llvm.ptrtoint %handle : !llvm.ptr to i64
      llhd.drv %sig, %val after %eps : i64
      llhd.halt
    }

    hw.output
  }
}
)MLIR";

static constexpr llvm::StringLiteral kContinuousCombOpsIR = R"MLIR(
module {
  hw.module @test(out rep: i4, out parity: i1, out shifted: i4, out mul: i4, out divu: i4, out modu: i4) {
    %c1 = hw.constant true
    %c1_i4 = hw.constant 1 : i4
    %c2_i4 = hw.constant 2 : i4
    %c3_i4 = hw.constant 3 : i4
    %rep = comb.replicate %c1 : (i1) -> i4
    %parity = comb.parity %rep : i4
    %shifted = comb.shl %c1_i4, %c1_i4 : i4
    %mul = comb.mul %c3_i4, %c2_i4 : i4
    %divu = comb.divu %mul, %c2_i4 : i4
    %modu = comb.modu %c3_i4, %c2_i4 : i4
    hw.output %rep, %parity, %shifted, %mul, %divu, %modu : i4, i1, i4, i4, i4, i4
  }
}
)MLIR";
TEST(LLHDProcessInterpreterToolTest, ProbeMuxedRefInCombinational) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<llhd::LLHDDialect>();
  context.loadDialect<comb::CombDialect>();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(kMuxedRefIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  llhd::CombinationalOp combOp;
  hwModule.walk([&](llhd::CombinationalOp op) { combOp = op; });
  ASSERT_TRUE(combOp);

  llvm::SmallVector<InterpretedValue, 4> results;
  EXPECT_TRUE(LLHDProcessInterpreterTest::evaluateCombinational(interpreter,
                                                                combOp, results));
  ASSERT_EQ(results.size(), 1u);
  EXPECT_FALSE(results[0].isX());
  EXPECT_EQ(results[0].getUInt64(), 0u);
}

TEST(LLHDProcessInterpreterToolTest, ProbeRefBlockArgument) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<llhd::LLHDDialect>();
  context.loadDialect<comb::CombDialect>();
  context.loadDialect<cf::ControlFlowDialect>();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(kRefBlockArgIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  llhd::ProcessOp procOp;
  hwModule.walk([&](llhd::ProcessOp op) { procOp = op; });
  ASSERT_TRUE(procOp);

  llhd::SignalOp outOp;
  hwModule.walk([&](llhd::SignalOp op) {
    auto nameAttr = op.getName();
    if (nameAttr && *nameAttr == "out")
      outOp = op;
  });
  ASSERT_TRUE(outOp);

  ProcessId procId =
      LLHDProcessInterpreterTest::getProcessId(interpreter, procOp);
  ASSERT_NE(procId, InvalidProcessId);
  LLHDProcessInterpreterTest::executeProcess(interpreter, procId);
  auto &eventScheduler = scheduler.getEventScheduler();
  for (int i = 0; i < 3; ++i)
    (void)eventScheduler.stepDelta();

  SignalId outId = interpreter.getSignalId(outOp.getResult());
  ASSERT_NE(outId, 0u);
  const SignalValue &outVal = scheduler.getSignalValue(outId);
  EXPECT_FALSE(outVal.isUnknown());
  EXPECT_EQ(outVal.getAPInt().getZExtValue(), 1u);
}

TEST(LLHDProcessInterpreterToolTest, ChildFirRegAsyncResetInit) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<llhd::LLHDDialect>();
  context.loadDialect<seq::SeqDialect>();
  context.loadDialect<comb::CombDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kChildFirRegResetIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto topModule = symbols.lookup<hw::HWModuleOp>("top");
  auto childModule = symbols.lookup<hw::HWModuleOp>("child");
  ASSERT_TRUE(topModule);
  ASSERT_TRUE(childModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(topModule)));

  seq::FirRegOp regOp;
  childModule.walk([&](seq::FirRegOp op) { regOp = op; });
  ASSERT_TRUE(regOp);

  SignalId regId = interpreter.getSignalId(regOp.getResult());
  ASSERT_NE(regId, 0u);
  const SignalValue &regVal = scheduler.getSignalValue(regId);
  EXPECT_FALSE(regVal.isUnknown());
  EXPECT_EQ(regVal.getAPInt().getZExtValue(), 0u);
}

TEST(LLHDProcessInterpreterToolTest, InstanceOutputMapping) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<llhd::LLHDDialect>();
  context.loadDialect<comb::CombDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kInstanceInputMappingIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("top");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  auto &eventScheduler = scheduler.getEventScheduler();
  for (int i = 0; i < 3; ++i)
    (void)eventScheduler.stepDelta();

  llhd::SignalOp out0Op;
  llhd::SignalOp out1Op;
  hwModule.walk([&](llhd::SignalOp op) {
    auto nameAttr = op.getName();
    if (!nameAttr)
      return;
    if (*nameAttr == "out0")
      out0Op = op;
    if (*nameAttr == "out1")
      out1Op = op;
  });
  ASSERT_TRUE(out0Op);
  ASSERT_TRUE(out1Op);

  SignalId out0Id = interpreter.getSignalId(out0Op.getResult());
  SignalId out1Id = interpreter.getSignalId(out1Op.getResult());
  ASSERT_NE(out0Id, 0u);
  ASSERT_NE(out1Id, 0u);

  const SignalValue &out0Val = scheduler.getSignalValue(out0Id);
  const SignalValue &out1Val = scheduler.getSignalValue(out1Id);
  EXPECT_FALSE(out0Val.isUnknown());
  EXPECT_FALSE(out1Val.isUnknown());
  EXPECT_EQ(out0Val.getAPInt().getZExtValue(), 1u);
  EXPECT_EQ(out1Val.getAPInt().getZExtValue(), 0u);
}

TEST(LLHDProcessInterpreterToolTest, InstanceProcessModuleDrive) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<llhd::LLHDDialect>();
  context.loadDialect<comb::CombDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kInstanceProcessModuleDriveIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("top");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  for (int i = 0; i < 5; ++i) {
    if (!scheduler.advanceTime())
      break;
  }

  llhd::SignalOp sig0Op;
  llhd::SignalOp sig1Op;
  llhd::SignalOp out0Op;
  llhd::SignalOp out1Op;
  hwModule.walk([&](llhd::SignalOp op) {
    auto nameAttr = op.getName();
    if (!nameAttr)
      return;
    if (*nameAttr == "sig0")
      sig0Op = op;
    if (*nameAttr == "sig1")
      sig1Op = op;
    if (*nameAttr == "out0")
      out0Op = op;
    if (*nameAttr == "out1")
      out1Op = op;
  });
  ASSERT_TRUE(sig0Op);
  ASSERT_TRUE(sig1Op);
  ASSERT_TRUE(out0Op);
  ASSERT_TRUE(out1Op);

  SignalId sig0Id = interpreter.getSignalId(sig0Op.getResult());
  SignalId sig1Id = interpreter.getSignalId(sig1Op.getResult());
  ASSERT_NE(sig0Id, 0u);
  ASSERT_NE(sig1Id, 0u);

  const SignalValue &sig0Val = scheduler.getSignalValue(sig0Id);
  const SignalValue &sig1Val = scheduler.getSignalValue(sig1Id);
  EXPECT_EQ(sig0Val.getAPInt().getZExtValue(), 0u);
  EXPECT_EQ(sig1Val.getAPInt().getZExtValue(), 1u);

  SignalId out0Id = interpreter.getSignalId(out0Op.getResult());
  SignalId out1Id = interpreter.getSignalId(out1Op.getResult());
  ASSERT_NE(out0Id, 0u);
  ASSERT_NE(out1Id, 0u);

  const SignalValue &out0Val = scheduler.getSignalValue(out0Id);
  const SignalValue &out1Val = scheduler.getSignalValue(out1Id);
  EXPECT_FALSE(out0Val.isUnknown());
  EXPECT_FALSE(out1Val.isUnknown());
  EXPECT_EQ(out0Val.getAPInt().getZExtValue(), 0u);
  EXPECT_EQ(out1Val.getAPInt().getZExtValue(), 1u);
}

TEST(LLHDProcessInterpreterToolTest, InstanceScopedSignalLookup) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<llhd::LLHDDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kInstanceScopedSignalLookupIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("top");
  auto childModule = symbols.lookup<hw::HWModuleOp>("child");
  ASSERT_TRUE(hwModule);
  ASSERT_TRUE(childModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  llhd::SignalOp localSigOp;
  childModule.walk([&](llhd::SignalOp op) { localSigOp = op; });
  ASSERT_TRUE(localSigOp);

  auto mappings = LLHDProcessInterpreterTest::getInstanceSignalIds(
      interpreter, localSigOp.getResult());
  ASSERT_EQ(mappings.size(), 2u);

  SignalId sig0 =
      interpreter.getSignalIdInInstance(localSigOp.getResult(),
                                        mappings[0].first);
  SignalId sig1 =
      interpreter.getSignalIdInInstance(localSigOp.getResult(),
                                        mappings[1].first);
  EXPECT_NE(sig0, 0u);
  EXPECT_NE(sig1, 0u);
  EXPECT_NE(sig0, sig1);

  EXPECT_EQ(interpreter.getSignalIdInInstance(localSigOp.getResult(), 0), 0u);
  EXPECT_EQ(interpreter.getSignalId(localSigOp.getResult()), 0u);
}

TEST(LLHDProcessInterpreterToolTest, InstanceChainedClockMapping) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<llhd::LLHDDialect>();
  context.loadDialect<seq::SeqDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kInstanceChainedClockIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("top");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  for (int i = 0; i < 5; ++i) {
    if (!scheduler.advanceTime())
      break;
  }

  llhd::SignalOp qSigOp;
  hwModule.walk([&](llhd::SignalOp op) {
    auto nameAttr = op.getName();
    if (nameAttr && *nameAttr == "q_sig")
      qSigOp = op;
  });
  ASSERT_TRUE(qSigOp);

  SignalId qSigId = interpreter.getSignalId(qSigOp.getResult());
  ASSERT_NE(qSigId, 0u);

  const SignalValue &qSigVal = scheduler.getSignalValue(qSigId);
  EXPECT_FALSE(qSigVal.isUnknown());
  EXPECT_EQ(qSigVal.getAPInt().getZExtValue(), 1u);
}

TEST(LLHDProcessInterpreterToolTest, ProbeEncodedUnknown) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<llhd::LLHDDialect>();
  context.loadDialect<comb::CombDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProbeEncodedUnknownIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  llhd::SignalOp sigOp;
  hwModule.walk([&](llhd::SignalOp op) {
    auto nameAttr = op.getName();
    if (nameAttr && *nameAttr == "sig")
      sigOp = op;
  });
  ASSERT_TRUE(sigOp);

  SignalId sigId = interpreter.getSignalId(sigOp.getResult());
  ASSERT_NE(sigId, 0u);
  scheduler.updateSignal(sigId, SignalValue::makeX(2));

  llhd::CombinationalOp combOp;
  hwModule.walk([&](llhd::CombinationalOp op) { combOp = op; });
  ASSERT_TRUE(combOp);

  llvm::SmallVector<InterpretedValue, 4> results;
  EXPECT_TRUE(LLHDProcessInterpreterTest::evaluateCombinational(interpreter,
                                                                combOp, results));
  ASSERT_EQ(results.size(), 1u);
  EXPECT_FALSE(results[0].isX());
  EXPECT_EQ(results[0].getUInt64(), 1u);
}

TEST(LLHDProcessInterpreterToolTest, LLVMSignalLoadStore) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<llhd::LLHDDialect>();
  context.loadDialect<LLVM::LLVMDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kLLVMSignalLoadStoreIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  llhd::SignalOp sigInOp;
  llhd::SignalOp sigOutOp;
  hwModule.walk([&](llhd::SignalOp op) {
    auto nameAttr = op.getName();
    if (!nameAttr)
      return;
    if (*nameAttr == "sig_in")
      sigInOp = op;
    else if (*nameAttr == "sig_out")
      sigOutOp = op;
  });
  ASSERT_TRUE(sigInOp);
  ASSERT_TRUE(sigOutOp);

  ASSERT_TRUE(scheduler.executeDeltaCycle());

  SignalId sigInId = interpreter.getSignalId(sigInOp.getResult());
  SignalId sigOutId = interpreter.getSignalId(sigOutOp.getResult());
  ASSERT_NE(sigInId, 0u);
  ASSERT_NE(sigOutId, 0u);

  const SignalValue &inVal = scheduler.getSignalValue(sigInId);
  const SignalValue &outVal = scheduler.getSignalValue(sigOutId);
  EXPECT_EQ(inVal.getAPInt(), outVal.getAPInt());
}

TEST(LLHDProcessInterpreterToolTest, LLVMSignalAggregateLayout) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<llhd::LLHDDialect>();
  context.loadDialect<LLVM::LLVMDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kLLVMSignalAggregateLayoutIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  llhd::SignalOp sigInOp;
  llhd::SignalOp sigOutOp;
  llhd::SignalOp sigValueOp;
  llhd::SignalOp sigUnknownOp;
  hwModule.walk([&](llhd::SignalOp op) {
    auto nameAttr = op.getName();
    if (!nameAttr)
      return;
    if (*nameAttr == "sig_in")
      sigInOp = op;
    else if (*nameAttr == "sig_out")
      sigOutOp = op;
    else if (*nameAttr == "sig_value")
      sigValueOp = op;
    else if (*nameAttr == "sig_unknown")
      sigUnknownOp = op;
  });
  ASSERT_TRUE(sigInOp);
  ASSERT_TRUE(sigOutOp);
  ASSERT_TRUE(sigValueOp);
  ASSERT_TRUE(sigUnknownOp);

  ASSERT_TRUE(scheduler.executeDeltaCycle());

  SignalId sigInId = interpreter.getSignalId(sigInOp.getResult());
  SignalId sigOutId = interpreter.getSignalId(sigOutOp.getResult());
  SignalId sigValueId = interpreter.getSignalId(sigValueOp.getResult());
  SignalId sigUnknownId = interpreter.getSignalId(sigUnknownOp.getResult());
  ASSERT_NE(sigInId, 0u);
  ASSERT_NE(sigOutId, 0u);
  ASSERT_NE(sigValueId, 0u);
  ASSERT_NE(sigUnknownId, 0u);

  Type sigInType =
      LLHDProcessInterpreterTest::getSignalValueType(interpreter, sigInId);
  ASSERT_TRUE(sigInType);
  EXPECT_TRUE(isa<hw::StructType>(sigInType));

  const SignalValue &outVal = scheduler.getSignalValue(sigOutId);
  const SignalValue &valueVal = scheduler.getSignalValue(sigValueId);
  const SignalValue &unknownVal = scheduler.getSignalValue(sigUnknownId);
  EXPECT_EQ(outVal.getAPInt().getZExtValue(), 2u);
  EXPECT_EQ(valueVal.getAPInt().getZExtValue(), 1u);
  EXPECT_EQ(unknownVal.getAPInt().getZExtValue(), 0u);
}

TEST(LLHDProcessInterpreterToolTest, LLVMWideLoadWithPadding) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<llhd::LLHDDialect>();
  context.loadDialect<LLVM::LLVMDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kLLVMWideLoadPaddingIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  llhd::SignalOp sigOp;
  hwModule.walk([&](llhd::SignalOp op) {
    auto nameAttr = op.getName();
    if (nameAttr && *nameAttr == "sig")
      sigOp = op;
  });
  ASSERT_TRUE(sigOp);

  ASSERT_TRUE(scheduler.executeDeltaCycle());

  SignalId sigId = interpreter.getSignalId(sigOp.getResult());
  ASSERT_NE(sigId, 0u);
  const SignalValue &sigVal = scheduler.getSignalValue(sigId);
  EXPECT_EQ(sigVal.getAPInt().getZExtValue(), 1u);
}

TEST(LLHDProcessInterpreterToolTest, ContinuousValueDeepChain) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<comb::CombDialect>();
  context.loadDialect<llhd::LLHDDialect>();

  std::string ir = buildDeepChainIR(600);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  hw::OutputOp outputOp;
  hwModule.walk([&](hw::OutputOp op) { outputOp = op; });
  ASSERT_TRUE(outputOp);
  ASSERT_EQ(outputOp.getNumOperands(), 1u);

  InterpretedValue val =
      LLHDProcessInterpreterTest::evaluateContinuousValue(
          interpreter, outputOp.getOperand(0));
  EXPECT_FALSE(val.isX());
  EXPECT_EQ(val.getUInt64(), 1u);
}

TEST(LLHDProcessInterpreterToolTest, CoverageDbStubs) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<llhd::LLHDDialect>();
  context.loadDialect<LLVM::LLVMDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kCoverageDbStubIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  llhd::SignalOp sigOp;
  hwModule.walk([&](llhd::SignalOp op) {
    auto nameAttr = op.getName();
    if (nameAttr && *nameAttr == "sig")
      sigOp = op;
  });
  ASSERT_TRUE(sigOp);

  ASSERT_TRUE(scheduler.executeDeltaCycle());

  SignalId sigId = interpreter.getSignalId(sigOp.getResult());
  ASSERT_NE(sigId, 0u);
  const SignalValue &sigVal = scheduler.getSignalValue(sigId);
  EXPECT_EQ(sigVal.getAPInt().getZExtValue(), 0u);
}

TEST(LLHDProcessInterpreterToolTest, ContinuousCombOps) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<comb::CombDialect>();
  context.loadDialect<llhd::LLHDDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kContinuousCombOpsIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  hw::OutputOp outputOp;
  hwModule.walk([&](hw::OutputOp op) { outputOp = op; });
  ASSERT_TRUE(outputOp);
  ASSERT_EQ(outputOp.getNumOperands(), 6u);

  InterpretedValue rep =
      LLHDProcessInterpreterTest::evaluateContinuousValue(
          interpreter, outputOp.getOperand(0));
  InterpretedValue parity =
      LLHDProcessInterpreterTest::evaluateContinuousValue(
          interpreter, outputOp.getOperand(1));
  InterpretedValue shifted =
      LLHDProcessInterpreterTest::evaluateContinuousValue(
          interpreter, outputOp.getOperand(2));
  InterpretedValue mul =
      LLHDProcessInterpreterTest::evaluateContinuousValue(
          interpreter, outputOp.getOperand(3));
  InterpretedValue divu =
      LLHDProcessInterpreterTest::evaluateContinuousValue(
          interpreter, outputOp.getOperand(4));
  InterpretedValue modu =
      LLHDProcessInterpreterTest::evaluateContinuousValue(
          interpreter, outputOp.getOperand(5));

  EXPECT_FALSE(rep.isX());
  EXPECT_FALSE(parity.isX());
  EXPECT_FALSE(shifted.isX());
  EXPECT_FALSE(mul.isX());
  EXPECT_FALSE(divu.isX());
  EXPECT_FALSE(modu.isX());
  EXPECT_EQ(rep.getUInt64(), 15u);
  EXPECT_EQ(parity.getUInt64(), 0u);
  EXPECT_EQ(shifted.getUInt64(), 2u);
  EXPECT_EQ(mul.getUInt64(), 6u);
  EXPECT_EQ(divu.getUInt64(), 3u);
  EXPECT_EQ(modu.getUInt64(), 1u);
}

//===----------------------------------------------------------------------===//
// Stack Overflow Regression Tests (Fix 0ec18eccf)
//===----------------------------------------------------------------------===//

// Helper to build IR with deeply nested llhd.combinational operations.
// This simulates the patterns found in large OpenTitan IPs (hmac_reg_top,
// rv_timer_reg_top, spi_host_reg_top) that caused stack overflow.
static std::string buildDeepCombinationalIR(unsigned nestingDepth) {
  std::string ir;
  llvm::raw_string_ostream os(ir);

  os << "module {\n";
  os << "  hw.module @test() {\n";
  os << "    %false = hw.constant false\n";
  os << "    %sig = llhd.sig %false : i1\n";
  os << "    %val0 = llhd.prb %sig : i1\n";

  // Build a chain of comb.xor operations inside nested combinational blocks
  for (unsigned i = 0; i < nestingDepth; ++i) {
    os << "    %comb" << i << ":1 = llhd.combinational -> i1 {\n";
    if (i == 0) {
      os << "      %r = comb.xor %val0, %false : i1\n";
    } else {
      os << "      %r = comb.xor %comb" << (i - 1) << "#0, %false : i1\n";
    }
    os << "      llhd.yield %r : i1\n";
    os << "    }\n";
  }

  os << "    hw.output\n";
  os << "  }\n";
  os << "}\n";

  return os.str();
}

TEST(LLHDProcessInterpreterToolTest, CollectSignalIdsDeepNesting) {
  // Regression test for commit 0ec18eccf:
  // "Fix stack overflow in collectSignalIds for large designs"
  //
  // The bug: collectSignalIds and collectSignalIdsFromCombinational had
  // mutual recursion that caused stack overflow on large OpenTitan IPs
  // with deep nesting (hmac_reg_top, rv_timer_reg_top, spi_host_reg_top).
  //
  // The fix: Inline collectSignalIdsFromCombinational logic into
  // collectSignalIds and add CombinationalOp operands directly to the
  // worklist instead of recursively calling collectSignalIds.
  //
  // This test verifies that deeply nested CombinationalOps can be
  // processed without stack overflow.

  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<comb::CombDialect>();
  context.loadDialect<llhd::LLHDDialect>();

  // Build IR with significant nesting depth.
  // Before the fix, even moderate depths (100-200) would cause stack overflow.
  // After the fix, this should handle much deeper nesting.
  constexpr unsigned kNestingDepth = 500;
  std::string ir = buildDeepCombinationalIR(kNestingDepth);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module) << "Failed to parse generated IR";

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);

  // The key test: initialization should complete without stack overflow.
  // Before the fix, this would crash on large designs.
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)))
      << "Initialization should succeed without stack overflow";

  // Verify that signals were properly registered
  EXPECT_GE(interpreter.getNumSignals(), 1u)
      << "At least the main signal should be registered";
}

TEST(LLHDProcessInterpreterToolTest, CollectSignalIdsWideCombinationalChain) {
  // Test that wide combinational chains (like those in OpenTitan reg_top
  // modules) are handled correctly without stack overflow.

  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<comb::CombDialect>();
  context.loadDialect<llhd::LLHDDialect>();

  // Use the existing buildDeepChainIR helper for a different test case
  // This creates a long chain of comb operations at module level
  std::string ir = buildDeepChainIR(1000);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module) << "Failed to parse generated IR";

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);

  // Should complete without stack overflow
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)))
      << "Initialization should succeed for wide operation chains";

  // Verify the output can be evaluated
  hw::OutputOp outputOp;
  hwModule.walk([&](hw::OutputOp op) { outputOp = op; });
  ASSERT_TRUE(outputOp);
  ASSERT_EQ(outputOp.getNumOperands(), 1u);

  InterpretedValue val =
      LLHDProcessInterpreterTest::evaluateContinuousValue(
          interpreter, outputOp.getOperand(0));

  // The chain alternates XOR with false, preserving the initial value of 1
  EXPECT_FALSE(val.isX());
  EXPECT_EQ(val.getUInt64(), 1u);
}

//===----------------------------------------------------------------------===//
// Reference Stability Regression Test (DenseMap-to-std::map fix)
//===----------------------------------------------------------------------===//

TEST(LLHDProcessInterpreterToolTest, ProcessStatesReferenceStability) {
  // Regression test: processStates was originally an llvm::DenseMap.
  // When evaluateCombinationalOp inserted temporary process states, a DenseMap
  // rehash could invalidate references held by callers (e.g., interpretWait's
  // `auto &state = processStates[procId]`), causing a segfault.
  //
  // The fix changed processStates to std::map, which guarantees that
  // references/pointers to elements remain valid after insertions and erasures.
  //
  // This test directly exercises that invariant: it takes a reference to an
  // entry in processStates, inserts enough additional entries to trigger what
  // would have been a DenseMap rehash, and verifies the original reference
  // is still valid.

  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<llhd::LLHDDialect>();

  // We only need a minimal module to construct the interpreter; the test
  // operates directly on the processStates map.
  static constexpr llvm::StringLiteral kMinimalIR = R"MLIR(
module {
  hw.module @test() {
    hw.output
  }
}
)MLIR";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kMinimalIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  auto &states =
      LLHDProcessInterpreterTest::getProcessStates(interpreter);

  // Insert an initial entry and take a reference to it.
  const ProcessId sentinelId = 42;
  states[sentinelId].halted = true;
  states[sentinelId].totalSteps = 12345;
  ProcessExecutionState &ref = states[sentinelId];

  // Verify the reference is valid before mass insertion.
  ASSERT_TRUE(ref.halted);
  ASSERT_EQ(ref.totalSteps, 12345u);

  // Insert many additional entries.  With the old DenseMap, this would
  // trigger one or more rehashes (DenseMap grows at 75% load, with an
  // initial bucket count of 64, so ~48 inserts would trigger the first
  // rehash).  We insert far more to be thorough.
  for (ProcessId i = 1000; i < 2000; ++i) {
    states[i].totalSteps = i;
  }

  // The critical check: the reference obtained before the insertions
  // must still point to valid, unchanged data.  With DenseMap this
  // would read garbage or segfault; with std::map it is guaranteed safe.
  EXPECT_TRUE(ref.halted)
      << "Reference to processStates entry was invalidated by insertion";
  EXPECT_EQ(ref.totalSteps, 12345u)
      << "Reference to processStates entry was invalidated by insertion";

  // Also verify that erasing other entries does not invalidate the reference.
  for (ProcessId i = 1000; i < 1500; ++i) {
    states.erase(i);
  }
  EXPECT_TRUE(ref.halted)
      << "Reference to processStates entry was invalidated by erasure";
  EXPECT_EQ(ref.totalSteps, 12345u)
      << "Reference to processStates entry was invalidated by erasure";
}

//===----------------------------------------------------------------------===//
// Layout Conversion Tests (convertLLVMToHWLayout / convertHWToLLVMLayout)
//===----------------------------------------------------------------------===//

// Test that convertLLVMToHWLayout correctly reverses field order for a flat
// struct, and that convertHWToLLVMLayout is the inverse.
//
// HW struct  !hw.struct<a: i8, b: i4>  stores a in high bits, b in low bits:
//   bits [11..4] = a, bits [3..0] = b
//
// LLVM struct !llvm.struct<(i8, i4)> stores field 0 in low bits, field 1 high:
//   bits [7..0] = field0 (a), bits [11..8] = field1 (b)
TEST(LLHDProcessInterpreterToolTest, ConvertLLVMToHWLayoutFlatStruct) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<llhd::LLHDDialect>();
  context.loadDialect<LLVM::LLVMDialect>();

  // We need a minimal module to construct the interpreter.
  static constexpr llvm::StringLiteral kMinimalIR = R"MLIR(
module {
  hw.module @test() {
    hw.output
  }
}
)MLIR";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kMinimalIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  // Build types:
  //   HW:   !hw.struct<a: i8, b: i4>   (12 bits total)
  //   LLVM: !llvm.struct<(i8, i4)>      (12 bits total)
  auto i8Ty = IntegerType::get(&context, 8);
  auto i4Ty = IntegerType::get(&context, 4);

  auto hwStructTy = hw::StructType::get(
      &context,
      {hw::StructType::FieldInfo{StringAttr::get(&context, "a"), i8Ty},
       hw::StructType::FieldInfo{StringAttr::get(&context, "b"), i4Ty}});

  auto llvmStructTy =
      LLVM::LLVMStructType::getLiteral(&context, {i8Ty, i4Ty});

  // LLVM layout: field0=a=0xAB (bits 7..0), field1=b=0x3 (bits 11..8)
  // Combined 12-bit LLVM value: 0x3AB
  APInt llvmValue(12, 0x3AB);

  // Expected HW layout: a in high bits, b in low bits
  // bits [11..4] = 0xAB, bits [3..0] = 0x3
  // Combined: 0xAB3
  APInt expectedHW(12, 0xAB3);

  APInt hwResult = LLHDProcessInterpreterTest::convertLLVMToHWLayout(
      interpreter, llvmValue, llvmStructTy, hwStructTy);
  EXPECT_EQ(hwResult, expectedHW)
      << "convertLLVMToHWLayout produced 0x"
      << llvm::utohexstr(hwResult.getZExtValue()) << " but expected 0x"
      << llvm::utohexstr(expectedHW.getZExtValue());

  // Verify round-trip: HW -> LLVM should recover the original.
  APInt llvmRoundTrip = LLHDProcessInterpreterTest::convertHWToLLVMLayout(
      interpreter, hwResult, hwStructTy, llvmStructTy);
  EXPECT_EQ(llvmRoundTrip, llvmValue)
      << "Round-trip HW->LLVM produced 0x"
      << llvm::utohexstr(llvmRoundTrip.getZExtValue()) << " but expected 0x"
      << llvm::utohexstr(llvmValue.getZExtValue());
}

// Test nested struct layout conversion.
//
// Outer HW:   !hw.struct<inner: !hw.struct<x: i4, y: i4>, z: i8>
//   Total 16 bits.  HW layout: inner in [15..8], z in [7..0].
//   Within inner: x in [15..12], y in [11..8].
//
// Outer LLVM: !llvm.struct<(!llvm.struct<(i4, i4)>, i8)>
//   LLVM layout: inner in [7..0], z in [15..8].
//   Within inner: x in [3..0], y in [7..4].
TEST(LLHDProcessInterpreterToolTest, ConvertLLVMToHWLayoutNestedStruct) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<llhd::LLHDDialect>();
  context.loadDialect<LLVM::LLVMDialect>();

  static constexpr llvm::StringLiteral kMinimalIR = R"MLIR(
module {
  hw.module @test() {
    hw.output
  }
}
)MLIR";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kMinimalIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  auto i4Ty = IntegerType::get(&context, 4);
  auto i8Ty = IntegerType::get(&context, 8);

  // Inner types
  auto hwInnerTy = hw::StructType::get(
      &context,
      {hw::StructType::FieldInfo{StringAttr::get(&context, "x"), i4Ty},
       hw::StructType::FieldInfo{StringAttr::get(&context, "y"), i4Ty}});
  auto llvmInnerTy =
      LLVM::LLVMStructType::getLiteral(&context, {i4Ty, i4Ty});

  // Outer types
  auto hwOuterTy = hw::StructType::get(
      &context,
      {hw::StructType::FieldInfo{StringAttr::get(&context, "inner"),
                                 hwInnerTy},
       hw::StructType::FieldInfo{StringAttr::get(&context, "z"), i8Ty}});
  auto llvmOuterTy =
      LLVM::LLVMStructType::getLiteral(&context, {llvmInnerTy, i8Ty});

  // Choose concrete field values:
  //   x = 0xA (4 bits), y = 0x5 (4 bits), z = 0xBC (8 bits)
  //
  // LLVM layout (low-to-high):
  //   bits [3..0]   = x = 0xA   (inner field 0)
  //   bits [7..4]   = y = 0x5   (inner field 1)
  //   bits [15..8]  = z = 0xBC  (outer field 1)
  //   Combined: 0xBC5A
  APInt llvmValue(16, 0xBC5A);

  // Expected HW layout (high-to-low):
  //   Outer: inner in [15..8], z in [7..0].
  //   Within inner: x in [15..12], y in [11..8].
  //   bits [15..12] = x = 0xA
  //   bits [11..8]  = y = 0x5
  //   bits [7..0]   = z = 0xBC
  //   Combined: 0xA5BC
  APInt expectedHW(16, 0xA5BC);

  APInt hwResult = LLHDProcessInterpreterTest::convertLLVMToHWLayout(
      interpreter, llvmValue, llvmOuterTy, hwOuterTy);
  EXPECT_EQ(hwResult, expectedHW)
      << "Nested struct: convertLLVMToHWLayout produced 0x"
      << llvm::utohexstr(hwResult.getZExtValue()) << " but expected 0x"
      << llvm::utohexstr(expectedHW.getZExtValue());

  // Verify round-trip.
  APInt llvmRoundTrip = LLHDProcessInterpreterTest::convertHWToLLVMLayout(
      interpreter, hwResult, hwOuterTy, llvmOuterTy);
  EXPECT_EQ(llvmRoundTrip, llvmValue)
      << "Nested struct round-trip: convertHWToLLVMLayout produced 0x"
      << llvm::utohexstr(llvmRoundTrip.getZExtValue()) << " but expected 0x"
      << llvm::utohexstr(llvmValue.getZExtValue());
}

// Test that plain integer types pass through unchanged (identity case).
TEST(LLHDProcessInterpreterToolTest, ConvertLayoutPlainIntegerIdentity) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<llhd::LLHDDialect>();
  context.loadDialect<LLVM::LLVMDialect>();

  static constexpr llvm::StringLiteral kMinimalIR = R"MLIR(
module {
  hw.module @test() {
    hw.output
  }
}
)MLIR";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kMinimalIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  auto i32Ty = IntegerType::get(&context, 32);
  APInt value(32, 0xDEADBEEF);

  // When both types are plain integers, conversion should be identity.
  APInt result = LLHDProcessInterpreterTest::convertLLVMToHWLayout(
      interpreter, value, i32Ty, i32Ty);
  EXPECT_EQ(result, value);

  APInt reverse = LLHDProcessInterpreterTest::convertHWToLLVMLayout(
      interpreter, value, i32Ty, i32Ty);
  EXPECT_EQ(reverse, value);
}

// End-to-end test: probe a nested-struct alloca-backed signal through
// an LLVM load/unrealized_conversion_cast path and verify the final
// HW-layout signal value has the correct bit ordering.
//
// The IR below:
//  1. Creates a signal with HW type
//     !hw.struct<a: i8, b: !hw.struct<c: i4, d: i4>>
//     initialized to {a=0xFF, b={c=0xA, d=0x5}} => HW bits = 0xFFA5
//  2. A process loads via LLVM ptr (gets LLVM layout), then stores to an
//     output signal.  The interpreter's probe handler should apply
//     convertLLVMToHWLayout so the output signal receives the correct
//     HW bits.
static constexpr llvm::StringLiteral kNestedStructProbeIR = R"MLIR(
module {
  hw.module @test() {
    %init = hw.aggregate_constant [255 : i8, [10 : i4, 5 : i4]] : !hw.struct<a: i8, b: !hw.struct<c: i4, d: i4>>
    %zero = hw.aggregate_constant [0 : i8, [0 : i4, 0 : i4]] : !hw.struct<a: i8, b: !hw.struct<c: i4, d: i4>>

    %sig_in = llhd.sig name "sig_in" %init : !hw.struct<a: i8, b: !hw.struct<c: i4, d: i4>>
    %sig_out = llhd.sig name "sig_out" %zero : !hw.struct<a: i8, b: !hw.struct<c: i4, d: i4>>

    llhd.process {
      %ptr_in = builtin.unrealized_conversion_cast %sig_in : !llhd.ref<!hw.struct<a: i8, b: !hw.struct<c: i4, d: i4>>> to !llvm.ptr
      %loaded = llvm.load %ptr_in : !llvm.ptr -> !llvm.struct<(i8, !llvm.struct<(i4, i4)>)>
      %ptr_out = builtin.unrealized_conversion_cast %sig_out : !llhd.ref<!hw.struct<a: i8, b: !hw.struct<c: i4, d: i4>>> to !llvm.ptr
      llvm.store %loaded, %ptr_out : !llvm.struct<(i8, !llvm.struct<(i4, i4)>)>, !llvm.ptr
      llhd.halt
    }

    hw.output
  }
}
)MLIR";

TEST(LLHDProcessInterpreterToolTest, ProbeNestedStructAllocaSignal) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect>();
  context.loadDialect<llhd::LLHDDialect>();
  context.loadDialect<LLVM::LLVMDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kNestedStructProbeIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbols(*module);
  auto hwModule = symbols.lookup<hw::HWModuleOp>("test");
  ASSERT_TRUE(hwModule);

  ProcessScheduler scheduler;
  LLHDProcessInterpreter interpreter(scheduler);
  ASSERT_TRUE(succeeded(interpreter.initialize(hwModule)));

  llhd::SignalOp sigInOp;
  llhd::SignalOp sigOutOp;
  hwModule.walk([&](llhd::SignalOp op) {
    auto nameAttr = op.getName();
    if (!nameAttr)
      return;
    if (*nameAttr == "sig_in")
      sigInOp = op;
    else if (*nameAttr == "sig_out")
      sigOutOp = op;
  });
  ASSERT_TRUE(sigInOp);
  ASSERT_TRUE(sigOutOp);

  // Execute the process (load from sig_in, store to sig_out).
  ASSERT_TRUE(scheduler.executeDeltaCycle());

  SignalId sigInId = interpreter.getSignalId(sigInOp.getResult());
  SignalId sigOutId = interpreter.getSignalId(sigOutOp.getResult());
  ASSERT_NE(sigInId, 0u);
  ASSERT_NE(sigOutId, 0u);

  const SignalValue &inVal = scheduler.getSignalValue(sigInId);
  const SignalValue &outVal = scheduler.getSignalValue(sigOutId);

  // The output should match the input: the load+store through LLVM types
  // should preserve the value because convertLLVMToHWLayout is applied
  // on the probe and convertHWToLLVMLayout on the drive.
  EXPECT_EQ(inVal.getAPInt(), outVal.getAPInt())
      << "sig_out should equal sig_in after load+store through LLVM types."
      << " sig_in=0x" << llvm::utohexstr(inVal.getAPInt().getZExtValue())
      << " sig_out=0x" << llvm::utohexstr(outVal.getAPInt().getZExtValue());
}

} // namespace
