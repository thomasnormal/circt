// RUN: env CIRCT_SIM_UVM_JIT_HOT_THRESHOLD=1 CIRCT_SIM_UVM_JIT_PROMOTION_BUDGET=1 CIRCT_SIM_UVM_JIT_TRACE_PROMOTIONS=1 circt-sim %s 2>&1 | FileCheck %s
//
// Verify hotness-gated UVM fast-path promotion hook marks exact registry
// actions as JIT promotion candidates.
//
// CHECK: [circt-sim] UVM JIT promotion candidate: registry.func.call.get_report_verbosity hits=1 threshold=1 budget_remaining=0
// CHECK: verbosity = 200

module {
  // If this body executes, it returns 0. Fast-path should return 200.
  func.func private @"uvm_pkg::uvm_report_object::get_report_verbosity_level"(
      %self: !llvm.ptr) -> i32 {
    %zero = arith.constant 0 : i32
    return %zero : i32
  }

  hw.module @main() {
    %fmtPrefix = sim.fmt.literal "verbosity = "
    %fmtNl = sim.fmt.literal "\0A"

    llhd.process {
      %self64 = arith.constant 4096 : i64
      %self = llvm.inttoptr %self64 : i64 to !llvm.ptr

      %result = func.call @"uvm_pkg::uvm_report_object::get_report_verbosity_level"(%self) :
          (!llvm.ptr) -> i32
      %dec = sim.fmt.dec %result signed : i32
      %line = sim.fmt.concat (%fmtPrefix, %dec, %fmtNl)
      sim.proc.print %line
      llhd.halt
    }
    hw.output
  }
}
