// RUN: env CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1 circt-sim %s 2>&1 | FileCheck %s
//
// Test function symbol lookup cache statistics for llvm.call operations.
//
// CHECK: result=42
// CHECK: [circt-sim] function symbol lookup cache: entries=
// CHECK-SAME: hits=
// CHECK-SAME: misses=
// CHECK-SAME: negative_hits=

module {
  llvm.func @get_value() -> i32 {
    %c42 = llvm.mlir.constant(42 : i32) : i32
    llvm.return %c42 : i32
  }

  hw.module @top() {
    %fmt_prefix = sim.fmt.literal "result="
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      cf.br ^entry
    ^entry:
      %v0 = llvm.call @get_value() : () -> i32
      %v1 = llvm.call @get_value() : () -> i32

      %v1_hw = builtin.unrealized_conversion_cast %v1 : i32 to i32
      %fmt_val = sim.fmt.dec %v1_hw : i32
      %line = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
      sim.proc.print %line
      llhd.halt
    }
    hw.output
  }
}
