// RUN: env CIRCT_SIM_DYNAMIC_STRING_MAX_ENTRIES=2 CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1 circt-sim %s 2>&1 | FileCheck %s
//
// Verify dynamic string registry capping telemetry.
//
// CHECK: latest=3
// CHECK: [circt-sim] Dynamic string registry: entries=2 max_entries=2 registrations=3 updates=0 evictions=1

llvm.func @__moore_int_to_string(i64) -> !llvm.struct<(ptr, i64)>

hw.module @top() {
  %c1_i64 = hw.constant 1 : i64
  %c2_i64 = hw.constant 2 : i64
  %c3_i64 = hw.constant 3 : i64
  %fmt_prefix = sim.fmt.literal "latest="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %s1 = llvm.call @__moore_int_to_string(%c1_i64) : (i64) -> !llvm.struct<(ptr, i64)>
    %s2 = llvm.call @__moore_int_to_string(%c2_i64) : (i64) -> !llvm.struct<(ptr, i64)>
    %s3 = llvm.call @__moore_int_to_string(%c3_i64) : (i64) -> !llvm.struct<(ptr, i64)>
    %latest = sim.fmt.dyn_string %s3 : !llvm.struct<(ptr, i64)>
    %line = sim.fmt.concat (%fmt_prefix, %latest, %fmt_nl)
    sim.proc.print %line
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
