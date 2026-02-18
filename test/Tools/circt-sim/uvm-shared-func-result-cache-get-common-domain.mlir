// RUN: env CIRCT_SIM_TRACE_FUNC_CACHE=1 circt-sim --mode=interpret %s 2>&1 | FileCheck %s
//
// Verify cacheable UVM singleton getters can hit the shared function-result
// cache across processes.
//
// CHECK-DAG: p1=1234
// CHECK-DAG: [FUNC-CACHE] shared hit func=get_common_domain
// CHECK-DAG: p2=1234

module {
  func.func private @get_common_domain() -> i64 {
    %v = arith.constant 1234 : i64
    return %v : i64
  }

  hw.module @main() {
    %fmtP1 = sim.fmt.literal "p1="
    %fmtP2 = sim.fmt.literal "p2="
    %fmtNl = sim.fmt.literal "\0A"

    llhd.process {
      %v = func.call @get_common_domain() : () -> i64
      %dec = sim.fmt.dec %v signed : i64
      %line = sim.fmt.concat (%fmtP1, %dec, %fmtNl)
      sim.proc.print %line
      llhd.halt
    }

    llhd.process {
      %v = func.call @get_common_domain() : () -> i64
      %dec = sim.fmt.dec %v signed : i64
      %line = sim.fmt.concat (%fmtP2, %dec, %fmtNl)
      sim.proc.print %line
      llhd.halt
    }

    hw.output
  }
}
