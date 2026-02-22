// RUN: circt-sim %s --top top --max-time=1000000 | FileCheck %s

// Regression: sim.fstring values passed through block arguments (cf.br,
// cf.cond_br, llhd.wait) must be resolved back to their defining ops.
//
// The bug: evaluateFormatString received a BlockArgument (no defining op)
// and returned "<unknown>" instead of tracing back through predecessors
// to find the original sim.fmt.literal / sim.fmt.concat.
//
// Fix: traceFStringBlockArg performs DFS through predecessor terminators
// (cf.br, cf.cond_br, llhd.wait) to find the original defining op.

// CHECK: VIA_BR=hello
// CHECK: VIA_COND=world

module {
  hw.module @top() {
    %fmt_hello = sim.fmt.literal "hello"
    %fmt_world = sim.fmt.literal "world"
    %fmt_br_label = sim.fmt.literal "VIA_BR="
    %fmt_cond_label = sim.fmt.literal "VIA_COND="
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      // Test 1: fstring through cf.br block argument.
      cf.br ^br_target(%fmt_hello : !sim.fstring)

    ^br_target(%arg0 : !sim.fstring):
      %out0 = sim.fmt.concat (%fmt_br_label, %arg0, %fmt_nl)
      sim.proc.print %out0

      // Test 2: fstring through cf.cond_br block argument (true branch).
      %true = llvm.mlir.constant(true) : i1
      cf.cond_br %true, ^cond_true(%fmt_world : !sim.fstring), ^cond_false(%fmt_hello : !sim.fstring)

    ^cond_true(%arg1 : !sim.fstring):
      %out1 = sim.fmt.concat (%fmt_cond_label, %arg1, %fmt_nl)
      sim.proc.print %out1
      cf.br ^done

    ^cond_false(%arg2 : !sim.fstring):
      %out2 = sim.fmt.concat (%fmt_cond_label, %arg2, %fmt_nl)
      sim.proc.print %out2
      cf.br ^done

    ^done:
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
