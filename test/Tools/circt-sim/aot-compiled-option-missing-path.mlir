// RUN: not circt-sim %s --compiled --max-time=1 2>&1 | FileCheck %s

// Regression: `--compiled` requires an explicit path. If omitted, the next
// option token must not be treated as a shared-object path.
// CHECK: [circt-sim] error: missing compiled module path for '--compiled' (got option token '--max-time=1'). Use --compiled=<path>

hw.module @top() {
  llhd.process {
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
