// RUN: not circt-bmc -b 1 --shared-libs=/tmp/libz3.so --module top %s 2>&1 | FileCheck %s
// CHECK: Unknown command line argument '--shared-libs=/tmp/libz3.so'

module {
  hw.module @top(in %clk : i1) {
    %c1 = hw.constant true
    verif.assert %c1 : i1
    hw.output
  }
}
