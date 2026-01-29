// REQUIRES: slang
// XFAIL: *
// NOTE: UVM tests currently fail during lowering
// RUN: env CIRCT_VERILOG=circt-verilog CIRCT_OPT=circt-opt CIRCT_LEC=circt-lec \
// RUN:   OUT=%t/results.txt TEST_FILTER=uvm \
// RUN:   INCLUDE_UVM_TAGS=1 LEC_SMOKE_ONLY=1 CIRCT_LEC_ARGS=--emit-mlir \
// RUN:   UVM_PATH=%S/../../../lib/Runtime/uvm \
// RUN:   %S/../../../utils/run_sv_tests_circt_lec.sh \
// RUN:   %S/Inputs/sv-tests-mini-uvm | FileCheck %s
// CHECK: sv-tests LEC summary: total=6 pass=6 fail=0 error=0 skip={{[0-9]+}}
