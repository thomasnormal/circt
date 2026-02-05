// REQUIRES: slang
// XFAIL: *
// NOTE: UVM tests currently fail during lowering
// RUN: env CIRCT_VERILOG=circt-verilog CIRCT_BMC=circt-bmc Z3_LIB=/dev/null \
// RUN:   OUT=%t/results.txt TEST_FILTER=uvm \
// RUN:   INCLUDE_UVM_TAGS=1 BMC_SMOKE_ONLY=1 CIRCT_BMC_ARGS=--emit-mlir \
// RUN:   UVM_PATH=%S/../../../lib/Runtime/uvm BOUND=1 IGNORE_ASSERTS_UNTIL=0 \
// RUN:   %S/../../../utils/run_sv_tests_circt_bmc.sh \
// RUN:   %S/Inputs/sv-tests-mini-uvm | FileCheck %s
// CHECK: sv-tests SVA summary: total=6 pass=6 fail=0 xfail=0 xpass=0 error=0 skip={{[0-9]+}}
