// REQUIRES: slang, uvm
// NOTE: UVM tests now pass during lowering
// RUN: env CIRCT_VERILOG=circt-verilog CIRCT_BMC=circt-bmc Z3_LIB=/dev/null \
// RUN:   OUT=%t/results.txt TAG_REGEX='uvm' TEST_FILTER=uvm-parsing$ \
// RUN:   FORCE_BMC=1 BMC_SMOKE_ONLY=1 CIRCT_BMC_ARGS=--emit-mlir \
// RUN:   UVM_PATH=%S/../../../lib/Runtime/uvm BOUND=1 IGNORE_ASSERTS_UNTIL=0 \
// RUN:   %S/../../../utils/run_sv_tests_circt_bmc.sh \
// RUN:   %S/Inputs/sv-tests-mini | FileCheck %s
// CHECK: sv-tests SVA summary: total=1 pass=1 fail=0 xfail=0 xpass=0 error=0 skip={{[0-9]+}}
