// REQUIRES: slang
// RUN: env CIRCT_VERILOG=circt-verilog CIRCT_BMC=circt-bmc Z3_LIB=/dev/null \
// RUN:   NO_PROPERTY_AS_SKIP=1 BMC_SMOKE_ONLY=1 CIRCT_BMC_ARGS=--emit-mlir \
// RUN:   TEST_FILTER='.*' %S/../../../utils/run_yosys_sva_circt_bmc.sh \
// RUN:   %S/Inputs/yosys-sva-mini-no-property | FileCheck %s
// CHECK: SKIP(no-property): no-property
// CHECK: yosys SVA summary: 1 tests, failures=0, xfail=0, xpass=0, skipped=1
