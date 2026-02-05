// REQUIRES: slang
// RUN: env CIRCT_VERILOG=circt-verilog CIRCT_BMC=circt-bmc \
// RUN:   BMC_SMOKE_ONLY=1 CIRCT_BMC_ARGS=--emit-mlir \
// RUN:   TEST_FILTER=basic00 \
// RUN:   %S/../../../utils/run_yosys_sva_circt_bmc.sh \
// RUN:   %S/Inputs/yosys-sva-mini | FileCheck %s
// CHECK: yosys SVA summary: 1 tests, failures=0, skipped=0
