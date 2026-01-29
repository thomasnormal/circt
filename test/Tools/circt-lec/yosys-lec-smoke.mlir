// REQUIRES: slang
// RUN: env CIRCT_VERILOG=circt-verilog CIRCT_OPT=circt-opt CIRCT_LEC=circt-lec \
// RUN:   LEC_SMOKE_ONLY=1 CIRCT_LEC_ARGS=--emit-mlir \
// RUN:   TEST_FILTER=basic00 \
// RUN:   %S/../../../utils/run_yosys_sva_circt_lec.sh \
// RUN:   %S/Inputs/yosys-sva-mini | FileCheck %s
// CHECK: yosys LEC summary: total=1 pass=1 fail=0 error=0 skip=0
