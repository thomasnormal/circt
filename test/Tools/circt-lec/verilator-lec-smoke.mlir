// REQUIRES: slang
// RUN: env CIRCT_VERILOG=circt-verilog CIRCT_OPT=circt-opt CIRCT_LEC=circt-lec \
// RUN:   OUT=%t/results.txt LEC_SMOKE_ONLY=1 CIRCT_LEC_ARGS=--emit-mlir \
// RUN:   %S/../../../utils/run_verilator_verification_circt_lec.sh \
// RUN:   %S/Inputs/verilator-mini | FileCheck %s
// CHECK: verilator-verification LEC summary: total=1 pass=1 fail=0 error=0 skip=0
