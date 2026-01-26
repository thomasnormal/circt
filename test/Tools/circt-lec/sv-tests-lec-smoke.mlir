// REQUIRES: slang
// RUN: env CIRCT_VERILOG=circt-verilog CIRCT_OPT=circt-opt CIRCT_LEC=circt-lec \
// RUN:   OUT=%t/results.txt TAG_REGEX='16\.10' TEST_FILTER=parsing \
// RUN:   FORCE_LEC=1 LEC_SMOKE_ONLY=1 CIRCT_LEC_ARGS=--emit-mlir \
// RUN:   %S/../../../utils/run_sv_tests_circt_lec.sh \
// RUN:   %S/Inputs/sv-tests-mini | FileCheck %s
// CHECK: sv-tests LEC summary: total=2 pass=1 fail=0 error=1 skip={{[0-9]+}}
