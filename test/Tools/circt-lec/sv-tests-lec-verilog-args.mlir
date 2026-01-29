// REQUIRES: slang
// RUN: env CIRCT_VERILOG=circt-verilog CIRCT_OPT=circt-opt CIRCT_LEC=circt-lec \
// RUN:   OUT=%t/results.txt TAG_REGEX='16\.10' TEST_FILTER=macro$ \
// RUN:   FORCE_LEC=1 LEC_SMOKE_ONLY=1 CIRCT_LEC_ARGS=--emit-mlir \
// RUN:   CIRCT_VERILOG_ARGS=-DENABLE \
// RUN:   %S/../../../utils/run_sv_tests_circt_lec.sh \
// RUN:   %S/Inputs/sv-tests-mini | FileCheck %s
// CHECK: sv-tests LEC summary: total=1 pass=1 fail=0 error=0 skip={{[0-9]+}}
