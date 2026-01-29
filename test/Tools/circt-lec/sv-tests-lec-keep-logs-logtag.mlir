// REQUIRES: slang
// RUN: rm -rf %t/logs && mkdir -p %t/logs
// RUN: env CIRCT_VERILOG=circt-verilog CIRCT_OPT=circt-opt CIRCT_LEC=circt-lec \
// RUN:   OUT=%t/results.txt TAG_REGEX='16\.10' TEST_FILTER=keep-logs \
// RUN:   FORCE_LEC=1 LEC_SMOKE_ONLY=1 CIRCT_LEC_ARGS=--emit-mlir \
// RUN:   KEEP_LOGS_DIR=%t/logs \
// RUN:   %S/../../../utils/run_sv_tests_circt_lec.sh \
// RUN:   %S/Inputs/sv-tests-mini-logtag
// RUN: ls %t/logs | FileCheck %s
// RUN: rm -rf %t/logs
// Output is sorted alphabetically
// CHECK-DAG: dir1__16.10--keep-logs.circt-lec.log
// CHECK-DAG: dir1__16.10--keep-logs.mlir
// CHECK-DAG: dir2__16.10--keep-logs.circt-lec.log
// CHECK-DAG: dir2__16.10--keep-logs.mlir
