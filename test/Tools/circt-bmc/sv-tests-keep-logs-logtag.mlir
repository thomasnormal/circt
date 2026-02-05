// REQUIRES: slang
// RUN: rm -rf %t/logs && mkdir -p %t/logs
// RUN: env CIRCT_VERILOG=circt-verilog CIRCT_BMC=circt-bmc Z3_LIB=/dev/null \
// RUN:   OUT=%t/results.txt TAG_REGEX='16\.10' TEST_FILTER=keep-logs \
// RUN:   FORCE_BMC=1 BMC_SMOKE_ONLY=1 CIRCT_BMC_ARGS=--emit-mlir \
// RUN:   KEEP_LOGS_DIR=%t/logs BOUND=1 IGNORE_ASSERTS_UNTIL=0 \
// RUN:   %S/../../../utils/run_sv_tests_circt_bmc.sh \
// RUN:   %S/Inputs/sv-tests-mini-logtag
// RUN: ls %t/logs | FileCheck %s
// RUN: rm -rf %t/logs
// Output is sorted alphabetically, so .circt-bmc.log comes before .circt-verilog.log before .mlir
// CHECK-DAG: dir1__16.10--keep-logs.circt-bmc.log
// CHECK-DAG: dir1__16.10--keep-logs.mlir
// CHECK-DAG: dir2__16.10--keep-logs.circt-bmc.log
// CHECK-DAG: dir2__16.10--keep-logs.mlir
