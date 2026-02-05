// REQUIRES: slang
// RUN: rm -rf %t/logs && mkdir -p %t/logs
// RUN: env CIRCT_VERILOG=circt-verilog CIRCT_BMC=circt-bmc Z3_LIB=/dev/null \
// RUN:   OUT=%t/results.txt TAG_REGEX='16\.10' TEST_FILTER=keep-logs \
// RUN:   FORCE_BMC=1 BMC_SMOKE_ONLY=1 CIRCT_BMC_ARGS=--emit-mlir \
// RUN:   KEEP_LOGS_DIR=%t/logs BOUND=1 IGNORE_ASSERTS_UNTIL=0 \
// RUN:   %S/../../../utils/run_sv_tests_circt_bmc.sh \
// RUN:   %S/Inputs/sv-tests-mini
// RUN: ls %t/logs | FileCheck %s
// RUN: rm -rf %t/logs
// CHECK: keep-logs.mlir
