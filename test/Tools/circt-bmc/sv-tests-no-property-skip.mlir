// REQUIRES: slang
// RUN: env CIRCT_VERILOG=circt-verilog CIRCT_BMC=circt-bmc Z3_LIB=/dev/null \
// RUN:   OUT=%t/results.txt TAG_REGEX='16\.10' TEST_FILTER=no-property \
// RUN:   FORCE_BMC=1 BMC_SMOKE_ONLY=1 CIRCT_BMC_ARGS=--emit-mlir \
// RUN:   NO_PROPERTY_AS_SKIP=1 BOUND=1 IGNORE_ASSERTS_UNTIL=0 \
// RUN:   %S/../../../utils/run_sv_tests_circt_bmc.sh \
// RUN:   %S/Inputs/sv-tests-mini-no-property | FileCheck %s
// CHECK: sv-tests SVA summary: total=1 pass=0 fail=0 xfail=0 xpass=0 error=0 skip=1
