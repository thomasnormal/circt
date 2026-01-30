// REQUIRES: slang
// RUN: env CIRCT_VERILOG=circt-verilog CIRCT_BMC=/bin/false Z3_LIB=/dev/null \
// RUN:   OUT=%t/results.txt TEST_FILTER='^[0-9]+\.[0-9]+--parsing$' %S/../../../utils/run_sv_tests_circt_bmc.sh \
// RUN:   %S/Inputs/sv-tests-mini | FileCheck %s
// CHECK: sv-tests SVA summary: total=1 pass=1 fail=0 xfail=0 xpass=0 error=0 skip={{[0-9]+}}
