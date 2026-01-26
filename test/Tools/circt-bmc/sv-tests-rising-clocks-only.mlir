// REQUIRES: slang
// RUN: env CIRCT_VERILOG=circt-verilog CIRCT_BMC=%S/Inputs/fake-bmc.sh \
// RUN:   RISING_CLOCKS_ONLY=1 Z3_LIB=/dev/null OUT=%t/results.txt \
// RUN:   %S/../../../utils/run_sv_tests_circt_bmc.sh %S/Inputs/sv-tests-mini | FileCheck %s
// CHECK: sv-tests SVA summary: total=7 pass=5 fail=0 xfail=0 xpass=1 error=1 skip={{[0-9]+}}
