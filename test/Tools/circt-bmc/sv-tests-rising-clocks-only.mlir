// REQUIRES: slang
// RUN: env CIRCT_VERILOG=circt-verilog CIRCT_BMC=%S/Inputs/fake-bmc.sh \
// RUN:   RISING_CLOCKS_ONLY=1 Z3_LIB=/dev/null OUT=%t/results.txt \
// RUN:   %S/../../../utils/run_sv_tests_circt_bmc.sh %S/Inputs/sv-tests-mini | FileCheck %s
// CHECK: sv-tests SVA summary: total=7 pass=6 fail=1 xfail=0 xpass=0 error=0 skip={{[0-9]+}}
