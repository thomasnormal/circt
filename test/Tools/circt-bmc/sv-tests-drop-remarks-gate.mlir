// REQUIRES: slang, uvm
// RUN: env CIRCT_VERILOG=circt-verilog CIRCT_BMC=circt-bmc Z3_LIB=/dev/null \
// RUN:   OUT=%t/results.txt TAG_REGEX='uvm' TEST_FILTER='uvm-local-var-mini$' \
// RUN:   ALLOW_MULTI_CLOCK=1 BMC_SMOKE_ONLY=1 CIRCT_BMC_ARGS=--emit-mlir \
// RUN:   UVM_PATH=%S/../../../lib/Runtime/uvm-core/src BOUND=1 IGNORE_ASSERTS_UNTIL=0 \
// RUN:   %S/../../../utils/run_sv_tests_circt_bmc.sh \
// RUN:   %S/Inputs/sv-tests-mini-uvm | FileCheck %s --check-prefix=SUMMARY
// RUN: printf '#!/usr/bin/env bash\nset -euo pipefail\ncirct-verilog "$@"\necho "warning: synthetic node will be dropped during lowering" >&2\n' > %t/wrap-circt-verilog.sh
// RUN: chmod +x %t/wrap-circt-verilog.sh
// RUN: not env CIRCT_VERILOG=%t/wrap-circt-verilog.sh CIRCT_BMC=circt-bmc Z3_LIB=/dev/null \
// RUN:   OUT=%t/results-gate.txt TAG_REGEX='uvm' TEST_FILTER='uvm-local-var-mini$' \
// RUN:   ALLOW_MULTI_CLOCK=1 BMC_SMOKE_ONLY=1 CIRCT_BMC_ARGS=--emit-mlir \
// RUN:   FAIL_ON_DROP_REMARKS=1 \
// RUN:   UVM_PATH=%S/../../../lib/Runtime/uvm-core/src BOUND=1 IGNORE_ASSERTS_UNTIL=0 \
// RUN:   %S/../../../utils/run_sv_tests_circt_bmc.sh \
// RUN:   %S/Inputs/sv-tests-mini-uvm 2>&1 | FileCheck %s --check-prefix=GATE

// SUMMARY: sv-tests SVA summary: total=1 pass=1 fail=0 xfail=0 xpass=0 error=0 skip={{[0-9]+}} unknown=0 timeout=0
// SUMMARY: sv-tests dropped-syntax summary: drop_remark_cases=0

// GATE: sv-tests dropped-syntax summary: drop_remark_cases=1
// GATE: FAIL_ON_DROP_REMARKS triggered: drop_remark_cases=1
