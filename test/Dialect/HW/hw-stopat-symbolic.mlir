// RUN: circt-opt --hw-stopat-symbolic='targets=*u_state_regs.state_o' %s | FileCheck %s --check-prefix=STOPAT
// RUN: not circt-opt --hw-stopat-symbolic='targets=*missing.state_o' %s 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: circt-opt --hw-stopat-symbolic='targets=*missing.state_o allow-unmatched=true' %s | FileCheck %s --check-prefix=ALLOW

module {
  hw.module.extern @child(out state_o: i8, out other: i1)

  hw.module @top(out out: i8) {
    %state, %other = hw.instance "u_state_regs" @child() -> (state_o: i8, other: i1)
    hw.output %state : i8
  }
}

// STOPAT: hw.module @top
// STOPAT: %[[STATE:.*]], %[[OTHER:.*]] = hw.instance "u_state_regs" @child() -> (state_o: i8, other: i1)
// STOPAT: %[[SYM:.*]] = verif.symbolic_value : i8
// STOPAT: hw.output %[[SYM]] : i8

// MISSING: error: unmatched stopat selectors: missing.state_o

// ALLOW: hw.module @top
// ALLOW: hw.output %{{.*state.*}} : i8
