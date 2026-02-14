// RUN: circt-opt --hw-stopat-symbolic='targets=*u_state_regs.state_o' %s | FileCheck %s --check-prefix=STOPAT
// RUN: not circt-opt --hw-stopat-symbolic='targets=*missing.state_o' %s 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: circt-opt --hw-stopat-symbolic='targets=*missing.state_o allow-unmatched=true' %s | FileCheck %s --check-prefix=ALLOW
// RUN: circt-opt --hw-stopat-symbolic='targets=*u_mid.u_state_regs.state_o' %s | FileCheck %s --check-prefix=HIER
// RUN: not circt-opt --hw-stopat-symbolic='targets=*u_mid.u_state_regs_amb.state_o' %s 2>&1 | FileCheck %s --check-prefix=AMBIG
// RUN: circt-opt --hw-stopat-symbolic='targets=*u_mid.u_state_regs_amb.state_o allow-unmatched=true' %s | FileCheck %s --check-prefix=AMBIG-ALLOW

module {
  hw.module.extern @child(out state_o: i8, out other: i1)
  hw.module.extern @leaf_unique(out state_o: i8)
  hw.module.extern @leaf_amb(out state_o: i8)

  hw.module @top(out out: i8) {
    %state, %other = hw.instance "u_state_regs" @child() -> (state_o: i8, other: i1)
    hw.output %state : i8
  }

  hw.module @mid_unique(out state_o: i8) {
    %state = hw.instance "u_state_regs" @leaf_unique() -> (state_o: i8)
    hw.output %state : i8
  }

  hw.module @top_hier(out out: i8) {
    %state = hw.instance "u_mid" @mid_unique() -> (state_o: i8)
    hw.output %state : i8
  }

  hw.module @mid_amb(out state_o: i8) {
    %state = hw.instance "u_state_regs_amb" @leaf_amb() -> (state_o: i8)
    hw.output %state : i8
  }

  hw.module @top_a(out out: i8) {
    %state = hw.instance "u_mid" @mid_amb() -> (state_o: i8)
    hw.output %state : i8
  }

  hw.module @top_b(out out: i8) {
    %state = hw.instance "u_mid" @mid_amb() -> (state_o: i8)
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

// HIER: hw.module @mid_unique
// HIER: %[[STATE:.*]] = hw.instance "u_state_regs" @leaf_unique() -> (state_o: i8)
// HIER: %[[SYM:.*]] = verif.symbolic_value : i8
// HIER: hw.output %[[SYM]] : i8

// AMBIG: error: unmatched stopat selectors: u_mid.u_state_regs_amb.state_o

// AMBIG-ALLOW: hw.module @mid_amb
// AMBIG-ALLOW: %[[STATE:.*]] = hw.instance "u_state_regs_amb" @leaf_amb() -> (state_o: i8)
// AMBIG-ALLOW: hw.output %[[STATE]] : i8
