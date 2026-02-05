// RUN: circt-opt %s

module {
  verif.bmc bound 1 num_regs 1 initial_values [unit]
      attributes {bmc_reg_clocks = [""]} init {
    verif.yield
  } loop {
    verif.yield
  } circuit {
  ^bb0(%state: i1):
    %true = hw.constant true
    verif.assert %true : i1
    verif.yield %state : i1
  }
}
