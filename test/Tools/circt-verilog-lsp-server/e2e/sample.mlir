// Sample CIRCT MLIR file for LSP testing
hw.module @counter(in %clk : i1, in %rst_n : i1, in %enable : i1, out count : i8) {
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %reg = seq.firreg %next clock %clk reset sync %rst_n, %c0_i8 : i8
  %next = comb.add %reg, %c1_i8 : i8
  %out = comb.mux %enable, %next, %reg : i8
  hw.output %out : i8
}

hw.module @top(in %clk : i1, in %rst_n : i1) {
  %true = hw.constant 1 : i1
  %count = hw.instance "u_counter" @counter(clk: %clk: i1, rst_n: %rst_n: i1, enable: %true: i1) -> (count: i8)
}
