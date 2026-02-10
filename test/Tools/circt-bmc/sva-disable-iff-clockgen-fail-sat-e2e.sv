// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc -b 10 --ignore-asserts-until=1 --module=sva_disable_iff_clockgen_fail_sat - | \
// RUN:   FileCheck %s --check-prefix=JIT
// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 10 --ignore-asserts-until=1 --module=sva_disable_iff_clockgen_fail_sat - | \
// RUN:   FileCheck %s --check-prefix=SMTLIB
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_disable_iff_clockgen_fail_sat_clk_gen(
    input logic rst,
    input logic clk,
    output logic out);
  initial out = 1'b0;

  always @(posedge clk or posedge rst) begin
    if (rst)
      out <= 1'b0;
    else
      out <= 1'b1;
  end
endmodule

module sva_disable_iff_clockgen_fail_sat();
  logic rst;
  logic clk;
  logic out;

  sva_disable_iff_clockgen_fail_sat_clk_gen dut(.rst(rst), .clk(clk), .out(out));

  initial begin
    clk = 1'b0;
    rst = 1'b1;
  end

  property p;
    @(posedge clk) disable iff (~rst) out;
  endproperty
  assert property (p);

  initial begin
    forever begin
      #50 clk = ~clk;
    end
  end
endmodule

// JIT: BMC_RESULT=SAT
// SMTLIB: BMC_RESULT=SAT
