// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test that interface instantiation with inout ports works correctly.
// This mimics the I3C AVIP pattern where an interface has direct inout ports.

// CHECK-LABEL: moore.interface @i3c_if
interface i3c_if(input pclk, input areset, inout scl, inout sda);
  logic scl_i;
  logic scl_o;
  logic scl_oen;
  logic sda_i;
  logic sda_o;
  logic sda_oen;

  // Tri-state buffer implementation
  assign scl = (scl_oen) ? scl_o : 1'bz;
  assign sda = (sda_oen) ? sda_o : 1'bz;

  // Used for sampling the interface signals
  assign scl_i = scl;
  assign sda_i = sda;
endinterface : i3c_if

// CHECK-LABEL: moore.module @hdl_top
module hdl_top;
  bit clk;
  bit rst;

  wire I3C_SCL;
  wire I3C_SDA;

  initial begin
    clk = 1'b0;
    forever #10 clk = ~clk;
  end

  initial begin
    rst = 1'b1;
    repeat (2) @(posedge clk);
    rst = 1'b0;
    repeat (2) @(posedge clk);
    rst = 1'b1;
  end

  // Interface instantiation with inout ports
  // CHECK: moore.interface.instance
  i3c_if intf_controller(.pclk(clk),
                         .areset(rst),
                         .scl(I3C_SCL),
                         .sda(I3C_SDA));

  // Second interface instance sharing same wires (bidirectional bus)
  i3c_if intf_target(.pclk(clk),
                     .areset(rst),
                     .scl(I3C_SCL),
                     .sda(I3C_SDA));

  // Pullup for I3C interface
  pullup p1 (I3C_SCL);
  pullup p2 (I3C_SDA);

endmodule : hdl_top
