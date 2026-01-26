// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test that interfaces with inout ports work like in I3C AVIP.
// This models the i3c_if interface pattern from the I3C AVIP.

// CHECK: moore.interface @i3c_if
interface i3c_if(input pclk, input areset, inout scl, inout sda);
  // Internal signals
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

// Module that uses the interface (like i3c_controller_driver_bfm)
// CHECK: moore.module
module controller_driver_bfm(
  input pclk,
  input areset,
  input scl_i,
  output scl_o,
  output scl_oen,
  input sda_i,
  output sda_o,
  output sda_oen
);
  // Drive default values
  assign scl_o = 1'b1;
  assign scl_oen = 1'b0;
  assign sda_o = 1'b1;
  assign sda_oen = 1'b0;
endmodule

// Module that wraps the interface (like i3c_controller_agent_bfm)
// CHECK: moore.module
module controller_agent_bfm(i3c_if intf);
  // Instantiate driver BFM with interface signals
  controller_driver_bfm drv_bfm(
    .pclk(intf.pclk),
    .areset(intf.areset),
    .scl_i(intf.scl_i),
    .scl_o(intf.scl_o),
    .scl_oen(intf.scl_oen),
    .sda_i(intf.sda_i),
    .sda_o(intf.sda_o),
    .sda_oen(intf.sda_oen)
  );
endmodule

// Top module (like hdl_top)
// CHECK: moore.module @hdl_top
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

  // Interface instantiation with inout ports connected to wires
  // CHECK: moore.interface.instance
  i3c_if intf_controller(.pclk(clk),
                         .areset(rst),
                         .scl(I3C_SCL),
                         .sda(I3C_SDA));

  // Second interface sharing same bidirectional bus
  i3c_if intf_target(.pclk(clk),
                     .areset(rst),
                     .scl(I3C_SCL),
                     .sda(I3C_SDA));

  // Pullups for the I3C bus
  pullup p1 (I3C_SCL);
  pullup p2 (I3C_SDA);

  // BFM agent instance
  // CHECK: moore.instance
  controller_agent_bfm controller_bfm(.intf(intf_controller));
endmodule
