// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test modport member access through interface port parameters.
// This tests the pattern used in AHB AVIP where a task/module has a modport
// parameter (e.g., simple_if.master port) and accesses signals through it
// (e.g., port.clk).

// Simple interface with modports - simulates AHB AVIP pattern
interface simple_if;
  logic clk;
  logic rst_n;
  logic [7:0] data;
  logic valid;
  logic ready;

  modport master(
    input clk,
    input rst_n,
    output data,
    output valid,
    input ready
  );

  modport slave(
    input clk,
    input rst_n,
    input data,
    input valid,
    output ready
  );
endinterface

// Module with modport interface port - accesses signals via port.signal
// CHECK-LABEL: moore.module private @master_driver
// CHECK-SAME: %port : !moore.ref<virtual_interface<@simple_if::@master>>
module master_driver(simple_if.master port);
  logic [7:0] local_data;

  // Read from modport member (rvalue access)
  // CHECK: moore.read %port
  // CHECK: moore.virtual_interface.signal_ref {{.*}}[@clk] : <@simple_if::@master>
  // CHECK: moore.read
  assign local_data = port.clk ? 8'h00 : 8'hFF;

  initial begin
    // Write to modport member (lvalue access)
    // CHECK: moore.virtual_interface.signal_ref {{.*}}[@data] : <@simple_if::@master>
    // CHECK: moore.blocking_assign
    port.data = 8'hAB;

    // CHECK: moore.virtual_interface.signal_ref {{.*}}[@valid] : <@simple_if::@master>
    // CHECK: moore.blocking_assign
    port.valid = 1'b1;

    // Read modport signals in expression
    // CHECK: moore.virtual_interface.signal_ref {{.*}}[@ready] : <@simple_if::@master>
    while (!port.ready) begin
      // Wait for clock edge via modport signal
      // CHECK: moore.virtual_interface.signal_ref {{.*}}[@clk] : <@simple_if::@master>
      // CHECK: moore.detect_event posedge
      @(posedge port.clk);
    end

    // Non-blocking assignment to modport signal
    // CHECK: moore.virtual_interface.signal_ref {{.*}}[@data] : <@simple_if::@master>
    // CHECK: moore.nonblocking_assign
    port.data <= 8'hCD;
  end
endmodule

// Module with slave modport
// CHECK-LABEL: moore.module private @slave_responder
// CHECK-SAME: %port : !moore.ref<virtual_interface<@simple_if::@slave>>
module slave_responder(simple_if.slave port);
  initial begin
    // Read from slave modport member
    // CHECK: moore.virtual_interface.signal_ref {{.*}}[@valid] : <@simple_if::@slave>
    if (port.valid) begin
      // Write to slave modport output
      // CHECK: moore.virtual_interface.signal_ref {{.*}}[@ready] : <@simple_if::@slave>
      // CHECK: moore.blocking_assign
      port.ready = 1'b1;
    end
  end
endmodule

// Top module for elaboration
// CHECK-LABEL: moore.module @top
module top;
  simple_if bus();

  master_driver u_master(.port(bus.master));
  slave_responder u_slave(.port(bus.slave));

  // Generate clock
  initial begin
    bus.clk = 0;
    forever #5 bus.clk = ~bus.clk;
  end

  // Reset
  initial begin
    bus.rst_n = 0;
    #20 bus.rst_n = 1;
  end
endmodule
