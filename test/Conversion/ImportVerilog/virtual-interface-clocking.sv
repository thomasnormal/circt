// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test virtual interface access to clocking blocks - UVM driver/monitor pattern
// This is a critical pattern used in UVM BFMs for synchronous signal access.

// Interface with clocking block
// CHECK-LABEL: moore.interface @clk_if
interface clk_if(input bit clk);
  logic [7:0] data_in;
  logic [7:0] data_out;
  logic       valid;
  logic       ready;

  // Driver clocking block - for driving signals
  clocking drv_cb @(posedge clk);
    output data_out;
    output valid;
    input  ready;
  endclocking

  // Monitor clocking block - for sampling signals
  clocking mon_cb @(posedge clk);
    input data_in;
    input data_out;
    input valid;
    input ready;
  endclocking

  // Task: wait_clocks - wait for N clock cycles
  // CHECK-LABEL: func.func private @"clk_if::wait_clocks"
  // CHECK-SAME: (%{{.*}}: !moore.virtual_interface<@clk_if>, %{{.*}}: !moore.i32)
  task wait_clocks(int n);
    repeat (n) @(posedge clk);
  endtask

  // Task: drive_data - drive data on output
  // CHECK-LABEL: func.func private @"clk_if::drive_data"
  // CHECK-SAME: (%{{.*}}: !moore.virtual_interface<@clk_if>, %{{.*}}: !moore.l8)
  // CHECK: moore.wait_event
  task drive_data(input logic [7:0] data);
    @(posedge clk);
    data_out <= data;
    valid <= 1'b1;
    // Wait for ready
    @(posedge clk);
    while (!ready) @(posedge clk);
    valid <= 1'b0;
  endtask

  // Task: sample_data - sample input data
  // CHECK-LABEL: func.func private @"clk_if::sample_data"
  task sample_data(output logic [7:0] data);
    @(posedge clk);
    while (!valid) @(posedge clk);
    data = data_in;
  endtask

endinterface

// Driver class using virtual interface with clocking
// CHECK-LABEL: moore.class.classdecl @clk_driver
class clk_driver;
  virtual clk_if vif;

  // Task: run - calls interface tasks through virtual interface
  // CHECK-LABEL: func.func private @"clk_driver::run"
  // CHECK: call @"clk_if::wait_clocks"
  // CHECK: call @"clk_if::drive_data"
  task run();
    forever begin
      vif.wait_clocks(10);
      vif.drive_data(8'hAB);
    end
  endtask
endclass

// Monitor class using virtual interface with clocking
// CHECK-LABEL: moore.class.classdecl @clk_monitor
class clk_monitor;
  virtual clk_if vif;

  // Task: monitor_transfer
  task monitor_transfer(output logic [7:0] data);
    // Call interface task through virtual interface
    vif.sample_data(data);
  endtask

  // Task: run
  // CHECK-LABEL: func.func private @"clk_monitor::run"
  // CHECK: call @"clk_if::sample_data"
  task run();
    logic [7:0] sampled_data;
    forever begin
      vif.sample_data(sampled_data);
      $display("Sampled: %0h", sampled_data);
    end
  endtask
endclass

// Testbench module
// CHECK-LABEL: moore.module @test_clocking_vif
module test_clocking_vif;
  bit clk;
  clk_if dut_if(clk);

  clk_driver drv;
  clk_monitor mon;

  initial begin
    drv = new();
    mon = new();
    drv.vif = dut_if;
    mon.vif = dut_if;

    fork
      drv.run();
      mon.run();
    join_none
  end

  always #5 clk = ~clk;
endmodule
