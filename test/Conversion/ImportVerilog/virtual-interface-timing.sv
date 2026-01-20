// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test virtual interface tasks with timing controls - patterns used in UVM BFMs
// This tests the critical patterns from AVIP BFM implementations.

// Package with common types
package vif_timing_pkg;
  typedef enum bit [1:0] {
    IDLE   = 2'b00,
    SETUP  = 2'b01,
    ACCESS = 2'b10,
    WAIT   = 2'b11
  } state_e;

  typedef struct {
    bit [31:0] addr;
    bit [31:0] data;
    bit        write;
  } transfer_s;
endpackage

// Interface with tasks containing timing controls
// CHECK-LABEL: moore.interface @apb_bfm_if
interface apb_bfm_if(input bit clk, input bit rst_n);
  import vif_timing_pkg::*;

  logic [31:0] paddr;
  logic [31:0] pwdata;
  logic [31:0] prdata;
  logic        pwrite;
  logic        penable;
  logic        pready;
  state_e      state;

  // Clocking block for synchronous access
  clocking cb @(posedge clk);
    input  prdata;
    input  pready;
    output paddr;
    output pwdata;
    output pwrite;
    output penable;
  endclocking

  // Task: wait_for_reset - wait for reset sequence
  // This pattern is common in BFMs: wait for negedge then posedge of reset
  // CHECK-LABEL: func.func private @"apb_bfm_if::wait_for_reset"
  // CHECK-SAME: (%[[IFACE:.*]]: !moore.virtual_interface<@apb_bfm_if>)
  // CHECK: moore.wait_event {
  // CHECK:   moore.virtual_interface.signal_ref %[[IFACE]][@rst_n]
  // CHECK:   moore.detect_event negedge
  // CHECK: }
  // CHECK: moore.wait_event {
  // CHECK:   moore.virtual_interface.signal_ref %[[IFACE]][@rst_n]
  // CHECK:   moore.detect_event posedge
  // CHECK: }
  task wait_for_reset();
    @(negedge rst_n);
    @(posedge rst_n);
  endtask

  // Task: drive_idle - drive interface to idle state
  // CHECK-LABEL: func.func private @"apb_bfm_if::drive_idle"
  // CHECK-SAME: (%[[IFACE:.*]]: !moore.virtual_interface<@apb_bfm_if>)
  // CHECK: moore.wait_event {
  // CHECK:   moore.virtual_interface.signal_ref %[[IFACE]][@clk]
  // CHECK:   moore.detect_event posedge
  // CHECK: }
  task drive_idle();
    @(posedge clk);
    paddr <= '0;
    pwdata <= '0;
    pwrite <= 1'b0;
    penable <= 1'b0;
    state = IDLE;
  endtask

  // Task: drive_setup - drive setup phase with struct parameter
  // CHECK-LABEL: func.func private @"apb_bfm_if::drive_setup"
  // CHECK-SAME: (%[[IFACE:.*]]: !moore.virtual_interface<@apb_bfm_if>, %[[TXN:.*]]: !moore.ustruct
  // CHECK: moore.wait_event {
  // CHECK:   moore.virtual_interface.signal_ref %[[IFACE]][@clk]
  // CHECK:   moore.detect_event posedge
  // CHECK: }
  task drive_setup(input transfer_s txn);
    @(posedge clk);
    paddr <= txn.addr;
    pwrite <= txn.write;
    if (txn.write) begin
      pwdata <= txn.data;
    end
    penable <= 1'b0;
    state = SETUP;
  endtask

  // Task: drive_access - drive access phase and wait for pready
  // This tests while loops with timing inside
  // CHECK-LABEL: func.func private @"apb_bfm_if::drive_access"
  // CHECK-SAME: (%[[IFACE:.*]]: !moore.virtual_interface<@apb_bfm_if>, %[[TXN:.*]]: !moore.ref<ustruct
  task drive_access(inout transfer_s txn);
    @(posedge clk);
    penable <= 1'b1;
    state = ACCESS;

    // Wait for slave ready (while loop with timing)
    while (pready == 0) begin
      @(posedge clk);
      state = WAIT;
    end

    // Capture read data
    if (!txn.write) begin
      txn.data = prdata;
    end

    // Return to idle
    penable <= 1'b0;
    state = IDLE;
  endtask

  // Task: perform_transaction - complete transaction calling other tasks
  // This tests task calls within interface tasks
  // CHECK-LABEL: func.func private @"apb_bfm_if::perform_transaction"
  // CHECK-SAME: (%[[IFACE:.*]]: !moore.virtual_interface<@apb_bfm_if>, %[[TXN:.*]]: !moore.ref<ustruct
  // CHECK: call @"apb_bfm_if::drive_setup"(%[[IFACE]],
  // CHECK: call @"apb_bfm_if::drive_access"(%[[IFACE]],
  task perform_transaction(inout transfer_s txn);
    drive_setup(txn);
    drive_access(txn);
  endtask

  // Function: is_idle - check if interface is idle
  function bit is_idle();
    return (state == IDLE);
  endfunction

endinterface

// Driver proxy class with virtual interface - UVM pattern
// CHECK-LABEL: moore.class.classdecl @apb_driver
class apb_driver;
  // Virtual interface handle
  virtual apb_bfm_if vif;

  // Task: run - main driver loop
  // This tests calling interface tasks through virtual interface
  // CHECK-LABEL: func.func private @"apb_driver::run"
  // CHECK-SAME: (%[[THIS:.*]]: !moore.class<@apb_driver>)
  task run();
    // Wait for reset through virtual interface
    vif.wait_for_reset();

    // Drive idle state
    vif.drive_idle();

    // Main loop
    forever begin
      vif_timing_pkg::transfer_s txn;
      txn.addr = 32'h1000;
      txn.data = 32'hDEAD_BEEF;
      txn.write = 1'b1;

      // Perform transaction through virtual interface
      vif.perform_transaction(txn);
    end
  endtask

  // Task: single_write - performs a single write transaction
  task single_write(input bit [31:0] addr, input bit [31:0] data);
    vif_timing_pkg::transfer_s txn;
    txn.addr = addr;
    txn.data = data;
    txn.write = 1'b1;
    vif.perform_transaction(txn);
  endtask

  // Task: single_read - performs a single read transaction
  task single_read(input bit [31:0] addr, output bit [31:0] data);
    vif_timing_pkg::transfer_s txn;
    txn.addr = addr;
    txn.write = 1'b0;
    vif.perform_transaction(txn);
    data = txn.data;
  endtask

endclass

// Top module for elaboration
// CHECK-LABEL: moore.module @test_top
module test_top;
  import vif_timing_pkg::*;

  bit clk;
  bit rst_n;

  // Interface instance
  apb_bfm_if apb_if(clk, rst_n);

  // Driver instance
  apb_driver drv;

  initial begin
    // Create driver and assign virtual interface
    drv = new();
    drv.vif = apb_if;

    // Run the driver
    fork
      drv.run();
    join_none
  end

  // Clock generation
  always #5 clk = ~clk;

  // Reset sequence
  initial begin
    rst_n = 0;
    #100 rst_n = 1;
  end

endmodule
