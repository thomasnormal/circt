// RUN: circt-verilog --ir-moore %s | FileCheck %s
// Test for interface tasks/functions - simulating BFM (Bus Functional Model) patterns

// Simplified global package for BFM testing
package bfm_global_pkg;
  parameter int DATA_WIDTH = 32;
  parameter int ADDR_WIDTH = 32;

  typedef enum bit {
    WRITE = 1'b1,
    READ  = 1'b0
  } tx_type_e;

  typedef enum bit[2:0] {
    IDLE,
    SETUP,
    ACCESS,
    WAIT_STATE
  } fsm_state_e;

  typedef struct {
    bit pwrite;
    bit pslverr;
    bit [DATA_WIDTH-1:0] prdata;
    bit [ADDR_WIDTH-1:0] paddr;
    bit [DATA_WIDTH-1:0] pwdata;
    int no_of_wait_states;
  } transfer_s;
endpackage : bfm_global_pkg

// APB-like interface with tasks for BFM testing
interface apb_bfm_if (input bit clk, input bit rst_n);
  import bfm_global_pkg::*;

  // Interface signals
  logic penable;
  logic pwrite;
  logic [ADDR_WIDTH-1:0] paddr;
  logic [DATA_WIDTH-1:0] pwdata;
  logic [DATA_WIDTH-1:0] prdata;
  logic pready;
  logic pslverr;

  // FSM state tracking
  fsm_state_e state;

  // Task: wait_for_reset
  // Wait for reset sequence
  task wait_for_reset();
    @(negedge rst_n);
    @(posedge rst_n);
  endtask : wait_for_reset

  // Task: drive_idle
  // Drive interface to idle state
  task drive_idle();
    @(posedge clk);
    penable <= 1'b0;
    pwrite <= 1'b0;
    paddr <= '0;
    pwdata <= '0;
    state = IDLE;
  endtask : drive_idle

  // Task: drive_setup
  // Drive setup phase of APB transaction
  task drive_setup(input transfer_s txn);
    @(posedge clk);
    paddr <= txn.paddr;
    pwrite <= txn.pwrite;
    if (txn.pwrite == WRITE) begin
      pwdata <= txn.pwdata;
    end
    penable <= 1'b0;
    state = SETUP;
  endtask : drive_setup

  // Task: drive_access
  // Drive access phase and wait for pready
  task drive_access(inout transfer_s txn);
    @(posedge clk);
    penable <= 1'b1;
    state = ACCESS;

    // Wait for slave ready
    while (pready == 0) begin
      @(posedge clk);
      state = WAIT_STATE;
      txn.no_of_wait_states++;
    end

    // Capture response
    if (txn.pwrite == READ) begin
      txn.prdata = prdata;
    end
    txn.pslverr = pslverr;

    // Return to idle
    penable <= 1'b0;
    state = IDLE;
  endtask : drive_access

  // Task: perform_transaction
  // Complete APB transaction
  task perform_transaction(inout transfer_s txn);
    drive_setup(txn);
    drive_access(txn);
  endtask : perform_transaction

  // Function: get_state
  // Return current FSM state
  function fsm_state_e get_state();
    return state;
  endfunction : get_state

  // Function: is_idle
  // Check if interface is idle
  function bit is_idle();
    return (state == IDLE);
  endfunction : is_idle

endinterface : apb_bfm_if

// Top module to instantiate the interface
module bfm_test_top;
  bit clk;
  bit rst_n;

  // Instantiate the BFM interface
  apb_bfm_if apb_if_inst(clk, rst_n);

  // Simple test sequence
  initial begin
    bfm_global_pkg::transfer_s txn;
    txn.paddr = 32'h1000;
    txn.pwdata = 32'hDEADBEEF;
    txn.pwrite = bfm_global_pkg::WRITE;
    txn.no_of_wait_states = 0;

    // Wait for reset
    apb_if_inst.wait_for_reset();

    // Perform a transaction
    apb_if_inst.perform_transaction(txn);

    // Check state
    if (apb_if_inst.is_idle()) begin
      $display("Transaction complete, interface idle");
    end
  end

  // Clock generation
  always #5 clk = ~clk;

endmodule : bfm_test_top

// Test that interface task that calls other interface tasks works correctly.
// The key pattern is: perform_transaction calls drive_setup and drive_access,
// and the interface instance is correctly passed through the call chain.

// CHECK: moore.interface @apb_bfm_if
// CHECK: func.func private @"apb_bfm_if::wait_for_reset"({{.*}}: !moore.virtual_interface<@apb_bfm_if>)
// CHECK: func.func private @"apb_bfm_if::drive_setup"({{.*}}: !moore.virtual_interface<@apb_bfm_if>
// CHECK: func.func private @"apb_bfm_if::drive_access"({{.*}}: !moore.virtual_interface<@apb_bfm_if>
// CHECK: func.func private @"apb_bfm_if::perform_transaction"(%[[IFACE:.*]]: !moore.virtual_interface<@apb_bfm_if>
// CHECK:   call @"apb_bfm_if::drive_setup"(%[[IFACE]]
// CHECK:   call @"apb_bfm_if::drive_access"(%[[IFACE]]
// CHECK: moore.module @bfm_test_top
// CHECK:   moore.interface.instance  @apb_bfm_if
// CHECK:   moore.procedure initial
// CHECK:     func.call @"apb_bfm_if::perform_transaction"
