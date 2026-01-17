// RUN: circt-verilog %s --ir-moore | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// AVIP APB Simulation Testbench - End-to-End Verification Flow
//===----------------------------------------------------------------------===//
//
// This testbench demonstrates a complete verification flow with:
// - APB transaction class with rand properties and constraints
// - Coverage collection with covergroups
// - Stimulus generation and response checking
//
// The design showcases CIRCT's support for:
// 1. Class-based transactions with randomization
// 2. Functional coverage with covergroups
// 3. Protocol modeling
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Package: APB Protocol Definitions
//===----------------------------------------------------------------------===//

package apb_pkg;
  // APB transfer types
  typedef enum bit {
    APB_READ  = 1'b0,
    APB_WRITE = 1'b1
  } apb_direction_e;

  // APB response types
  typedef enum bit {
    APB_OKAY  = 1'b0,
    APB_ERROR = 1'b1
  } apb_response_e;

  // Configuration parameters
  parameter int APB_ADDR_WIDTH = 32;
  parameter int APB_DATA_WIDTH = 32;
endpackage : apb_pkg

// CHECK: moore.package @apb_pkg

//===----------------------------------------------------------------------===//
// Transaction Class: APB Transaction
//===----------------------------------------------------------------------===//

class apb_transaction;
  // Random properties for stimulus generation
  rand bit [31:0] addr;
  rand bit [31:0] wdata;
  rand bit direction;  // 0=read, 1=write
  rand bit [3:0] strb;  // Byte strobes

  // Non-random response fields (filled by responder)
  bit [31:0] rdata;
  bit response;  // 0=okay, 1=error

  // Transaction ID for tracking
  int unsigned id;
  static int unsigned id_counter = 0;

  // Constraints for valid APB transactions
  constraint c_addr_align {
    // Word-aligned addresses
    addr[1:0] == 2'b00;
  }

  constraint c_addr_range {
    // Valid memory region: limited range for 16-word memory
    addr[31:6] == 26'b0;  // Only use addr[5:0]
  }

  constraint c_strb_valid {
    // At least one byte strobe active for writes
    direction == 1'b1 -> strb != 4'b0000;
    // All strobes active for reads (simplification)
    direction == 1'b0 -> strb == 4'b1111;
  }

  // Constructor
  function new();
    id = id_counter++;
    addr = 0;
    wdata = 0;
    rdata = 0;
    direction = 0;
    strb = 4'b1111;
    response = 0;
  endfunction

  // Display transaction
  function void display(string prefix);
    $display("%s[TXN %0d] %s addr=0x%08h data=0x%08h strb=%04b",
             prefix, id,
             direction ? "WR" : "RD",
             addr,
             direction ? wdata : rdata,
             strb);
  endfunction

  // Copy transaction
  function apb_transaction copy();
    apb_transaction c = new();
    c.addr = this.addr;
    c.wdata = this.wdata;
    c.direction = this.direction;
    c.strb = this.strb;
    c.rdata = this.rdata;
    c.response = this.response;
    c.id = this.id;
    return c;
  endfunction
endclass

// CHECK: moore.class.classdecl @apb_transaction
// CHECK:   moore.class.propertydecl @addr : !moore.i32 rand_mode rand
// CHECK:   moore.class.propertydecl @wdata : !moore.i32 rand_mode rand
// CHECK:   moore.class.propertydecl @direction : !moore.i1 rand_mode rand
// CHECK:   moore.class.propertydecl @strb : !moore.i4 rand_mode rand
// CHECK:   moore.constraint.block @c_addr_align
// CHECK:   moore.constraint.block @c_addr_range
// CHECK:   moore.constraint.block @c_strb_valid

//===----------------------------------------------------------------------===//
// Coverage: APB Transaction Coverage
//===----------------------------------------------------------------------===//

class apb_coverage;
  // Coverage variables
  bit [31:0] addr;
  bit        direction;
  bit [3:0]  strb;
  bit        response;

  // Covergroup for APB transactions
  covergroup cg_apb_txn;
    // Address coverage with bins
    cp_addr: coverpoint addr[5:2] {
      bins low_addr  = {[0:3]};
      bins mid_addr  = {[4:11]};
      bins high_addr = {[12:15]};
    }

    // Direction coverage
    cp_direction: coverpoint direction {
      bins read  = {0};
      bins write = {1};
    }

    // Byte strobe coverage
    cp_strb: coverpoint strb {
      bins all_bytes   = {4'b1111};
      bins low_bytes   = {4'b0011};
      bins high_bytes  = {4'b1100};
      bins single_byte = {4'b0001, 4'b0010, 4'b0100, 4'b1000};
    }

    // Response coverage
    cp_response: coverpoint response {
      bins okay  = {0};
      bins error = {1};
    }

    // Cross coverage: direction x address region
    cross_dir_addr: cross cp_direction, cp_addr;
  endgroup

  function new();
    cg_apb_txn = new();
  endfunction

  // Sample transaction
  function void sample(apb_transaction txn);
    this.addr      = txn.addr;
    this.direction = txn.direction;
    this.strb      = txn.strb;
    this.response  = txn.response;
    cg_apb_txn.sample();
  endfunction

  // Report coverage
  function void report();
    $display("=== APB Transaction Coverage ===");
    // Note: get_coverage() requires runtime support - using placeholder
    $display("  Coverage collected - see runtime for details");
  endfunction
endclass

// CHECK: moore.class.classdecl @apb_coverage
// CHECK:   moore.class.propertydecl @cg_apb_txn : !moore.covergroup<@cg_apb_txn>
// CHECK: moore.covergroup.decl @cg_apb_txn
// CHECK:   moore.coverpoint.decl @cp_addr
// CHECK:   moore.coverpoint.decl @cp_direction
// CHECK:   moore.coverpoint.decl @cp_strb
// CHECK:   moore.coverpoint.decl @cp_response
// CHECK:   moore.covercross.decl @cross_dir_addr

//===----------------------------------------------------------------------===//
// Scoreboard: Transaction Checker
//===----------------------------------------------------------------------===//

class apb_scoreboard;
  // Expected values from reference model
  bit [31:0] ref_memory [16];

  // Statistics
  int unsigned checks_passed;
  int unsigned checks_failed;
  int unsigned transactions_seen;

  function new();
    checks_passed = 0;
    checks_failed = 0;
    transactions_seen = 0;
    // Initialize reference memory
    for (int i = 0; i < 16; i++) begin
      ref_memory[i] = 32'hDEAD0000 + i;
    end
  endfunction

  // Check transaction
  function void check(apb_transaction txn);
    automatic int idx = txn.addr[5:2];
    transactions_seen++;

    if (txn.response == 1'b1) begin
      // Don't check data on error response
      $display("  [SCB] TXN %0d: Error response (not checking data)", txn.id);
      return;
    end

    if (txn.direction == 1'b1) begin
      // Update reference model for writes
      if (txn.strb[0]) ref_memory[idx][7:0]   = txn.wdata[7:0];
      if (txn.strb[1]) ref_memory[idx][15:8]  = txn.wdata[15:8];
      if (txn.strb[2]) ref_memory[idx][23:16] = txn.wdata[23:16];
      if (txn.strb[3]) ref_memory[idx][31:24] = txn.wdata[31:24];
      checks_passed++;
      $display("  [SCB] TXN %0d: Write OK", txn.id);
    end else begin
      // Check read data
      if (txn.rdata == ref_memory[idx]) begin
        checks_passed++;
        $display("  [SCB] TXN %0d: Read OK", txn.id);
      end else begin
        checks_failed++;
        $display("  [SCB] TXN %0d: Read MISMATCH", txn.id);
      end
    end
  endfunction

  // Report results
  function void report();
    $display("=== Scoreboard Summary ===");
    $display("  Transactions: %0d", transactions_seen);
    $display("  Passed:       %0d", checks_passed);
    $display("  Failed:       %0d", checks_failed);
    if (checks_failed == 0) begin
      $display("  Result:       PASS");
    end else begin
      $display("  Result:       FAIL");
    end
  endfunction
endclass

// CHECK: moore.class.classdecl @apb_scoreboard

//===----------------------------------------------------------------------===//
// Memory Model: Simple APB Slave
//===----------------------------------------------------------------------===//

class apb_memory;
  // Simple memory model (16 words)
  bit [31:0] memory [16];

  function new();
    // Initialize memory with pattern
    for (int i = 0; i < 16; i++) begin
      memory[i] = 32'hDEAD0000 + i;
    end
  endfunction

  // Process transaction (functional model)
  function void process(apb_transaction txn);
    automatic int idx = txn.addr[5:2];

    if (txn.direction == 1'b1) begin
      // Write with byte strobes
      if (txn.strb[0]) memory[idx][7:0]   = txn.wdata[7:0];
      if (txn.strb[1]) memory[idx][15:8]  = txn.wdata[15:8];
      if (txn.strb[2]) memory[idx][23:16] = txn.wdata[23:16];
      if (txn.strb[3]) memory[idx][31:24] = txn.wdata[31:24];
    end else begin
      // Read
      txn.rdata = memory[idx];
    end
    txn.response = 0;  // Always okay
  endfunction
endclass

// CHECK: moore.class.classdecl @apb_memory

//===----------------------------------------------------------------------===//
// Testbench: Top-Level Test Environment
//===----------------------------------------------------------------------===//

module avip_apb_tb;
  // Test components
  apb_coverage   coverage;
  apb_scoreboard scoreboard;
  apb_memory     mem;

  // Test configuration
  int num_transactions = 20;

  // Main test
  initial begin
    apb_transaction txn;
    int success;
    int txn_count;

    $display("========================================");
    $display("  AVIP APB Simulation Test");
    $display("========================================");

    // Create components
    coverage   = new();
    scoreboard = new();
    mem        = new();

    txn_count = 0;

    $display("\n--- Starting Transaction Sequence ---");

    // Generate and execute transactions
    for (int i = 0; i < num_transactions; i++) begin
      txn = new();

      // Randomize transaction
      success = txn.randomize();
      if (!success) begin
        $display("ERROR: Randomization failed for transaction %0d", i);
        continue;
      end

      // Display generated transaction
      txn.display("  [GEN] ");

      // Process through memory model
      mem.process(txn);

      // Sample coverage
      coverage.sample(txn);

      // Check in scoreboard
      scoreboard.check(txn);

      txn_count++;
    end

    $display("\n--- Transaction Sequence Complete ---\n");

    // Report results
    coverage.report();
    scoreboard.report();

    $display("\n========================================");
    $display("  Total Transactions: %0d", txn_count);
    $display("========================================");

    // End simulation
    $finish;
  end

endmodule

// CHECK: moore.module @avip_apb_tb
// CHECK:   moore.procedure initial
// CHECK:     moore.class.new : <@apb_coverage>
// CHECK:     moore.class.new : <@apb_scoreboard>
// CHECK:     moore.class.new : <@apb_memory>
// CHECK:     moore.class.new : <@apb_transaction>
// CHECK:     moore.randomize

//===----------------------------------------------------------------------===//
// End of AVIP APB Simulation Testbench
//===----------------------------------------------------------------------===//
