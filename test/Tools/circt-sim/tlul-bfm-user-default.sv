// Test that TL_H2D_DEFAULT localparam struct initialization works correctly.
// This is a simplified test that doesn't use the full BFM to avoid slow simulation.
//
// The timeout ensures we don't hang.
//
// RUN: circt-verilog --no-uvm-auto-include --ir-hw -I %S/../../../utils/opentitan_wrappers %s -o %t.mlir
// RUN: circt-sim %t.mlir --timeout=30 | FileCheck %s

package prim_mubi_pkg;
  typedef enum logic [3:0] {
    MuBi4True = 4'b0101,
    MuBi4False = 4'b1010
  } mubi4_t;
endpackage

package tlul_pkg;
  typedef enum logic [2:0] {
    PutFullData    = 3'h0,
    PutPartialData = 3'h1,
    Get            = 3'h4
  } tl_a_op_e;

  typedef struct packed {
    logic [1:0] cmd_intg;
    logic [1:0] data_intg;
    prim_mubi_pkg::mubi4_t instr_type;
  } tl_a_user_t;

  typedef struct packed {
    logic         a_valid;
    tl_a_op_e     a_opcode;
    logic [2:0]   a_param;
    logic [1:0]   a_size;
    logic [1:0]   a_source;
    logic [7:0]   a_address;
    logic [3:0]   a_mask;
    logic [31:0]  a_data;
    tl_a_user_t   a_user;
    logic         d_ready;
  } tl_h2d_t;

  localparam tl_a_user_t TL_A_USER_DEFAULT = '{
    cmd_intg: 2'b11,
    data_intg: 2'b11,
    instr_type: prim_mubi_pkg::MuBi4False
  };

  localparam tl_h2d_t TL_H2D_DEFAULT = '{
    d_ready: 1'b1,
    a_user: TL_A_USER_DEFAULT,
    a_opcode: tl_a_op_e'(0),
    default: '0
  };
endpackage

// Simplified test that just checks the struct default values directly
module tlul_bfm_user_default_smoke;
  import tlul_pkg::*;

  tl_h2d_t tl_i;

  initial begin
    // Initialize with the TL_H2D_DEFAULT localparam
    tl_i = TL_H2D_DEFAULT;

    // Check d_ready field (should be 1)
    if (tl_i.d_ready !== 1'b1) begin
      $display("TLUL BFM d_ready mismatch: %b", tl_i.d_ready);
    end else begin
      $display("TLUL BFM d_ready ok");
    end

    // Check a_user.instr_type field (should be MuBi4False = 4'b1010)
    if (tl_i.a_user.instr_type !== prim_mubi_pkg::MuBi4False) begin
      $display("TLUL BFM instr_type mismatch: 0x%0x", tl_i.a_user.instr_type);
    end else begin
      $display("TLUL BFM instr_type ok");
    end

    $finish;
  end
endmodule

// CHECK: TLUL BFM d_ready ok
// CHECK: TLUL BFM instr_type ok
