// Test that tlul_bfm exits early when a_ready never asserts.
// TODO: BFM timeout messages not printed — task-level $display in while loop not working.
// This avoids waiting for d_valid on a request that never handshakes.
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

  typedef struct packed {
    logic        a_ready;
    logic        d_valid;
    logic [31:0] d_data;
    logic        d_error;
  } tl_d2h_t;

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

  function automatic logic [1:0] get_data_intg(input logic [31:0] data);
    get_data_intg = ^data ? 2'b10 : 2'b01;
  endfunction

  function automatic logic [1:0] get_cmd_intg(input tl_h2d_t tl);
    get_cmd_intg = ^tl.a_address ? 2'b10 : 2'b01;
  endfunction
endpackage

`include "tlul_bfm.sv"

module tlul_bfm_a_ready_timeout_short_circuit;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;

  logic clk_i = 0;
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] data;
  logic err;

  always #5 clk_i = ~clk_i;

  initial begin
    tlul_init(tl_i);
    tl_o = '0;

    tlul_read32(clk_i, tl_i, tl_o, 32'h0, data);
    $display("READ_DONE data=0x%08x", data);

    tlul_write32(clk_i, tl_i, tl_o, 32'h4, 32'hdeadbeef, 4'hF, err);
    $display("WRITE_DONE err=%0d", err);

    $finish;
  end

  initial begin
    #4000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule

// CHECK: TLUL BFM read timeout waiting for a_ready
// CHECK-NOT: TLUL BFM read timeout waiting for d_valid
// CHECK: READ_DONE data=0x{{0+}}
// CHECK: TLUL BFM write timeout waiting for a_ready
// CHECK-NOT: TLUL BFM write timeout waiting for d_valid
// CHECK: WRITE_DONE err=1
