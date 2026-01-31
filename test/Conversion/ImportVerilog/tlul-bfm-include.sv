// RUN: circt-verilog --parse-only --no-uvm-auto-include -I%S/../../../utils/opentitan_wrappers %s
// Test that tlul_bfm.sv can be included with a compatible tlul_pkg definition.

// Stub for prim_mubi_pkg required by tlul_bfm.sv
package prim_mubi_pkg;
  typedef logic [3:0] mubi4_t;
  localparam mubi4_t MuBi4False = 4'b1001;
endpackage

package tlul_pkg;
  import prim_mubi_pkg::*;

  typedef enum logic [2:0] {
    Get = 3'b100,
    PutFullData = 3'b000,
    PutPartialData = 3'b001
  } tl_a_op_e;

  // User field type for integrity checking
  typedef struct packed {
    logic [6:0] data_intg;
    logic [6:0] cmd_intg;
    mubi4_t instr_type;
  } tl_a_user_t;

  typedef struct packed {
    logic a_valid;
    tl_a_op_e a_opcode;
    logic [31:0] a_address;
    logic [2:0] a_size;
    logic [3:0] a_mask;
    logic [31:0] a_data;
    tl_a_user_t a_user;
    logic d_ready;
  } tl_h2d_t;

  typedef struct packed {
    logic d_valid;
    logic [31:0] d_data;
    logic d_error;
    logic a_ready;
  } tl_d2h_t;

  localparam tl_h2d_t TL_H2D_DEFAULT = '0;
  localparam tl_a_user_t TL_A_USER_DEFAULT = '0;

  // Stub integrity calculation functions
  function automatic logic [6:0] get_data_intg(input logic [31:0] data);
    return '0;
  endfunction

  function automatic logic [6:0] get_cmd_intg(input tl_h2d_t cmd);
    return '0;
  endfunction
endpackage

`include "tlul_bfm.sv"

module tlul_bfm_smoke;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;

  logic clk;
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] data;
  logic err;

  initial begin
    tlul_init(tl_i);
    tlul_read32(clk, tl_i, tl_o, 32'h0, data);
    tlul_write32(clk, tl_i, tl_o, 32'h4, 32'h1, 4'hF, err);
  end
endmodule
