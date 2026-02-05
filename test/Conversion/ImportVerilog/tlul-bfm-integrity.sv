// RUN: circt-verilog --no-uvm-auto-include --ir-hw -I %S/../../../utils/opentitan_wrappers %s

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
    logic         d_valid;
    logic [31:0]  d_data;
    logic         d_error;
    logic         a_ready;
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

  function automatic logic [1:0] get_cmd_intg(tl_h2d_t tl);
    logic unused_tl;
    unused_tl = ^tl;
    return 2'b01;
  endfunction

  function automatic logic [1:0] get_data_intg(logic [31:0] data);
    logic unused_data;
    unused_data = ^data;
    return 2'b10;
  endfunction
endpackage

`include "tlul_bfm.sv"

module tlul_bfm_integrity_smoke;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;

  logic clk_i = 0;
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  always #5 clk_i = ~clk_i;

  initial begin
    tlul_init(tl_i);
    tl_o = '0;
    tl_o.a_ready = 1'b1;
    tl_o.d_valid = 1'b1;
    tl_o.d_data = 32'h12345678;
    tl_o.d_error = 1'b0;
    tlul_read32(clk_i, tl_i, tl_o, 32'h10, tlul_rdata);
    tlul_write32(clk_i, tl_i, tl_o, 32'h20, 32'hdeadbeef, 4'hF, tlul_err);
  end
endmodule
