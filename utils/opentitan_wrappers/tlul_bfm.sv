// Simple TL-UL BFM helpers for OpenTitan smoke tests.
package tlul_bfm_pkg;
  import tlul_pkg::*;
  import prim_mubi_pkg::*;

  task automatic tlul_init(output tl_h2d_t tl_i);
    tl_i = TL_H2D_DEFAULT;
  endtask

  task automatic tlul_read32(
      ref logic clk_i,
      inout tl_h2d_t tl_i,
      ref tl_d2h_t tl_o,
      input logic [31:0] addr,
      output logic [31:0] data);
    int unsigned wait_cycles;
    bit got_a_ready;
    bit got_resp;
    tl_h2d_t tl_tmp;
    tl_a_user_t a_user;
    data = '0;

    @(negedge clk_i);
    tl_tmp = TL_H2D_DEFAULT;
    tl_tmp.d_ready = 1'b1;
    tl_tmp.a_valid = 1'b1;
    tl_tmp.a_opcode = Get;
    tl_tmp.a_address = addr;
    tl_tmp.a_size = 2;  // 4 bytes
    tl_tmp.a_mask = 4'hF;
    tl_tmp.a_data = '0;
    a_user = TL_A_USER_DEFAULT;
    tl_tmp.a_user = a_user;
    a_user.data_intg = get_data_intg(tl_tmp.a_data);
    a_user.cmd_intg = get_cmd_intg(tl_tmp);
    a_user.instr_type = MuBi4False;
    tl_tmp.a_user = a_user;
    tl_i = tl_tmp;

    got_a_ready = 0;
    for (wait_cycles = 0; wait_cycles < 50; wait_cycles++) begin
      @(posedge clk_i);
      if (tl_o.a_ready === 1'b1) begin
        got_a_ready = 1;
        break;
      end
    end

    if (!got_a_ready) begin
      tl_i.a_valid = 1'b0;
      $display("TLUL BFM read timeout waiting for a_ready (addr=0x%08x, a_valid=%b, a_ready=%b, d_valid=%b)",
               addr, tl_i.a_valid, tl_o.a_ready, tl_o.d_valid);
    end
    if (got_a_ready) begin
      @(negedge clk_i);
      tl_i.a_valid = 1'b0;
    end

    got_resp = 0;
    for (wait_cycles = 0; wait_cycles < 50; wait_cycles++) begin
      @(posedge clk_i);
      if (tl_o.d_valid) begin
        data = tl_o.d_data;
        got_resp = 1;
        break;
      end
    end

    if (!got_resp) begin
      $display("TLUL BFM read timeout waiting for d_valid (addr=0x%08x, a_valid=%b, a_ready=%b, d_valid=%b)",
               addr, tl_i.a_valid, tl_o.a_ready, tl_o.d_valid);
    end
  endtask

  task automatic tlul_write32(
      ref logic clk_i,
      inout tl_h2d_t tl_i,
      ref tl_d2h_t tl_o,
      input logic [31:0] addr,
      input logic [31:0] data,
      input logic [3:0] mask,
      output logic err);
    int unsigned wait_cycles;
    bit got_a_ready;
    bit got_resp;
    tl_h2d_t tl_tmp;
    tl_a_user_t a_user;
    err = 1'b0;

    @(negedge clk_i);
    tl_tmp = TL_H2D_DEFAULT;
    tl_tmp.d_ready = 1'b1;
    tl_tmp.a_valid = 1'b1;
    if (mask == 4'hF) begin
      tl_tmp.a_opcode = PutFullData;
    end else begin
      tl_tmp.a_opcode = PutPartialData;
    end
    tl_tmp.a_address = addr;
    tl_tmp.a_size = 2;  // 4 bytes
    tl_tmp.a_mask = mask;
    tl_tmp.a_data = data;
    a_user = TL_A_USER_DEFAULT;
    tl_tmp.a_user = a_user;
    a_user.data_intg = get_data_intg(tl_tmp.a_data);
    a_user.cmd_intg = get_cmd_intg(tl_tmp);
    a_user.instr_type = MuBi4False;
    tl_tmp.a_user = a_user;
    tl_i = tl_tmp;

    got_a_ready = 0;
    for (wait_cycles = 0; wait_cycles < 50; wait_cycles++) begin
      @(posedge clk_i);
      if (tl_o.a_ready === 1'b1) begin
        got_a_ready = 1;
        break;
      end
    end

    if (!got_a_ready) begin
      tl_i.a_valid = 1'b0;
      err = 1'b1;
      $display("TLUL BFM write timeout waiting for a_ready (addr=0x%08x, a_valid=%b, a_ready=%b, d_valid=%b)",
               addr, tl_i.a_valid, tl_o.a_ready, tl_o.d_valid);
    end
    if (got_a_ready) begin
      @(negedge clk_i);
      tl_i.a_valid = 1'b0;
    end

    got_resp = 0;
    for (wait_cycles = 0; wait_cycles < 50; wait_cycles++) begin
      @(posedge clk_i);
      if (tl_o.d_valid) begin
        err = tl_o.d_error;
        got_resp = 1;
        break;
      end
    end

    if (!got_resp) begin
      err = 1'b1;
      $display("TLUL BFM write timeout waiting for d_valid (addr=0x%08x, a_valid=%b, a_ready=%b, d_valid=%b)",
               addr, tl_i.a_valid, tl_o.a_ready, tl_o.d_valid);
    end
  endtask
endpackage
