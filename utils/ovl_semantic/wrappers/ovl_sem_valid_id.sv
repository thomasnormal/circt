module ovl_sem_valid_id(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
`ifdef FAIL
  logic issued = 1'bx;
`else
  logic issued = 1'b0;
`endif
  logic returned = 1'b0;
  logic flush = 1'b0;
  logic [1:0] issued_id = 2'b00;
  logic [1:0] returned_id = 2'b00;
  logic [1:0] flush_id = 2'b00;
  logic [1:0] issued_count = 2'b00;

  ovl_valid_id #(
      .min_cks(1),
      .max_cks(1),
      .width(2),
      .max_id_instances(2),
      .max_ids(2),
      .max_instances_per_id(2),
      .instance_count_width(2)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .issued(issued),
      .issued_id(issued_id),
      .returned(returned),
      .returned_id(returned_id),
      .flush(flush),
      .flush_id(flush_id),
      .issued_count(issued_count),
      .fire());
endmodule
