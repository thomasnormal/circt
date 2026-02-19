// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top fork_struct_field_last_write 2>&1 | FileCheck %s

// Regression: struct field updates inside a join_none child must not create
// multi-driver X semantics across parent/child procedural writers.
// This shape models I3C monitor code that updates no_of_i3c_bits_transfer and
// writeData in a forked child while parent writes address/operation.
`timescale 1ns/1ps
module fork_struct_field_last_write;
  typedef enum bit[1:0] { POSEDGE = 2'b01, NEGEDGE = 2'b10 } edge_detect_e;
  typedef struct {
    bit [6:0] targetAddress;
    bit operation;
    bit [7:0] writeData[4];
    int no_of_i3c_bits_transfer;
  } pkt_t;

  bit pclk = 0;
  bit scl_i = 1;
  bit sda_i = 1;
  pkt_t pkt;

  always #5 pclk = ~pclk;

  task automatic detectEdge_scl(input edge_detect_e edgeSCL);
    bit [1:0] scl_local;
    scl_local = 2'b11;
    do begin
      @(negedge pclk);
      scl_local = {scl_local[0], scl_i};
    end while (!(scl_local == edgeSCL));
  endtask

  task automatic sample_write_data(inout pkt_t p, input int idx);
    int k;
    int bit_no;
    bit [7:0] wdata;
    for (k = 0; k < 8; k++) begin
      bit_no = 7 - k;
      detectEdge_scl(POSEDGE);
      wdata[bit_no] = sda_i;
      p.no_of_i3c_bits_transfer++;
    end
    p.writeData[idx] = wdata;
  endtask

  task automatic wrDetect_stop();
    bit [1:0] scl_local;
    bit [1:0] sda_local;
    do begin
      @(negedge pclk);
      scl_local = {scl_local[0], scl_i};
      sda_local = {sda_local[0], sda_i};
    end while (!(sda_local == POSEDGE && scl_local == 2'b11));
  endtask

  task automatic sampleWriteDataAndACK(inout pkt_t p);
    fork
      begin
        sample_write_data(p, 0);
      end
    join_none
    wrDetect_stop();
    disable fork;
  endtask

  task automatic sample_data(inout pkt_t p);
    p.targetAddress = 7'h68;
    p.operation = 1'b0;
    sampleWriteDataAndACK(p);
  endtask

  task automatic drive_bit(input bit b);
    @(negedge pclk);
    scl_i = 0;
    sda_i = b;
    @(negedge pclk);
    scl_i = 1;
  endtask

  task automatic drive_tx(input bit [7:0] data);
    int i;
    @(negedge pclk);
    scl_i = 1;
    sda_i = 0;
    for (i = 7; i >= 0; i--)
      drive_bit(data[i]);
    @(negedge pclk);
    scl_i = 1;
    sda_i = 0;
    @(negedge pclk);
    scl_i = 1;
    sda_i = 1;
  endtask

  initial begin
    pkt = '{default:0};
    fork
      sample_data(pkt);
      drive_tx(8'h4e);
    join

    // CHECK: addr=68 op=0 no=8 wd0=4e
    $display("addr=%0h op=%0d no=%0d wd0=%0h", pkt.targetAddress, pkt.operation,
             pkt.no_of_i3c_bits_transfer, pkt.writeData[0]);

    if (pkt.targetAddress !== 7'h68 || pkt.operation !== 1'b0 ||
        pkt.no_of_i3c_bits_transfer != 8 || pkt.writeData[0] !== 8'h4e) begin
      $display("FAIL");
      $fatal(1);
    end

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule
