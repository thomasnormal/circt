// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top sig_extract_struct_array_bit_memory_layout_tb | FileCheck %s

// Regression: drives to nested refs of the form
//   llhd.sig.extract(llhd.sig.array_get(llhd.sig.struct_extract(...)))
// must preserve HW<->LLVM array layout mapping for memory-backed refs.

// CHECK: no_bits=8 w0=fb
// CHECK: PASS
// CHECK-NOT: FAIL

module sig_extract_struct_array_bit_memory_layout_tb;
  typedef struct {
    int unsigned no_bits;
    bit [7:0] writeData [0:3];
  } pkt_t;

  reg clk = 1'b0;
  always #5 clk = ~clk;

  task automatic detect_edge();
    @(posedge clk);
  endtask

  task automatic sample_write_data(inout pkt_t pkt, input int i, input bit [7:0] w);
    int k;
    for (k = 0; k < 8; k++) begin
      detect_edge();
      pkt.no_bits++;
      pkt.writeData[i][7-k] = w[7-k];
    end
  endtask

  initial begin
    pkt_t p;
    p.no_bits = 0;
    p.writeData[0] = 8'h00;
    sample_write_data(p, 0, 8'hFB);
    $display("no_bits=%0d w0=%0h", p.no_bits, p.writeData[0]);
    if (p.no_bits != 8 || p.writeData[0] != 8'hFB)
      $display("FAIL");
    else
      $display("PASS");
    $finish;
  end
endmodule
