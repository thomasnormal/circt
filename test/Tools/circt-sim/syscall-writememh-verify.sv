// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $writememh and then verify with $readmemh — round-trip check
module top;
  reg [7:0] mem_out [0:3];
  reg [7:0] mem_in [0:3];
  integer fd;

  initial begin
    // Populate memory
    mem_out[0] = 8'hDE;
    mem_out[1] = 8'hAD;
    mem_out[2] = 8'hBE;
    mem_out[3] = 8'hEF;

    // Allow signal values to propagate
    #1;

    // Write memory to file
    $writememh("writemem_verify.dat", mem_out);

    // Verify by checking the file was created and has content
    fd = $fopen("writemem_verify.dat", "r");
    // CHECK: file_created=1
    $display("file_created=%0d", fd != 0);
    $fclose(fd);

    // Read back with $readmemh
    $readmemh("writemem_verify.dat", mem_in);

    // Verify round-trip values
    // CHECK: rt0=de
    $display("rt0=%h", mem_in[0]);
    // CHECK: rt1=ad
    $display("rt1=%h", mem_in[1]);
    // CHECK: rt2=be
    $display("rt2=%h", mem_in[2]);
    // CHECK: rt3=ef
    $display("rt3=%h", mem_in[3]);

    $finish;
  end
endmodule
