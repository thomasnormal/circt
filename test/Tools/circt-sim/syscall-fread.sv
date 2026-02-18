// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  integer fd, code;
  reg [7:0] data [0:3];
  reg [31:0] word;

  initial begin
    // Write binary data to file
    fd = $fopen("fread_test.bin", "wb");
    $fwrite(fd, "%c%c%c%c", 8'hDE, 8'hAD, 8'hBE, 8'hEF);
    $fclose(fd);

    // Read it back with $fread into array
    fd = $fopen("fread_test.bin", "rb");
    code = $fread(data, fd);
    // CHECK: fread returned 4
    $display("fread returned %0d", code);
    // CHECK: data[0]=de
    $display("data[0]=%h", data[0]);
    // CHECK: data[1]=ad
    $display("data[1]=%h", data[1]);
    // CHECK: data[2]=be
    $display("data[2]=%h", data[2]);
    // CHECK: data[3]=ef
    $display("data[3]=%h", data[3]);
    $fclose(fd);

    // Read into a single reg
    fd = $fopen("fread_test.bin", "rb");
    code = $fread(word, fd);
    // CHECK: word=deadbeef
    $display("word=%h", word);
    $fclose(fd);

    $finish;
  end
endmodule
