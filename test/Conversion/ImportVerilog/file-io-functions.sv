// RUN: circt-verilog --ir-moore %s | FileCheck %s
// Tests for file I/O system functions: $feof and $fgetc

module FileIOFunctions;
  int fd;
  int ch;
  int eof_status;

  initial begin
    fd = $fopen("test.txt", "r");

    // CHECK: moore.builtin.fgetc
    ch = $fgetc(fd);

    // CHECK: moore.builtin.feof
    eof_status = $feof(fd);

    // Test in a loop pattern
    while (!$feof(fd)) begin
      // CHECK: moore.builtin.fgetc
      ch = $fgetc(fd);
    end

    $fclose(fd);
  end
endmodule
