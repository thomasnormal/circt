// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// TODO: Comprehensive file I/O test — individual ops work but combined sequence fails.
// Test $feof, $fgetc, $fgets, $ftell, $fseek, $rewind, $fflush
module top;
  integer fd, c, pos, count;
  reg [8*32-1:0] line;

  initial begin
    // Create a test file
    fd = $fopen("feof_test.dat", "w");
    $fwrite(fd, "AB\n");
    $fclose(fd);

    // Open for reading
    fd = $fopen("feof_test.dat", "r");

    // $feof should return 0 (not at end)
    // CHECK: feof_start=0
    $display("feof_start=%0d", $feof(fd));

    // $fgetc reads one character
    c = $fgetc(fd);
    // CHECK: fgetc=65
    $display("fgetc=%0d", c);  // 'A' = 65

    // $ftell returns current position
    pos = $ftell(fd);
    // CHECK: ftell=1
    $display("ftell=%0d", pos);

    // $fseek to beginning
    $fseek(fd, 0, 0);
    pos = $ftell(fd);
    // CHECK: ftell_after_seek=0
    $display("ftell_after_seek=%0d", pos);

    // $fgets reads a line
    count = $fgets(line, fd);
    // CHECK: fgets_count=3
    $display("fgets_count=%0d", count);

    // $feof should return non-zero at end
    // CHECK: feof_end=1
    $display("feof_end=%0d", $feof(fd) != 0);

    // $rewind goes to beginning
    $rewind(fd);
    pos = $ftell(fd);
    // CHECK: ftell_after_rewind=0
    $display("ftell_after_rewind=%0d", pos);

    $fclose(fd);

    // $fflush with no args (flush all)
    $fflush;

    // CHECK: done
    $display("done");
    $finish;
  end
endmodule
