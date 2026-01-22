// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test for file I/O system calls with output arguments

module file_io_test;
  int fd;
  string str;
  integer errno;

  // CHECK-LABEL: moore.module @file_io_test
  initial begin
    // Test $ferror - returns error code and writes message to str
    // CHECK: %[[FD1:.*]] = moore.read %fd
    // CHECK: moore.builtin.ferror %[[FD1]], %str : <string>
    fd = $fopen("test.txt", "w");
    errno = $ferror(fd, str);

    // Test $fgets - reads line from file into string
    // CHECK: moore.builtin.fgets %str, %[[FD2:.*]] : <string>
    $fgets(str, fd);

    // Test $ungetc - pushes character back to stream
    // CHECK: moore.builtin.ungetc
    $ungetc(65, fd);

    $fclose(fd);
  end
endmodule
