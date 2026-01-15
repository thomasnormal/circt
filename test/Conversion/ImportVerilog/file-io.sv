// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// IEEE 1800-2017 Section 21.3 "Opening and closing files"

// CHECK-LABEL: moore.module @FileIOTest
module FileIOTest;
  int fd;

  initial begin
    // CHECK: [[FILENAME1:%.+]] = moore.constant_string "test.txt"
    // CHECK: [[STR1:%.+]] = moore.int_to_string [[FILENAME1]]
    // CHECK: [[FD1:%.+]] = moore.builtin.fopen [[STR1]]
    // CHECK: moore.blocking_assign %fd, [[FD1]]
    fd = $fopen("test.txt");

    // CHECK: [[FILENAME2:%.+]] = moore.constant_string "output.log"
    // CHECK: [[STR2:%.+]] = moore.int_to_string [[FILENAME2]]
    // CHECK: [[MODE2:%.+]] = moore.constant_string "w"
    // CHECK: [[MODESTR2:%.+]] = moore.int_to_string [[MODE2]]
    // CHECK: [[FD2:%.+]] = moore.builtin.fopen [[STR2]], [[MODESTR2]]
    // CHECK: moore.blocking_assign %fd, [[FD2]]
    fd = $fopen("output.log", "w");

    // CHECK: [[FILENAME3:%.+]] = moore.constant_string "append.txt"
    // CHECK: [[STR3:%.+]] = moore.int_to_string [[FILENAME3]]
    // CHECK: [[MODE3:%.+]] = moore.constant_string "a"
    // CHECK: [[MODESTR3:%.+]] = moore.int_to_string [[MODE3]]
    // CHECK: [[FD3:%.+]] = moore.builtin.fopen [[STR3]], [[MODESTR3]]
    // CHECK: moore.blocking_assign %fd, [[FD3]]
    fd = $fopen("append.txt", "a");

    // CHECK: [[FILENAME4:%.+]] = moore.constant_string "readwrite.dat"
    // CHECK: [[STR4:%.+]] = moore.int_to_string [[FILENAME4]]
    // CHECK: [[MODE4:%.+]] = moore.constant_string "r+"
    // CHECK: [[MODESTR4:%.+]] = moore.int_to_string [[MODE4]]
    // CHECK: [[FD4:%.+]] = moore.builtin.fopen [[STR4]], [[MODESTR4]]
    // CHECK: moore.blocking_assign %fd, [[FD4]]
    fd = $fopen("readwrite.dat", "r+");
  end
endmodule
