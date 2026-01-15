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

// IEEE 1800-2017 Section 21.3 "File I/O tasks and functions"
// CHECK-LABEL: func.func private @FileIOBuiltins(
// CHECK-SAME: %arg0: !moore.i32, %arg1: !moore.i32, %arg2: !moore.f64
function void FileIOBuiltins(int fd, int x, real r);
  // CHECK: [[TMP1:%.+]] = moore.fmt.literal "hello"
  // CHECK: moore.builtin.fwrite %arg0, [[TMP1]] : i32
  $fwrite(fd, "hello");

  // CHECK: [[TMP1:%.+]] = moore.fmt.literal "hello"
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.fwrite %arg0, [[TMP3]] : i32
  $fdisplay(fd, "hello");

  // CHECK: [[TMP1:%.+]] = moore.fmt.literal "value: "
  // CHECK: [[TMP2:%.+]] = moore.fmt.int decimal %arg1, align right, pad space signed : i32
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.fwrite %arg0, [[TMP3]] : i32
  $fwrite(fd, "value: %d", x);

  // Test with binary format suffix
  // CHECK: [[TMP1:%.+]] = moore.fmt.int binary %arg1, align right, pad zero : i32
  // CHECK: moore.builtin.fwrite %arg0, [[TMP1]] : i32
  $fwriteb(fd, x);

  // Test with octal format suffix
  // CHECK: [[TMP1:%.+]] = moore.fmt.int octal %arg1, align right, pad zero : i32
  // CHECK: moore.builtin.fwrite %arg0, [[TMP1]] : i32
  $fwriteo(fd, x);

  // Test with hex format suffix
  // CHECK: [[TMP1:%.+]] = moore.fmt.int hex_lower %arg1, align right, pad zero : i32
  // CHECK: moore.builtin.fwrite %arg0, [[TMP1]] : i32
  $fwriteh(fd, x);

  // Test $fdisplay variants
  // CHECK: [[TMP1:%.+]] = moore.fmt.int binary %arg1, align right, pad zero : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.fwrite %arg0, [[TMP3]] : i32
  $fdisplayb(fd, x);

endfunction
