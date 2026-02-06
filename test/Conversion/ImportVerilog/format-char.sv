// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// CHECK-LABEL: moore.module @charFormatConversion
module charFormatConversion();
  int x = 65;

  // CHECK: procedure initial
  // CHECK-NEXT: [[READ:%.+]] = moore.read %x : <i32>
  // CHECK-NEXT: [[FMT:%.+]] = moore.fmt.char [[READ]] : i32
  // CHECK-NEXT: [[LINEBREAK:%.+]] = moore.fmt.literal "\0A"
  // CHECK-NEXT: [[CONCAT:%.+]] = moore.fmt.concat ([[FMT]], [[LINEBREAK]])
  // CHECK-NEXT: moore.builtin.display [[CONCAT]]
  initial begin
    $display("%c", x);
  end
endmodule
