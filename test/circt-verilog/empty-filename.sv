// RUN: circt-verilog "" %s 2>&1 | FileCheck %s --check-prefix=CHECK-EMPTY
// REQUIRES: slang

// Check that empty filenames are gracefully handled with a warning
// CHECK-EMPTY: warning: ignoring empty input filename

module Foo;
endmodule
