// RUN: circt-translate --import-verilog --verify-diagnostics --split-input-file %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// expected-error @below {{expected ';'}}
module Foo 4;
endmodule

// -----
// expected-note @below {{expanded from macro 'FOO'}}
`define FOO input
// expected-note @below {{expanded from macro 'BAR'}}
`define BAR `FOO
// expected-error @below {{expected identifier}}
module Bar(`BAR);
endmodule

// -----
module Foo;
  mailbox a;
  string b;
  // expected-error @below {{value of type 'string' cannot be assigned to type 'mailbox'}}
  initial a = b;
endmodule

// -----
// nettype is now supported (ignored)
module Foo;
  nettype real x;
endmodule

// -----
// interconnect is now supported
module Foo;
  interconnect x;
endmodule

// -----
module Foo;
  int x;
  bit y;
  // expected-error @below {{unsupported non-blocking assignment timing control: SignalEvent}}
  initial x <= @y x;
endmodule

// -----
module Foo;
  int x;
  // expected-error @below {{implicit events cannot be used here}}
  initial x = @* x;
endmodule

// -----
module Foo;
  int a;
  // expected-remark @below {{release statement ignored (simplified simulation semantics)}}
  initial release a;
endmodule

// -----
// unpacked arrays in inside expressions are now supported
module Foo;
  int a, b[3];
  int c = a inside { b };
endmodule

// -----
module Foo;
  int a, b, c;
  int j;
  initial begin
    // expected-error @below {{streaming operator target size 32 does not fit source size 96}}
    j = {>>{ a, b, c }}; // error: j is 32 bits < 96 bits
  end
endmodule


// -----
module Foo;
  int a, b, c;
  int j;
  initial begin
    // expected-error @below {{streaming operator target size 96 does not fit source size 23}}
    {>>{ a, b, c }} = 23'b1;
  end
endmodule

// -----
module Foo;
  initial begin
    logic [15:0] vec_0;
    logic [47:0] vec_1;
    logic arr [63:0];
    int c;
    // expected-error @below {{Moore only support streaming concatenation with fixed size 'with expression'}}
    vec_1 = {<<byte{vec_0, arr with [c:0]}};
  end
endmodule

// -----
module Foo;
  // expected-remark @below {{hello}}
  $info("hello");
  // expected-warning @below {{hello}}
  $warning("hello");
endmodule

// -----
module Foo;
  // expected-error @below {{hello}}
  $error("hello");
endmodule

// -----
module Foo;
  // expected-error @below {{hello}}
  $fatal(0, "hello");
endmodule

// -----
function Foo;
  // expected-error @below {{unsupported format specifier `%l`}}
  $write("%l");
endfunction

// -----
// String format specifier with width is now supported
function Foo;
  $write("%42s", "foo");
endfunction

// -----
function time Foo;
  // expected-error @below {{time value is larger than 18446744073709549568 fs}}
  return 100000s;
endfunction

// -----
// associative arrays with wildcard index are now supported
module Foo;
  int x[*];
endmodule

// -----
// Queue slices with $ are now supported
function void foo();
  int q[$];
  q = q[2:$]; // No longer an error - queue slicing now works
endfunction

// -----
// Associative array element select is now supported
function void foo;
  int a[string];
  a["foo"] = 1;
endfunction

// -----
module TimeTypeConversion1;
  struct packed { time t; } a;
  int b;
  // expected-error @below {{contains a time type}}
  assign a = b;
endmodule

// -----
module TimeTypeConversion2;
  int a;
  struct packed { time t; } b;
  // expected-error @below {{contains a time type}}
  assign a = b;
endmodule
