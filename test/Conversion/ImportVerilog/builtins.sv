// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

function void dummyA(int x); endfunction
function void dummyB(real x); endfunction
function void dummyC(shortreal x); endfunction
function void dummyD(string x); endfunction
function void dummyE(byte x); endfunction
function void dummyF(bit x); endfunction

// IEEE 1800-2017 § 20.2 "Simulation control system tasks"
// CHECK-LABEL: func.func private @SimulationControlBuiltins(
function void SimulationControlBuiltins(bit x);
  // CHECK: moore.builtin.finish_message false
  // CHECK: moore.builtin.stop
  $stop;
  // CHECK-NOT: moore.builtin.finish_message
  // CHECK: moore.builtin.stop
  $stop(0);
  // CHECK: moore.builtin.finish_message true
  // CHECK: moore.builtin.stop
  $stop(2);

  // CHECK: moore.builtin.finish_message false
  // CHECK: moore.builtin.finish 0
  // CHECK: moore.unreachable
  if (x) $finish;
  // CHECK-NOT: moore.builtin.finish_message
  // CHECK: moore.builtin.finish 0
  // CHECK: moore.unreachable
  if (x) $finish(0);
  // CHECK: moore.builtin.finish_message true
  // CHECK: moore.builtin.finish 0
  // CHECK: moore.unreachable
  if (x) $finish(2);

  // Ignore `$exit` until we have support for programs.
  // CHECK-NOT: moore.builtin.finish
  $exit;
endfunction

// IEEE 1800-2017 § 20.10 "Severity tasks"
// IEEE 1800-2017 § 21.2 "Display system tasks"
// CHECK-LABEL: func.func private @DisplayAndSeverityBuiltins(
// CHECK-SAME: [[X:%.+]]: !moore.i32
// CHECK-SAME: [[R:%.+]]: !moore.f64
function void DisplayAndSeverityBuiltins(int x, real r);
  // CHECK: [[TMP:%.+]] = moore.fmt.literal "\0A"
  // CHECK: moore.builtin.display [[TMP]]
  $display;
  // CHECK: [[TMP1:%.+]] = moore.fmt.literal "hello"
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $display("hello");

  // CHECK-NOT: moore.builtin.display
  $write;
  $write(,);
  // CHECK: [[TMP:%.+]] = moore.fmt.literal "hello\0A world \\ foo ! bar % \22"
  // CHECK: moore.builtin.display [[TMP]]
  $write("hello\n world \\ foo \x21 bar %% \042");

  // CHECK: [[TMP1:%.+]] = moore.fmt.literal "foo "
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "bar"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $write("foo %s", "bar");

  // CHECK: moore.fmt.int binary [[X]], align right, pad zero : i32
  $write("%b", x);
  // CHECK: moore.fmt.int binary [[X]], align right, pad zero : i32
  $write("%B", x);
  // CHECK: moore.fmt.int binary [[X]], align right, pad zero width 0 : i32
  $write("%0b", x);
  // CHECK: moore.fmt.int binary [[X]], align right, pad zero width 42 : i32
  $write("%42b", x);
  // CHECK: moore.fmt.int binary [[X]], align left, pad zero width 42 : i32
  $write("%-42b", x);

  // CHECK: moore.fmt.int octal [[X]], align right, pad zero : i32
  $write("%o", x);
  // CHECK: moore.fmt.int octal [[X]], align right, pad zero : i32
  $write("%O", x);
  // CHECK: moore.fmt.int octal [[X]], align right, pad zero width 0 : i32
  $write("%0o", x);
  // CHECK: moore.fmt.int octal [[X]], align right, pad zero width 19 : i32
  $write("%19o", x);
  // CHECK: moore.fmt.int octal [[X]], align left, pad zero width 19 : i32
  $write("%-19o", x);

  // CHECK: moore.fmt.real float [[R]], align right : f64
  $write("%f", r);
  // CHECK: moore.fmt.real exponential [[R]], align right : f64
  $write("%e", r);
  // CHECK: moore.fmt.real general [[R]], align right : f64
  $write("%g", r);
  // CHECK: moore.fmt.real float [[R]], align left fracDigits 1 : f64
  $write("%-.f", r);
  // CHECK: moore.fmt.real exponential [[R]], align right fieldWidth 10 : f64
  $write("%10e", r);
  // CHECK: moore.fmt.real general [[R]], align right fracDigits 5 : f64
  $write("%0.5g", r);
  // CHECK: moore.fmt.real float [[R]], align left fieldWidth 20 fracDigits 5 : f64
  $write("%-20.5f", r);
  // CHECK: moore.fmt.real exponential [[R]], align right fieldWidth 10 fracDigits 1 : f64
  $write("%10.e", r);
  // CHECK: moore.fmt.real general [[R]], align right fieldWidth 9 fracDigits 8 : f64
  $write("%9.8g", r);

  // CHECK: [[XR:%.+]] = moore.sint_to_real [[X]] : i32 -> f64
  // CHECK: [[TMP:%.+]] = moore.fmt.real float [[XR]]
  // CHECK: moore.builtin.display [[TMP]]
  $write("%f", x);

  // CHECK: moore.fmt.int decimal [[X]], align right, pad space signed : i32
  $write("%d", x);
  // CHECK: moore.fmt.int decimal [[X]], align right, pad space signed : i32
  $write("%D", x);
  // CHECK: moore.fmt.int decimal [[X]], align right, pad space width 0 signed : i32
  $write("%0d", x);
  // CHECK: moore.fmt.int decimal [[X]], align right, pad space width 19 signed : i32
  $write("%19d", x);
  // CHECK: moore.fmt.int decimal [[X]], align left, pad space width 19 signed : i32
  $write("%-19d", x);

  // CHECK: moore.fmt.int hex_lower [[X]], align right, pad zero : i32
  $write("%h", x);
  // CHECK: moore.fmt.int hex_lower [[X]], align right, pad zero : i32
  $write("%x", x);
  // CHECK: moore.fmt.int hex_upper [[X]], align right, pad zero : i32
  $write("%H", x);
  // CHECK: moore.fmt.int hex_upper [[X]], align right, pad zero : i32
  $write("%X", x);
  // CHECK: moore.fmt.int hex_lower [[X]], align right, pad zero width 0 : i32
  $write("%0h", x);
  // CHECK: moore.fmt.int hex_lower [[X]], align right, pad zero width 19 : i32
  $write("%19h", x);
  // CHECK: moore.fmt.int hex_lower [[X]], align right, pad zero width 19 : i32
  $write("%019h", x);
  // CHECK: moore.fmt.int hex_lower [[X]], align left, pad zero width 19 : i32
  $write("%-19h", x);
  // CHECK: moore.fmt.int hex_lower [[X]], align left, pad zero width 19 : i32
  $write("%-019h", x);

  // Test %p format specifier (pointer/handle format)
  // CHECK: moore.fmt.literal "<ptr>"
  $write("%p", x);
  // CHECK: moore.fmt.literal "<ptr>"
  $write("%0p", x);

  // CHECK: [[TMP:%.+]] = moore.fmt.int decimal [[X]], align right, pad space signed : i32
  // CHECK: moore.builtin.display [[TMP]]
  $write(x);
  // CHECK: [[TMP:%.+]] = moore.fmt.int binary [[X]], align right, pad zero : i32
  // CHECK: moore.builtin.display [[TMP]]
  $writeb(x);
  // CHECK: [[TMP:%.+]] = moore.fmt.int octal [[X]], align right, pad zero : i32
  // CHECK: moore.builtin.display [[TMP]]
  $writeo(x);
  // CHECK: [[TMP:%.+]] = moore.fmt.int hex_lower [[X]], align right, pad zero : i32
  // CHECK: moore.builtin.display [[TMP]]
  $writeh(x);

  // CHECK: [[TMP1:%.+]] = moore.fmt.int decimal [[X]], align right, pad space signed : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $display(x);
  // CHECK: [[TMP1:%.+]] = moore.fmt.int binary [[X]], align right, pad zero : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $displayb(x);
  // CHECK: [[TMP1:%.+]] = moore.fmt.int octal [[X]], align right, pad zero : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $displayo(x);
  // CHECK: [[TMP1:%.+]] = moore.fmt.int hex_lower [[X]], align right, pad zero : i32
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $displayh(x);

  // CHECK: [[TMP1:%.+]] = moore.fmt.real float [[R]]
  // CHECK: [[TMP2:%.+]] = moore.fmt.literal "\0A"
  // CHECK: [[TMP3:%.+]] = moore.fmt.concat ([[TMP1]], [[TMP2]])
  // CHECK: moore.builtin.display [[TMP3]]
  $display("%f", r);

  // CHECK: [[TMP:%.+]] = moore.fmt.literal ""
  // CHECK: moore.builtin.severity info [[TMP]]
  $info;
  // CHECK: [[TMP:%.+]] = moore.fmt.int
  // CHECK: moore.builtin.severity info [[TMP]]
  $info("%d", x);
  // CHECK: [[TMP:%.+]] = moore.fmt.real
  // CHECK: moore.builtin.severity info [[TMP]]
  $info("%f", r);
  // CHECK: [[TMP:%.+]] = moore.fmt.literal ""
  // CHECK: moore.builtin.severity warning [[TMP]]
  $warning;
  // CHECK: [[TMP:%.+]] = moore.fmt.int
  // CHECK: moore.builtin.severity warning [[TMP]]
  $warning("%d", x);
  // CHECK: [[TMP:%.+]] = moore.fmt.real
  // CHECK: moore.builtin.severity warning [[TMP]]
  $warning("%f", r);
  // CHECK: [[TMP:%.+]] = moore.fmt.literal ""
  // CHECK: moore.builtin.severity error [[TMP]]
  $error;
  // CHECK: [[TMP:%.+]] = moore.fmt.int
  // CHECK: moore.builtin.severity error [[TMP]]
  $error("%d", x);
  // CHECK: [[TMP:%.+]] = moore.fmt.real
  // CHECK: moore.builtin.severity error [[TMP]]
  $error("%f", r);
  // Dead code blocks (if (0)) are optimized away by slang, so no $fatal operations appear in output
endfunction

// IEEE 1800-2017 § 20.8 "Math functions"
// CHECK-LABEL: func.func private @MathBuiltins(
// CHECK-SAME: [[X:%.+]]: !moore.i32
// CHECK-SAME: [[Y:%.+]]: !moore.l42
// CHECK-SAME: [[R:%.+]]: !moore.f64
function void MathBuiltins(int x, logic [41:0] y, real r);
  // CHECK: moore.builtin.clog2 [[X]] : i32
  dummyA($clog2(x));
  // CHECK: moore.builtin.clog2 [[Y]] : l42
  dummyA($clog2(y));

  // CHECK:  moore.builtin.ln [[R]] : f64
  dummyB($ln(r));
  // CHECK:  moore.builtin.log10 [[R]] : f64
  dummyB($log10(r));
  // CHECK:  moore.builtin.exp [[R]] : f64
  dummyB($exp(r));
  // CHECK:  moore.builtin.sqrt [[R]] : f64
  dummyB($sqrt(r));
  // CHECK:  moore.builtin.floor [[R]] : f64
  dummyB($floor(r));
  // CHECK:  moore.builtin.ceil [[R]] : f64
  dummyB($ceil(r));
  // CHECK:  moore.builtin.sin [[R]] : f64
  dummyB($sin(r));
  // CHECK:  moore.builtin.cos [[R]] : f64
  dummyB($cos(r));
  // CHECK:  moore.builtin.tan [[R]] : f64
  dummyB($tan(r));
  // CHECK:  moore.builtin.asin [[R]] : f64
  dummyB($asin(r));
  // CHECK:  moore.builtin.acos [[R]] : f64
  dummyB($acos(r));
  // CHECK:  moore.builtin.atan [[R]] : f64
  dummyB($atan(r));
  // CHECK:  moore.builtin.sinh [[R]] : f64
  dummyB($sinh(r));
  // CHECK:  moore.builtin.cosh [[R]] : f64
  dummyB($cosh(r));
  // CHECK:  moore.builtin.tanh [[R]] : f64
  dummyB($tanh(r));
  // CHECK:  moore.builtin.asinh [[R]] : f64
  dummyB($asinh(r));
  // CHECK:  moore.builtin.acosh [[R]] : f64
  dummyB($acosh(r));
  // CHECK:  moore.builtin.atanh [[R]] : f64
  dummyB($atanh(r));

endfunction

// IEEE 1800-2017 Section 20.9 "Bit vector system functions"
// CHECK-LABEL: func.func private @BitVectorBuiltins(
// CHECK-SAME: [[X:%.+]]: !moore.i32
// CHECK-SAME: [[Y:%.+]]: !moore.l42
function void BitVectorBuiltins(int x, logic [41:0] y);
  // CHECK: [[ISUNK_X:%.+]] = moore.builtin.isunknown [[X]] : i32
  dummyF($isunknown(x));
  // CHECK: [[ISUNK_Y:%.+]] = moore.builtin.isunknown [[Y]] : l42
  dummyF($isunknown(y));
  // CHECK: [[CNT_X:%.+]] = moore.builtin.countones [[X]] : i32
  dummyA($countones(x));
  // CHECK: [[CNT_Y:%.+]] = moore.builtin.countones [[Y]] : l42
  dummyA($countones(y));
  // CHECK: [[ONEHOT_X:%.+]] = moore.builtin.onehot [[X]] : i32
  dummyF($onehot(x));
  // CHECK: [[ONEHOT_Y:%.+]] = moore.builtin.onehot [[Y]] : l42
  dummyF($onehot(y));
  // CHECK: [[ONEHOT0_X:%.+]] = moore.builtin.onehot0 [[X]] : i32
  dummyF($onehot0(x));
  // CHECK: [[ONEHOT0_Y:%.+]] = moore.builtin.onehot0 [[Y]] : l42
  dummyF($onehot0(y));
  // CHECK: [[CNTBITS_ONES_X:%.+]] = moore.builtin.countbits [[X]], 2 : i32
  dummyA($countbits(x, 1));
  // CHECK: [[CNTBITS_ZEROS_Y:%.+]] = moore.builtin.countbits [[Y]], 1 : l42
  dummyA($countbits(y, 0));
  // CHECK: [[CNTBITS_BOTH_X:%.+]] = moore.builtin.countbits [[X]], 3 : i32
  dummyA($countbits(x, 0, 1));
  // CHECK: [[CNTBITS_XZ_Y:%.+]] = moore.builtin.countbits [[Y]], 12 : l42
  dummyA($countbits(y, 'x, 'z));
  // CHECK: [[CNTBITS_ALL_X:%.+]] = moore.builtin.countbits [[X]], 15 : i32
  dummyA($countbits(x, '0, '1, 'x, 'z));
endfunction

// CHECK-LABEL: func.func private @RandomBuiltins(
// CHECK-SAME: [[X:%.+]]: !moore.i32, [[Y:%.+]]: !moore.i32) -> !moore.l1 {
function RandomBuiltins(int x, int y);
  // CHECK: [[URAND0:%.+]] = moore.builtin.urandom
  // CHECK-NEXT: call @dummyA([[URAND0]]) : (!moore.i32) -> ()
  dummyA($urandom());
  // CHECK: [[URAND1:%.+]] = moore.builtin.urandom [[X]]
  // CHECK-NEXT: call @dummyA([[URAND1]]) : (!moore.i32) -> ()
  dummyA($urandom(x));
  // CHECK: [[RAND0:%.+]] = moore.builtin.random
  // CHECK-NEXT: call @dummyA([[RAND0]]) : (!moore.i32) -> ()
  dummyA($random());
  // CHECK: [[RAND1:%.+]] = moore.builtin.random [[X]]
  // CHECK-NEXT: call @dummyA([[RAND1]]) : (!moore.i32) -> ()
  dummyA($random(x));
  // CHECK: [[URAND_RANGE0:%.+]] = moore.builtin.urandom_range [[X]]
  // CHECK-NEXT: call @dummyA([[URAND_RANGE0]]) : (!moore.i32) -> ()
  dummyA($urandom_range(x));
  // CHECK: [[URAND_RANGE1:%.+]] = moore.builtin.urandom_range [[X]], [[Y]]
  // CHECK-NEXT: call @dummyA([[URAND_RANGE1]]) : (!moore.i32) -> ()
  dummyA($urandom_range(x, y));
endfunction

// CHECK-LABEL: func.func private @TimeBuiltins(
function TimeBuiltins();
  // CHECK: [[TIME:%.+]] = moore.builtin.time
  // CHECK-NEXT: [[TIMETOLOGIC:%.+]] = moore.time_to_logic [[TIME]]
  dummyA($time());
  // CHECK: [[STIME:%.+]] = moore.builtin.time
  dummyA($stime());
  // CHECK: [[REALTIME:%.+]] = moore.builtin.time
  // TODO: There is no int-to-real conversion yet; change this to dummyB once int-to-real works!
  dummyA($realtime());
endfunction

// CHECK-LABEL: func.func private @ConversionBuiltins(
// CHECK-SAME: [[SINT:%.+]]: !moore.i32
// CHECK-SAME: [[LINT:%.+]]: !moore.i64
// CHECK-SAME: [[SR:%.+]]: !moore.f32
// CHECK-SAME: [[R:%.+]]: !moore.f64
function void ConversionBuiltins(int shortint_in, longint longint_in,
                                 shortreal shortreal_in, real real_in);
  // CHECK: [[B2SR:%.+]] = moore.builtin.bitstoshortreal [[SINT]] : i32
  dummyC($bitstoshortreal(shortint_in));
  // CHECK: [[B2R:%.+]] = moore.builtin.bitstoreal [[LINT]] : i64
  dummyB($bitstoreal(longint_in));
  // CHECK: [[R2B:%.+]] = moore.builtin.realtobits [[R]]
  dummyA($realtobits(real_in));
  // CHECK: [[SR2B:%.+]] = moore.builtin.shortrealtobits [[SR]]
  dummyA($shortrealtobits(shortreal_in));
endfunction

// CHECK-LABEL: func.func private @SeverityBuiltinEdgeCase(
// CHECK-SAME: [[STR:%.+]]: !moore.string
function void SeverityBuiltinEdgeCase(string testStr);
  // CHECK: [[CONST:%.+]] = moore.constant 1234 : i32
  // CHECK-NEXT: [[INTVAR:%.+]] = moore.variable [[CONST]] : <i32>
  int val = 1234;
  // CHECK: moore.read [[INTVAR]] : <i32>
  // CHECK: [[INTVAL1:%.+]] = moore.read [[INTVAR]] : <i32>
  // CHECK-NEXT: [[FMTINT1:%.+]] = moore.fmt.int binary [[INTVAL1]], align right, pad zero : i32
  // CHECK-NEXT: [[LINE1:%.+]] = moore.fmt.literal "\0A"
  // CHECK-NEXT: [[CONCAT1:%.+]] = moore.fmt.concat ([[FMTINT1]], [[LINE1]])
  // CHECK-NEXT: moore.builtin.display [[CONCAT1]]
  $displayb(val);
  // CHECK: moore.read [[INTVAR]] : <i32>
  // CHECK: [[INTVAL2:%.+]] = moore.read [[INTVAR]] : <i32>
  // CHECK-NEXT: [[FMTINT2:%.+]] = moore.fmt.int octal [[INTVAL2]], align right, pad zero : i32
  // CHECK-NEXT: [[LINE2:%.+]] = moore.fmt.literal "\0A"
  // CHECK-NEXT: [[CONCAT2:%.+]] = moore.fmt.concat ([[FMTINT2]], [[LINE2]])
  // CHECK-NEXT: moore.builtin.display [[CONCAT2]]
  $displayo(val);
  // CHECK: moore.read [[INTVAR]] : <i32>
  // CHECK: [[INTVAL3:%.+]] = moore.read [[INTVAR]] : <i32>
  // CHECK-NEXT: [[FMTINT3:%.+]] = moore.fmt.int hex_lower [[INTVAL3]], align right, pad zero : i32
  // CHECK-NEXT: [[LINE3:%.+]] = moore.fmt.literal "\0A"
  // CHECK-NEXT: [[CONCAT3:%.+]] = moore.fmt.concat ([[FMTINT3]], [[LINE3]])
  // CHECK-NEXT: moore.builtin.display [[CONCAT3]]
  $displayh(val);
  // CHECK: [[FMTSTR:%.+]] = moore.fmt.string [[STR]]
  // CHECK-NEXT: [[FMTLIT:%.+]] = moore.fmt.literal " 23"
  // CHECK-NEXT: [[FMTCONCAT:%.+]] = moore.fmt.concat ([[FMTSTR]], [[FMTLIT]])
  // CHECK-NEXT: moore.builtin.severity fatal [[FMTCONCAT]]
  // CHECK: moore.unreachable
  $fatal(1, $sformatf("%s 23", testStr));
endfunction

// CHECK-LABEL: moore.module @SampleValueBuiltins(
// CHECK-SAME: in [[CLK:%.+]] : !moore.l1
module SampleValueBuiltins #() (
    input clk_i
);
  // CHECK: [[CLKWIRE:%.+]] = moore.net name "clk_i" wire : <l1>
  // CHECK: moore.past
  // CHECK: moore.not
  // CHECK: moore.and
  // CHECK: ltl.implication
  // CHECK: verif.{{(clocked_)?}}assert
  rising_clk: assert property (@(posedge clk_i) clk_i |=> $rose(clk_i));
  // CHECK: moore.not
  // CHECK: moore.and
  // CHECK: ltl.implication
  // CHECK: verif.{{(clocked_)?}}assert
  falling_clk: assert property (@(posedge clk_i) clk_i |=> $fell(clk_i));
  // CHECK: moore.case_eq
  // CHECK: ltl.implication
  // CHECK: verif.{{(clocked_)?}}assert
  stable_clk: assert property (@(posedge clk_i) clk_i |=> $stable(clk_i));

endmodule

// CHECK-LABEL: func.func private @StringBuiltins(
// CHECK-SAME: [[STR:%.+]]: !moore.string
// CHECK-SAME: [[INT:%.+]]: !moore.i32
function void StringBuiltins(string string_in, int int_in);
  // CHECK: [[STRVAR:%.+]] = moore.variable
  string result;
  // CHECK: [[LEN:%.+]] = moore.string.len [[STR]]
  dummyA(string_in.len());
  // CHECK: [[LEN:%.+]] = moore.string.toupper [[STR]]
  dummyD(string_in.toupper());
  // CHECK: [[LEN:%.+]] = moore.string.tolower [[STR]]
  dummyD(string_in.tolower());
  // CHECK: [[CHAR:%.+]] = moore.string.getc [[STR]]{{\[}}[[INT]]]
  dummyE(string_in.getc(int_in));
  // CHECK: [[INTCONV:%.+]] = moore.int_to_logic [[INT]] : i32
  // CHECK: moore.string.itoa [[STRVAR]], [[INTCONV]] : <string>, l32
  result.itoa(int_in);
  // CHECK: [[CHARVAL:%.+]] = moore.constant 65 : i8
  // CHECK: moore.string.putc [[STRVAR]]{{\[}}[[INT]]], [[CHARVAL]] : <string>
  result[int_in] = 8'd65;
  // String to integer conversion methods (IEEE 1800-2017 Section 6.16.8)
  // CHECK: [[ATOI:%.+]] = moore.string.atoi [[STR]]
  dummyA(string_in.atoi());
  // CHECK: [[ATOHEX:%.+]] = moore.string.atohex [[STR]]
  dummyA(string_in.atohex());
  // CHECK: [[ATOOCT:%.+]] = moore.string.atooct [[STR]]
  dummyA(string_in.atooct());
  // CHECK: [[ATOBIN:%.+]] = moore.string.atobin [[STR]]
  dummyA(string_in.atobin());
  // String to real conversion method (IEEE 1800-2017 Section 6.16.9)
  // CHECK: [[ATOREAL:%.+]] = moore.string.atoreal [[STR]]
  dummyB(string_in.atoreal());
  // Integer to string conversion methods (IEEE 1800-2017 Section 6.16.9)
  // CHECK: [[INTCONV:%.+]] = moore.int_to_logic [[INT]] : i32
  // CHECK: moore.string.hextoa [[STRVAR]], [[INTCONV]] : <string>, l32
  result.hextoa(int_in);
  // CHECK: [[INTCONV:%.+]] = moore.int_to_logic [[INT]] : i32
  // CHECK: moore.string.octtoa [[STRVAR]], [[INTCONV]] : <string>, l32
  result.octtoa(int_in);
  // CHECK: [[INTCONV:%.+]] = moore.int_to_logic [[INT]] : i32
  // CHECK: moore.string.bintoa [[STRVAR]], [[INTCONV]] : <string>, l32
  result.bintoa(int_in);
  // Real to string conversion method (IEEE 1800-2017 Section 6.16.9)
  // CHECK: [[REALVAL:%.+]] = moore.constant_real 3.140000e+00 : f64
  // CHECK: moore.string.realtoa [[STRVAR]], [[REALVAL]] : <string>, f64
  result.realtoa(3.14);
  // putc method (IEEE 1800-2017 Section 6.16.2)
  // CHECK: [[CHARVAL:%.+]] = moore.constant 66 : i8
  // CHECK: moore.string.putc [[STRVAR]]{{\[}}[[INT]]], [[CHARVAL]] : <string>
  result.putc(int_in, 8'd66);
  // substr method (IEEE 1800-2017 Section 6.16.7)
  // CHECK: [[START:%.+]] = moore.constant 1 : i32
  // CHECK: [[LEN2:%.+]] = moore.constant 3 : i32
  // CHECK: [[SUBSTR:%.+]] = moore.string.substr [[STR]]{{\[}}[[START]], [[LEN2]]]
  // CHECK: call @dummyD([[SUBSTR]]) : (!moore.string) -> ()
  dummyD(string_in.substr(1, 3));
endfunction

// IEEE 1800-2017 Section 6.19.5.5 "Enum .name() method"
typedef enum logic [2:0] {
  ENUM_RED = 3'd0,
  ENUM_GREEN = 3'd1,
  ENUM_BLUE = 3'd2
} color_enum_t;

// CHECK-LABEL: func.func private @EnumNameBuiltin(
// CHECK-SAME: [[COLOR:%.+]]: !moore.l3
function void EnumNameBuiltin(color_enum_t color);
  // CHECK: moore.constant_string "" : i8
  // CHECK: moore.int_to_string
  // CHECK: moore.constant 2 : l3
  // CHECK: moore.eq [[COLOR]], %{{.+}} : l3
  // CHECK: moore.conditional %{{.+}} : l1 -> string {
  // CHECK:   moore.constant_string "ENUM_BLUE" : i72
  // CHECK:   moore.int_to_string
  // CHECK:   moore.yield %{{.+}} : string
  // CHECK: } {
  // CHECK:   moore.yield %{{.+}} : string
  // CHECK: }
  // CHECK: call @dummyD
  dummyD(color.name());
endfunction

// IEEE 1800-2017 Section 20.6.1 "$typename"
// CHECK-LABEL: func.func private @TypenameBuiltin(
// CHECK-SAME: [[INT_ARG:%.+]]: !moore.i32
// CHECK-SAME: [[REAL_ARG:%.+]]: !moore.f64
// CHECK-SAME: [[STR_ARG:%.+]]: !moore.string
// CHECK-SAME: [[BYTE_ARG:%.+]]: !moore.i8
// CHECK-SAME: [[LOGIC_ARG:%.+]]: !moore.l8
function void TypenameBuiltin(int int_arg, real real_arg, string str_arg, byte byte_arg, logic [7:0] logic_arg);
  // CHECK: [[INT_TYPE:%.+]] = moore.constant_string "int" : i24
  // CHECK: [[INT_STR:%.+]] = moore.int_to_string [[INT_TYPE]]
  // CHECK: call @dummyD([[INT_STR]]) : (!moore.string) -> ()
  dummyD($typename(int_arg));

  // CHECK: [[REAL_TYPE:%.+]] = moore.constant_string "real" : i32
  // CHECK: [[REAL_STR:%.+]] = moore.int_to_string [[REAL_TYPE]]
  // CHECK: call @dummyD([[REAL_STR]]) : (!moore.string) -> ()
  dummyD($typename(real_arg));

  // CHECK: [[STR_TYPE:%.+]] = moore.constant_string "string" : i48
  // CHECK: [[STR_STR:%.+]] = moore.int_to_string [[STR_TYPE]]
  // CHECK: call @dummyD([[STR_STR]]) : (!moore.string) -> ()
  dummyD($typename(str_arg));

  // CHECK: [[BYTE_TYPE:%.+]] = moore.constant_string "byte" : i32
  // CHECK: [[BYTE_STR:%.+]] = moore.int_to_string [[BYTE_TYPE]]
  // CHECK: call @dummyD([[BYTE_STR]]) : (!moore.string) -> ()
  dummyD($typename(byte_arg));

  // CHECK: [[LOGIC_TYPE:%.+]] = moore.constant_string "logic[7:0]" : i80
  // CHECK: [[LOGIC_STR:%.+]] = moore.int_to_string [[LOGIC_TYPE]]
  // CHECK: call @dummyD([[LOGIC_STR]]) : (!moore.string) -> ()
  dummyD($typename(logic_arg));
endfunction

// IEEE 1800-2017 Section 21.2.1.3 "Hierarchical name format"
// Test %m format specifier for hierarchical names
// CHECK-LABEL: moore.module @HierarchicalNameModule
module HierarchicalNameModule;
  int x;
  // CHECK: moore.fmt.literal "HierarchicalNameModule"
  // CHECK: moore.fmt.literal "\0A"
  // CHECK: moore.fmt.concat
  // CHECK: moore.builtin.display
  initial $display("%m");

  // CHECK: moore.fmt.literal "Path: "
  // CHECK: moore.fmt.literal "HierarchicalNameModule"
  // CHECK: moore.fmt.literal "\0A"
  // CHECK: moore.fmt.concat
  // CHECK: moore.builtin.display
  initial $display("Path: %m");
endmodule

// IEEE 1800-2017 Section 21.2.1.2 "Format specifications"
// Test %p format specifier for class handles
class FormatTestClass;
  int value;
endclass

// CHECK-LABEL: func.func private @ClassFormatBuiltin(
// CHECK-SAME: [[OBJ:%.+]]: !moore.class<@FormatTestClass>
function void ClassFormatBuiltin(FormatTestClass obj);
  // CHECK: moore.fmt.class [[OBJ]] : <@FormatTestClass>
  // CHECK: moore.fmt.literal "\0A"
  // CHECK: moore.fmt.concat
  // CHECK: moore.builtin.display
  $display("%p", obj);
endfunction

// IEEE 1800-2017 Section 20.14 "Coverage control system functions"
// Test coverage control system functions (stubbed implementations)
// CHECK-LABEL: func.func private @CoverageBuiltins(
function void CoverageBuiltins();
  int unsigned i;
  real r;
  // Coverage control functions return 0 (stubbed)
  // CHECK: [[CC1:%.+]] = moore.constant 0 : i32
  i = $coverage_control(0, 0, 0, "");
  // CHECK: [[CGM:%.+]] = moore.constant 0 : i32
  i = $coverage_get_max(0, 0, "");
  // CHECK: [[CM:%.+]] = moore.constant 0 : i32
  i = $coverage_merge(0, "");
  // CHECK: [[CS:%.+]] = moore.constant 0 : i32
  i = $coverage_save(0, "");
  // Coverage get functions return 0.0 (stubbed)
  // CHECK: [[CG:%.+]] = moore.constant_real 0.0{{.*}}
  r = $coverage_get(0, 0, "");
  // CHECK: [[GC:%.+]] = moore.constant_real 0.0{{.*}}
  r = $get_coverage();
  // Coverage database tasks are no-ops
  $set_coverage_db_name("coverage.db");
  $load_coverage_db("coverage.db");
endfunction
