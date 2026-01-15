// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// Time Type Handling Tests
//===----------------------------------------------------------------------===//
// Test that the time type is correctly handled throughout the conversion
// pipeline, including:
// - Variable declarations
// - Function return types
// - Default value generation (Mem2Reg getDefaultValue)

// Test basic time variable declaration
// CHECK-LABEL: moore.module @TimeVariableBasic() {
// Time literals are converted with the correct scale (100ns -> fs)
// CHECK:   moore.constant_time {{[0-9]+}} fs
// CHECK:   %t = moore.variable : <time>
// CHECK:   moore.procedure initial {
// CHECK:     moore.blocking_assign %t
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module TimeVariableBasic;
    time t;
    initial begin
        t = 100;
    end
endmodule

// Test time variable with realtime (should be same type)
// Unused variables may be optimized away
// CHECK-LABEL: moore.module @RealtimeVariable() {
// CHECK:   moore.output
// CHECK: }

module RealtimeVariable;
    realtime rt;
endmodule

// Test time type in arithmetic operations
// CHECK-LABEL: moore.module @TimeArithmetic() {
// CHECK:   %t1 = moore.variable : <time>
// CHECK:   %t2 = moore.variable : <time>
// CHECK:   %result = moore.variable : <time>
// CHECK:   moore.procedure initial {
// Time operations require conversion to/from logic
// CHECK:     moore.read %t1
// CHECK:     moore.read %t2
// CHECK:     moore.time_to_logic
// CHECK:     moore.add
// CHECK:     moore.logic_to_time
// CHECK:     moore.blocking_assign %result
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module TimeArithmetic;
    time t1;
    time t2;
    time result;
    initial begin
        result = t1 + t2;
    end
endmodule

// Test time comparison
// CHECK-LABEL: moore.module @TimeComparison() {
// CHECK:   %t1 = moore.variable : <time>
// CHECK:   %t2 = moore.variable : <time>
// CHECK:   %flag = moore.variable : <i1>
// CHECK:   moore.procedure initial {
// CHECK:     moore.read %t1
// CHECK:     moore.read %t2
// Time comparison via time_to_logic
// CHECK:     moore.time_to_logic
// CHECK:     moore.ult
// CHECK:     moore.blocking_assign %flag
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module TimeComparison;
    time t1;
    time t2;
    bit flag;
    initial begin
        flag = (t1 < t2);
    end
endmodule

// Test time in function argument
// CHECK: func.func private @timeArgFunc(%arg0: !moore.time) -> !moore.time {
// CHECK:   return %arg0 : !moore.time
// CHECK: }

function time timeArgFunc(time t);
    return t;
endfunction

// CHECK-LABEL: moore.module @TimeFunction() {
// CHECK:   %t = moore.variable : <time>
// CHECK:   %result = moore.variable : <time>
// CHECK:   moore.procedure initial {
// CHECK:     func.call @timeArgFunc
// CHECK:     moore.blocking_assign %result
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module TimeFunction;
    time t;
    time result;
    initial begin
        result = timeArgFunc(t);
    end
endmodule

// Test time in struct (packed struct with time member)
// Unused variables may be optimized away
// CHECK-LABEL: moore.module @TimeInStruct() {
// CHECK:   moore.output
// CHECK: }

module TimeInStruct;
    struct packed {
        time t;
        int value;
    } s;
endmodule

// Test time system functions
// CHECK-LABEL: moore.module @TimeSystemFuncs() {
// CHECK:   %current_time = moore.variable : <time>
// CHECK:   moore.procedure initial {
// $time is lowered to moore.builtin.time
// CHECK:     moore.builtin.time
// CHECK:     moore.blocking_assign %current_time
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module TimeSystemFuncs;
    time current_time;
    initial begin
        current_time = $time;
    end
endmodule

// Test time literal values
// CHECK-LABEL: moore.module @TimeLiterals() {
// Large time values are correctly represented
// CHECK:   moore.constant_time {{[0-9]+}} fs
// CHECK:   %t = moore.variable : <time>
// CHECK:   moore.procedure initial {
// CHECK:     moore.blocking_assign %t
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module TimeLiterals;
    time t;
    initial begin
        t = 1000000;  // Large time value
    end
endmodule
