// RUN: circt-verilog %s --ir-hw --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// CHECK: wand_logic_0=0
// CHECK: wand_logic_1=1
// CHECK: wor_logic_0=0
// CHECK: wor_logic_1=1
// CHECK: wand_integer=8
// CHECK: wor_integer=11

module top;
    // wand: multiple drivers resolved via bitwise AND
    wand logic wand_logic_0;
    assign wand_logic_0 = 0;
    assign wand_logic_0 = 1;
    // 0 AND 1 = 0

    wand logic wand_logic_1;
    assign wand_logic_1 = 1;
    assign wand_logic_1 = 1;
    // 1 AND 1 = 1

    // wor: multiple drivers resolved via bitwise OR
    wor logic wor_logic_0;
    assign wor_logic_0 = 0;
    assign wor_logic_0 = 0;
    // 0 OR 0 = 0

    wor logic wor_logic_1;
    assign wor_logic_1 = 1;
    assign wor_logic_1 = 0;
    // 1 OR 0 = 1

    // Multi-bit wand/wor
    wand integer wand_integer;
    assign wand_integer = 4'b1001;
    assign wand_integer = 4'b1010;
    // 1001 AND 1010 = 1000 = 8

    wor integer wor_integer;
    assign wor_integer = 4'b1001;
    assign wor_integer = 4'b1010;
    // 1001 OR 1010 = 1011 = 11

    initial begin
        #1;
        $display("wand_logic_0=%0d", wand_logic_0);
        $display("wand_logic_1=%0d", wand_logic_1);
        $display("wor_logic_0=%0d", wor_logic_0);
        $display("wor_logic_1=%0d", wor_logic_1);
        $display("wand_integer=%0d", wand_integer);
        $display("wor_integer=%0d", wor_integer);
        $finish;
    end
endmodule
