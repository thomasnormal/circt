// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --max-time=1000000 2>&1 | FileCheck %s

class Counter;
  int count;
  
  function new();
    count = 0;
  endfunction
  
  function void get_count(output int val);
    val = count;  // This is a write to output - works
  endfunction
  
  function int read_ref(ref int val);
    return val;  // This is a read from ref - needs testing
  endfunction
  
  function void increment();
    count = count + 1;
  endfunction
endclass

module top;
  initial begin
    Counter c;
    int x, result;
    
    c = new();
    c.increment();
    c.increment();
    
    // Test reading ref parameter
    x = 42;
    result = c.read_ref(x);
    
    // CHECK: result = 42
    $display("result = %d", result);
    
    if (result == 42) begin
      // CHECK: PASS
      $display("PASS");
    end else begin
      $display("FAIL");
    end
    
    $finish;
  end
endmodule
