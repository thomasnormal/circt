// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// TODO: $isunbounded with class type parameters — compilation or runtime failure.
module top;
  // $isunbounded returns 1 if the argument is $ (unbounded), 0 otherwise
  // The $ literal is only valid in certain type parameter contexts

  class bounded_queue #(int MAX = $);
    int q[$:MAX];
    function int is_max_unbounded();
      return $isunbounded(MAX);
    endfunction
  endclass

  initial begin
    // Unbounded case: default MAX = $
    bounded_queue #() unbounded_q = new();
    // CHECK: unbounded=1
    $display("unbounded=%0d", unbounded_q.is_max_unbounded());

    // Bounded case: MAX = 10
    bounded_queue #(10) bounded_q = new();
    // CHECK: bounded=0
    $display("bounded=%0d", bounded_q.is_max_unbounded());

    $finish;
  end
endmodule
