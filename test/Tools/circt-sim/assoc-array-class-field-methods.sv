// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir && circt-sim %t.mlir --top top | FileCheck %s

class Node;
  bit pred[Node];
  bit succ[Node];

  function void link_to(Node other);
    succ[other] = 1;
    other.pred[this] = 1;
  endfunction

  function void insert_before(Node target, Node newnode);
    Node p;
    foreach (target.pred[p]) begin
      p.succ.delete(target);
      p.succ[newnode] = 1;
    end
    newnode.pred = target.pred;
    target.pred.delete();
    target.pred[newnode] = 1;
    newnode.succ.delete();
    newnode.succ[target] = 1;
  endfunction
endclass

module top;
  Node start;
  Node finish;
  Node mid;

  initial begin
    start = new();
    finish = new();
    mid = new();

    start.link_to(finish);
    start.insert_before(finish, mid);

    // CHECK: start_has_finish=0
    $display("start_has_finish=%0d", start.succ.exists(finish));
    // CHECK: start_has_mid=1
    $display("start_has_mid=%0d", start.succ.exists(mid));
    // CHECK: mid_has_start_pred=1
    $display("mid_has_start_pred=%0d", mid.pred.exists(start));
    // CHECK: finish_has_start_pred=0
    $display("finish_has_start_pred=%0d", finish.pred.exists(start));
    // CHECK: finish_has_mid_pred=1
    $display("finish_has_mid_pred=%0d", finish.pred.exists(mid));

    $finish;
  end
endmodule
