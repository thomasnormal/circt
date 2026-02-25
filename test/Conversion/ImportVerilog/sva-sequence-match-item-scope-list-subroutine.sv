// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemScopeListSubroutine(input logic clk, a);
  sequence s_scope;
    (1, $scope(SVASequenceMatchItemScopeListSubroutine)) ##1 a;
  endsequence
  sequence s_list;
    (1, $list(SVASequenceMatchItemScopeListSubroutine)) ##1 a;
  endsequence

  // Scope/list debug tasks should be recognized and not ignored in match-items.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemScopeListSubroutine
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk) s_scope);
  assert property (@(posedge clk) s_list);
endmodule

// DIAG-NOT: ignoring system subroutine `$scope`
// DIAG-NOT: ignoring system subroutine `$list`
