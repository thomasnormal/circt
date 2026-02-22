// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-translate --import-verilog %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemCoverageSdfStaticSubroutine(input logic clk, a);
  sequence s;
    (1, $set_coverage_db_name("cov.db"), $load_coverage_db("cov.db"),
     $sdf_annotate("top.sdf"), $static_assert(1)) ##1 a;
  endsequence

  // Coverage/SDF/static tasks should be recognized in match-items.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemCoverageSdfStaticSubroutine
  // CHECK: verif.assert
  assert property (@(posedge clk) s);
endmodule

// DIAG: warning: $sdf_annotate is not supported in circt-sim (SDF timing annotation not implemented)
// DIAG-NOT: ignoring system subroutine `$set_coverage_db_name`
// DIAG-NOT: ignoring system subroutine `$load_coverage_db`
// DIAG-NOT: ignoring system subroutine `$sdf_annotate`
// DIAG-NOT: ignoring system subroutine `$static_assert`
