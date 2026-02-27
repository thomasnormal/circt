// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s --check-prefix=IR
// REQUIRES: slang

module DelayCycleSupported;
  bit clk;
  int x;

  default clocking cb @(posedge clk);
  endclocking

  initial begin
    ##2 x = 1;
  end
endmodule

// DIAG-NOT: unsupported delay control: CycleDelay

// IR-LABEL: moore.module @DelayCycleSupported
// IR: %[[ONE:.+]] = moore.constant 1 : i32
// IR: %[[TWO:.+]] = moore.constant 2 : i32
// IR: moore.procedure initial {
// IR: cf.br ^[[LOOP:bb[0-9]+]](%[[TWO]] : !moore.i32)
// IR: ^[[LOOP]](%[[COUNT:.+]]: !moore.i32)
// IR: %[[HAS_ITERS:.+]] = moore.bool_cast %[[COUNT]] : i32 -> i1
// IR: %[[HAS_ITERS_BUILTIN:.+]] = moore.to_builtin_bool %[[HAS_ITERS]] : i1
// IR: cf.cond_br %[[HAS_ITERS_BUILTIN]], ^[[WAIT:bb[0-9]+]], ^[[DONE:bb[0-9]+]]
// IR: ^[[WAIT]]
// IR: moore.wait_event {
// IR: moore.detect_event posedge
// IR: }
// IR: %[[DEC:.+]] = moore.sub %[[COUNT]], %[[ONE]] : i32
// IR: cf.br ^[[LOOP]](%[[DEC]] : !moore.i32)
// IR: ^[[DONE]]
// IR: moore.blocking_assign %x, %[[ONE]] : i32
