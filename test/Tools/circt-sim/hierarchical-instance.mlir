// RUN: circt-sim %s --top=top --max-time=100000000 --sim-stats 2>&1 | FileCheck %s

// This test verifies that circt-sim correctly handles hierarchical module
// instantiation by walking into hw.instance operations and registering
// processes from child modules.

// CHECK: Registered {{[0-9]+}} LLHD signals and 3 LLHD processes
// CHECK: Simulation completed

// Child module with a process that runs a counter
hw.module private @counter(in %clk : !llhd.ref<i1>, in %rst : !llhd.ref<i1>) {
  %c1_i8 = hw.constant 1 : i8
  %c0_i8 = hw.constant 0 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %count = llhd.sig %c0_i8 : i8

  // Counter process - increments on each clock edge
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %clkVal = llhd.prb %clk : i1
    llhd.wait (%clkVal : i1), ^bb2
  ^bb2:
    %clkNew = llhd.prb %clk : i1
    %rstVal = llhd.prb %rst : i1
    // Reset check
    %notRst = comb.xor bin %rstVal, %true : i1
    cf.cond_br %notRst, ^bb3, ^bb1
  ^bb3:
    // Increment on posedge
    %oldCount = llhd.prb %count : i8
    %newCount = comb.add %oldCount, %c1_i8 : i8
    llhd.drv %count, %newCount after %0 : i8
    cf.br ^bb1
  }
  %true = hw.constant true
  hw.output
}

// Top module with clock generation and counter instance
hw.module @top() {
  %c10000000_i64 = hw.constant 10000000 : i64
  %c50000000_i64 = hw.constant 50000000 : i64
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %true = hw.constant true

  // Clock signal
  %clk = llhd.sig %false : i1

  // Reset signal
  %rst = llhd.sig %true : i1

  // Clock generator process
  llhd.process {
    llhd.drv %clk, %false after %0 : i1
    cf.br ^bb1
  ^bb1:
    %1 = llhd.int_to_time %c10000000_i64
    llhd.wait delay %1, ^bb2
  ^bb2:
    %2 = llhd.prb %clk : i1
    %3 = comb.xor %2, %true : i1
    llhd.drv %clk, %3 after %0 : i1
    cf.br ^bb1
  }

  // Reset sequence process
  llhd.process {
    llhd.drv %rst, %true after %0 : i1
    %1 = llhd.int_to_time %c50000000_i64
    llhd.wait delay %1, ^bb1
  ^bb1:
    llhd.drv %rst, %false after %0 : i1
    llhd.halt
  }

  // Instantiate the counter module
  hw.instance "counter0" @counter(clk: %clk: !llhd.ref<i1>, rst: %rst: !llhd.ref<i1>) -> ()

  hw.output
}
