// RUN: circt-sim %s --max-time=1000000 --max-process-steps=100 2>&1 | FileCheck %s

// Null self-pointer BaudClk helper calls should not spin in no-op wait_event
// loops. The fast path stalls these calls behind a non-waking waiter.
//
// CHECK: done
// CHECK: [circt-sim] Simulation completed

module {
  llvm.mlir.global internal @"Toy::BaudClkGenerator::count"(0 : i32) {
    addr_space = 0 : i32
  } : i32

  func.func private @"Toy::BaudClkGenerator"(%arg0: !llvm.ptr, %arg1: i32) {
    %c-1_i32 = hw.constant -1 : i32
    %true = hw.constant true
    %c0_i32 = hw.constant 0 : i32
    %c1_i32 = hw.constant 1 : i32
    %count = llvm.mlir.addressof @"Toy::BaudClkGenerator::count" : !llvm.ptr
    cf.br ^bb1
  ^bb1:  // 3 preds: ^bb0, ^bb2, ^bb3
    moore.wait_event {
      %clk_ptr = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"toy_if", (i1, i8, i8, i8, i1)>
      %clk_pos = llvm.load %clk_ptr : !llvm.ptr -> i1
      %clk_pos_m = builtin.unrealized_conversion_cast %clk_pos : i1 to !moore.l1
      moore.detect_event posedge %clk_pos_m : l1
      %clk_neg = llvm.load %clk_ptr : !llvm.ptr -> i1
      %clk_neg_m = builtin.unrealized_conversion_cast %clk_neg : i1 to !moore.l1
      moore.detect_event negedge %clk_neg_m : l1
    }
    %cur = llvm.load %count : !llvm.ptr -> i32
    %target = comb.add %arg1, %c-1_i32 : i32
    %is_toggle = comb.icmp eq %cur, %target : i32
    cf.cond_br %is_toggle, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.store %c0_i32, %count : i32, !llvm.ptr
    %out_ptr = llvm.getelementptr %arg0[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"toy_if", (i1, i8, i8, i8, i1)>
    %out_cur = llvm.load %out_ptr : !llvm.ptr -> i1
    %out_next = comb.xor %out_cur, %true : i1
    llvm.store %out_next, %out_ptr : i1, !llvm.ptr
    cf.br ^bb1
  ^bb3:  // pred: ^bb1
    %next = comb.add %cur, %c1_i32 : i32
    llvm.store %next, %count : i32, !llvm.ptr
    cf.br ^bb1
  }

  hw.module @test() {
    %c2_i32 = hw.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %fmt = sim.fmt.literal "done\0A"

    // Null-handle call should not run as a hot no-op loop.
    llhd.process {
      func.call @"Toy::BaudClkGenerator"(%null, %c2_i32) : (!llvm.ptr, i32) -> ()
      llhd.halt
    }

    // Independent terminator process. Without the null-self stall path above,
    // the BaudClk process can hit step overflow before this executes.
    llhd.process {
      %d = llhd.int_to_time %c1_i64
      llhd.wait delay %d, ^done
    ^done:
      sim.proc.print %fmt
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
