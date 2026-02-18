// RUN: circt-sim %s --max-time=1000000 --max-process-steps=200 2>&1 | FileCheck %s

// If the BaudClkGenerator count global is externally visible, the fast path
// must preserve memory writes for that global.
//
// CHECK: count=1
// CHECK: [circt-sim] Simulation completed

module {
  llvm.func @malloc(i64) -> !llvm.ptr

  llvm.mlir.global internal @"Toy::BaudClkGenerator::count"(0 : i32) {
    addr_space = 0 : i32
  } : i32

  llvm.mlir.global internal @"toy::obj_ptr"() {addr_space = 0 : i32} : !llvm.ptr

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
    %c1_i64 = arith.constant 1 : i64
    %c10_i64 = arith.constant 10 : i64
    %c80_i64 = arith.constant 80 : i64
    %c8_i64 = arith.constant 8 : i64
    %c0_i32 = hw.constant 0 : i32
    %c1_i32 = hw.constant 1 : i32
    %c2_i32 = hw.constant 2 : i32
    %c5_i32 = hw.constant 5 : i32
    %false = hw.constant false
    %true = hw.constant true
    %c0_i8 = hw.constant 0 : i8

    %fmt_pre = sim.fmt.literal "count="
    %fmt_nl = sim.fmt.literal "\0A"

    // Setup object fields.
    llhd.process {
      %obj = llvm.call @malloc(%c8_i64) : (i64) -> !llvm.ptr
      %gptr = llvm.mlir.addressof @"toy::obj_ptr" : !llvm.ptr
      llvm.store %obj, %gptr : !llvm.ptr, !llvm.ptr
      %clk_ptr = llvm.getelementptr %obj[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"toy_if", (i1, i8, i8, i8, i1)>
      %out_ptr = llvm.getelementptr %obj[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"toy_if", (i1, i8, i8, i8, i1)>
      llvm.store %false, %clk_ptr : i1, !llvm.ptr
      llvm.store %false, %out_ptr : i1, !llvm.ptr
      %f1 = llvm.getelementptr %obj[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"toy_if", (i1, i8, i8, i8, i1)>
      %f2 = llvm.getelementptr %obj[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"toy_if", (i1, i8, i8, i8, i1)>
      %f3 = llvm.getelementptr %obj[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"toy_if", (i1, i8, i8, i8, i1)>
      llvm.store %c0_i8, %f1 : i8, !llvm.ptr
      llvm.store %c0_i8, %f2 : i8, !llvm.ptr
      llvm.store %c0_i8, %f3 : i8, !llvm.ptr
      llhd.halt
    }

    // Clock driver: five edge events.
    llhd.process {
      %start = llhd.int_to_time %c1_i64
      llhd.wait delay %start, ^entry
    ^entry:
      %gptr = llvm.mlir.addressof @"toy::obj_ptr" : !llvm.ptr
      %obj = llvm.load %gptr : !llvm.ptr -> !llvm.ptr
      cf.br ^loop(%c0_i32, %false, %obj : i32, i1, !llvm.ptr)
    ^loop(%iter: i32, %clk: i1, %obj_loop: !llvm.ptr):
      %done = comb.icmp eq %iter, %c5_i32 : i32
      cf.cond_br %done, ^halt, ^step
    ^step:
      %delay = llhd.int_to_time %c10_i64
      llhd.wait delay %delay, ^toggle(%iter, %clk, %obj_loop : i32, i1, !llvm.ptr)
    ^toggle(%iter_t: i32, %clk_t: i1, %obj_t: !llvm.ptr):
      %next_clk = comb.xor %clk_t, %true : i1
      %clk_ptr = llvm.getelementptr %obj_t[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"toy_if", (i1, i8, i8, i8, i1)>
      llvm.store %next_clk, %clk_ptr : i1, !llvm.ptr
      %next_iter = comb.add %iter_t, %c1_i32 : i32
      cf.br ^loop(%next_iter, %next_clk, %obj_t : i32, i1, !llvm.ptr)
    ^halt:
      llhd.halt
    }

    // Divider process under test.
    llhd.process {
      %start = llhd.int_to_time %c1_i64
      llhd.wait delay %start, ^entry
    ^entry:
      %gptr = llvm.mlir.addressof @"toy::obj_ptr" : !llvm.ptr
      %obj = llvm.load %gptr : !llvm.ptr -> !llvm.ptr
      func.call @"Toy::BaudClkGenerator"(%obj, %c2_i32) : (!llvm.ptr, i32) -> ()
      llhd.halt
    }

    // Checker reads the count global. This must observe fast-path updates.
    llhd.process {
      %check_delay = llhd.int_to_time %c80_i64
      llhd.wait delay %check_delay, ^check
    ^check:
      %count_ptr = llvm.mlir.addressof @"Toy::BaudClkGenerator::count" : !llvm.ptr
      %count = llvm.load %count_ptr : !llvm.ptr -> i32
      %fmt_val = sim.fmt.dec %count : i32
      %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
      sim.proc.print %fmt_out
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
