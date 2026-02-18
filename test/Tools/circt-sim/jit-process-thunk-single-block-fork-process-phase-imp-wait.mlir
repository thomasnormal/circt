// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=16 --jit-fail-on-deopt --max-time=10 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG-DAG: parent_done
// LOG-DAG: slot=16384
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func @"uvm_pkg::uvm_build_phase::new"(%self: !llvm.ptr) {
    return
  }

  func.func @"uvm_pkg::uvm_connect_phase::new"(%self: !llvm.ptr) {
    return
  }

  func.func @"uvm_pkg::uvm_phase_hopper::process_phase"(%hopper: !llvm.ptr, %phase: !llvm.ptr) {
    return
  }

  llvm.mlir.global internal @"phase_vtable"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"uvm_pkg::uvm_build_phase::new"], [1, @"uvm_pkg::uvm_connect_phase::new"], [2, @"uvm_pkg::uvm_phase_hopper::process_phase"]]
  } : !llvm.array<3 x ptr>

  llvm.mlir.global internal @"phase_obj"(#llvm.zero) {
    addr_space = 0 : i32
  } : !llvm.array<64 x i8>

  hw.module @top() {
    %build_addr_i64 = hw.constant 12288 : i64
    %connect_addr_i64 = hw.constant 16384 : i64
    %null_i64 = hw.constant 0 : i64
    %fmt_parent = sim.fmt.literal "parent_done\0A"
    %fmt_slot = sim.fmt.literal "slot="
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      %h = sim.fork join_type "join_none" {
        %build_imp = llvm.inttoptr %build_addr_i64 : i64 to !llvm.ptr
        %connect_imp = llvm.inttoptr %connect_addr_i64 : i64 to !llvm.ptr
        %phase_mem = llvm.mlir.addressof @"phase_obj" : !llvm.ptr
        %phase_imp_slot = llvm.getelementptr %phase_mem[0, 44] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x i8>
        llvm.store %connect_addr_i64, %phase_imp_slot : i64, !llvm.ptr
        %loaded = llvm.load %phase_imp_slot : !llvm.ptr -> i64
        %slot_val = sim.fmt.dec %loaded : i64
        %slot_out = sim.fmt.concat (%fmt_slot, %slot_val, %fmt_nl)
        sim.proc.print %slot_out

        %vt = llvm.mlir.addressof @"phase_vtable" : !llvm.ptr
        %slot0 = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
        %f0_ptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
        %f0 = builtin.unrealized_conversion_cast %f0_ptr : !llvm.ptr to (!llvm.ptr) -> ()
        func.call_indirect %f0(%build_imp) : (!llvm.ptr) -> ()

        %slot1 = llvm.getelementptr %vt[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
        %f1_ptr = llvm.load %slot1 : !llvm.ptr -> !llvm.ptr
        %f1 = builtin.unrealized_conversion_cast %f1_ptr : !llvm.ptr to (!llvm.ptr) -> ()
        func.call_indirect %f1(%connect_imp) : (!llvm.ptr) -> ()

        %slot2 = llvm.getelementptr %vt[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
        %f2_ptr = llvm.load %slot2 : !llvm.ptr -> !llvm.ptr
        %f2 = builtin.unrealized_conversion_cast %f2_ptr : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()
        %null = llvm.inttoptr %null_i64 : i64 to !llvm.ptr
        func.call_indirect %f2(%null, %phase_mem) : (!llvm.ptr, !llvm.ptr) -> ()

        sim.fork.terminator
      }

      sim.proc.print %fmt_parent
      llhd.halt
    }

    hw.output
  }
}
