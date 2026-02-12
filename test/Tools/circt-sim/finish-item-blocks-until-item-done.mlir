// RUN: circt-sim %s --max-time 100000000 2>&1 | FileCheck %s
//
// Test that finish_item blocks until the driver calls item_done.
// This is the standard UVM sequence-driver handshake:
//
// Sequence: start_item -> finish_item [blocks until item_done]
// Driver: get_next_item -> drive -> item_done
//
// CHECK: sequence: start_item
// CHECK: sequence: finish_item (blocking)
// CHECK: driver: got item
// CHECK: driver: item_done
// CHECK: sequence: finish_item returned
// CHECK: [circt-sim] Simulation completed

module {
  // Storage for the item and sequencer address.
  llvm.mlir.global internal @item_storage(0 : i64) : i64
  llvm.mlir.global internal @sequencer_addr(42 : i64) : i64

  // Sequence vtable: slot 0 = start_item, slot 1 = finish_item
  llvm.mlir.global internal @seq_vtable(#llvm.zero) {addr_space = 0 : i32, circt.vtable_entries = [[0, @"uvm_pkg::uvm_sequence_base::start_item"], [1, @"uvm_pkg::uvm_sequence_base::finish_item"]]} : !llvm.array<2 x ptr>

  // Port vtable: slot 0 = get_next_item, slot 1 = (unused), slot 2 = item_done
  llvm.mlir.global internal @port_vtable(#llvm.zero) {addr_space = 0 : i32, circt.vtable_entries = [[0, @"uvm_pkg::uvm_seq_item_pull_port::get_next_item"], [2, @"uvm_pkg::uvm_seq_item_pull_port::item_done"]]} : !llvm.array<3 x ptr>

  // Intercepted functions with dummy bodies (interceptors fire before body).
  func.func @"uvm_pkg::uvm_sequence_base::start_item"(%self: !llvm.ptr, %item: !llvm.ptr, %pri: i32, %sqr: !llvm.ptr) {
    return
  }
  func.func @"uvm_pkg::uvm_sequence_base::finish_item"(%self: !llvm.ptr, %item: !llvm.ptr, %pri: i32) {
    return
  }
  func.func @"uvm_pkg::uvm_seq_item_pull_port::get_next_item"(%port: !llvm.ptr, %ref: !llvm.ptr) {
    return
  }
  func.func @"uvm_pkg::uvm_seq_item_pull_port::item_done"(%port: !llvm.ptr, %rsp: !llvm.ptr) {
    return
  }

  hw.module @top() {
    %t2ns = llhd.constant_time <2ns, 0d, 0e>
    %t1ns = llhd.constant_time <1ns, 0d, 0e>
    %fmt_si = sim.fmt.literal "sequence: start_item\0A"
    %fmt_fi = sim.fmt.literal "sequence: finish_item (blocking)\0A"
    %fmt_fi_ret = sim.fmt.literal "sequence: finish_item returned\0A"
    %fmt_got = sim.fmt.literal "driver: got item\0A"
    %fmt_done = sim.fmt.literal "driver: item_done\0A"

    // Sequence process: start_item then finish_item (should block until item_done).
    llhd.process {
      cf.br ^start
    ^start:
      %item = llvm.mlir.addressof @item_storage : !llvm.ptr
      %sqr = llvm.mlir.addressof @sequencer_addr : !llvm.ptr
      %null = llvm.mlir.zero : !llvm.ptr
      %neg1 = llvm.mlir.constant(-1 : i32) : i32

      // Load vtable for sequence
      %vt = llvm.mlir.addressof @seq_vtable : !llvm.ptr

      // start_item via func.call_indirect (vtable slot 0)
      %si_slot = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %si_fn = llvm.load %si_slot : !llvm.ptr -> !llvm.ptr
      %si_cast = builtin.unrealized_conversion_cast %si_fn : !llvm.ptr to (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
      func.call_indirect %si_cast(%null, %item, %neg1, %sqr) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
      sim.proc.print %fmt_si

      // finish_item via func.call_indirect (vtable slot 1)
      sim.proc.print %fmt_fi
      %fi_slot = llvm.getelementptr %vt[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fi_fn = llvm.load %fi_slot : !llvm.ptr -> !llvm.ptr
      %fi_cast = builtin.unrealized_conversion_cast %fi_fn : !llvm.ptr to (!llvm.ptr, !llvm.ptr, i32) -> ()
      func.call_indirect %fi_cast(%null, %item, %neg1) : (!llvm.ptr, !llvm.ptr, i32) -> ()
      sim.proc.print %fmt_fi_ret
      llhd.halt
    }

    // Driver process: delay 2ns, then get_next_item + 1ns delay + item_done.
    llhd.process {
      cf.br ^start
    ^start:
      llhd.wait delay %t2ns, ^get
    ^get:
      %port = llvm.mlir.addressof @port_vtable : !llvm.ptr
      %one = llvm.mlir.constant(1 : i64) : i64
      %ref = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
      %null = llvm.mlir.zero : !llvm.ptr

      // get_next_item via func.call_indirect (vtable slot 0)
      %gni_slot = llvm.getelementptr %port[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
      %gni_fn = llvm.load %gni_slot : !llvm.ptr -> !llvm.ptr
      %gni_cast = builtin.unrealized_conversion_cast %gni_fn : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()
      func.call_indirect %gni_cast(%port, %ref) : (!llvm.ptr, !llvm.ptr) -> ()
      sim.proc.print %fmt_got

      // Simulate driving pins
      llhd.wait delay %t1ns, ^done
    ^done:
      %port2 = llvm.mlir.addressof @port_vtable : !llvm.ptr
      %null2 = llvm.mlir.zero : !llvm.ptr

      // item_done via func.call_indirect (vtable slot 2)
      %id_slot = llvm.getelementptr %port2[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
      %id_fn = llvm.load %id_slot : !llvm.ptr -> !llvm.ptr
      %id_cast = builtin.unrealized_conversion_cast %id_fn : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()
      func.call_indirect %id_cast(%port2, %null2) : (!llvm.ptr, !llvm.ptr) -> ()
      sim.proc.print %fmt_done
      llhd.halt
    }

    hw.output
  }
}
