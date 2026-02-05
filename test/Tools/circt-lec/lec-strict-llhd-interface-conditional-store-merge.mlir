// RUN: circt-opt --strip-llhd-interface-signals='strict-llhd=1' %s | FileCheck %s

// CHECK: comb.mux
// CHECK: verif.assert
// CHECK-NOT: llvm.store
// CHECK-NOT: llhd.prb

module {
  llvm.mlir.global internal @iface_storage() : !llvm.struct<(i1)>

  hw.module @top(in %c0 : i1, in %c1 : i1) {
    %ptr = llvm.mlir.addressof @iface_storage : !llvm.ptr
    %sig = llhd.sig %ptr : !llvm.ptr
    %probe = llhd.prb %sig : !llvm.ptr
    %field = llvm.getelementptr %probe[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1)>
    %true = hw.constant true
    %false = hw.constant false
    scf.execute_region {
      cf.cond_br %c0, ^bb_then, ^bb_else
    ^bb_then:
      cf.cond_br %c1, ^bb_t1, ^bb_t2
    ^bb_t1:
      cf.br ^bb_join
    ^bb_t2:
      cf.br ^bb_join
    ^bb_join:
      llvm.store %true, %field : i1, !llvm.ptr
      cf.br ^bb_end
    ^bb_else:
      llvm.store %false, %field : i1, !llvm.ptr
      cf.br ^bb_end
    ^bb_end:
      scf.yield
    }
    %ref = builtin.unrealized_conversion_cast %field : !llvm.ptr to !llhd.ref<i1>
    %val = llhd.prb %ref : i1
    verif.assert %val : i1
    hw.output
  }
}
