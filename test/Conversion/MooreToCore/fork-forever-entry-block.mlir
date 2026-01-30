// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test that the entry block of a sim.fork region has no predecessors after
// conversion, even when the fork body contains a forever loop (cf.br back-edge).
// This tests the fix from commit bfe727841 which restructures entry blocks
// that would otherwise have predecessors from forever loops inside fork/join_none.
// The fix also adds a sim.proc.print anti-elision op to entry blocks that
// contain only a branch (to prevent the MLIR printer from eliding them).

func.func private @work() -> ()

// Test 1: Fork with forever loop where entry block only has a branch.
// The entry block should get a sim.proc.print to prevent elision, and
// the back-edge should go to a separate loop header block (not the entry).

// CHECK-LABEL: hw.module @ForkForeverEntryBranch
moore.module @ForkForeverEntryBranch() {
  moore.procedure initial {
    // CHECK:       sim.fork join_type "join_none" {
    //   Entry block: has sim.proc.print (anti-elision), branch to loop header.
    // CHECK-NEXT:    sim.proc.print
    // CHECK-NEXT:    cf.br ^[[LOOP:bb[0-9]+]]
    //   Loop header block has the back-edge to itself (2 preds: entry + self).
    // CHECK-NEXT:  ^[[LOOP]]:
    // CHECK-NEXT:    func.call @work
    // CHECK-NEXT:    cf.br ^[[LOOP]]
    // CHECK-NEXT:  }
    moore.fork join_none {
      cf.br ^loop
    ^loop:
      func.call @work() : () -> ()
      cf.br ^loop
    }
    moore.return
  }
}

// Test 2: Fork with forever loop where entry block has a side-effecting op.
// No sim.proc.print is needed since func.call already prevents elision.

// CHECK-LABEL: hw.module @ForkForeverEntryContent
moore.module @ForkForeverEntryContent() {
  moore.procedure initial {
    // CHECK:       sim.fork join_type "join_none" {
    //   Entry block has func.call (side effects prevent elision) and branch to loop header.
    // CHECK-NEXT:    func.call @work
    // CHECK-NEXT:    cf.br ^[[LOOP:bb[0-9]+]]
    //   Loop header block has the back-edge to itself.
    // CHECK-NEXT:  ^[[LOOP]]:
    // CHECK-NEXT:    func.call @work
    // CHECK-NEXT:    cf.br ^[[LOOP]]
    // CHECK-NEXT:  }
    moore.fork join_none {
      func.call @work() : () -> ()
      cf.br ^loop
    ^loop:
      func.call @work() : () -> ()
      cf.br ^loop
    }
    moore.return
  }
}

// Test 3: Simple fork with no loop - entry block should NOT get sim.proc.print
// (no elision risk since the block has operations with side effects).

// CHECK-LABEL: hw.module @ForkSimpleNoLoop
moore.module @ForkSimpleNoLoop() {
  moore.procedure initial {
    // CHECK:       sim.fork join_type "join_none" {
    // CHECK-NEXT:    func.call @work
    // CHECK-NOT:     sim.proc.print
    // CHECK:       }
    moore.fork join_none {
      func.call @work() : () -> ()
    }
    moore.return
  }
}
