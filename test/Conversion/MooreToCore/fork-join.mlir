// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-LABEL: hw.module @ForkJoinConversion
moore.module @ForkJoinConversion() {
  moore.procedure initial {
    %c1 = moore.constant 1 : i32
    %c2 = moore.constant 2 : i32

    // CHECK: sim.fork {
    // CHECK-NOT: join_type
    moore.fork join {
      %v1 = moore.variable : <i32>
      moore.blocking_assign %v1, %c1 : i32
    }, {
      %v2 = moore.variable : <i32>
      moore.blocking_assign %v2, %c2 : i32
    }

    moore.return
  }
}

// CHECK-LABEL: hw.module @ForkJoinAnyConversion
moore.module @ForkJoinAnyConversion() {
  moore.procedure initial {
    %c1 = moore.constant 1 : i32
    %c2 = moore.constant 2 : i32

    // CHECK: sim.fork
    // CHECK-SAME: join_type "join_any"
    moore.fork join_any {
      %v1 = moore.variable : <i32>
      moore.blocking_assign %v1, %c1 : i32
    }, {
      %v2 = moore.variable : <i32>
      moore.blocking_assign %v2, %c2 : i32
    }

    moore.return
  }
}

// CHECK-LABEL: hw.module @ForkJoinNoneConversion
moore.module @ForkJoinNoneConversion() {
  moore.procedure initial {
    %c = moore.constant 100 : i32

    // CHECK: sim.fork
    // CHECK-SAME: join_type "join_none"
    moore.fork join_none {
      %v = moore.variable : <i32>
      moore.blocking_assign %v, %c : i32
    }

    moore.return
  }
}

// CHECK-LABEL: hw.module @WaitForkConversion
moore.module @WaitForkConversion() {
  moore.procedure initial {
    %c = moore.constant 10 : i32

    moore.fork join_none {
      %v = moore.variable : <i32>
      moore.blocking_assign %v, %c : i32
    }

    // CHECK: sim.wait_fork
    moore.wait_fork

    moore.return
  }
}

// CHECK-LABEL: hw.module @DisableForkConversion
moore.module @DisableForkConversion() {
  moore.procedure initial {
    %c = moore.constant 100 : i32

    moore.fork join_none {
      %v = moore.variable : <i32>
      moore.blocking_assign %v, %c : i32
    }

    // CHECK: sim.disable_fork
    moore.disable_fork

    moore.return
  }
}

// CHECK-LABEL: hw.module @NamedBlockConversion
moore.module @NamedBlockConversion() {
  moore.procedure initial {
    // CHECK: sim.named_block "my_block"
    // CHECK: sim.disable "my_block"
    moore.named_block "my_block" {
      moore.disable "my_block"
    }

    moore.return
  }
}

// CHECK-LABEL: hw.module @NestedNamedBlockConversion
moore.module @NestedNamedBlockConversion() {
  moore.procedure initial {
    // CHECK: sim.named_block "outer"
    // CHECK: sim.named_block "inner"
    // CHECK: sim.disable "outer"
    moore.named_block "outer" {
      moore.named_block "inner" {
        moore.disable "outer"
      }
    }

    moore.return
  }
}
