// RUN: circt-opt --verify-diagnostics --verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: moore.module @ForkJoinBasic
moore.module @ForkJoinBasic() {
  moore.procedure initial {
    %delay1 = moore.constant_time 10000000 fs
    %delay2 = moore.constant_time 20000000 fs
    %delay3 = moore.constant_time 15000000 fs

    // CHECK: moore.fork {
    // CHECK:   moore.wait_delay %{{.*}}
    // CHECK: }, {
    // CHECK:   moore.wait_delay %{{.*}}
    // CHECK: }
    moore.fork join {
      moore.wait_delay %delay1
    }, {
      moore.wait_delay %delay2
    }

    // CHECK: moore.fork join_any {
    // CHECK:   moore.wait_delay %{{.*}}
    // CHECK: }, {
    // CHECK:   moore.wait_delay %{{.*}}
    // CHECK: }
    moore.fork join_any {
      moore.wait_delay %delay1
    }, {
      moore.wait_delay %delay2
    }

    // CHECK: moore.fork join_none {
    // CHECK:   moore.wait_delay %{{.*}}
    // CHECK: }
    moore.fork join_none {
      moore.wait_delay %delay3
    }

    moore.return
  }
}

// CHECK-LABEL: moore.module @ForkJoinWithName
moore.module @ForkJoinWithName() {
  moore.procedure initial {
    %delay = moore.constant_time 10000000 fs

    // CHECK: moore.fork name "my_fork" {
    // CHECK:   moore.wait_delay %{{.*}}
    // CHECK: }
    moore.fork join name "my_fork" {
      moore.wait_delay %delay
    }

    moore.return
  }
}

// CHECK-LABEL: moore.module @WaitForkBasic
moore.module @WaitForkBasic() {
  moore.procedure initial {
    %delay1 = moore.constant_time 10000000 fs
    %delay2 = moore.constant_time 20000000 fs

    // Spawn multiple background tasks
    moore.fork join_none {
      moore.wait_delay %delay1
    }
    moore.fork join_none {
      moore.wait_delay %delay2
    }

    // CHECK: moore.wait_fork
    // Wait for all spawned processes
    moore.wait_fork

    moore.return
  }
}

// CHECK-LABEL: moore.module @DisableForkBasic
moore.module @DisableForkBasic() {
  moore.procedure initial {
    %delay1 = moore.constant_time 100000000 fs
    %delay2 = moore.constant_time 200000000 fs

    // Spawn background tasks
    moore.fork join_none {
      moore.wait_delay %delay1
    }
    moore.fork join_none {
      moore.wait_delay %delay2
    }

    // CHECK: moore.disable_fork
    // Terminate all child processes
    moore.disable_fork

    moore.return
  }
}

// CHECK-LABEL: moore.module @NamedBlockBasic
moore.module @NamedBlockBasic() {
  moore.procedure initial {
    // CHECK: moore.named_block "outer_loop"
    // CHECK:   moore.named_block "inner_loop"
    // CHECK:     moore.disable "outer_loop"
    moore.named_block "outer_loop" {
      moore.named_block "inner_loop" {
        // Exit both loops
        moore.disable "outer_loop"
      }
    }

    moore.return
  }
}

// CHECK-LABEL: moore.module @DisableBasic
moore.module @DisableBasic() {
  moore.procedure initial {
    // CHECK: moore.named_block "search"
    // CHECK:   moore.disable "search"
    moore.named_block "search" {
      moore.disable "search"
    }

    moore.return
  }
}

// CHECK-LABEL: moore.module @MultipleBranches
moore.module @MultipleBranches() {
  moore.procedure initial {
    %delay1 = moore.constant_time 10000000 fs
    %delay2 = moore.constant_time 20000000 fs
    %delay3 = moore.constant_time 30000000 fs
    %delay4 = moore.constant_time 40000000 fs

    // CHECK: moore.fork {
    // CHECK: }, {
    // CHECK: }, {
    // CHECK: }, {
    // CHECK: }
    moore.fork join {
      moore.wait_delay %delay1
    }, {
      moore.wait_delay %delay2
    }, {
      moore.wait_delay %delay3
    }, {
      moore.wait_delay %delay4
    }

    moore.return
  }
}
