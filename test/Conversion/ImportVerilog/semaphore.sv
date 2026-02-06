// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// CHECK-LABEL: moore.module @semaphoreOps
module semaphoreOps();
  semaphore sem;

  // CHECK: procedure initial
  initial begin
    // CHECK: llvm.call @__moore_semaphore_create
    sem = new(2);
    // CHECK: llvm.call @__moore_semaphore_get
    sem.get(1);
    // CHECK: llvm.call @__moore_semaphore_put
    sem.put(1);
    // CHECK: llvm.call @__moore_semaphore_try_get
    if (sem.try_get(1))
      $display("got key");
  end
endmodule
