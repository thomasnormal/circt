// RUN: circt-sim %s | FileCheck %s
// This test verifies basic semaphore create/get/put/try_get operations.

// External semaphore runtime functions
llvm.func @__moore_semaphore_create(i64, i32) -> ()
llvm.func @__moore_semaphore_get(i64, i32) -> ()
llvm.func @__moore_semaphore_put(i64, i32) -> ()
llvm.func @__moore_semaphore_try_get(i64, i32) -> i1

hw.module @test() {
  %nl = sim.fmt.literal "\0A"
  %lbl1 = sim.fmt.literal "try_get(1)="
  %lbl2 = sim.fmt.literal "try_get(2)="

  llhd.process {
    // Use address 0x100 as semaphore ID
    %sem_addr = llvm.mlir.constant(256 : i64) : i64
    %key1 = llvm.mlir.constant(1 : i32) : i32
    %key2 = llvm.mlir.constant(2 : i32) : i32

    // Create semaphore with 2 initial keys
    llvm.call @__moore_semaphore_create(%sem_addr, %key2) : (i64, i32) -> ()

    // try_get(1) should succeed (2 keys available -> 1 remaining)
    %r1 = llvm.call @__moore_semaphore_try_get(%sem_addr, %key1) : (i64, i32) -> i1
    %fmt_r1 = sim.fmt.dec %r1 : i1
    %out1 = sim.fmt.concat (%lbl1, %fmt_r1, %nl)
    // CHECK: try_get(1)=1
    sim.proc.print %out1

    // try_get(1) should succeed (1 key available -> 0 remaining)
    %r2 = llvm.call @__moore_semaphore_try_get(%sem_addr, %key1) : (i64, i32) -> i1
    %fmt_r2 = sim.fmt.dec %r2 : i1
    %out2 = sim.fmt.concat (%lbl1, %fmt_r2, %nl)
    // CHECK: try_get(1)=1
    sim.proc.print %out2

    // try_get(1) should fail (0 keys available)
    %r3 = llvm.call @__moore_semaphore_try_get(%sem_addr, %key1) : (i64, i32) -> i1
    %fmt_r3 = sim.fmt.dec %r3 : i1
    %out3 = sim.fmt.concat (%lbl1, %fmt_r3, %nl)
    // CHECK: try_get(1)=0
    sim.proc.print %out3

    // put(1) adds a key back (0 -> 1)
    llvm.call @__moore_semaphore_put(%sem_addr, %key1) : (i64, i32) -> ()

    // try_get(1) should succeed again (1 -> 0)
    %r4 = llvm.call @__moore_semaphore_try_get(%sem_addr, %key1) : (i64, i32) -> i1
    %fmt_r4 = sim.fmt.dec %r4 : i1
    %out4 = sim.fmt.concat (%lbl1, %fmt_r4, %nl)
    // CHECK: try_get(1)=1
    sim.proc.print %out4

    // put(2) adds 2 keys back (0 -> 2)
    llvm.call @__moore_semaphore_put(%sem_addr, %key2) : (i64, i32) -> ()

    // try_get(2) should succeed (2 -> 0)
    %r5 = llvm.call @__moore_semaphore_try_get(%sem_addr, %key2) : (i64, i32) -> i1
    %fmt_r5 = sim.fmt.dec %r5 : i1
    %out5 = sim.fmt.concat (%lbl2, %fmt_r5, %nl)
    // CHECK: try_get(2)=1
    sim.proc.print %out5

    // try_get(1) should fail (0 keys)
    %r6 = llvm.call @__moore_semaphore_try_get(%sem_addr, %key1) : (i64, i32) -> i1
    %fmt_r6 = sim.fmt.dec %r6 : i1
    %out6 = sim.fmt.concat (%lbl1, %fmt_r6, %nl)
    // CHECK: try_get(1)=0
    sim.proc.print %out6

    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
