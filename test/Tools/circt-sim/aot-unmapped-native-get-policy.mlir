// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_STATS=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE_NAMES=uvm_pkg::uvm_component::get_child circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=ALLOWNAMES
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_DENY_UNMAPPED_NATIVE_ALL=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE_NAMES=uvm_pkg::uvm_component::get_child circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=DENYALLALLOW
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=ALLOW
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE_ZEROARG_HELPERS_UNSAFE=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=ALLOWUNSAFE
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE=1 CIRCT_AOT_DENY_UNMAPPED_NATIVE_NAMES=get_* circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=ALLOWDENY

// Regression: direct func.call to a compiled function that has no FuncId
// mapping must follow unmapped-native policy.
//
// COMPILE: [circt-compile] Functions: 3 total, 0 external, 0 rejected, 3 compilable
// COMPILE: [circt-compile] 3 functions + 0 processes ready for codegen
//
// DEFAULT: Unmapped native func.call policy: default deny uvm_pkg::* and pointer-typed get_/set_/create_/m_initialize* (allow others)
// DEFAULT: Compiled function calls:          1
// DEFAULT: Interpreted function calls:       2
// DEFAULT: direct_calls_native:              1
// DEFAULT: direct_calls_interpreted:         2
// DEFAULT: Top interpreted func.call fallback reasons (top 50):
// DEFAULT: 1x get_ptr_1 [unmapped-policy=1]
// DEFAULT: 1x uvm_pkg::uvm_component::get_child [unmapped-policy=1]
// DEFAULT: out=47 uvm=107
//
// ALLOWNAMES: Unmapped native func.call policy: default deny uvm_pkg::* and pointer-typed get_/set_/create_/m_initialize* (allow others){{.*}}with allow list 'uvm_pkg::uvm_component::get_child'
// ALLOWNAMES: Compiled function calls:          2
// ALLOWNAMES: Interpreted function calls:       1
// ALLOWNAMES: direct_calls_native:              2
// ALLOWNAMES: direct_calls_interpreted:         1
// ALLOWNAMES: Top interpreted func.call fallback reasons (top 50):
// ALLOWNAMES: 1x get_ptr_1 [unmapped-policy=1]
// ALLOWNAMES: out=47 uvm=107
//
// DENYALLALLOW: Unmapped native func.call policy: deny-all{{.*}}with allow list 'uvm_pkg::uvm_component::get_child'
// DENYALLALLOW: Compiled function calls:          1
// DENYALLALLOW: Interpreted function calls:       2
// DENYALLALLOW: direct_calls_native:              1
// DENYALLALLOW: direct_calls_interpreted:         2
// DENYALLALLOW: Top interpreted func.call fallback reasons (top 50):
// DENYALLALLOW: 1x get_2160 [unmapped-policy=1]
// DENYALLALLOW: 1x get_ptr_1 [unmapped-policy=1]
// DENYALLALLOW: out=47 uvm=107
//
// ALLOW: Unmapped native func.call policy: allow-all
// ALLOW: Compiled function calls:          2
// ALLOW: Interpreted function calls:       1
// ALLOW: direct_calls_native:              2
// ALLOW: direct_calls_interpreted:         1
// ALLOW: out=47 uvm=107
//
// ALLOWUNSAFE: Unmapped native func.call policy: allow-all
// ALLOWUNSAFE: Compiled function calls:          3
// ALLOWUNSAFE: Interpreted function calls:       0
// ALLOWUNSAFE: direct_calls_native:              3
// ALLOWUNSAFE: direct_calls_interpreted:         0
// ALLOWUNSAFE: out=47 uvm=107
//
// ALLOWDENY: Unmapped native func.call policy: allow-all{{.*}}with deny list 'get_*'
// ALLOWDENY: Compiled function calls:          1
// ALLOWDENY: Interpreted function calls:       2
// ALLOWDENY: direct_calls_native:              1
// ALLOWDENY: direct_calls_interpreted:         2
// ALLOWDENY: Top interpreted func.call fallback reasons (top 50):
// ALLOWDENY: 1x get_2160 [unmapped-policy=1]
// ALLOWDENY: 1x get_ptr_1 [unmapped-policy=1]
// ALLOWDENY: out=47 uvm=107

func.func @get_ptr_1() -> !llvm.ptr {
  %0 = llvm.mlir.zero : !llvm.ptr
  return %0 : !llvm.ptr
}

func.func @get_2160(%x: i32) -> i32 {
  %c42 = arith.constant 42 : i32
  %r = arith.addi %x, %c42 : i32
  return %r : i32
}

func.func @"uvm_pkg::uvm_component::get_child"(%this: !llvm.ptr, %key: i64) -> i64 {
  %c100 = arith.constant 100 : i64
  %r = arith.addi %key, %c100 : i64
  return %r : i64
}

llvm.mlir.global internal @dummy_this(0 : i8) : i8

hw.module @top() {
  %fmtPrefix = sim.fmt.literal "out="
  %fmtMid = sim.fmt.literal " uvm="
  %fmtNl = sim.fmt.literal "\0A"
  %c5 = hw.constant 5 : i32
  %c7 = hw.constant 7 : i64
  %c10_i64 = hw.constant 10000000 : i64
  %this_ptr = llvm.mlir.addressof @dummy_this : !llvm.ptr

  llhd.process {
    %p = func.call @get_ptr_1() : () -> !llvm.ptr
    %r = func.call @get_2160(%c5) : (i32) -> i32
    %uvm = func.call @"uvm_pkg::uvm_component::get_child"(%this_ptr, %c7) : (!llvm.ptr, i64) -> i64
    %fmtV = sim.fmt.dec %r signed : i32
    %fmtU = sim.fmt.dec %uvm signed : i64
    %fmtOut = sim.fmt.concat (%fmtPrefix, %fmtV, %fmtMid, %fmtU, %fmtNl)
    sim.proc.print %fmtOut
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %c10_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
