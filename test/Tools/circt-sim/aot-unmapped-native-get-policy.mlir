// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_STATS=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=ALLOW
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE=1 CIRCT_AOT_DENY_UNMAPPED_NATIVE_NAMES=get_* circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=ALLOWDENY

// Regression: direct func.call to a compiled function that has no FuncId
// mapping must follow unmapped-native policy.
//
// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
//
// DEFAULT: Unmapped native func.call policy: default allow-all
// DEFAULT: Compiled function calls:          1
// DEFAULT: Interpreted function calls:       0
// DEFAULT: direct_calls_native:              1
// DEFAULT: direct_calls_interpreted:         0
// DEFAULT: out=47
//
// ALLOW: Unmapped native func.call policy: allow-all
// ALLOW: Compiled function calls:          1
// ALLOW: Interpreted function calls:       0
// ALLOW: direct_calls_native:              1
// ALLOW: direct_calls_interpreted:         0
// ALLOW: out=47
//
// ALLOWDENY: Unmapped native func.call policy: allow-all with deny list 'get_*'
// ALLOWDENY: Compiled function calls:          0
// ALLOWDENY: Interpreted function calls:       1
// ALLOWDENY: direct_calls_native:              0
// ALLOWDENY: direct_calls_interpreted:         1
// ALLOWDENY: out=47

func.func @get_2160(%x: i32) -> i32 {
  %c42 = arith.constant 42 : i32
  %r = arith.addi %x, %c42 : i32
  return %r : i32
}

hw.module @top() {
  %fmtPrefix = sim.fmt.literal "out="
  %fmtNl = sim.fmt.literal "\0A"
  %c5 = hw.constant 5 : i32
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %r = func.call @get_2160(%c5) : (i32) -> i32
    %fmtV = sim.fmt.dec %r signed : i32
    %fmtOut = sim.fmt.concat (%fmtPrefix, %fmtV, %fmtNl)
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
