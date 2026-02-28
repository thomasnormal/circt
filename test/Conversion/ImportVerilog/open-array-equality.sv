// REQUIRES: circt-sim
// RUN: circt-verilog --ir-moore %s | FileCheck %s --check-prefix=IR
// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s --check-prefix=SIM

// IR-LABEL: func.func private @arr_eq
// IR: moore.array.size
// IR: moore.array.locator all
// IR: moore.and
// IR: return
// IR-LABEL: func.func private @arr_ne
// IR: moore.array.locator all
// IR: moore.not
// IR: return

module top;
  int a[];
  int b[];
  bit eqv;
  bit nev;

  function automatic bit arr_eq(input int lhs[], input int rhs[]);
    return lhs == rhs;
  endfunction

  function automatic bit arr_ne(input int lhs[], input int rhs[]);
    return lhs != rhs;
  endfunction

  initial begin
    a = new[2];
    b = new[2];
    a[0] = 11;
    a[1] = 22;
    b[0] = 11;
    b[1] = 22;
    eqv = arr_eq(a, b);
    nev = arr_ne(a, b);
    // SIM: eq_same=1 ne_same=0
    $display("eq_same=%0d ne_same=%0d", eqv, nev);

    b[1] = 33;
    eqv = arr_eq(a, b);
    nev = arr_ne(a, b);
    // SIM: eq_diff=0 ne_diff=1
    $display("eq_diff=%0d ne_diff=%0d", eqv, nev);

    b = new[1];
    b[0] = 11;
    eqv = arr_eq(a, b);
    nev = arr_ne(a, b);
    // SIM: eq_size=0 ne_size=1
    $display("eq_size=%0d ne_size=%0d", eqv, nev);

    $finish;
  end
endmodule
