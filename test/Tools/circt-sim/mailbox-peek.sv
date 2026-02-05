// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --max-time=1000000000 2>&1 | FileCheck %s

// Test mailbox.peek() and mailbox.try_peek().
// IEEE 1800-2017 Section 15.4 "Mailboxes"

module test_mailbox_peek;
  mailbox #(int) m;
  int v;
  bit ok;

  initial begin
    m = new();

    v = 0;
    ok = m.try_peek(v);
    if (ok)
      $display("TRYPEEK_EMPTY_UNEXPECTED");
    else
      $display("TRYPEEK_EMPTY_OK");

    fork
      begin
        m.peek(v);
        $display("PEEK_WAIT_OK %0d", v);
      end
      begin
        #1;
        m.put(7);
      end
    join

    v = 0;
    ok = m.try_peek(v);
    if (ok)
      $display("TRYPEEK_OK %0d", v);
    else
      $display("TRYPEEK_FAIL");

    v = 0;
    m.get(v);
    $display("GET_OK %0d", v);

    $finish;
  end
endmodule

// CHECK: TRYPEEK_EMPTY_OK
// CHECK: PEEK_WAIT_OK 7
// CHECK: TRYPEEK_OK 7
// CHECK: GET_OK 7
// CHECK-NOT: TRYPEEK_FAIL
// CHECK-NOT: TRYPEEK_EMPTY_UNEXPECTED
