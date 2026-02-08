// RUN: circt-verilog %s --ir-hw --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test associative array iteration with string keys using first() and next().
// Verifies that iterated keys are usable in subsequent operations such as
// $display and string comparison, and that the iteration count matches the
// number of entries.

module top;
  int data[string];
  string key;
  int count;
  int found;

  initial begin
    // Populate the associative array with three entries.
    data["alpha"] = 10;
    data["beta"]  = 20;
    data["gamma"] = 30;

    // Iterate using first() and next().
    // std::map orders keys lexicographically: alpha < beta < gamma
    count = 0;

    found = data.first(key);
    while (found) begin
      count = count + 1;

      // Verify the key is non-empty by checking its length.
      if (key.len() == 0) begin
        $display("ERROR: empty key at count %0d", count);
      end

      // Display the iterated key and its associated value.
      $display("key=%s val=%0d", key, data[key]);

      found = data.next(key);
    end

    // Verify we iterated over exactly 3 entries.
    $display("count=%0d", count);

    // CHECK: key=alpha val=10
    // CHECK: key=beta val=20
    // CHECK: key=gamma val=30
    // CHECK: count=3

    // Verify string comparison works on the last iterated key.
    // After the loop, key holds the last key visited ("gamma").
    if (key == "gamma") begin
      // CHECK: last_key_ok
      $display("last_key_ok");
    end else begin
      $display("ERROR: expected last key gamma, got %s", key);
    end

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule
