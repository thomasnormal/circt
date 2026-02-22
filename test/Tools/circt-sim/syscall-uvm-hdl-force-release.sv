// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test uvm_hdl_force/uvm_hdl_release actually change and restore signal values.
// Bug: uvm_hdl_force and uvm_hdl_release are stubs that always return 1 but
// do not perform any VPI-based signal forcing — the signal is unchanged.

// The DPI-C signatures match the UVM reference implementation.
// uvm_hdl_force: path + packed value → returns 1 on success.
// uvm_hdl_release: path → returns 1 on success.
import "DPI-C" function int uvm_hdl_force(string path, input bit [31:0] val);
import "DPI-C" function int uvm_hdl_release(string path);

module top;
  reg [31:0] sig;
  int force_ok, release_ok;

  initial begin
    // Establish a known driven value (not 0, not the forced value)
    sig = 32'd42;

    // Force to a distinct sentinel that cannot be confused with default or 0
    force_ok = uvm_hdl_force("top.sig", 32'd99);
    // CHECK: force_returns_ok=1
    $display("force_returns_ok=%0d", force_ok);
    #1;

    // If uvm_hdl_force actually worked, sig must now read 99 not 42 or 0.
    // A stub that returns 1 but does nothing leaves sig == 42.
    // CHECK: sig_while_forced=99
    $display("sig_while_forced=%0d", sig);

    // Release — signal should revert to the last procedural driver value (42).
    release_ok = uvm_hdl_release("top.sig");
    // CHECK: release_returns_ok=1
    $display("release_returns_ok=%0d", release_ok);
    #1;

    // A correct release reverts to 42; a no-op release leaves sig at 99.
    // CHECK: sig_after_release=42
    $display("sig_after_release=%0d", sig);

    $finish;
  end
endmodule
