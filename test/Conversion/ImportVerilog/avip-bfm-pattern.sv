// RUN: circt-verilog %s --ir-moore

// AVIP BFM pattern: Class instances inside interfaces with virtual interface access

// Forward declare the proxy class
typedef class apb_master_driver_proxy;

interface apb_master_driver_bfm(input logic clk);
  // Class instance inside interface - this is the pattern from AVIP BFMs
  apb_master_driver_proxy apb_master_drv_proxy_h;

  logic [31:0] paddr;
  logic [31:0] pwdata;
  logic pwrite;
  logic psel;
  logic penable;
endinterface

// Proxy class that uses virtual interface to access interface members
class apb_master_driver_proxy;
  virtual apb_master_driver_bfm apb_master_drv_bfm_h;
  int transaction_count;

  function void set_proxy();
    // This is the key pattern: assign 'this' to a class member inside the interface
    apb_master_drv_bfm_h.apb_master_drv_proxy_h = this;
  endfunction

  function apb_master_driver_proxy get_proxy();
    // Read the class member back from the interface
    return apb_master_drv_bfm_h.apb_master_drv_proxy_h;
  endfunction

  function void drive_transfer(logic [31:0] addr, logic [31:0] data);
    // Access interface signals through virtual interface (already supported)
    apb_master_drv_bfm_h.paddr = addr;
    apb_master_drv_bfm_h.pwdata = data;
    apb_master_drv_bfm_h.pwrite = 1'b1;
    apb_master_drv_bfm_h.psel = 1'b1;
    apb_master_drv_bfm_h.penable = 1'b0;
    transaction_count = transaction_count + 1;
  endfunction
endclass

module test_top;
  logic clk;
  apb_master_driver_proxy proxy;
  virtual apb_master_driver_bfm vif;

  initial begin
    proxy = new();
    proxy.apb_master_drv_bfm_h = vif;
    proxy.set_proxy();
    proxy = proxy.get_proxy();  // Read class member back from interface
    proxy.drive_transfer(32'h1000, 32'hDEADBEEF);
  end
endmodule
