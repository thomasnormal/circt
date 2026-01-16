// AVIP AXI4 Constraint Patterns - Extracted from axi4_master_tx.sv and axi4_slave_tx.sv
// Tests soft constraints, inside constraints, complex if-else, enum != constraints

parameter int ADDRESS_WIDTH = 32;
parameter int DATA_WIDTH = 32;
parameter int LENGTH = 8;
parameter int STROBE_WIDTH = DATA_WIDTH/8;

typedef enum bit [1:0] {
  WRITE_FIXED    = 2'b00,
  WRITE_INCR     = 2'b01,
  WRITE_WRAP     = 2'b10,
  WRITE_RESERVED = 2'b11
} awburst_e;

typedef enum bit [1:0] {
  READ_FIXED    = 2'b00,
  READ_INCR     = 2'b01,
  READ_WRAP     = 2'b10,
  READ_RESERVED = 2'b11
} arburst_e;

typedef enum bit [2:0] {
  AWSIZE_1B   = 3'b000,
  AWSIZE_2B   = 3'b001,
  AWSIZE_4B   = 3'b010,
  AWSIZE_8B   = 3'b011,
  AWSIZE_16B  = 3'b100,
  AWSIZE_32B  = 3'b101,
  AWSIZE_64B  = 3'b110,
  AWSIZE_128B = 3'b111
} awsize_e;

typedef enum bit [2:0] {
  ARSIZE_1B   = 3'b000,
  ARSIZE_2B   = 3'b001,
  ARSIZE_4B   = 3'b010,
  ARSIZE_8B   = 3'b011,
  ARSIZE_16B  = 3'b100,
  ARSIZE_32B  = 3'b101,
  ARSIZE_64B  = 3'b110,
  ARSIZE_128B = 3'b111
} arsize_e;

typedef enum bit {
  WRITE_NORMAL_ACCESS    = 1'b0,
  WRITE_EXCLUSIVE_ACCESS = 1'b1
} awlock_e;

typedef enum bit {
  READ_NORMAL_ACCESS    = 1'b0,
  READ_EXCLUSIVE_ACCESS = 1'b1
} arlock_e;

typedef enum bit {
  LITTLE_ENDIAN = 1'b0,
  BIG_ENDIAN    = 1'b1
} endian_e;

typedef enum bit [1:0] {
  WRITE_OKAY   = 2'b00,
  WRITE_EXOKAY = 2'b01,
  WRITE_SLVERR = 2'b10,
  WRITE_DECERR = 2'b11
} bresp_e;

typedef enum bit [1:0] {
  READ_OKAY   = 2'b00,
  READ_EXOKAY = 2'b01,
  READ_SLVERR = 2'b10,
  READ_DECERR = 2'b11
} rresp_e;

// AXI4 Master Transaction - extracted constraints
class axi4_master_tx;
  // Write address channel
  rand bit [ADDRESS_WIDTH-1:0] awaddr;
  rand bit [LENGTH-1:0] awlen;
  rand awsize_e awsize;
  rand awburst_e awburst;
  rand awlock_e awlock;

  // Write data channel
  rand bit [DATA_WIDTH-1:0] wdata[$:2**LENGTH];
  rand bit [(DATA_WIDTH/8)-1:0] wstrb[$:2**LENGTH];

  // Read address channel
  rand bit [ADDRESS_WIDTH-1:0] araddr;
  rand bit [LENGTH-1:0] arlen;
  rand arsize_e arsize;
  rand arburst_e arburst;
  rand arlock_e arlock;

  // Memory
  rand endian_e endian;
  rand int no_of_wait_states;

  // Constraint: soft address alignment (complex expression)
  constraint awaddr_c0 { soft awaddr == (awaddr % (2**awsize)) == 0; }

  // Constraint: enum inequality (not equal to reserved)
  constraint awburst_c1 { awburst != WRITE_RESERVED; }

  // Constraint: if-else with inside range
  constraint awlength_c2 {
    if (awburst == WRITE_FIXED || awburst == WRITE_WRAP)
      awlen inside {[0:15]};
    else if (awburst == WRITE_INCR)
      awlen inside {[0:255]};
  }

  // Constraint: if with inside set (discrete values)
  constraint awlength_c3 {
    if (awburst == WRITE_WRAP)
      awlen + 1 inside {2, 4, 8, 16};
  }

  // Constraint: soft enum equality
  constraint awlock_c4 { soft awlock == WRITE_NORMAL_ACCESS; }

  // Constraint: soft enum equality
  constraint awburst_c5 { soft awburst == WRITE_INCR; }

  // Constraint: soft inside range
  constraint awsize_c6 { soft awsize inside {[0:2]}; }

  // Constraint: queue size equals expression
  constraint wdata_c1 { wdata.size() == awlen + 1; }

  // Constraint: queue size equals expression
  constraint wstrb_c2 { wstrb.size() == awlen + 1; }

  // Constraint: foreach with inequality
  constraint wstrb_c3 { foreach(wstrb[i]) wstrb[i] != 0; }

  // Constraint: foreach with $countones and power expression
  constraint wstrb_c4 { foreach(wstrb[i]) $countones(wstrb[i]) == 2**awsize; }

  // Read address constraints (similar patterns)
  constraint araddr_c0 { soft araddr == (araddr % (2**arsize)) == 0; }
  constraint arburst_c1 { arburst != READ_RESERVED; }

  constraint arlength_c2 {
    if (arburst == READ_FIXED || arburst == READ_WRAP)
      arlen inside {[0:15]};
    else if (arburst == READ_INCR)
      arlen inside {[0:255]};
  }

  constraint arlength_c3 {
    if (arburst == READ_WRAP)
      arlen + 1 inside {2, 4, 8, 16};
  }

  constraint arlock_c4 { soft arlock == READ_NORMAL_ACCESS; }
  constraint arburst_c5 { soft arburst == READ_INCR; }
  constraint arsize_c6 { soft arsize inside {[0:2]}; }

  // Constraint: inside range for wait states
  constraint no_of_wait_states_c3 { no_of_wait_states inside {[0:3]}; }

  // Constraint: soft endian selection
  constraint endian_c1 { soft endian == LITTLE_ENDIAN; }
endclass

// AXI4 Slave Transaction - extracted constraints
class axi4_slave_tx;
  rand rresp_e rresp;
  rand bresp_e bresp;

  // Constraint: soft enum equality for response
  constraint rresp_c1 { soft rresp == READ_OKAY; }
  constraint bresp_c1 { soft bresp == WRITE_OKAY; }
endclass

module test_axi4_constraints;
  initial begin
    axi4_master_tx master;
    axi4_slave_tx slave;

    master = new();
    slave = new();

    // Test randomization
    if (master.randomize()) begin
      $display("AXI4 Master randomize success");
      $display("  awburst=%s awlen=%d awsize=%s",
               master.awburst.name(), master.awlen, master.awsize.name());
      $display("  wdata.size=%d wstrb.size=%d",
               master.wdata.size(), master.wstrb.size());
    end

    if (slave.randomize()) begin
      $display("AXI4 Slave randomize success");
      $display("  rresp=%s bresp=%s", slave.rresp.name(), slave.bresp.name());
    end
  end
endmodule
