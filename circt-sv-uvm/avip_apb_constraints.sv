// AVIP APB Constraint Patterns - Extracted from apb_master_tx.sv and apb_slave_tx.sv
// Tests soft constraints, inside constraints, range constraints, and $countones

parameter int NO_OF_SLAVES = 1;
parameter int ADDRESS_WIDTH = 32;
parameter int DATA_WIDTH = 32;

typedef enum bit[31:0] {
  BIT_8  = 32'd8,
  BIT_16 = 32'd16,
  BIT_24 = 32'd24,
  BIT_32 = 32'd32
} transfer_size_e;

typedef enum bit {
  NO_ERROR = 1'b0,
  ERROR_FLAG = 1'b1
} slave_error_e;

typedef enum bit {
  WRITE = 1'b1,
  READ  = 1'b0
} tx_type_e;

typedef enum logic[2:0] {
  NORMAL_SECURE_DATA              = 3'b000,
  NORMAL_SECURE_INSTRUCTION       = 3'b001,
  NORMAL_NONSECURE_DATA           = 3'b010,
  NORMAL_NONSECURE_INSTRUCTION    = 3'b011,
  PRIVILEGED_SECURE_DATA          = 3'b100,
  PRIVILEGED_SECURE_INSTRUCTION   = 3'b101,
  PRIVILEGED_NONSECURE_DATA       = 3'b110,
  PRIVILEGED_NONSECURE_INSTUCTION = 3'b111
} protection_type_e;

typedef enum bit [15:0] {
  SLAVE_0  = 16'b0000_0000_0000_0001,
  SLAVE_1  = 16'b0000_0000_0000_0010
} slave_no_e;

// APB Master Transaction - extracted constraints
class apb_master_tx;
  rand bit [ADDRESS_WIDTH-1:0] paddr;
  rand protection_type_e pprot;
  rand slave_no_e pselx;
  rand tx_type_e pwrite;
  rand transfer_size_e transfer_size;
  rand bit [DATA_WIDTH-1:0] pwdata;
  rand bit [(DATA_WIDTH/8)-1:0] pstrb;

  // Constraint: $countones - one-hot encoding
  constraint pselx_c1  { $countones(pselx) == 1; }

  // Constraint: range with comparison operators
  constraint pselx_c2 { pselx > 0 && pselx < 2**NO_OF_SLAVES; }

  // Constraint: soft + inside + range
  constraint pwdata_c3 { soft pwdata inside {[0:100]}; }

  // Constraint: if-else with $countones
  constraint transfer_size_c4 {
    if (transfer_size == BIT_8)
      $countones(pstrb) == 1;
    else if (transfer_size == BIT_16)
      $countones(pstrb) == 2;
    else if (transfer_size == BIT_24)
      $countones(pstrb) == 3;
    else
      $countones(pstrb) == 4;
  }
endclass

// APB Slave Transaction - extracted constraints
class apb_slave_tx;
  rand slave_error_e pslverr;
  rand bit [DATA_WIDTH-1:0] prdata;
  rand protection_type_e pprot;
  rand bit [2:0] no_of_wait_states;
  rand bit choose_packet_data;

  // Constraint: soft + inside + range
  constraint wait_states_c1 { soft no_of_wait_states inside {[0:3]}; }

  // Constraint: soft + equality with enum
  constraint pslverr_c2 { soft pslverr == NO_ERROR; }

  // Constraint: soft + equality
  constraint choose_data_packet_c3 { soft choose_packet_data == 1; }
endclass

module test_apb_constraints;
  initial begin
    apb_master_tx master;
    apb_slave_tx slave;

    master = new();
    slave = new();

    // Test randomization
    if (master.randomize()) begin
      $display("APB Master randomize success");
      $display("  paddr=%h pselx=%h pwdata=%h pstrb=%b",
               master.paddr, master.pselx, master.pwdata, master.pstrb);
    end

    if (slave.randomize()) begin
      $display("APB Slave randomize success");
      $display("  pslverr=%s no_of_wait_states=%d",
               slave.pslverr.name(), slave.no_of_wait_states);
    end
  end
endmodule
