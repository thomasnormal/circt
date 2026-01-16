// AVIP AHB Constraint Patterns - Extracted from AhbMasterTransaction.sv and AhbSlaveTransaction.sv
// Tests soft constraints, if-else constraints, queue constraints, foreach constraints

parameter int ADDR_WIDTH = 32;
parameter int DATA_WIDTH = 32;
parameter int NO_OF_SLAVES = 1;
parameter int HMASTER_WIDTH = 4;
parameter int LENGTH = 4;

typedef enum bit [2:0] {
  SINGLE = 3'b000,
  INCR   = 3'b001,
  WRAP4  = 3'b010,
  INCR4  = 3'b011,
  WRAP8  = 3'b100,
  INCR8  = 3'b101,
  WRAP16 = 3'b110,
  INCR16 = 3'b111
} ahbBurstEnum;

typedef enum bit [2:0] {
  BYTE       = 3'b000,
  HALFWORD   = 3'b001,
  WORD       = 3'b010,
  DOUBLEWORD = 3'b011
} ahbHsizeEnum;

typedef enum bit [1:0] {
  IDLE   = 2'b00,
  BUSY   = 2'b01,
  NONSEQ = 2'b10,
  SEQ    = 2'b11
} ahbTransferEnum;

typedef enum bit {
  AHB_READ  = 1'b0,
  AHB_WRITE = 1'b1
} ahbOperationEnum;

typedef enum bit [3:0] {
  OPCODE_FETCH              = 4'b0001,
  DATA_ACCESS               = 4'b0010,
  USER_ACCESS               = 4'b0100,
  PRIVILEGED_ACCESS         = 4'b1000
} ahbProtectionEnum;

typedef enum bit {
  OKAY  = 1'b0,
  ERROR_RESP = 1'b1
} ahbRespEnum;

// AHB Master Transaction - extracted constraints
class AhbMasterTransaction;
  rand bit [ADDR_WIDTH-1:0] haddr;
  rand bit [NO_OF_SLAVES-1:0] hselx;
  rand ahbBurstEnum hburst;
  rand bit hmastlock;
  rand ahbProtectionEnum hprot;
  rand ahbHsizeEnum hsize;
  rand bit hnonsec;
  rand bit hexcl;
  rand bit [HMASTER_WIDTH-1:0] hmaster;
  rand ahbTransferEnum htrans;
  rand bit [DATA_WIDTH-1:0] hwdata[$:2**LENGTH];
  rand bit [(DATA_WIDTH/8)-1:0] hwstrb[$:2**LENGTH];
  rand ahbOperationEnum hwrite;
  rand bit hexokay;
  rand bit busyControl[];

  // Constraint: foreach with if-else and $countones
  constraint strobleValue {
    foreach(hwstrb[i]) {
      if (hsize == BYTE) $countones(hwstrb[i]) == 1;
      else if (hsize == HALFWORD) $countones(hwstrb[i]) == 2;
      else if (hsize == WORD) $countones(hwstrb[i]) == 4;
      else if (hsize == DOUBLEWORD) $countones(hwstrb[i]) == 8;
    }
  }

  // Constraint: if-else with queue.size()
  constraint burstsize {
    if (hburst == WRAP4 || hburst == INCR4) hwdata.size() == 4;
    else if (hburst == WRAP8 || hburst == INCR8) hwdata.size() == 8;
    else if (hburst == WRAP16 || hburst == INCR16) hwdata.size() == 16;
    else hwdata.size() == 1;
  }

  // Constraint: if-else with queue.size() for strobe
  constraint strobesize {
    if (hburst == WRAP4 || hburst == INCR4) hwstrb.size() == 4;
    else if (hburst == WRAP8 || hburst == INCR8) hwstrb.size() == 8;
    else if (hburst == WRAP16 || hburst == INCR16) hwstrb.size() == 16;
    else hwstrb.size() == 1;
  }

  // Constraint: if-else with dynamic array size
  constraint busyState {
    if (hburst == WRAP4 || hburst == INCR4) busyControl.size() == 4;
    else if (hburst == WRAP8 || hburst == INCR8) busyControl.size() == 8;
    else if (hburst == WRAP16 || hburst == INCR16) busyControl.size() == 16;
    else busyControl.size() == 1;
  }
endclass

// AHB Slave Transaction - extracted constraints
class AhbSlaveTransaction;
  rand bit [DATA_WIDTH-1:0] hwdata[$:2**LENGTH];
  rand bit [(DATA_WIDTH/8)-1:0] hwstrb[$:2**LENGTH];
  rand bit [DATA_WIDTH-1:0] hrdata[$:2**LENGTH];
  rand bit hreadyout;
  rand bit hexokay;
  rand bit choosePacketData;
  rand int noOfWaitStates;

  // Constraint: soft with equality
  constraint chooseDataPacketC1 { soft choosePacketData == 0; }

  // Constraint: queue size with literal
  constraint readDataSize { hrdata.size() == 16; }
  constraint writeDataSize { hwdata.size() == 16; }
  constraint hwstrbSize { hwstrb.size() == 16; }

  // Constraint: soft with equality (int type)
  constraint waitState { soft noOfWaitStates == 0; }
endclass

module test_ahb_constraints;
  initial begin
    AhbMasterTransaction master;
    AhbSlaveTransaction slave;

    master = new();
    slave = new();

    // Test randomization
    if (master.randomize()) begin
      $display("AHB Master randomize success");
      $display("  hburst=%s hsize=%s hwdata.size=%d",
               master.hburst.name(), master.hsize.name(), master.hwdata.size());
    end

    if (slave.randomize()) begin
      $display("AHB Slave randomize success");
      $display("  noOfWaitStates=%d choosePacketData=%d",
               slave.noOfWaitStates, slave.choosePacketData);
    end
  end
endmodule
