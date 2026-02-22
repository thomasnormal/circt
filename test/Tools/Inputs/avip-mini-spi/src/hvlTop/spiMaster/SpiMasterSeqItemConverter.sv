function void do_print();
  printer.print_field($sformatf("masterInSlaveOut[%0d]",i,),packetStruct.masterInSlaveOut[i],8,UVM_HEX);
endfunction
