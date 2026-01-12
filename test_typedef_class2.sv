package test_pkg;
  class registry #(type T=int);
    static function registry get();
      registry r = new();
      return r;
    endfunction
  endclass
  class my_class;
  endclass

  // Direct usage without typedef
  function registry#(my_class) get_direct();
    return registry#(my_class)::get();
  endfunction
endpackage
