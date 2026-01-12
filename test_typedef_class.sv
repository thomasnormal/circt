package test_pkg;
  class registry #(type T=int);
    static function registry get();
      registry r = new();
      return r;
    endfunction
  endclass
  class my_class;
    typedef registry#(my_class) type_id;
    static function type_id get_type();
      return type_id::get();
    endfunction
  endclass
endpackage
