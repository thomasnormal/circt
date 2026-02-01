`timescale 1ns/1ps

class Container;
  static string arr[string];

  static function void add(string key, string value);
    $display("  Adding arr[%s] = %s", key, value);
    arr[key] = value;
  endfunction

  static function string get(string key);
    if (arr.exists(key)) begin
      $display("  Getting arr[%s] = %s", key, arr[key]);
      return arr[key];
    end else begin
      $display("  Key %s not found, returning empty", key);
      return "";
    end
  endfunction
endclass

module test;
  initial begin
    string val;

    $display("Step 1: Add some entries");
    Container::add("first", "AAA");
    Container::add("second", "BBB");

    $display("Step 2: Get entries");
    val = Container::get("first");
    $display("  Got: %s", val);

    val = Container::get("second");
    $display("  Got: %s", val);

    val = Container::get("third");
    $display("  Got: %s", val);

    $display("Done");
    $finish;
  end
endmodule
