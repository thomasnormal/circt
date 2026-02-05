// Sample SystemVerilog file for LSP testing
module counter #(
    parameter WIDTH = 8
) (
    input  logic clk,
    input  logic rst_n,
    input  logic enable,
    output logic [WIDTH-1:0] count
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count <= '0;
        else if (enable)
            count <= count + 1'b1;
    end

endmodule

module top (
    input  logic clk,
    input  logic rst_n
);
    logic enable;
    logic [7:0] cnt_value;

    counter #(.WIDTH(8)) u_counter (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .count(cnt_value)
    );

    assign enable = 1'b1;

endmodule
