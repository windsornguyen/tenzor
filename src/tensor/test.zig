//====================================================================
// File: src/tensor/test.zig
// Author(s): windsornguyen
// 
// Testing suite for the Tensor struct
//====================================================================

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const Datatype = @import("tensor.zig").Datatype;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectApproxEqAbs = std.testing.expectApproxEqAbs;

test "Tensor initialization" {
    var allocator = std.testing.allocator;
    const shape = [_]usize{ 2, 3, 4 };
    var tensor = try Tensor.init(&allocator, &shape, .f32);
    defer tensor.deinit();

    std.debug.print("\n=== Tensor Initialization Test ===\n", .{});
    std.debug.print("Shape: {any}\n", .{tensor.shape});
    std.debug.print("Data length: {d} bytes\n", .{tensor.data.len});

    try expectEqual(tensor.shape.len, 3);
    try expectEqual(tensor.shape[0], 2);
    try expectEqual(tensor.shape[1], 3);
    try expectEqual(tensor.shape[2], 4);
    try expectEqual(tensor.data.len, 2 * 3 * 4 * @sizeOf(f32));
    try expectEqual(tensor.dtype, Datatype.f32);

    std.debug.print("Tensor initialization test passed\n", .{});
}

test "Tensor fill and sum" {
    var allocator = std.testing.allocator;
    const shape = [_]usize{ 2, 3 };
    var tensor = try Tensor.init(&allocator, &shape, .f32);
    defer tensor.deinit();

    const fill_value: f64 = 2.5;
    tensor.fill(fill_value);
    const sum = tensor.sum();
    const expected_sum = fill_value * 2 * 3;

    std.debug.print("\n=== Tensor Fill and Sum Test ===\n", .{});
    std.debug.print("Fill value: {d}\n", .{fill_value});
    std.debug.print("Sum: {d}\n", .{sum});
    std.debug.print("Expected sum: {d}\n", .{expected_sum});

    try expectApproxEqAbs(sum, expected_sum, 1e-6);

    std.debug.print("Tensor fill and sum test passed\n", .{});
}

test "Tensor element-wise addition" {
    var allocator = std.testing.allocator;
    const shape = [_]usize{ 2, 2 };
    var a = try Tensor.init(&allocator, &shape, .f32);
    defer a.deinit();
    var b = try Tensor.init(&allocator, &shape, .f32);
    defer b.deinit();

    a.fill(2.0);
    b.fill(3.0);

    std.debug.print("\n=== Tensor Element-wise Addition Test ===\n", .{});

    try a.add(&b);
    try expectApproxEqAbs(a.sum(), 20.0, 1e-6);
    std.debug.print("Addition test passed\n", .{});
}

test "Tensor data types" {
    var allocator = std.testing.allocator;
    const shape = [_]usize{4};

    std.debug.print("\n=== Tensor Data Types Test ===\n", .{});

    inline for (.{ .f16, .f32, .f64, .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64, .bool }) |dtype| {
        var tensor = try Tensor.init(&allocator, &shape, dtype);
        defer tensor.deinit();

        tensor.fill(1.0);
        const sum = tensor.sum();

        std.debug.print("Datatype: {s}, Sum: {d}\n", .{ @tagName(dtype), sum });
        try expectApproxEqAbs(sum, 4.0, 1e-6);
    }

    std.debug.print("Tensor data types test passed\n", .{});
}

test "Tensor error handling" {
    var allocator = std.testing.allocator;
    const shape1 = [_]usize{ 2, 2 };
    const shape2 = [_]usize{ 2, 3 };
    var tensor1 = try Tensor.init(&allocator, &shape1, .f32);
    defer tensor1.deinit();
    var tensor2 = try Tensor.init(&allocator, &shape2, .f32);
    defer tensor2.deinit();

    std.debug.print("\n=== Tensor Error Handling Test ===\n", .{});

    // Test shape mismatch error
    const add_result = tensor1.add(&tensor2);
    std.debug.print("Add result: {!}\n", .{add_result});
    try expect(add_result == error.ShapeMismatch);

    std.debug.print("Tensor error handling test passed\n", .{});
}

test "Tensor operations (placeholders)" {
    std.debug.print("\n=== Tensor Operations Placeholders ===\n", .{});
    std.debug.print("The following operations are not yet implemented:\n", .{});
    std.debug.print("- Subtraction\n", .{});
    std.debug.print("- Multiplication\n", .{});
    std.debug.print("- Division\n", .{});
    std.debug.print("- Matrix multiplication\n", .{});
    std.debug.print("- Transposition\n", .{});
    std.debug.print("- Indexing and slicing\n", .{});
    std.debug.print("- Broadcasting\n", .{});
    std.debug.print("- Reduction operations (e.g., mean, standard deviation)\n", .{});
    std.debug.print("- Gradients and autograd\n", .{});
}

pub fn main() !void {
    std.debug.print("Run `zig build test` to execute the comprehensive Tensor test suite.\n", .{});
}