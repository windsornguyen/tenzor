//====================================================================
// File: src/tensor/main.zig
// Author(s): windsornguyen
//
// The Tensor struct
//====================================================================

const std = @import("std");
const math = std.math;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectApproxEqAbs = std.testing.expectApproxEqAbs;

pub const Datatype = enum {
    // Floating point types
    f16, // 16-bit floating point
    f32, // 32-bit floating point
    f64, // 64-bit floating point
    // bf16,   // 16-bit brain floating point (bfloat16) - TODO: Not natively supported in Zig

    // Complex types - Not natively supported in Zig
    // complex32,  // 32-bit complex
    // complex64,  // 64-bit complex
    // complex128, // 128-bit complex

    // Signed integer types
    i8, // 8-bit signed integer
    i16, // 16-bit signed integer
    i32, // 32-bit signed integer
    i64, // 64-bit signed integer

    // Unsigned integer types
    u8, // 8-bit unsigned integer
    u16, // 16-bit unsigned integer
    u32, // 32-bit unsigned integer
    u64, // 64-bit unsigned integer

    // Boolean type
    bool, // Boolean

    // Quantized types - Not natively supported in Zig
    // quint8,  // quantized 8-bit unsigned integer
    // qint8,   // quantized 8-bit signed integer
    // qint32,  // quantized 32-bit signed integer
    // quint4x2,// quantized 4-bit unsigned integer

    // 8-bit floating point types - Not natively supported in Zig
    // float8_e4m3fn, // 8-bit floating point, e4m3
    // float8_e5m2,   // 8-bit floating point, e5m2

    pub fn sizeOf(self: Datatype) usize {
        return switch (self) {
            .f16 => 2,
            .f32 => 4,
            .f64 => 8,
            .i8, .u8 => 1,
            .i16, .u16 => 2,
            .i32, .u32 => 4,
            .i64, .u64 => 8,
            .bool => 1,
        };
    }

    pub fn isFloat(self: Datatype) bool {
        return switch (self) {
            .f16, .f32, .f64 => true,
            else => false,
        };
    }

    pub fn isInt(self: Datatype) bool {
        return switch (self) {
            .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64 => true,
            else => false,
        };
    }

    pub fn isSigned(self: Datatype) bool {
        return switch (self) {
            .i8, .i16, .i32, .i64, .f16, .f32, .f64 => true,
            else => false,
        };
    }
};

pub const Tensor = struct {
    data: []u8, // Raw byte array to store data
    shape: []usize, // Dimension of tensor
    strides: []usize, // Number of elements to skip in each dimension
    dtype: Datatype, // Data type of the tensor's elements
    allocator: *std.mem.Allocator, // Memory allocator

    fn get_size(comptime dtype: Datatype) comptime_int {
        return switch (dtype) {
            .bool, .i8, .u8 => 1,
            .f16, .i16, .u16 => 2,
            .f32, .i32, .u32 => 4,
            .f64, .i64, .u64 => 8,
        };
    }

    pub fn init(allocator: *std.mem.Allocator, shape: []const usize, comptime dtype: Datatype) !Tensor {
        const total_elements = blk: {
            var prod: usize = 1;
            for (shape) |dim| {
                prod *= dim;
            }
            break :blk prod;
        };

        const element_size = comptime get_size(dtype);

        const data = try allocator.alloc(u8, total_elements * element_size);
        errdefer allocator.free(data);

        const shape_copy = try allocator.dupe(usize, shape);
        errdefer allocator.free(shape_copy);

        const strides = try allocator.alloc(usize, shape.len);
        errdefer allocator.free(strides);

        var stride: usize = 1;
        var i: usize = shape.len;
        while (i > 0) {
            i -= 1;
            strides[i] = stride;
            stride *= shape[i];
        }

        return Tensor{
            .data = data,
            .shape = shape_copy,
            .strides = strides,
            .dtype = dtype,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape);
        self.allocator.free(self.strides);
    }

    // TODO: These operations can probably be vectorized later on if we have GPU support
    pub fn fill(self: *Tensor, value: f64) void {
        switch (self.dtype) {
            .f16 => {
                const slice = std.mem.bytesAsSlice(f16, self.data);
                for (slice) |*item| {
                    item.* = @as(f16, @floatCast(value));
                }
            },
            .f32 => {
                const slice = std.mem.bytesAsSlice(f32, self.data);
                for (slice) |*item| {
                    item.* = @as(f32, @floatCast(value));
                }
            },
            .f64 => {
                const slice = std.mem.bytesAsSlice(f64, self.data);
                for (slice) |*item| {
                    item.* = value;
                }
            },
            .i8 => {
                const slice = std.mem.bytesAsSlice(i8, self.data);
                for (slice) |*item| {
                    item.* = @as(i8, @intFromFloat(value));
                }
            },
            .i16 => {
                const slice = std.mem.bytesAsSlice(i16, self.data);
                for (slice) |*item| {
                    item.* = @as(i16, @intFromFloat(value));
                }
            },
            .i32 => {
                const slice = std.mem.bytesAsSlice(i32, self.data);
                for (slice) |*item| {
                    item.* = @as(i32, @intFromFloat(value));
                }
            },
            .i64 => {
                const slice = std.mem.bytesAsSlice(i64, self.data);
                for (slice) |*item| {
                    item.* = @as(i64, @intFromFloat(value));
                }
            },
            .u8 => {
                const slice = std.mem.bytesAsSlice(u8, self.data);
                for (slice) |*item| {
                    item.* = @as(u8, @intFromFloat(@max(0, value)));
                }
            },
            .u16 => {
                const slice = std.mem.bytesAsSlice(u16, self.data);
                for (slice) |*item| {
                    item.* = @as(u16, @intFromFloat(@max(0, value)));
                }
            },
            .u32 => {
                const slice = std.mem.bytesAsSlice(u32, self.data);
                for (slice) |*item| {
                    item.* = @as(u32, @intFromFloat(@max(0, value)));
                }
            },
            .u64 => {
                const slice = std.mem.bytesAsSlice(u64, self.data);
                for (slice) |*item| {
                    item.* = @as(u64, @intFromFloat(@max(0, value)));
                }
            },
            .bool => {
                const slice = std.mem.bytesAsSlice(bool, self.data);
                for (slice) |*item| {
                    item.* = value != 0;
                }
            },
        }
    }

    pub fn sum(self: *const Tensor) f64 {
        var result: f64 = 0;

        switch (self.dtype) {
            .f16 => {
                const slice = std.mem.bytesAsSlice(f16, self.data);
                for (slice) |item| {
                    result += @as(f64, @floatCast(item));
                }
            },
            .f32 => {
                const slice = std.mem.bytesAsSlice(f32, self.data);
                for (slice) |item| {
                    result += item;
                }
            },
            .f64 => {
                const slice = std.mem.bytesAsSlice(f64, self.data);
                for (slice) |item| {
                    result += item;
                }
            },
            .i8 => {
                const slice = std.mem.bytesAsSlice(i8, self.data);
                for (slice) |item| {
                    result += @as(f64, @floatFromInt(item));
                }
            },
            .i16 => {
                const slice = std.mem.bytesAsSlice(i16, self.data);
                for (slice) |item| {
                    result += @as(f64, @floatFromInt(item));
                }
            },
            .i32 => {
                const slice = std.mem.bytesAsSlice(i32, self.data);
                for (slice) |item| {
                    result += @as(f64, @floatFromInt(item));
                }
            },
            .i64 => {
                const slice = std.mem.bytesAsSlice(i64, self.data);
                for (slice) |item| {
                    result += @as(f64, @floatFromInt(item));
                }
            },
            .u8 => {
                const slice = std.mem.bytesAsSlice(u8, self.data);
                for (slice) |item| {
                    result += @as(f64, @floatFromInt(item));
                }
            },
            .u16 => {
                const slice = std.mem.bytesAsSlice(u16, self.data);
                for (slice) |item| {
                    result += @as(f64, @floatFromInt(item));
                }
            },
            .u32 => {
                const slice = std.mem.bytesAsSlice(u32, self.data);
                for (slice) |item| {
                    result += @as(f64, @floatFromInt(item));
                }
            },
            .u64 => {
                const slice = std.mem.bytesAsSlice(u64, self.data);
                for (slice) |item| {
                    result += @as(f64, @floatFromInt(item));
                }
            },
            .bool => {
                const slice = std.mem.bytesAsSlice(bool, self.data);
                for (slice) |item| {
                    result += if (item) 1 else 0;
                }
            },
        }

        return result;
    }

    pub fn add(self: *Tensor, other: *const Tensor) !void {
        if (!std.mem.eql(usize, self.shape, other.shape)) {
            return error.ShapeMismatch;
        }
        if (self.dtype != other.dtype) {
            return error.DtypeMismatch;
        }

        switch (self.dtype) {
            .f16 => {
                const self_slice = std.mem.bytesAsSlice(f16, self.data);
                const other_slice = std.mem.bytesAsSlice(f16, other.data);
                for (self_slice, other_slice) |*addend_1, addend_2| {
                    addend_1.* += addend_2;
                }
            },
            .f32 => {
                const self_slice = std.mem.bytesAsSlice(f32, self.data);
                const other_slice = std.mem.bytesAsSlice(f32, other.data);
                for (self_slice, other_slice) |*addend_1, addend_2| {
                    addend_1.* += addend_2;
                }
            },
            .f64 => {
                const self_slice = std.mem.bytesAsSlice(f64, self.data);
                const other_slice = std.mem.bytesAsSlice(f64, other.data);
                for (self_slice, other_slice) |*addend_1, addend_2| {
                    addend_1.* += addend_2;
                }
            },
            .i8 => {
                const self_slice = std.mem.bytesAsSlice(i8, self.data);
                const other_slice = std.mem.bytesAsSlice(i8, other.data);
                for (self_slice, other_slice) |*addend_1, addend_2| {
                    addend_1.* = @addWithOverflow(addend_1.*, addend_2)[0];
                }
            },
            .i16 => {
                const self_slice = std.mem.bytesAsSlice(i16, self.data);
                const other_slice = std.mem.bytesAsSlice(i16, other.data);
                for (self_slice, other_slice) |*addend_1, addend_2| {
                    addend_1.* = @addWithOverflow(addend_1.*, addend_2)[0];
                }
            },
            .i32 => {
                const self_slice = std.mem.bytesAsSlice(i32, self.data);
                const other_slice = std.mem.bytesAsSlice(i32, other.data);
                for (self_slice, other_slice) |*addend_1, addend_2| {
                    addend_1.* = @addWithOverflow(addend_1.*, addend_2)[0];
                }
            },
            .i64 => {
                const self_slice = std.mem.bytesAsSlice(i64, self.data);
                const other_slice = std.mem.bytesAsSlice(i64, other.data);
                for (self_slice, other_slice) |*addend_1, addend_2| {
                    addend_1.* = @addWithOverflow(addend_1.*, addend_2)[0];
                }
            },
            .u8 => {
                const self_slice = std.mem.bytesAsSlice(u8, self.data);
                const other_slice = std.mem.bytesAsSlice(u8, other.data);
                for (self_slice, other_slice) |*addend_1, addend_2| {
                    addend_1.* = @addWithOverflow(addend_1.*, addend_2)[0];
                }
            },
            .u16 => {
                const self_slice = std.mem.bytesAsSlice(u16, self.data);
                const other_slice = std.mem.bytesAsSlice(u16, other.data);
                for (self_slice, other_slice) |*addend_1, addend_2| {
                    addend_1.* = @addWithOverflow(addend_1.*, addend_2)[0];
                }
            },
            .u32 => {
                const self_slice = std.mem.bytesAsSlice(u32, self.data);
                const other_slice = std.mem.bytesAsSlice(u32, other.data);
                for (self_slice, other_slice) |*addend_1, addend_2| {
                    addend_1.* = @addWithOverflow(addend_1.*, addend_2)[0];
                }
            },
            .u64 => {
                const self_slice = std.mem.bytesAsSlice(u64, self.data);
                const other_slice = std.mem.bytesAsSlice(u64, other.data);
                for (self_slice, other_slice) |*addend_1, addend_2| {
                    addend_1.* = @addWithOverflow(addend_1.*, addend_2)[0];
                }
            },
            .bool => {
                const self_slice = std.mem.bytesAsSlice(bool, self.data);
                const other_slice = std.mem.bytesAsSlice(bool, other.data);
                for (self_slice, other_slice) |*addend_1, addend_2| {
                    addend_1.* = addend_1.* or addend_2;
                }
            },
        }
    }
};

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
    std.debug.print("Compiled successfully! Run `zig build test` to run the tests.\n", .{});
}
