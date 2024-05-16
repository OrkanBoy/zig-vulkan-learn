const std = @import("std");
const img = @import("image.zig");
const print = std.debug.print;

fn bytesAsValue(comptime T: type, bytes: []const u8) T {
    return @as(*const T, @ptrCast(@alignCast(bytes))).*;
}

fn valueAsBytes(value: anytype) []u8 {
    return @as([*] u8, @ptrCast(@constCast(&value)))[0..@sizeOf(@TypeOf(value))];
}

pub fn main() !void {
    const cwd = std.fs.cwd();

    const out = try cwd.createFile("assets.hex", .{});

    const file = try cwd.openFile("images/statue.bmp", .{
        .mode = .read_only,
    });
    defer file.close();

    var header: [0x50]u8 = undefined;
    if (try file.read(&header) != header.len) {
        return error.HeaderReadFail;
    }

    // check 0x46 .. 0x4A
    var image: img.Image = undefined;
    image.format = .b8g8r8a8_srgb;
    image.width = bytesAsValue(u16, header[0x12..0x14]);
    image.height = bytesAsValue(u16, header[0x16..0x18]);
    
    const bytes_per_pixel = header[0x1C] / 0x8;

    const pixels_len = @as(u32, image.width) * @as(u32, image.height);
    const bytes_to_read = pixels_len * bytes_per_pixel;

    const allocator = std.heap.page_allocator;
    image.size = pixels_len * 0x4;
    const alloc = try allocator.alloc(u8, image.size);
    image.ptr = alloc.ptr;
    defer allocator.free(alloc);

    var read_i = image.size - bytes_to_read;
    
    const pixels_offset = bytesAsValue(u16, header[0xA..0xE]);
    _ = try file.seekTo(pixels_offset);

    if (try file.read(image.ptr[read_i .. image.size]) != bytes_to_read) {
        return error.PixelsReadFail;
    }

    if (bytes_per_pixel == 0x3) {
        var write_i: usize = 0;

        while (read_i != image.size) {
            image.ptr[write_i] = image.ptr[read_i];
            write_i += 1;
            read_i += 1;

            image.ptr[write_i] = image.ptr[read_i];
            write_i += 1;
            read_i += 1;

            image.ptr[write_i] = image.ptr[read_i];
            write_i += 1;
            read_i += 1;

            image.ptr[write_i] = 0xFF;
            write_i += 1;            
        }
    } else if (bytes_per_pixel != 0x4) {
        return error.UnsupportedBytesPerPixel;
    }

    try out.writeAll(@as([*]u8, @ptrCast(&image))[0 .. img.Image.header_bytes]);
    try out.writeAll(image.ptr[0 .. image.size]);
}