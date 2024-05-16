const vk = @import("vk");
const std = @import("std");
const print = std.debug.print;

pub const Image = packed struct {
    // =====header begin======
    width: u16,
    height: u16,
    format: vk.Format,
    size: usize,
    // =====header end======
    ptr: [*]u8,

    pub const header_bytes = @offsetOf(Self, "ptr");

    const Self = @This();

    pub fn load(file: std.fs.File, allocator: std.mem.Allocator) !Self {
        var self: Self = undefined;

        if (try file.read(@as([*]u8, @ptrCast(&self))[0 .. header_bytes]) != header_bytes) {
            return error.HeaderReadFailed;
        }

        const alloc = try allocator.alloc(u8, self.size);
        self.ptr = alloc.ptr;

        if (try file.read(alloc) != self.size) {
            return error.ImageReadFailed;
        }
        return self;
    }
};