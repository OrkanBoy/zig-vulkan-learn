const std = @import("std").math;

pub const Affine3 = extern struct {    
    xx: f32,
    yx: f32,
    zx: f32,
    _x: f32,
    
    xy: f32,
    yy: f32,
    zy: f32,
    _y: f32,

    xz: f32,
    yz: f32,
    zz: f32,
    _z: f32,

    const Self = @This();
    pub const identity = Self {
        .xx = 1.0,
        .yx = 0.0,
        .zx = 0.0,
        ._x = 0.0,
        .xy = 0.0,
        .yy = 1.0,
        .zy = 0.0,
        ._y = 0.0,
        .xz = 0.0,
        .yz = 0.0,
        .zz = 1.0,
        ._z = 0.0,
    };

    pub fn compose(self: Self, other: Self) Self {
        return .{
            .xx = self.xx * other.xx + self.xy * other.yx + self.xz * other.zx,
            .yx = self.yx * other.xx + self.yy * other.yx + self.yz * other.zx,
            .zx = self.zx * other.xx + self.zy * other.yx + self.zz * other.zx,
            ._x = self._x * other.xx + self._y * other.yx + self._z * other.zx + other._x,

            .xy = self.xx * other.xy + self.xy * other.yy + self.xz * other.zy,
            .yy = self.yx * other.xy + self.yy * other.yy + self.yz * other.zy,
            .zy = self.zx * other.xy + self.zy * other.yy + self.zz * other.zy,
            ._y = self._x * other.xy + self._y * other.yy + self._z * other.zy + other._y,

            .xz = self.xx * other.xz + self.xy * other.yz + self.xz * other.zz,
            .yz = self.yx * other.xz + self.yy * other.yz + self.yz * other.zz,
            .zz = self.zx * other.xz + self.zy * other.yz + self.zz * other.zz,
            ._z = self._x * other.xz + self._y * other.yz + self._z * other.zz + other._z,
        };
    }

    pub fn scale(self: *Self, s: Scale3) *Self {
        self.xx *= s.x;
        self.yx *= s.x;
        self.zx *= s.x;
        self._x *= s.x;

        self.xy *= s.y;
        self.yy *= s.y;
        self.zy *= s.y;
        self._y *= s.y;

        self.xz *= s.z;
        self.yz *= s.z;
        self.zz *= s.z;
        self._z *= s.z;
        return self;
    }

    pub fn translate(self: *Self, v: Vector3) *Self {
        self._x += v.x;
        self._y += v.y;
        self._z += v.z;
        return self;
    }

    pub fn rotate(self: *Self, norm: f32, b: Bivector3) *Self {
        const zx_yz = b.zx * b.yz;
        const yz_xy = b.yz * b.xy;
        const xy_zx = b.xy * b.zx;

        const yz_yz = b.yz * b.yz;
        const zx_zx = b.zx * b.zx;
        const xy_xy = b.xy * b.xy;

        const cos = @cos(norm);
        const sin = @sin(norm);
        const one_sub_cos = 1.0 - cos;

        const yz_sin = b.yz * sin;
        const zx_sin = b.zx * sin;
        const xy_sin = b.xy * sin;

        const zx_yz_one_sub_cos = zx_yz * one_sub_cos;
        const yz_xy_one_sub_cos = yz_xy * one_sub_cos;
        const xy_zx_one_sub_cos = xy_zx * one_sub_cos;

        const xx = (1.0 - yz_yz) * cos + yz_yz;
        const xy = zx_yz_one_sub_cos + xy_sin;
        const xz = yz_xy_one_sub_cos - zx_sin;

        const yx = zx_yz_one_sub_cos - xy_sin;
        const yy = (1.0 - zx_zx) * cos + zx_zx;
        const yz = xy_zx_one_sub_cos + yz_sin;

        const zx = yz_xy_one_sub_cos + zx_sin;
        const zy = xy_zx_one_sub_cos - yz_sin;
        const zz = (1.0 - xy_xy) * cos + xy_xy;

        const new_self = Self {
            .xx = self.xx * xx + self.xy * yx + self.xz * zx,
            .yx = self.yx * xx + self.yy * yx + self.yz * zx,
            .zx = self.zx * xx + self.zy * yx + self.zz * zx,
            ._x = self._x * xx + self._y * yx + self._z * zx,

            .xy = self.xx * xy + self.xy * yy + self.xz * zy,
            .yy = self.yx * xy + self.yy * yy + self.yz * zy,
            .zy = self.zx * xy + self.zy * yy + self.zz * zy,
            ._y = self._x * xy + self._y * yy + self._z * zy,

            .xz = self.xx * xz + self.xy * yz + self.xz * zz,
            .yz = self.yx * xz + self.yy * yz + self.yz * zz,
            .zz = self.zx * xz + self.zy * yz + self.zz * zz,
            ._z = self._x * xz + self._y * yz + self._z * zz,
        };
        self.* = new_self;

        return self;
    }

};

pub const Vector3 = extern struct {
    const Self = @This();
    x: f32,
    y: f32,
    z: f32,

    pub const identity = Self {
        .x = 0.0,
        .y = 0.0,
        .z = 0.0,
    };

    pub fn add(self: Self, other: Self) Self {
        return .{
            .x = self.x + other.x,
            .y = self.y + other.y,
            .z = self.z + other.z,
        };
    }

    pub fn sub(self: Self, other: Self) Self {
        return .{
            .x = self.x - other.x,
            .y = self.y - other.y,
            .z = self.z - other.z,
        };
    }

    pub fn mul_f32(self: Self, other: f32) Self {
        return .{
            .x = self.x * other,
            .y = self.y * other,
            .z = self.z * other,
        };
    }

    pub fn div_f32(self: Self, other: f32) Self {
        return .{
            .x = self.x / other,
            .y = self.y / other,
            .z = self.z / other,
        };
    }

    pub fn neg(self: Self) Self {
        return .{
            .x = -self.x,
            .y = -self.y,
            .z = -self.z,
        };
    }

    pub fn wedge(self: Self, other: Self) Bivector3 {
        return .{
            .xy = self.x * other.y - self.y * other.x,
            .yz = self.y * other.z - self.z * other.y,
            .zx = self.z * other.x - self.x * other.z,
        };
    }

    pub fn dot(self: Self, other: Self) f32 {
        return self.x * other.x + self.y * other.y + self.z * other.z;
    }

    pub fn apply(self: Self, a: Affine3) Self {
        return .{
            .x = self.x * a.xx + self.y * a.yx + self.z * a.zx + a._x,
            .y = self.x * a.xy + self.y * a.yy + self.z * a.zy + a._y,
            .z = self.x * a.xz + self.y * a.yz + self.z * a.zz + a._z,
        };
    }

    pub fn add_assign(self: *Self, other: Self) void {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }

    pub fn sub_assign(self: *Self, other: Self) void {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
};


pub const Bivector3 = extern struct  {
    xy: f32,
    yz: f32,
    zx: f32,

    const Self = @This();
    const identity = Self {
        .xy = 0.0,
        .yz = 0.0,
        .zx = 0.0,
    };

    pub fn commute(self: Self, other: Bivector3) Bivector3 {
        return .{
            .xy = self.yz * other.zx - other.yz * self.zx,
            .yz = self.zx * other.xy - other.zx * self.xy,
            .zx = self.xy * other.yz - other.xy * self.yz,
        };
    }

    pub fn norm_sqr(self: *Self) f32 {
        return self.xy * self.xy + self.yz * self.yz + self.zx * self.zx;
    }

    /// In R3 the biVector squares to a negative scalar
    /// hence we can factor the BiVector to a scalar and unit biVector
    pub fn exp(self: *Self) Rotor3 {
        const _norm_sqr = self.norm_sqr();
        if (_norm_sqr == 0.0) {
            return Rotor3.identity;
        }
        const norm = _norm_sqr.sqrt();
        const b = self.mul(@sin(norm) / norm);

        return .{
            ._1 = @cos(norm),
            .xy = b.xy,
            .yz = b.yz,
            .zx = b.zx,
        };
    }

    pub fn add_assign(self: *Self, other: Self) void {
        self.xy += other.xy;
        self.yz += other.yz;
        self.zx += other.zx;
    }

    pub fn mul_f32(self: Self, other: f32) Self {
        return .{
            .xy = self.xy * other,
            .yz = self.yz * other,
            .zx = self.zx * other,
        };
    }

    pub fn div_f32(self: Self, other: f32) Self {
        return Self {
            .xy = self.xy / other,
            .yz = self.yz / other,
            .zx = self.zx / other,
        };
    }
};

pub const Rotor3 = extern struct {
    _1: f32,
    xy: f32,
    yz: f32,
    zx: f32,

    const Self = @This();
    const identity = Self {
        ._1 = 1.0,
        .xy = 0.0,
        .yz = 0.0,
        .zx = 0.0,
    };

    pub fn mul(self: Self, other: Rotor3) Self {
        return .{
            ._1 = -self.xy * other.xy - self.yz * other.yz - self.zx * other.zx,
            .xy =  self.xy * other._1 + self.yz * other.zx - self.zx * other.yz,
            .yz = -self.xy * other.zx + self.yz * other._1 + self.zx * other.xy,
            .zx =  self.yz * other.xy - self.xy * other.yz + self.zx * other._1,
        };
    }

    pub fn norm_sqr(self: Self) f32 {
        return self._1 * self._1 + self.xy * self.xy + self.yz * self.yz + self.zx * self.zx;
    }

    pub fn div_assign_f32(self: *Self, other: f32) void {
        self._1 /= other;
        self.xy /= other;
        self.yz /= other;
        self.zx /= other;
    }
};

pub const Scale3 = struct {
    x: f32,
    y: f32,
    z: f32,

    const Self = @This();
    const identity = Self {
        .x = 1.0,
        .y = 1.0,
        .z = 1.0,
    };
};

pub const Vector2 = extern struct {
    const Self = @This();
    x: f32,
    y: f32,

    pub const identity = Self {
        .x = 0.0,
        .y = 0.0,
    };

    pub fn add(self: Self, other: Self) Self {
        return .{
            .x = self.x + other.x,
            .y = self.y + other.y,
        };
    }

    pub fn sub(self: Self, other: Self) Self {
        return .{
            .x = self.x - other.x,
            .y = self.y - other.y,
        };
    }

    pub fn mul_f32(self: Self, other: f32) Self {
        return .{
            .x = self.x * other,
            .y = self.y * other,
        };
    }

    pub fn div_f32(self: Self, other: f32) Self {
        return .{
            .x = self.x / other,
            .y = self.y / other,
        };
    }

    pub fn neg(self: Self) Self {
        return .{
            .x = -self.x,
            .y = -self.y,
        };
    }

    pub fn dot(self: Self, other: Self) f32 {
        return self.x * other.x + self.y * other.y;
    }

    pub fn add_assign(self: *Self, other: Self) void {
        self.x += other.x;
        self.y += other.y;
    }

    pub fn sub_assign(self: *Self, other: Self) void {
        self.x -= other.x;
        self.y -= other.y;
    }
};