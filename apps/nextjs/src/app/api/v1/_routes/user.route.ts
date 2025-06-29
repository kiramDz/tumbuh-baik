import { Hono } from "hono";
import { User } from "@/lib/database/schema/users.model";
import db from "@/lib/database/db";
import { parseError } from "@/lib/utils";

const userRoute = new Hono();

// GET - Get all users
userRoute.get("/", async (c) => {
  try {
    await db();

    const page = Number(c.req.query("page")) || 1;
    const pageSize = Number(c.req.query("pageSize")) || 10;

    const total = await User.countDocuments();

    const users = await User.find()
      .select("name email emailVerified role createdAt updatedAt") // Select only needed fields, exclude sensitive data
      .skip((page - 1) * pageSize)
      .limit(pageSize)
      .sort({ createdAt: -1 });

    return c.json({
      message: "Users retrieved successfully",
      data: {
        items: users,
        total,
        currentPage: page,
        totalPages: Math.ceil(total / pageSize),
        pageSize,
      },
    });
  } catch (error) {
    console.error("Get users error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

// PUT - Update user role
userRoute.put("/:id/role", async (c) => {
  try {
    await db();
    const userId = c.req.param("id");
    const body = await c.req.json();

    // Validasi input
    if (!body.role) {
      return c.json({ message: "Role is required" }, 400);
    }

    if (!["user", "admin"].includes(body.role)) {
      return c.json({ message: "Role must be either 'user' or 'admin'" }, 400);
    }

    // Cek apakah user exists
    const existingUser = await User.findById(userId);
    if (!existingUser) {
      return c.json({ message: "User not found" }, 404);
    }

    // Update role
    const updatedUser = await User.findByIdAndUpdate(userId, { role: body.role }, { new: true }).select("name email emailVerified role createdAt updatedAt");

    return c.json({
      message: "User role updated successfully",
      data: updatedUser,
    });
  } catch (error) {
    console.error("Update user role error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

export default userRoute;