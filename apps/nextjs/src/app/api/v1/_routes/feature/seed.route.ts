import { Hono } from "hono";
import { Seed } from "@/lib/database/schema/feature/seed.model";
import db from "@/lib/database/db";
import { parseError } from "@/lib/utils";

const seedRoute = new Hono();

// GET - Get all seeds
seedRoute.get("/", async (c) => {
  try {
    await db();

    const page = Number(c.req.query("page")) || 1;
    const pageSize = Number(c.req.query("pageSize")) || 10;

    const total = await Seed.countDocuments();

    const seeds = await Seed.find()
      .skip((page - 1) * pageSize)
      .limit(pageSize)
      .sort({ createdAt: -1 });

    return c.json({
      message: "Seeds retrieved successfully",
      data: {
        items: seeds,
        total,
        currentPage: page,
        totalPages: Math.ceil(total / pageSize),
        pageSize,
      },
    });
  } catch (error) {
    console.error("Get seeds error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

seedRoute.post("/", async (c) => {
  try {
    await db();
    const body = await c.req.json();

    // Validasi input
    if (!body.name || !body.duration) {
      return c.json({ message: "Name and duration are required" }, 400);
    }

    if (body.duration <= 0) {
      return c.json({ message: "Duration must be greater than 0" }, 400);
    }

    const newSeed = await Seed.create({
      name: body.name.trim(),
      duration: Number(body.duration),
    });

    return c.json(
      {
        message: "Seed created successfully",
        data: newSeed,
      },
      201
    );
  } catch (error) {
    console.error("Create seed error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});
seedRoute.delete("/:id", async (c) => {
  try {
    await db();
    const id = c.req.param("id");
    const deletedSeed = await Seed.findByIdAndDelete(id);

    if (!deletedSeed) {
      return c.json({ message: "Seed not found" }, 404);
    }

    return c.json({ message: "Deleted" });
  } catch (error) {
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

export default seedRoute;
