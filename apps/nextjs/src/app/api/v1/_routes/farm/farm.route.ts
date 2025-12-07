import { Hono } from "hono";
import { Farm } from "@/lib/database/schema/model/farm.model";
import db from "@/lib/database/db";
import { parseError } from "@/lib/utils";

const farmRoute = new Hono();

// CREATE - Tambah farm baru
farmRoute.post("/", async (c) => {
    try {
        await db();
        const body = await c.req.json();
        
        // Validasi required fields
        if (!body.name || !body.location || !body.userId) {
            return c.json(
                { message: "Validation Error", description: "name, location, and userId are required" },
                { status: 400 }
            );
        }

        const newFarm = new Farm(body);
        const savedFarm = await newFarm.save();
        
        return c.json(savedFarm, { status: 201 });
    } catch (err: any) {
        const errMsg = parseError(err);
        return c.json({ message: "Error", description: errMsg }, { status: 500 });
    }
});

// READ ALL - Get semua farms (bisa filter by userId)
farmRoute.get("/", async (c) => {
    try {
        await db();
        const userId = c.req.query("userId"); // Optional filter by user
        
        const query = userId ? { userId } : {};
        const allFarms = await Farm.find(query)
            .populate('userId', 'name email') // Populate user data
            .sort({ createdAt: -1 })
            .lean();
        
        console.log("All Farms:", allFarms);
        return c.json(allFarms, { status: 200 });
    } catch (err: any) {
        const errMsg = parseError(err);
        return c.json({ message: "Error", description: errMsg }, { status: 500 });
    }
});

// READ ONE - Get farm by ID
farmRoute.get("/:id", async (c) => {
    try {
        await db();
        const id = c.req.param("id");
        
        const farm = await Farm.findById(id)
            .populate('userId', 'name email')
            .lean();
        
        if (!farm) {
            return c.json(
                { message: "Not Found", description: "Farm not found" },
                { status: 404 }
            );
        }
        
        return c.json(farm, { status: 200 });
    } catch (err: any) {
        const errMsg = parseError(err);
        return c.json({ message: "Error", description: errMsg }, { status: 500 });
    }
});

// UPDATE - Update farm by ID
farmRoute.put("/:id", async (c) => {
    try {
        await db();
        const id = c.req.param("id");
        const body = await c.req.json();
        
        const updatedFarm = await Farm.findByIdAndUpdate(
            id,
            body,
            { new: true, runValidators: true }
        )
            .populate('userId', 'name email')
            .lean();
        
        if (!updatedFarm) {
            return c.json(
                { message: "Not Found", description: "Farm not found" },
                { status: 404 }
            );
        }
        
        return c.json(updatedFarm, { status: 200 });
    } catch (err: any) {
        const errMsg = parseError(err);
        return c.json({ message: "Error", description: errMsg }, { status: 500 });
    }
});

// DELETE - Delete farm by ID
farmRoute.delete("/:id", async (c) => {
    try {
        await db();
        const id = c.req.param("id");
        
        const deletedFarm = await Farm.findByIdAndDelete(id).lean();
        
        if (!deletedFarm) {
            return c.json(
                { message: "Not Found", description: "Farm not found" },
                { status: 404 }
            );
        }
        
        return c.json(
            { message: "Success", description: "Farm deleted successfully" },
            { status: 200 }
        );
    } catch (err: any) {
        const errMsg = parseError(err);
        return c.json({ message: "Error", description: errMsg }, { status: 500 });
    }
});

// GET by User ID - Get all farms for specific user
farmRoute.get("/user/:userId", async (c) => {
    try {
        await db();
        const userId = c.req.param("userId");
        
        const farms = await Farm.find({ userId })
            .sort({ createdAt: -1 })
            .lean();
        
        return c.json(farms, { status: 200 });
    } catch (err: any) {
        const errMsg = parseError(err);
        return c.json({ message: "Error", description: errMsg }, { status: 500 });
    }
});

export { farmRoute };