import { Hono } from "hono";
import { KuesionerPetani, KuesionerManajemen } from "@/lib/database/schema/dataset/kuesioner-petani";
import db from "@/lib/database/db";
import { parseError } from "@/lib/utils";

const kuesionerRoute = new Hono();

// kuesionerRoute.post("/petani", async (c) => {
//     try {
//         await db();
//         const body = await c.req.json();
//         const newEntry = new KuesionerPetani(body);
//         const savedEntry = await newEntry.save();
//         return c.json( savedEntry,  { status: 201 } );
//     } catch (err: any) {
//         const errMsg = parseError(err);
//         return c.json({ message: "Error", description: errMsg }, { status: 500 });
//     }
// });

kuesionerRoute.get("/petani", async (c) => {
    try {
        await db();
        const allEntries = await KuesionerPetani.find().sort({ createdAt: -1 }).lean();
        console.log("All Kuesioner Entries:", allEntries);
        return c.json( allEntries,  { status: 200 } );
    } catch (err: any) {
        const errMsg = parseError(err);
        return c.json({ message: "Error", description: errMsg }, { status: 500 });
    }
});

// kuesionerRoute.get("/petani/:id", async (c) => {
//     try {
//         await db();
//         const id = c.req.param("id");
//         const entry = await KuesionerPetani.findById(id).lean();
//         if (!entry) {
//             return c.json({ message: "Not Found", description: "Entry not found" }, { status: 404 });
//         }
//         return c.json(entry, { status: 200 });
//     } catch (err: any) {
//         const errMsg = parseError(err);
//         return c.json({ message: "Error", description: errMsg }, { status: 500 });
//     }
// });

// kuesionerRoute.put("/petani/:id", async (c) => {
//     try {
//         await db();
//         const id = c.req.param("id");
//         const body = await c.req.json();
//         const updatedEntry = await KuesionerPetani.findByIdAndUpdate(id, body, { new: true }).lean();
//         if (!updatedEntry) {
//             return c.json({ message: "Not Found", description: "Entry not found" }, { status: 404 });
//         }
//         return c.json(updatedEntry, { status: 200 });
//     } catch (err: any) {
//         const errMsg = parseError(err);
//         return c.json({ message: "Error", description: errMsg }, { status: 500 });
//     }
// });

// kuesionerRoute.delete("/petani/:id", async (c) => {
//     try {
//         await db();
//         const id = c.req.param("id");
//         const deletedEntry = await KuesionerPetani.findByIdAndDelete(id).lean();
//         if (!deletedEntry) {
//             return c.json({ message: "Not Found", description: "Entry not found" }, { status: 404 });
//         }
//         return c.json({ message: "Success", description: "Entry deleted successfully" }, { status: 200 });
//     } catch (err: any) {
//         const errMsg = parseError(err);
//         return c.json({ message: "Error", description: errMsg }, { status: 500 });
//     }
// });

const kuesionerManajemenRoute = new Hono();

kuesionerManajemenRoute.get("/manajemen", async (c) => {
    try {
        await db();
        const allEntries = await KuesionerManajemen.find().sort({ createdAt: -1 }).lean();
        console.log("All Kuesioner Manajemen Entries:", allEntries);
        return c.json( allEntries,  { status: 200 } );
    } catch (err: any) {
        const errMsg = parseError(err);
        return c.json({ message: "Error", description: errMsg }, { status: 500 });
    }
});


export { kuesionerRoute, kuesionerManajemenRoute };