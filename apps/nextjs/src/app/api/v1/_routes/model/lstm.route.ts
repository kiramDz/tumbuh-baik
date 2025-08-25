import { Hono } from "hono";
import db from "@/lib/database/db";
import { LSTMDaily } from "@/lib/database/schema/model/lstm.model";
import { parseError } from "@/lib/utils";

const lstm = new Hono();

lstm.get("/daily", async (c) => {
    try {
        await db();
        const page = Number(c.req.query("page")) || 1;
        const pageSize = Number(c.req.query("pageSize")) || 10;
        const totalData = await LSTMDaily.countDocuments();

        console.log("Collection name:", LSTMDaily.collection.name);
        console.log("Total documents:", totalData);

        const data = await LSTMDaily.find()
        .skip((page - 1) * pageSize)
        .limit(pageSize)
        .sort({ forecast_date: -1 })
        .lean();

        console.log("Retrieved documents:", data.length);

        return c.json({
            message: 'Success',
            data: {
                items: data,
                total: totalData,
                currentPage: page,
                totalPages: Math.ceil(totalData / pageSize),
                pageSize,
            },
        });
    } catch (error) {
        const err = parseError(error);
        return c.json({ message: err.message }, 500);
    }
});

lstm.get("/daily/grid", async (c) => {
    try {
        await db();
        const data = await LSTMDaily.find()
        .sort({ forecast_date: -1 })
        .lean();
    
    console.log("Retrieved all documents:", data.length);

    return c.json({
        message: "Success",
        data: data,
    });
} catch (error) {
    const err = parseError(error);
    return c.json({ message: err.message }, 500);
}
});

export default lstm;