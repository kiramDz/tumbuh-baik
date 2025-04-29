import { z } from "zod";

// We're keeping a simple non-relational schema here.
// IRL, you will have a schema for your data models.
export const paymentSchmea = z.object({
  id: z.number(),
  amount: z.number(),
  status: z.enum(["backlog", "todo", "in progress", "done", "canceled"]),
  email: z.string(),
  fullName: z.string(),
});

export const BmkgSchema = z.object({
  _id: z.string(),
  timestamp: z.string().refine((val) => !isNaN(Date.parse(val)), {
    message: "Invalid date string",
  }),
  // Tambahkan field sesuai data BMKG Anda
  city: z.string(),
  temperature: z.number(),
  humidity: z.number(),
  windSpeed: z.number(),
  lat: z.number(),
  lon: z.number(),
  pressure: z.number(),
  // Tambahkan field lain sesuai kebutuhan
});

export type BmkgDataType = z.infer<typeof BmkgSchema>
export type PaymentType = z.infer<typeof paymentSchmea>;
