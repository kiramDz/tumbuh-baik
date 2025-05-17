import { z } from "zod";

export const paymentSchmea = z.object({
  id: z.number(),
  amount: z.number(),
  status: z.enum(["backlog", "todo", "in progress", "done", "canceled"]),
  email: z.string(),
  fullName: z.string(),
});

export const statusSchmea = z.object({
  id: z.number(),
  source: z.string(),
  status: z.enum(["backlog", "todo", "in progress", "done", "canceled"]),
  record: z.number(),
  date: z.string().refine((val) => !isNaN(Date.parse(val)), {
    message: "Invalid date string",
  }),
});

export const BmkgSchema = z.object({
  _id: z.string(),
  timestamp: z.string().refine((val) => !isNaN(Date.parse(val)), {
    message: "Invalid date string",
  }),
  city: z.string(),
  temperature: z.number(),
  humidity: z.number(),
  windSpeed: z.number(),
  lat: z.number(),
  lon: z.number(),
  pressure: z.number(),
});

export const BMKGDataItemSchema = z.object({
  local_datetime: z.string(),
  t: z.number(), 
  hu: z.number(), 
  weather_desc: z.string(), 
  ws: z.number(), 
  wd: z.string(), 
  tcc: z.number(), 
  vs_text: z.string(), 
});

export const BMKGApi = z.object({
  kode_gampong: z.string(),
  nama_gampong: z.string(),
  tanggal_data: z.string(), // format: YYYY-MM-DD
  analysis_date: z.date(),
  data: z.array(BMKGDataItemSchema),
});

export type BmkgDataType = z.infer<typeof BmkgSchema>
export type PaymentType = z.infer<typeof paymentSchmea>;
export type Statustype = z.infer<typeof statusSchmea>
export type BMKGDataItem = z.infer<typeof BMKGDataItemSchema>;
export type BMKGApiData = z.infer<typeof BMKGApi>;
