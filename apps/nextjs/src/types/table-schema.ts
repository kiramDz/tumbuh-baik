import { z } from "zod";

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

//  untuk bmkg-summary (tanam)
export const PlantSummaryDataSchema = z.object({
  month: z.string(), // YYYY-MM
  curah_hujan_total: z.number(),
  kelembapan_avg: z.number(),
  status: z.string(),
  timestamp: z.string(),
});

//show holt winter daily data in dashboard
export const HoltWinterDataSchema = z.object({
  _id: z.union([z.string(), z.object({ $oid: z.string() })]),
  timestamp: z.string().refine((val) => !isNaN(Date.parse(val)), {
    message: "Invalid timestamp",
  }),
  forecast_date: z.string().refine((val) => !isNaN(Date.parse(val)), {
    message: "Invalid forecast_date",
  }),
  parameters: z.object({
    RR: z.object({
      forecast_value: z.number(),
      model_metadata: z.object({
        alpha: z.number(),
        beta: z.number().nullable(),
        gamma: z.number(),
      }),
    }),
    RH_AVG: z.object({
      forecast_value: z.number(),
      model_metadata: z.object({
        alpha: z.number(),
        beta: z.number().nullable(),
        gamma: z.number(),
      }),
    }),
  }),
});

export const SeedSchema = z.object({
  _id: z.union([z.string(), z.object({ $oid: z.string() })]),
  name: z.string(),
  duration: z.number(),
  createdAt: z.string().refine((val) => !isNaN(Date.parse(val)), {
    message: "Invalid createdAt date",
  }),
  updatedAt: z.string().refine((val) => !isNaN(Date.parse(val)), {
    message: "Invalid updatedAt date",
  }),
});

export const UserSchema = z.object({
  _id: z.union([z.string(), z.object({ $oid: z.string() })]),
  name: z.string(),
  email: z.string().email(),
  emailVerified: z.boolean(),
  image: z.string().optional(),
  role: z.enum(["user", "admin"]),
  createdAt: z.string(),
  updatedAt: z.string(),
});

export type HoltWinterDataType = z.infer<typeof HoltWinterDataSchema>;
export type BMKGDataItem = z.infer<typeof BMKGDataItemSchema>;
export type BMKGApiData = z.infer<typeof BMKGApi>;
export type SeedType = z.infer<typeof SeedSchema>;
export type UserType = z.infer<typeof UserSchema>;
export type PlantSummaryData = z.infer<typeof PlantSummaryDataSchema>;
