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
  _id: z.union([z.string(), z.object({ $oid: z.string() })]), // tergantung dari loader Mongo
  Date: z.string().refine((val) => !isNaN(Date.parse(val)), {
    message: "Invalid date string",
  }),
  Year: z.number(),
  Month: z.string(),
  Day: z.number(),
  TN: z.number(), // Suhu minimum
  TX: z.number(), // Suhu maksimum
  TAVG: z.number(), // Suhu rata-rata
  RH_AVG: z.number(), // Kelembapan rata-rata
  RR: z.number(), // Curah hujan
  SS: z.number(), // Lama penyinaran matahari
  FF_X: z.union([z.string(), z.number()]), // Bisa kosong string atau numeric
  DDD_X: z.number(), // Arah angin maksimum
  FF_AVG: z.number(), // Kecepatan angin rata-rata
  DDD_CAR: z.string(), // Arah angin utama
  Season: z.string(), // Musim
  is_RR_missing: z.number(), // Penanda apakah data RR missing
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

export type BmkgDataType = z.infer<typeof BmkgSchema>;
export type HoltWinterDataType = z.infer<typeof HoltWinterDataSchema>;
export type PaymentType = z.infer<typeof paymentSchmea>;
export type Statustype = z.infer<typeof statusSchmea>;
export type BMKGDataItem = z.infer<typeof BMKGDataItemSchema>;
export type BMKGApiData = z.infer<typeof BMKGApi>;
export type SeedType = z.infer<typeof SeedSchema>;
