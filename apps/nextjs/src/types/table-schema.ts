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
  tanggal_data: z.string(),
  analysis_date: z.date(),
  data: z.array(BMKGDataItemSchema),
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

export const RecycleBinSchema = z.object({
  _id: z.union([z.string(), z.object({ $oid: z.string() })]),
  name: z.string(),
  source: z.string(),
  collectionName: z.string(),
  description: z.string().optional(),
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

export type BMKGDataItem = z.infer<typeof BMKGDataItemSchema>;
export type BMKGApiData = z.infer<typeof BMKGApi>;
export type SeedType = z.infer<typeof SeedSchema>;
export type UserType = z.infer<typeof UserSchema>;
export type RecycleBinType = z.infer<typeof RecycleBinSchema>;
