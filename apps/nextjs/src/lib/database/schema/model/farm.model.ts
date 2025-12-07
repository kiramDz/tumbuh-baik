import mongoose, { Schema, Document } from "mongoose";

export interface IFarm extends Document {
  name: string;
  location: string;
  luasTanam: number;
  hasilPanen: number;
  biaya: number;
  keuntungan: number;
  season: string;
  plantingDate: Date;
  harvestDate: Date;
  userId: mongoose.Types.ObjectId;
  createdAt: Date;
  updatedAt: Date;
}

const FarmSchema = new Schema(
  {
    name: { type: String, required: true },
    location: { type: String, required: true },
    luasTanam: { type: Number, required: true },
    hasilPanen: { type: Number, required: true },
    biaya: { type: Number, required: true },
    keuntungan: { type: Number, required: true },
    season: { type: String, required: true },
    plantingDate: { type: Date, required: true },
    harvestDate: { type: Date, required: true },
    userId: { 
      type: Schema.Types.ObjectId, 
      ref: "User",
      required: true, 
      index: true  // Cukup satu index di sini
    },
  },
  { 
    collection: "farms", 
    timestamps: true 
  }
);

export const Farm = mongoose.models.Farm || mongoose.model<IFarm>("Farm", FarmSchema, "farms");