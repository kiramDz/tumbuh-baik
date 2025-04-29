import mongoose, { Schema, Document, Model } from "mongoose";

export interface IFile {
  _id: string;
  pinataId: string; 
  name: string; 
  cid: string; 
  size: number; 
  mimeType: string; 
  userInfo: { id: mongoose.Types.ObjectId | string; name: string };
  groupId?: string; 
  sharedWith: {
    email: string;
    permissions: ("file:read" | "file:update" | "file:delete")[]; 
  }[];
  category: { type: string; enum: ["bmkg-station", "citra-satelit", "temperatur-laut", "daily-weather"]; required: true };
  createdAt: Date;
  updatedAt: Date;
}

// Extend Mongoose's Document and override _id type
interface FileModel extends Omit<IFile, "_id">, Document {
  _id: string;
}

const fileSchema: Schema<FileModel> = new Schema(
  {
    pinataId: { type: String, required: true },
    name: { type: String, required: true },
    cid: { type: String, required: true },
    size: { type: Number, required: true },
    mimeType: { type: String, required: true },
    category: { type: String, enum: ["BMKG", "Citra Satelit", "Buoys", "Daily Weather"], required: true },
    userInfo: {
      id: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
      name: { type: String, required: true },
    },
    groupId: { type: String },
    sharedWith: [
      {
        email: { type: String, required: true },
        permissions: [
          {
            type: String,
            enum: ["file:read", "file:update", "file:delete"],
            required: true,
          },
        ],
      },
    ],
  },
  { timestamps: true }
);




export const File: Model<FileModel> = mongoose.models.File || mongoose.model<FileModel>("File", fileSchema);
