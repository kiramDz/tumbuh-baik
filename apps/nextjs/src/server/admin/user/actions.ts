"use server";

import { revalidatePath } from "next/cache";
import { z } from "zod";
import { adminProcedure } from "@/lib/safe-actions";
import { UserSchema } from "@/types/table-schema";
import db from "@/lib/database/db";
import { User } from "@/lib/database/schema/users.model";

export const updateUser = adminProcedure
  .createServerAction()
  .input(UserSchema.extend({ id: z.string() }))
  .handler(async ({ input: { id, ...input } }) => {
    // Pastikan koneksi MongoDB aktif
    await db(); // Panggil fungsi koneksi Anda

    const user = await User.findByIdAndUpdate(
      id,
      input,
      { new: true } // Return data terbaru
    );

    if (!user) throw new Error("User not found");

    revalidatePath("/admin/users");
    return user;
  });
