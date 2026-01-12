"use server";

import { revalidatePath } from "next/cache";
import { z } from "zod";
import { adminProcedure } from "@/lib/safe-actions";
import db from "@/lib/database/db";
import { User } from "@/lib/database/schema/users.model";

export const updateUser = adminProcedure
  .createServerAction()
  .input(
    z.object({
      id: z.string(),
      role: z.enum(["user", "admin"]),
    })
  )
  .handler(async ({ input: { id, role } }) => {
    await db(); // koneksi ke database

    const user = await User.findByIdAndUpdate(
      id,
      { role },
      { new: true }
    );

    if (!user) throw new Error("User not found");

    revalidatePath("/admin/users");
    return user;
  });
