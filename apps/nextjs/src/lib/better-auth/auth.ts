import { betterAuth } from "better-auth";
import { nextCookies } from "better-auth/next-js";
import { mongodbAdapter } from "better-auth/adapters/mongodb";
import { createAuthMiddleware, magicLink, admin } from "better-auth/plugins";
import client from "./db";
import { GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET } from "../env";
import db from "../database/db";
import { Subscription } from "../database/schema/subscription.model";
import { ObjectId } from "mongodb";

const dbClient = client.db();

export const auth = betterAuth({
  database: mongodbAdapter(dbClient),
  socialProviders: {
    google: {
      clientId: GOOGLE_CLIENT_ID,
      clientSecret: GOOGLE_CLIENT_SECRET,
    },
  },
  emailAndPassword: {
    enabled: true,
  },
  session: {
    cookieCache: {
      enabled: true,
    },
  },
  hooks: {
    after: createAuthMiddleware(async ({ context }) => {
      const newSession = context.newSession;
      const user = newSession?.user;

      if (newSession && user) {
        try {
          await db();

          const isSubAvil = await Subscription.findOne({ subscriber: user.id });
          if (!isSubAvil) {
            const subs = await Subscription.create({
              subscriber: user.id,
              status: "activated",
            });

            const userCollection = dbClient.collection("user");
            await userCollection.updateOne({ _id: new ObjectId(user.id) }, { $set: { subscription: subs._id } });
          }
        } catch (error) {
          console.error("Error setting subscription:", error);
          throw context.redirect("/sign-in");
        }
      }
    }),
  },
  plugins: [
    magicLink({
      sendMagicLink: async ({ email, url }) => {
        console.log(`Send login link to ${email}: ${url}`);
        // TODO: implement email sender
      },
    }),
    // oneTimeToken(), // Bisa kamu nonaktifkan sementara
    admin(), // Aktifkan sistem admin role
    nextCookies(),
  ],
});
