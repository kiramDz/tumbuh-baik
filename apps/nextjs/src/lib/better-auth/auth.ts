import { betterAuth } from "better-auth";
import { nextCookies } from "better-auth/next-js";
import { mongodbAdapter } from "better-auth/adapters/mongodb";
import { magicLink, admin } from "better-auth/plugins";
import { getClient } from "./db";
import { GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET } from "../env";

const dbClient = (await getClient()).db();

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
    // 🔥 KONFIGURASI COOKIE UNTUK PRODUCTION (cross-domain)
    cookie: {
      sameSite: "none", // memungkinkan cookie dikirim ke domain berbeda (ngrok)
      secure: true, // wajib jika sameSite=none, hanya melalui HTTPS
      httpOnly: true, // default, amankan dari XSS
      // path: "/",
      // domain: ".zonapetik.tech", // jika ingin cookie tersedia di semua subdomain custom domain
    },
  },
  plugins: [
    magicLink({
      sendMagicLink: async ({ email, url }) => {
        console.log(`Send login link to ${email}: ${url}`);
        // TODO: implement email sender
      },
    }),
    admin(),
    nextCookies(),
  ],
});
