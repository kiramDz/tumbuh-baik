// import { createAuthClient } from "better-auth/react";
// import { BASE_URL } from "../env";

// export const authClient = createAuthClient({
//   baseURL: BASE_URL,
// });

// export const { useSession, signOut, signUp, signIn } = authClient;

import { createAuthClient } from "better-auth/react";
import { adminClient, magicLinkClient } from "better-auth/client/plugins";
import { BASE_URL } from "../env";

export const { useSession, signOut, signUp, signIn, admin } = createAuthClient({
  baseURL: BASE_URL,
  plugins: [adminClient(), magicLinkClient()],
});
