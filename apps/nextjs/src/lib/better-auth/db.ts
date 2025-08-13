import { MongoClient, ServerApiVersion } from "mongodb";
import { MONGODB_URI } from "../env";

if (!MONGODB_URI) {
  throw new Error('Invalid/Missing environment variable: "MONGODB_URI"');
}

const options = {
  serverApi: {
    version: ServerApiVersion.v1,
    strict: true,
    deprecationErrors: true,
  },
};

let client: MongoClient | null = null;

export async function getClient() {
  if (!client) {
    client = new MongoClient(MONGODB_URI, options);
    await client.connect();
  }
  return client;
}
