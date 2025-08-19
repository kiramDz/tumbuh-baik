import { MongoClient, ServerApiVersion } from "mongodb";
import { MONGODB_URI } from "../env";

if (!MONGODB_URI) {
  throw new Error('Invalid/Missing environment variable: "MONGODB_URI"');
}

// Validate MongoDB URI format
if (!MONGODB_URI.startsWith("mongodb://") && !MONGODB_URI.startsWith("mongodb+srv://")) {
  throw new Error('Invalid MongoDB URI format. Must start with "mongodb://" or "mongodb+srv://"');
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
    try {
      client = new MongoClient(MONGODB_URI, options);
      await client.connect();
    } catch (error) {
      console.error("Failed to connect to MongoDB:", error);
      throw error;
    }
  }
  return client;
}
