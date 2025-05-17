import { Hono } from "hono";
import { handle } from "hono/vercel";
import fileRoute from "../_routes/file.route";
import paddleRoute from "../_routes/paddle.route";
import visualizationRoute from "../_routes/visualization.route";
import bmkgRoute from "../_routes/bmkg.route";
import bmkgApiRoute from "../_routes/bmkgApi.route";
import bmkgFetcherRoute from "../_routes/bmkgFetcher.route";

export const runtime = "nodejs";

const app = new Hono().basePath("/api/v1");

app.route("/files", fileRoute);

app.route("/webhook", paddleRoute);

app.route("/visualization", visualizationRoute);

app.route("/bmkg", bmkgRoute);
app.route("/bmkg-api", bmkgApiRoute);
//fetch : http://localhost:3000/api/v1/bmkg-fetch
app.route("/bmkg-fetch", bmkgFetcherRoute);

export const GET = handle(app);
export const POST = handle(app);

//https://http://localhost:3000/api/v1/visualization/daily-weather
