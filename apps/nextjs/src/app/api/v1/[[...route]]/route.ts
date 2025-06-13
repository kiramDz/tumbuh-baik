import { Hono } from "hono";
import { handle } from "hono/vercel";
import fileRoute from "../_routes/file.route";
import paddleRoute from "../_routes/paddle.route";
import visualizationRoute from "../_routes/visualization.route";
import bmkgRoute from "../_routes/bmkg.route";
import bmkgApiRoute from "../_routes/bmkgApi.route";
import bmkgFetcherRoute from "../_routes/bmkgFetcher.route";
import bmkgSummaryRoute from "../_routes/bmkg-summary.route";
import bmkgDailyRoute from "../_routes/bmkg-daily.route";
export const runtime = "nodejs";

const app = new Hono().basePath("/api/v1");

app.route("/files", fileRoute);

app.route("/webhook", paddleRoute);

app.route("/visualization", visualizationRoute);

// TODO: Review and remove this route if unused
app.route("/bmkg", bmkgRoute);
// TODO: Review and remove this route if unused
app.route("/bmkg-api", bmkgApiRoute);


//fetch : http://localhost:3000/api/v1/bmkg-fetch
app.route("/bmkg-fetch", bmkgFetcherRoute);
app.route("/bmkg-summary", bmkgSummaryRoute);
app.route("/bmkg-daily", bmkgDailyRoute);

export const GET = handle(app);
export const POST = handle(app);

