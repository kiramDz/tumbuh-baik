import { Hono } from "hono";
import { handle } from "hono/vercel";
import bmkgRoute from "../_routes/dataset/bmkg.route";
import buoysRoute from "../_routes/dataset/buoys.route";
import bmkgApiRoute from "../_routes/dataset/bmkgApi.route";
import bmkgFetcherRoute from "../_routes/dataset/bmkgFetcher.route";
import bmkgSummaryRoute from "../_routes/model/bmkg-summary.route";
import bmkgDailyRoute from "../_routes/model/bmkg-daily.route";
import seedRoute from "../_routes/feature/seed.route";
import userRoute from "../_routes/user.route";
import exportRoute from "../_routes/feature/export-csv.route";
import datasetMetaRoute from "../_routes/feature/dataset-meta.route";

export const runtime = "nodejs";

const app = new Hono().basePath("/api/v1");

// dont forget to integrate to app

// TODO: Review and remove this route if unused
app.route("/bmkg", bmkgRoute);
app.route("/buoys", buoysRoute);

app.route("/bmkg-api", bmkgApiRoute); //api untuk ambil dari mongdo, tampilin d UI

//fetch : http://localhost:3000/api/v1/bmkg-fetch

app.route("/bmkg-fetch", bmkgFetcherRoute); //api untuk ambik dari bmkg api dan masukin ke mongodb
app.route("/bmkg-summary", bmkgSummaryRoute);
app.route("/bmkg-daily", bmkgDailyRoute);
app.route("/seeds", seedRoute);
app.route("/user", userRoute);
app.route("/export-csv", exportRoute);
app.route("/dataset-meta", datasetMetaRoute);

export const GET = handle(app);
export const POST = handle(app);
export const PUT = handle(app);
