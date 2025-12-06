import { Hono } from "hono";
import { handle } from "hono/vercel";
import bmkgApiRoute from "../_routes/dataset/bmkgApi.route";
import bmkgFetcherRoute from "../_routes/dataset/bmkgFetcher.route";
import bmkgSummaryRoute from "../_routes/model/bmkg-summary.route";
import bmkgDailyRoute from "../_routes/model/bmkg-daily.route";
import seedRoute from "../_routes/feature/seed.route";
import userRoute from "../_routes/user.route";
import exportRoute from "../_routes/feature/export-csv.route";
import datasetMetaRoute from "../_routes/feature/dataset-meta.route";
import forecastConfigRoute from "../_routes/feature/forecast-config";
import holtWinter from "../_routes/model/holt-winter.route";
import bmkgLiveRoute from "../_routes/dataset/newBmkg.route";
export const runtime = "nodejs";

const app = new Hono().basePath("/api/v1");

app.route("/bmkg-api", bmkgApiRoute); //api untuk ambil dari mongdo, tampilin d UI

//fetch : http://localhost:3000/api/v1/bmkg-fetch

app.route("/bmkg-live", bmkgLiveRoute);
app.route("/bmkg-fetch", bmkgFetcherRoute); //api untuk ambik dari bmkg api dan masukin ke mongodb
app.route("/bmkg-summary", bmkgSummaryRoute);
app.route("/bmkg-daily", bmkgDailyRoute);
app.route("/seeds", seedRoute);
app.route("/user", userRoute);
app.route("/export-csv", exportRoute);
app.route("/dataset-meta", datasetMetaRoute);
app.route("/forecast-config", forecastConfigRoute);
app.route("/hw", holtWinter);

export const GET = handle(app);
export const POST = handle(app);
export const PUT = handle(app);
export const DELETE = handle(app);
export const PATCH = handle(app);
