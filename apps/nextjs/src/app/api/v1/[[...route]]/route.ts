import { Hono } from "hono";
import { handle } from "hono/vercel";
import bmkgApiRoute from "../_routes/dataset/bmkgApi.route";
import bmkgFetcherRoute from "../_routes/dataset/bmkgFetcher.route";
import bmkgSummaryRoute from "../_routes/model/bmkg-summary.route";
import bmkgDailyRoute from "../_routes/model/bmkg-daily.route";
import seedRoute from "../_routes/feature/seed.route";
import userRoute from "../_routes/user.route";
import exportRoute from "../_routes/feature/export-csv.route";
import datasetMetaRoute from "../_routes/dataset/meta/dataset-meta.route";
import forecastConfigRoute from "../_routes/feature/forecast-config";
// import bmkgOnlineRoute from "../_routes/dataset/bmkgOnline.route";
import holtWinter from "../_routes/model/holt-winter.route";
import nasaPowerRoute from "../_routes/dataset/nasa-power.route";
import lstmConfigRoute from "../_routes/feature/lstm-config";
import lstm from "../_routes/model/lstm.route";
import bmkgLiveRoute from "../_routes/dataset/newBmkg.route";
import { kuesionerRoute, kuesionerManajemenRoute, kuesionerPeriodeRoute } from "../_routes/dataset/kuesioner.route";
import { farmRoute } from "../_routes/farm/farm.route";
import decomposeLstmRoute from "../_routes/feature/decompose-lstm.route";
import historicalLstmRoute from "../_routes/feature/historical-lstm.route";

export const runtime = "nodejs";

const app = new Hono().basePath("/api/v1");

app.route("/bmkg-api", bmkgApiRoute); 



app.route("/bmkg-live", bmkgLiveRoute);
app.route("/bmkg-fetch", bmkgFetcherRoute); 
app.route("/bmkg-summary", bmkgSummaryRoute);
app.route("/bmkg-daily", bmkgDailyRoute);
app.route("/seeds", seedRoute);
app.route("/user", userRoute);
app.route("/export-csv", exportRoute);
app.route("/dataset-meta", datasetMetaRoute);
app.route("/forecast-config", forecastConfigRoute);
app.route("/hw", holtWinter);
app.route("/nasa-power", nasaPowerRoute);
// app.route("/dataset/bmkg", bmkgOnlineRoute);
app.route("/lstm-config", lstmConfigRoute);
app.route("/lstm", lstm);
app.route("/kuesioner", kuesionerRoute);
app.route("/kuesioner-manajemen", kuesionerManajemenRoute);
app.route("/kuesioner-periode", kuesionerPeriodeRoute);
app.route("/farm", farmRoute); 
app.route("/decompose-lstm", decomposeLstmRoute);
app.route("/historical-lstm", historicalLstmRoute); 

export const GET = handle(app);
export const POST = handle(app);
export const PUT = handle(app);
export const DELETE = handle(app); 
export const PATCH = handle(app);
