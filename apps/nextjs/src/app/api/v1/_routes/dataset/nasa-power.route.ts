import { Hono } from "hono";
import axios from "axios";
import db from "@/lib/database/db";
import { DatasetMeta } from "@/lib/database/schema/feature/dataset-meta.model";
import mongoose from "mongoose";
import { parseError } from "@/lib/utils";
import { constructFromSymbol } from "date-fns/constants";

const nasaPowerRoute = new Hono();

// GET - Fetch NASA Power data based on query parameters
nasaPowerRoute.get("/", async (c) => {
  try {
    const start = c.req.query("start");
    const end = c.req.query("end");
    const latitude = Number(c.req.query("latitude"));
    const longitude = Number(c.req.query("longitude"));
    const parameters = c.req.query("parameters")?.split(",") || [];
    const community = c.req.query("community") || "ag";

    console.log("NASA POWER API Request:", {
      start,
      end,
      latitude,
      longitude,
      parameters,
      community,
    });

    if (!start || !end || !latitude || !longitude || parameters.length === 0) {
      return c.json(
        {
          message:
            "Missing required parameters: start, end, latitude, longitude, parameters",
        },
        400
      );
    }

    // Validate date format (YYYYMMDD)
    const dateRegex = /^\d{8}$/;
    if (!dateRegex.test(start) || !dateRegex.test(end)) {
      return c.json(
        {
          message:
            "Invalid date format. Use YYYYMMDD format for start and end dates.",
        },
        400
      );
    }

    // Validate latitude and longitude
    if (
      latitude < -90 ||
      latitude > 90 ||
      longitude < -180 ||
      longitude > 180
    ) {
      return c.json(
        {
          message: "Invalid latitude or longitude values.",
        },
        400
      );
    }

    const baseUrl = "https://power.larc.nasa.gov/api/temporal/daily/point";

    // Build request parameters
    const requestParams = {
      start,
      end,
      latitude,
      longitude,
      parameters: parameters.join(","),
      community,
      format: "JSON",
      user: "tumbuhbaik", // Use dash instead of underscore
      header: "true",
      "time-standard": "LST",
    };

    console.log("NASA POWER API Request parameters:", requestParams);

    const response = await axios.get(baseUrl, {
      params: requestParams,
      timeout: 60000, // Increase timeout to 60 seconds
    });

    console.log(
      "NASA POWER API Response received with status:",
      response.status
    );

    return c.json(
      {
        message: "NASA Power data fetched successfully",
        data: response.data,
      },
      200
    );
  } catch (error: any) {
    console.error("Error fetching NASA Power data:", error);

    // Enhanced error logging
    if (error.response) {
      console.error("NASA API Response Error:", {
        status: error.response.status,
        data: error.response.data,
        headers: error.response.headers,
      });
    }

    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

nasaPowerRoute.post("/save", async (c) => {
  try {
    await db();
    const body = await c.req.json();

    const {
      name,
      collectionName = `nasa_power_${Date.now()}`,
      description = "",
      status = "processed",
      nasaParams,
    } = body;

    if (!name || !nasaParams) {
      return c.json(
        { message: "Missing required fields: name, nasaParams" },
        400
      );
    }

    // Add community from nasaParams or use default
    const community = nasaParams.community || "ag";

    console.log("NASA POWER API Request for saving:", {
      start: nasaParams.start,
      end: nasaParams.end,
      latitude: nasaParams.latitude,
      longitude: nasaParams.longitude,
      parameters: nasaParams.parameters,
      community,
    });

    // Fetch data from NASA POWER API
    const baseUrl = "https://power.larc.nasa.gov/api/temporal/daily/point";
    const nasaResponse = await axios.get(baseUrl, {
      params: {
        start: nasaParams.start,
        end: nasaParams.end,
        latitude: nasaParams.latitude,
        longitude: nasaParams.longitude,
        parameters: nasaParams.parameters.join(","),
        community: community, // Use lowercase and from params
        format: "JSON",
        user: "tumbuhbaik", // Use consistent user identifier (dash instead of underscore)
        header: "true",
        "time-standard": "LST",
      },
      timeout: 60000, // Increased timeout for larger date ranges
    });

    console.log(
      "NASA POWER API Response received with status:",
      nasaResponse.status
    );

    if (
      !nasaResponse.data ||
      !nasaResponse.data.properties ||
      !nasaResponse.data.properties.parameter
    ) {
      return c.json(
        { message: "Invalid data received from NASA POWER API" },
        400
      );
    }

    // Transform NASA POWER data into array of objects
    const parameters = nasaParams.parameters;
    const properties = nasaResponse.data.properties;
    const paramData = properties.parameter;
    const timespan = Object.keys(paramData[parameters[0]]);

    const records = timespan.map((date) => {
      const record: Record<string, any> = { Date: date };

      parameters.forEach((param: string) => {
        if (paramData[param] && paramData[param][date] !== undefined) {
          record[param] = paramData[param][date];
        }
      });
      return record;
    });

    // Log the number of records being saved
    console.log(
      `Transforming ${records.length} records to save to ${collectionName}`
    );

    // Save data to dynamic collection
    const dynamicModel = mongoose.model(
      collectionName,
      new mongoose.Schema({}, { strict: false }),
      collectionName
    );

    await dynamicModel.insertMany(records);
    console.log(
      `Successfully inserted ${records.length} records into ${collectionName}`
    );

    // Calculate metadata
    const fileSize = Buffer.byteLength(JSON.stringify(records), "utf8");
    const totalRecords = records.length;
    const columns = records.length > 0 ? Object.keys(records[0]) : [];

    // Create dataset metadata entry
    const newDatasetMeta = await DatasetMeta.create({
      name: name.trim(),
      source: body.source || "Data NASA (https://power.larc.nasa.gov/)",
      filename: `${name}.json`,
      collectionName,
      fileType: "json",
      status,
      description,
      fileSize,
      totalRecords,
      columns,
      metadata: {
        nasaParams,
        header: properties.header,
      },
    });

    console.log("Dataset metadata created with ID:", newDatasetMeta._id);

    return c.json(
      {
        message: "Dataset metadata saved successfully",
        data: {
          meta: newDatasetMeta,
          records: records.slice(0, 5),
        },
      },
      201
    );
  } catch (error) {
    console.error("Error saving NASA Power dataset:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});
export default nasaPowerRoute;
