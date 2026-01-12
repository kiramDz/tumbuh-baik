import { Hono } from "hono";
import axios from "axios";
import db from "@/lib/database/db";
import { DatasetMeta } from "@/lib/database/schema/feature/dataset-meta.model";
import mongoose from "mongoose";
import { parseError } from "@/lib/utils";
import { constructFromSymbol } from "date-fns/constants";

interface NasaPowerRecord {
  Date?: Date;
  Year?: number;
  month?: number;
  day?: number;
  [key: string]: any;
}
interface DatasetMetaDocument {
  _id: mongoose.Types.ObjectId;
  name: string;
  source: string;
  filename: string;
  collectionName: string;
  fileSize: number;
  totalRecords: number;
  fileType: string;
  status: string;
  columns: string[];
  description?: string;
  uploadDate?: Date;
  errorMessage?: string;
  isAPI: boolean;
  lastUpdated?: Date;
  apiConfig?: {
    type: string;
    params: {
      start: string;
      end: string;
      latitude: number;
      longitude: number;
      parameters: string[];
      community?: string;
    };
  };
  metadata?: any;
  __v: number;
}

const BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point";

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

    const response = await axios.get(BASE_URL, {
      params: requestParams,
      timeout: 30000, // Increase timeout to 60 seconds
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

    const { name, description = "", status = "processed", nasaParams } = body;

    if (!name || !nasaParams) {
      return c.json(
        { message: "Missing required fields: name, nasaParams" },
        400
      );
    }

    // Log the original name
    console.log("Original name:", name);

    // Generate collectionName from name - PRESERVING SPACES, just removing invalid chars
    const collectionName = name
      .trim()
      .replace(/[^a-zA-Z0-9\s]/g, "") // Keep spaces, only remove special chars
      .replace(/\s+/g, " "); // Normalize multiple spaces to single space

    // Log the processed collection name
    console.log("Generated collection name:", collectionName);

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
    const nasaResponse = await axios.get(BASE_URL, {
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
      timeout: 30000, // Increased timeout for larger date ranges
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

    // Convert YYYYMMDDD string date to MongoDB Date object
    const records = timespan.map((date) => {
      const year = parseInt(date.substring(0, 4));
      const month = parseInt(date.substring(4, 6)) - 1; // JS months are 0-indexed
      const day = parseInt(date.substring(6, 8));

      const record: Record<string, any> = {
        Date: new Date(Date.UTC(year, month, day)),
        Year: year,
        month: month + 1,
        day: day,
      };
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
    // const dynamicModel = mongoose.model(
    //   collectionName,
    //   new mongoose.Schema({}, { strict: false }),
    //   collectionName
    // );
    const dynamicModel =
      mongoose.models[collectionName] ||
      mongoose.model(
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
      status: "raw",
      description,
      fileSize,
      totalRecords,
      columns,
      isAPI: true,
      lastUpdated: new Date(),
      apiConfig: {
        type: "nasa-power",
        params: nasaParams,
      }, // Store API configuration for future updates
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

// GET - Fetch all NASA POWER datasets that can be updated
nasaPowerRoute.get("/refreshable", async (c) => {
  try {
    await db();
    const datasets = await DatasetMeta.find({
      isAPI: true,
      "apiConfig.type": "nasa-power",
    }).lean();

    // create current date refresh status
    const today = new Date();
    const todayFormatted = today.toISOString().slice(0, 10).replace(/-/g, "");

    // Process each dataset to check its latest record date
    const datasetsWithRefreshInfo = await Promise.all(
      datasets.map(async (dataset) => {
        // Initialize variables => refresh status
        let lastRecordDate = null;
        let canRefresh = false;
        let startDate = null;
        let daysSinceLastRecord = null;

        try {
          // Get the dynamic model for this dataset
          const dynamicModel =
            mongoose.models[dataset.collectionName] ||
            mongoose.model(
              dataset.collectionName,
              new mongoose.Schema({}, { strict: false }),
              dataset.collectionName
            );
          // Find the most recent record in collectionName
          const latestRecord = (await dynamicModel
            .findOne({})
            .sort({ Date: -1 })
            .lean()) as unknown as NasaPowerRecord;
          if (latestRecord && latestRecord.Date) {
            // If we found a record with a date, use that as the last record date
            lastRecordDate = new Date(latestRecord.Date);

            // Create start date for next day after the latest record
            const nextDate = new Date(lastRecordDate);
            nextDate.setDate(nextDate.getDate() + 1);
            startDate = nextDate.toISOString().slice(0, 10).replace(/-/g, "");

            // Calculate days since the last record
            daysSinceLastRecord = Math.floor(
              (today.getTime() - lastRecordDate.getTime()) /
                (1000 * 60 * 60 * 24)
            );

            // Can refresh if the last record is at least 1 day old
            canRefresh = daysSinceLastRecord >= 1;
          } else {
            // No records found, fallback to metadata dates
            if (dataset.lastUpdated) {
              // Use lastUpdated from metadata
              lastRecordDate = new Date(dataset.lastUpdated);
              const nextDate = new Date(lastRecordDate);
              nextDate.setDate(nextDate.getDate() + 1);
              startDate = nextDate.toISOString().slice(0, 10).replace(/-/g, "");
              daysSinceLastRecord = Math.floor(
                (today.getTime() - lastRecordDate.getTime()) /
                  (1000 * 60 * 60 * 24)
              );
              canRefresh = daysSinceLastRecord >= 1;
            } else if (dataset.apiConfig?.params?.end) {
              // Use end date from API params as fallback
              const endDate = dataset.apiConfig.params.end;
              const year = parseInt(endDate.substring(0, 4));
              const month = parseInt(endDate.substring(4, 6)) - 1;
              const day = parseInt(endDate.substring(6, 8));
              lastRecordDate = new Date(year, month, day);

              const nextDate = new Date(lastRecordDate);
              nextDate.setDate(nextDate.getDate() + 1);
              startDate = nextDate.toISOString().slice(0, 10).replace(/-/g, "");
              daysSinceLastRecord = Math.floor(
                (today.getTime() - lastRecordDate.getTime()) /
                  (1000 * 60 * 60 * 24)
              );
              canRefresh = daysSinceLastRecord >= 1;
            }
          }
        } catch (error) {
          console.error(
            `Error while checking records for ${dataset.name}:`,
            error
          );
          // Use metadaata dates as fallback
          if (dataset.lastUpdated) {
            lastRecordDate = new Date(dataset.lastUpdated);
            const nextDate = new Date(lastRecordDate);
            nextDate.setDate(nextDate.getDate() + 1);
            startDate = nextDate.toISOString().slice(0, 10).replace(/-/g, "");
            daysSinceLastRecord = Math.floor(
              (today.getTime() - lastRecordDate.getTime()) /
                (1000 * 60 * 60 * 24)
            );
            canRefresh = daysSinceLastRecord >= 1;
          } else if (dataset.apiConfig?.params?.end) {
            const endDate = dataset.apiConfig.params.end;
            const year = parseInt(endDate.substring(0, 4));
            const month = parseInt(endDate.substring(4, 6)) - 1;
            const day = parseInt(endDate.substring(6, 8));
            lastRecordDate = new Date(year, month, day);

            const nextDate = new Date(lastRecordDate);
            nextDate.setDate(nextDate.getDate() + 1);
            startDate = nextDate.toISOString().slice(0, 10).replace(/-/g, "");
            daysSinceLastRecord = Math.floor(
              (today.getTime() - lastRecordDate.getTime()) /
                (1000 * 60 * 60 * 24)
            );
            canRefresh = daysSinceLastRecord >= 1;
          }
        }
        return {
          ...dataset,
          refreshInfo: {
            canRefresh,
            lastRecordDate: lastRecordDate?.toISOString(),
            startDate, // When to start fetching new data
            endDate: todayFormatted, // Today's date (to fetch up to)
            daysSinceLastRecord, // How many days since the last record
          },
        };
      })
    );
    return c.json(
      {
        message: "Retrieved NASA POWER datasets that can be refreshed",
        data: datasetsWithRefreshInfo,
      },
      200
    );
  } catch (error) {
    console.error("Error fetching refreshable NASA POWER datasets:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

nasaPowerRoute.post("/refresh/:id", async (c) => {
  const requestedId = c.req.param("id");
  try {
    await db();

    // Find dastaset by Id
    const dataset = (await DatasetMeta.findById(
      requestedId
    ).lean()) as DatasetMetaDocument;

    const today = new Date();
    const todayFormatted = today.toISOString().slice(0, 10).replace(/-/g, "");

    try {
      const dynamicModel =
        mongoose.models[dataset.collectionName] ||
        mongoose.model(
          dataset.collectionName,
          new mongoose.Schema({}, { strict: false }),
          dataset.collectionName
        );
      const latestRecord = (await dynamicModel
        .findOne({})
        .sort({ Date: -1 })
        .lean()) as unknown as NasaPowerRecord;
      let startDateFormatted;
      if (latestRecord && latestRecord.Date) {
        const latestDate = new Date(latestRecord.Date);
        latestDate.setDate(latestDate.getDate() + 1);
        startDateFormatted = latestDate
          .toISOString()
          .slice(0, 10)
          .replace(/-/g, "");
        console.log(`Latest record date found: ${latestRecord.Date}`);
        console.log(`Starting refresh from: ${startDateFormatted}`);
      }
      // If no records found, fallback to metadata
      else if (dataset.lastUpdated) {
        const lastUpdated = new Date(dataset.lastUpdated);
        lastUpdated.setDate(lastUpdated.getDate() + 1);
        startDateFormatted = lastUpdated
          .toISOString()
          .slice(0, 10)
          .replace(/-/g, "");
        console.log(
          `No records found, using lastUpdated from metadata: ${dataset.lastUpdated}`
        );
        console.log(`Starting refresh from: ${startDateFormatted}`);
      }
      // Final fallback to original API end date
      else if (dataset.apiConfig?.params?.end) {
        const endDate = dataset.apiConfig.params.end;
        const year = parseInt(endDate.substring(0, 4));
        const month = parseInt(endDate.substring(4, 6)) - 1; // JS months are 0-based
        const day = parseInt(endDate.substring(6, 8)) + 1;
        const nextDate = new Date(year, month, day);
        startDateFormatted = nextDate
          .toISOString()
          .slice(0, 10)
          .replace(/-/g, "");

        console.log(
          `No records or lastUpdated found, using original end date: ${dataset.apiConfig?.params?.end}`
        );
        console.log(`Starting refresh from: ${startDateFormatted}`);
      } else {
        return c.json(
          { message: "Cannot determine start date for refresh" },
          400
        );
      }
      // Check if dataset is already up to date
      if (startDateFormatted > todayFormatted) {
        return c.json(
          {
            message: "dataset is up to date",
            data: {
              name: dataset.name,
              lastUpdated: dataset.lastUpdated,
              status: dataset.status,
            },
          },
          200
        );
      }
      console.log(
        `Refreshing dataset '${dataset.name}' from ${startDateFormatted} to ${todayFormatted}`
      );

      const response = await axios.get(BASE_URL, {
        params: {
          start: startDateFormatted,
          end: todayFormatted,
          latitude: dataset.apiConfig?.params.latitude,
          longitude: dataset.apiConfig?.params.longitude,
          parameters: dataset.apiConfig?.params.parameters.join(","),
          community: dataset.apiConfig?.params.community || "ag",
          format: "JSON",
          user: "tumbuhbaik", // Use dash instead of underscore
          header: "true",
          "time-standard": "LST",
        },
        timeout: 30000,
      });
      if (
        !response.data ||
        !response.data.properties ||
        !response.data.properties.parameter
      ) {
        return c.json(
          { message: "Invalid data received from NASA POWER API" },
          400
        );
      }
      if (!dataset.apiConfig?.params?.parameters) {
        return c.json(
          { message: "Dataset API configuration is missing or incomplete" },
          400
        );
      }
      // Transform NASA POWER data into array of objects (same as original data)
      const parameters = dataset.apiConfig.params.parameters;
      const properties = response.data.properties;
      const paramData = properties.parameter;
      const timespan = Object.keys(paramData[parameters[0]]);

      // If no data available
      if (timespan.length === 0) {
        // Don't use "no-new-data" as status
        return c.json(
          {
            message: "No new data available for this dataset",
            data: {
              name: dataset.name,
              lastUpdated: dataset.lastUpdated,
              status: dataset.status, // Use the actual dataset status
              refreshResult: "no-new-data", // Add this property to indicate refresh outcome
            },
          },
          200
        );
      }
      // Convert date strings to records (Date Objects MONGODB)
      const newRecords = timespan.map((date) => {
        const year = parseInt(date.substring(0, 4));
        const month = parseInt(date.substring(4, 6)) - 1;
        const day = parseInt(date.substring(6, 8));

        const record: Record<string, any> = {
          Date: new Date(Date.UTC(year, month, day)),
          Year: year,
          month: month + 1,
          day: day,
        };
        parameters.forEach((param: string) => {
          if (paramData[param] && paramData[param][date] !== undefined) {
            record[param] = paramData[param][date];
          }
        });
        return record;
      });
      console.log(
        `Found ${newRecords.length} new records added to dataset '${dataset.name}'`
      );
      // Await the new newRecords
      await dynamicModel.insertMany(newRecords);

      const totalRecords = await dynamicModel.countDocuments();
      const updatedDataset = await DatasetMeta.findByIdAndUpdate(
        requestedId,
        {
          $set: {
            lastUpdated: today,
            totalRecords,
            "apiConfig.params.end": todayFormatted,
            // status: "latest",
            status:
              dataset.status === "preprocessed" ||
              dataset.status === "validated"
                ? "raw"
                : "latest",
          },
        },
        { new: true }
      );
      return c.json(
        {
          message: `Successfully refreshed dataset with ${newRecords.length} new records`,
          data: {
            dataset: updatedDataset,
            newRecordsCount: newRecords.length,
            newRecordsSample: newRecords.slice(0, 3), // Sample of new records
          },
        },
        200
      );
    } catch (error: any) {
      console.error(`Error refreshing dataset ${requestedId}:`, error);

      // Handle specific API errors
      if (error.response) {
        console.error("NASA API Response Error:", {
          status: error.response.status,
          data: error.response.data,
          headers: error.response.headers,
        });

        if (error.response.status === 429) {
          return c.json(
            {
              message:
                "NASA POWER API rate limit exceeded. Please try again later.",
            },
            429
          );
        }
      }

      // Handle MongoDB errors
      if (error.name === "MongoServerError") {
        return c.json(
          {
            message: `Database error while refreshing dataset: ${error.message}`,
            code: error.code,
          },
          500
        );
      }

      // Handle network/timeout errors
      if (error.code === "ECONNABORTED" || error.message.includes("timeout")) {
        return c.json(
          {
            message:
              "Connection timed out while fetching data from NASA POWER API. Try with a smaller date range.",
          },
          504
        );
      }

      return c.json(
        {
          message: `Error refreshing dataset: ${
            error.message || "Unknown error"
          }`,
          details: error.stack
            ? error.stack.split("\n").slice(0, 3).join("\n")
            : "No stack trace available",
        },
        500
      );
    }
  } catch (error: any) {
    console.error("Error in refresh/:id endpoint:", error);

    // Handle missing or invalid dataset
    if (error.name === "CastError" && error.kind === "ObjectId") {
      return c.json(
        {
          message: "Invalid dataset ID format",
          details: `Provided ID '${requestedId}' is not a valid ObjectId`,
        },
        400
      );
    }

    // Handle database connection errors
    if (
      error.name === "MongooseError" &&
      error.message.includes("failed to connect")
    ) {
      return c.json(
        {
          message: "Database connection failed",
          details: "Could not connect to MongoDB server",
        },
        503
      );
    }

    // Use parseError utility for consistent error handling
    const { message, status } = parseError(error);
    return c.json(
      {
        message,
        endpoint: "refresh/:id",
        id: c.req.param("id"),
      },
      status
    );
  }
});

// POST - Refresh all NASA POWER at once
nasaPowerRoute.post("/refresh-all", async (c) => {
  try {
    await db();

    // Find all NASA POWER datasets
    const datasets = (await DatasetMeta.find({
      isAPI: true,
      "apiConfig.type": "nasa-power",
    }).lean()) as DatasetMetaDocument[];

    if (datasets.length === 0) {
      return c.json(
        { message: "No NASA POWER datasets found to refresh" },
        200
      );
    }
    const today = new Date();
    const todayFormatted = today.toISOString().slice(0, 10).replace(/-/g, "");

    const results = {
      total: datasets.length,
      refreshed: 0,
      alreadyUpToDate: 0,
      failed: 0,
      details: [] as any[],
    };

    // Process each dataset
    for (const dataset of datasets) {
      try {
        if (!dataset.apiConfig?.params) {
          results.details.push({
            id: dataset._id,
            name: dataset.name,
            status: dataset.status, // Keep actual DB status
            refreshResult: "failed", // Use refreshResult for operation outcome
            reason: "Missing API parameters",
          });
          continue;
        }

        // Get dynamic model dataset
        const dynamicModel =
          mongoose.models[dataset.collectionName] ||
          mongoose.model(
            dataset.collectionName,
            new mongoose.Schema({}, { strict: false }),
            dataset.collectionName
          );
        // Find latest record in collectionName
        const latestRecord = (await dynamicModel
          .findOne({})
          .sort({ Date: -1 })
          .lean()) as unknown as NasaPowerRecord;
        let startDateFormatted;

        // Determine start date based on latest record
        if (latestRecord && latestRecord.Date) {
          // Use latest record date
          const latestDate = new Date(latestRecord.Date);
          latestDate.setDate(latestDate.getDate() + 1);
          startDateFormatted = latestDate
            .toISOString()
            .slice(0, 10)
            .replace(/-/g, "");
        } else if (dataset.lastUpdated) {
          // Use day after lastUpdated as fallback
          const lastUpdated = new Date(dataset.lastUpdated);
          lastUpdated.setDate(lastUpdated.getDate() + 1);
          startDateFormatted = lastUpdated
            .toISOString()
            .slice(0, 10)
            .replace(/-/g, "");
        } else if (dataset.apiConfig?.params?.end) {
          // Use day after original end date as last resort
          const endDate = dataset.apiConfig.params.end;
          const year = parseInt(endDate.substring(0, 4));
          const month = parseInt(endDate.substring(4, 6)) - 1;
          const day = parseInt(endDate.substring(6, 8)) + 1;
          const nextDate = new Date(year, month, day);
          startDateFormatted = nextDate
            .toISOString()
            .slice(0, 10)
            .replace(/-/g, "");
        } else {
          results.failed++;
          results.details.push({
            id: dataset._id,
            name: dataset.name,
            status: dataset.status,
            reason: "Cannot determine start date",
          });
          continue;
        }

        // Check if dataset is already up to date
        if (startDateFormatted > todayFormatted) {
          results.alreadyUpToDate++;
          results.details.push({
            id: dataset._id,
            name: dataset.name,
            status: dataset.status,
            lastRecord:
              latestRecord?.Date?.toISOString() || dataset.lastUpdated,
          });
          continue;
        }
        console.log(
          `Refreshing dataset '${dataset.name}' from ${startDateFormatted} to ${todayFormatted}`
        );
        // Fetch new data from NASA POWER API
        const response = await axios.get(BASE_URL, {
          params: {
            start: startDateFormatted,
            end: todayFormatted,
            latitude: dataset.apiConfig.params.latitude,
            longitude: dataset.apiConfig.params.longitude,
            parameters: dataset.apiConfig.params.parameters.join(","),
            community: dataset.apiConfig.params.community || "ag",
            format: "JSON",
            user: "tumbuhbaik",
            header: "true",
            "time-standard": "LST",
          },
          timeout: 30000,
        });
        if (
          !response.data ||
          !response.data.properties ||
          !response.data.properties.parameter
        ) {
          results.failed++;
          results.details.push({
            id: dataset._id,
            name: dataset.name,
            status: dataset.status,
            reason: "Invalid data received from NASA POWER API",
          });
          continue;
        }
        // Transfrom data into array of objects
        const parameters = dataset.apiConfig.params.parameters;
        const properties = response.data.properties;
        const paramData = properties.parameter;
        const timespan = Object.keys(paramData[parameters[0]]);

        // If no new data available
        if (timespan.length === 0) {
          results.alreadyUpToDate++;
          results.details.push({
            id: dataset._id,
            name: dataset.name,
            status: dataset.status, // Keep using the actual status from database
            refreshResult: "no-new-data", // Use a separate field for the refresh result
            lastRecord:
              latestRecord?.Date?.toISOString() || dataset.lastUpdated,
          });
          continue;
        }
        const newRecords = timespan.map((date) => {
          const year = parseInt(date.substring(0, 4));
          const month = parseInt(date.substring(4, 6)) - 1;
          const day = parseInt(date.substring(6, 8));

          const record: Record<string, any> = {
            Date: new Date(Date.UTC(year, month, day)),
            Year: year,
            month: month + 1,
            day: day,
          };
          parameters.forEach((param: string) => {
            if (paramData[param] && paramData[param][date] !== undefined) {
              record[param] = paramData[param][date];
            }
          });
          return record;
        });

        // Insert new records
        await dynamicModel.insertMany(newRecords);
        // Update  metadata
        const totalRecords = await dynamicModel.countDocuments();
        await DatasetMeta.findByIdAndUpdate(dataset._id, {
          $set: {
            lastUpdated: today,
            totalRecords,
            "apiConfig.params.end": todayFormatted,
            status:
              dataset.status === "preprocessed" ||
              dataset.status === "validated"
                ? "raw"
                : "latest",
          },
        });
        results.refreshed++;
        results.details.push({
          id: dataset._id,
          name: dataset.name,
          status:
            dataset.status === "preprocessed" || dataset.status === "validated"
              ? "raw"
              : "latest",

          refreshResult: "success", // Operation result
          newRecordsCount: newRecords.length,
          lastRecord:
            newRecords.length > 0
              ? newRecords[newRecords.length - 1].Date?.toISOString()
              : null,
        });
      } catch (error: any) {
        console.error(`Error refreshing dataset ${dataset._id}:`, error);
        results.failed++;
        results.details.push({
          id: dataset._id,
          name: dataset.name,
          status: dataset.status,
          reason: error.message || "Unknown error",
        });
      }
    }
    return c.json(
      {
        message: `Completed refresh operation: ${results.refreshed} refreshed, ${results.alreadyUpToDate} up-to-date, ${results.failed} failed`,
        data: results,
      },
      200
    );
  } catch (error) {
    console.error("Error in batch refresh endpoint:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

// // PATCH - Move NASA POWER dataset to recycle bin if status: archived
// nasaPowerRoute.patch("/:collectionName/recycle-bin", async (c) => {
//   try {
//     await db();
//     const { collectionName } = c.req.param();

//     // Find dataset meta
//     const meta = await DatasetMeta.findOne({ collectionName }).lean();
//     if (!meta) {
//       return c.json({ message: "Dataset not found" }, 404);
//     }

//     // Only allow soft delete if status is archived and isAPI NASA POWER
//     if (
//       meta.isAPI &&
//       meta.apiConfig?.type === "nasa-power" &&
//       meta.status === "archived"
//     ) {
//       // Set deletedAt timestamp
//       const updated = await DatasetMeta.findOneAndUpdate(
//         { collectionName },
//         { $set: { deletedAt: new Date() } },
//         { new: true }
//       );
//       return c.json(
//         {
//           message: "NASA POWER dataset moved to recycle bin",
//           data: updated,
//         },
//         200
//       );
//     } else {
//       return c.json(
//         {
//           message:
//             "Only NASA POWER datasets with status 'archived' can be moved to recycle bin",
//         },
//         400
//       );
//     }
//   } catch (error) {
//     console.error("Soft delete NASA dataset error:", error);
//     const { message, status } = parseError(error);
//     return c.json({ message }, status);
//   }
// });

export default nasaPowerRoute;
