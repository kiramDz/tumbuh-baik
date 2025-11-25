import axios from "axios";
import { trackAllowedDynamicAccess } from "next/dist/server/app-render/dynamic-rendering";

export interface DatasetMetaType {
  _id: string;
  name: string;
  source: string;
  collectionName: string;
  description?: string;
  status: string;
  uploadDate: string;
  isAPI: boolean; // Added API flag for fetching NASA
  lastUpdated?: string; // Dated latest updating
  apiConfig?: Record<string, any>; // API config
  refreshInfo: {
    canRefresh: boolean;
    daysSinceLastRecord: number;
    lastRecordDate: string;
    startDate: string;
    endDate: string;
  };
}

export interface RefreshResult {
  id: string;
  name: string;
  status: string;
  refreshResult: string;
  newRecordsCount?: number;
  lastRecord?: string;
  reason?: string;
}

export interface RefreshAllResponse {
  message: string;
  data: {
    total: number;
    refreshed: number;
    alreadyUpToDate: number;
    failed: number;
    details: RefreshResult[];
  };
}

export async function exportDatasetCsv(
  collectionName: string,
  sortBy = "Date",
  sortOrder = "asc"
) {
  try {
    // Fetch all data from the existing endpoint (without pagination)
    const response = await axios.get(`/api/v1/dataset-meta/${collectionName}`, {
      params: {
        page: 1,
        pageSize: 999999, // Get all records
        sortBy,
        sortOrder,
      },
    });

    if (response.status === 200 && response.data?.data?.items) {
      const data = response.data.data.items;

      if (data.length === 0) {
        return { success: false, message: "No data to export" };
      }

      // Get column headers from the first item
      const headers = Object.keys(data[0]).filter((key) => key !== "_id");

      // Create CSV content
      let csvContent = headers.join(",") + "\n";

      // Add data rows
      data.forEach((item: any) => {
        const row = headers
          .map((header) => {
            let value = item[header];

            // Handle Date formatting specifically
            if (header.toLowerCase() === "date" || header === "Date") {
              if (typeof value === "string") {
                // Convert DD/MM/YYYY to YYYY-MM-DD
                if (/^\d{1,2}\/\d{1,2}\/\d{4}$/.test(value)) {
                  const [day, month, year] = value.split("/");
                  value = `${year}-${month.padStart(2, "0")}-${day.padStart(
                    2,
                    "0"
                  )}`;
                }
                // Handle ISO string format
                else if (value.includes("T")) {
                  value = value.split("T")[0];
                }
              } else if (value instanceof Date) {
                value = value.toISOString().split("T")[0];
              }
            }

            // Handle other data types
            if (value === null || value === undefined) {
              return "";
            }
            if (typeof value === "object") {
              return JSON.stringify(value);
            }

            // Escape commas and quotes in CSV
            const stringValue = String(value);
            if (
              stringValue.includes(",") ||
              stringValue.includes('"') ||
              stringValue.includes("\n")
            ) {
              return `"${stringValue.replace(/"/g, '""')}"`;
            }

            return stringValue;
          })
          .join(",");

        csvContent += row + "\n";
      });

      // Create and download the CSV file
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");

      const filename = `${collectionName}_data_${
        new Date().toISOString().split("T")[0]
      }.csv`;

      link.href = url;
      link.download = filename;
      link.style.display = "none";

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      return {
        success: true,
        message: `Successfully exported ${data.length} records`,
      };
    }

    return { success: false, message: "No data found" };
  } catch (error) {
    console.error("Error exporting CSV:", error);
    return {
      success: false,
      message: error instanceof Error ? error.message : "Export failed",
    };
  }
}

// export async function exportDatasetCsv(
//   collectionName: string,
//   sortBy = "Date",
//   sortOrder = "desc"
// ) {
//   try {
//     const response = await axios.get("/api/v1/export-csv/dataset-meta", {
//       params: { category: collectionName, sortBy, sortOrder }, // category → collectionName
//       responseType: "blob",
//     });

//     if (response.status === 200) {
//       const blob = new Blob([response.data], { type: "text/csv" });
//       const url = window.URL.createObjectURL(blob);
//       const link = document.createElement("a");

//       const contentDisposition = response.headers["content-disposition"];
//       let filename = `${collectionName}_data_${
//         new Date().toISOString().split("T")[0]
//       }.csv`;

//       if (contentDisposition) {
//         const filenameMatch = contentDisposition.match(/filename="(.+)"/);
//         if (filenameMatch) {
//           filename = filenameMatch[1];
//         }
//       }

//       link.href = url;
//       link.download = filename;
//       link.style.display = "none";

//       document.body.appendChild(link);
//       link.click();
//       document.body.removeChild(link);
//       window.URL.revokeObjectURL(url);

//       return { success: true, message: "File downloaded successfully" };
//     }
//   } catch (error) {
//     console.error("Error exporting CSV:", error);
//     return { success: false, message: "Export failed" };
//   }
// }
export async function exportHoltWinterCsv(
  sortBy = "forecast_date",
  sortOrder = "asc"
) {
  try {
    const response = await axios.get("/api/v1/export-csv/hw-daily", {
      params: { sortBy, sortOrder },
      responseType: "blob",
    });

    if (response.status === 200) {
      const blob = new Blob([response.data], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");

      const filename = `holt_winter_daily_${
        new Date().toISOString().split("T")[0]
      }.csv`;
      link.href = url;
      link.download = filename;
      link.style.display = "none";

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      return { success: true, message: "File downloaded successfully" };
    }
  } catch (error: any) {
    console.error(
      "Export HW CSV failed:",
      error?.response?.data || error.message
    );
    return { success: false, message: "Export failed" };
  }
}

export const getBmkgLive = async () => {
  try {
    const response = await axios.get("/api/v1/bmkg-live/all");
    return response.data;
  } catch (error: any) {
    console.error("Error fetching BMKG live data:", error);
    throw new Error(
      error?.response?.data?.description || "Failed to fetch BMKG live data"
    );
  }
};

//bmkg api
export const getBmkgApi = async () => {
  try {
    const response = await axios.get("/api/v1/bmkg-api/all");
    return response.data;
  } catch (error: any) {
    console.error("Error fetching BMKG API data:", error);
    throw new Error(
      error?.response?.data?.description || "Failed to fetch BMKG data"
    );
  }
};

// holt winter table
export const getHoltWinterDaily = async (page = 1, pageSize = 10) => {
  try {
    const res = await axios.get("/api/v1/hw/daily", {
      params: { page, pageSize },
    });
    if (res.status === 200) {
      console.log("🟢 Retrieved documents:", res.data.length);
      console.log("✅ HW API response:", res.data.data);
      return (
        res.data.data || {
          items: [],
          total: 0,
          currentPage: 1,
          totalPages: 1,
          pageSize,
        }
      );
    }
  } catch (error) {
    console.error("Error fetching HW Daily:", error);
    return {
      items: [],
      total: 0,
      currentPage: 1,
      totalPages: 1,
      pageSize,
    };
  }
};

export const getSeeds = async (page = 1, pageSize = 10) => {
  try {
    const res = await axios.get("/api/v1/seeds", {
      params: { page, pageSize },
    });

    if (res.status === 200) {
      return (
        res.data.data || {
          items: [],
          total: 0,
          currentPage: 1,
          totalPages: 1,
          pageSize,
        }
      );
    }
  } catch (error) {
    console.error("Get seeds error:", error);
    return {
      items: [],
      total: 0,
      currentPage: 1,
      totalPages: 1,
      pageSize,
    };
  }
};

export const createSeed = async (data: { name: string; duration: number }) => {
  try {
    const res = await axios.post("/api/v1/seeds", data);
    return res.data.data;
  } catch (error) {
    console.error("Create seed error:", error);
    throw error;
  }
};

export const deleteSeed = async (id: string) => {
  await axios.delete(`/api/v1/seeds/${id}`);
};

export const getUsers = async (page = 1, pageSize = 10) => {
  try {
    const res = await axios.get("/api/v1/user", {
      params: { page, pageSize },
    });

    if (res.status === 200) {
      return (
        res.data.data || {
          items: [],
          total: 0,
          currentPage: 1,
          totalPages: 1,
          pageSize,
        }
      );
    }
  } catch (error) {
    console.error("Get users error:", error);
    return {
      items: [],
      total: 0,
      currentPage: 1,
      totalPages: 1,
      pageSize,
    };
  }
};

export const updateUserRole = async (
  userId: string,
  role: "user" | "admin"
) => {
  try {
    const res = await axios.put(`/api/v1/user/${userId}/role`, { role });
    return res.data.data;
  } catch (error) {
    console.error("Update user role error:", error);
    throw error;
  }
};

// for dataset card
export const GetAllDatasetMeta = async (): Promise<DatasetMetaType[]> => {
  try {
    const res = await axios.get("/api/v1/dataset-meta");
    return res.data.data;
  } catch (error) {
    console.error("Get all dataset meta error:", error);
    throw error;
  }
};

export const GetRecycleBinDatasets = async (page = 1, pageSize = 10) => {
  try {
    console.log("[GetRecycleBinDatasets] Fetching recycle bin datasets", {
      page,
      pageSize,
    });

    const res = await axios.get("/api/v1/dataset-meta/recycle-bin", {
      params: { page, pageSize },
    });

    console.log("[GetRecycleBinDatasets] Response status:", res.status);
    console.log("[GetRecycleBinDatasets] Response data:", res.data);

    if (res.status === 200) {
      const result = res.data.data || {
        items: [],
        total: 0,
        currentPage: 1,
        totalPages: 1,
        pageSize,
      };

      console.log("[GetRecycleBinDatasets] Final result:", result);
      return result;
    }
  } catch (error) {
    console.error("Get recycle bin error:", error);
    throw error;
  }
};

//for dataset detail page
export const GetDatasetBySlug = async (
  slug: string
): Promise<{ meta: any; items: any[] }> => {
  const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3000";
  const res = await fetch(`${baseUrl}/api/v1/dataset-meta/${slug}`, {
    cache: "no-store", // optional jika ingin real-time
  });

  if (!res.ok) throw new Error("Failed to fetch dataset");

  const json = await res.json();
  return json.data;
};

export const GetChartDataBySlug = async (
  slug: string
): Promise<{
  items: any[];
  numericColumns: string[];
  dateColumn: string;
} | null> => {
  const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3000";
  const res = await fetch(`${baseUrl}/api/v1/dataset-meta/${slug}/chart-data`, {
    cache: "no-store",
  });

  if (!res.ok) throw new Error("Failed to fetch chart data");

  const json = await res.json();

  return json.data;
};

export const SoftDeleteDataset = async (collectionName: string) => {
  try {
    const res = await axios.patch(
      `/api/v1/dataset-meta/${collectionName}/delete`
    );
    return res.data.data;
  } catch (error) {
    console.error("Soft delete dataset error:", error);
    throw error;
  }
};

// API function - tambah Promise type

export const RestoreDataset = async (collectionName: string) => {
  try {
    const res = await axios.patch(
      `/api/v1/dataset-meta/${collectionName}/restore`
    );
    return res.data.data;
  } catch (error) {
    console.error("Restore dataset error:", error);
    throw error;
  }
};

// export const PermanentDeleteDataset = async (collectionName: string) => {
//   try {
//     const res = await axios.delete(`/api/v1/dataset-meta/${collectionName}`);
//     return res.data;
//   } catch (error) {
//     console.error("Permanent delete dataset error:", error);
//     throw error;
//   }
// };

// for dataset table
// lib/fetch/files.fetch.ts
export async function getDynamicDatasetData(
  slug: string,
  page = 1,
  pageSize = 10,
  sortBy = "Date",
  sortOrder: "asc" | "desc" = "desc"
) {
  try {
    const res = await axios.get(`/api/v1/dataset-meta/${slug}`, {
      params: { page, pageSize, sortBy, sortOrder },
    });

    if (res.status === 200) {
      return (
        res.data.data || {
          items: [],
          total: 0,
          currentPage: 1,
          totalPages: 1,
          pageSize,
          sortBy,
          sortOrder,
        }
      );
    }
  } catch (error) {
    console.error(`❌ Error fetching dataset ${slug}:`, error);
    return {
      items: [],
      total: 0,
      currentPage: 1,
      totalPages: 1,
      pageSize,
      sortBy,
      sortOrder,
    };
  }
}

export const AddDatasetMeta = async (data: {
  name: string;
  source: string;
  fileType: "csv" | "json";
  filename?: string;
  collectionName?: string;
  status?: string;
  description?: string;
  records: Record<string, any>[]; // hasil parsing CSV atau JSON
}) => {
  try {
    const res = await axios.post("/api/v1/dataset-meta", {
      name: data.name,
      source: data.source,
      fileType: data.fileType,
      filename: data.filename || `${data.name}.${data.fileType}`,
      collectionName: data.collectionName,
      description: data.description || "",
      status: data.status || "raw",
      data: data.records,
    });
    return res.data.data;
  } catch (error) {
    console.error("Add dataset meta error:", error);
    throw error;
  }
};

export const UpdateDatasetMeta = async (
  _id: string,
  data: {
    name?: string;
    source?: string;
    fileType?: "csv" | "json";
    filename?: string;
    collectionName?: string;
    status?: string;
    description?: string;
  }
) => {
  try {
    const res = await axios.put(`/api/v1/dataset-meta/${_id}`, data, {
      headers: {
        "Content-Type": "application/json",
      },
    });

    console.log("API response:", res.data);
    return res.data.data;
  } catch (error) {
    console.error("❌ Update dataset meta error:", error);
    throw error;
  }
};

export const AddXlsxDatasetMeta = async (data: {
  name: string;
  source: string;
  description?: string;
  status?: string;
  file?: File;
  files?: File[];
  isMultiFile?: boolean;
}) => {
  try {
    // Validation: Must provide either file or files
    if (data.isMultiFile) {
      if (!data.files || data.files.length === 0) {
        throw new Error("Files array is required for multi-file upload");
      }
      if (data.files.length > 50) {
        throw new Error("Maximum 50 files allowed for multi-file upload");
      }
    } else {
      if (!data.file) {
        throw new Error("File is required for single file upload");
      }
    }

    // Create FormData for multipart upload
    const formData = new FormData();
    formData.append("name", data.name.trim());
    formData.append("source", data.source.trim());
    formData.append("description", data.description || "");
    formData.append("status", data.status || "raw");
    formData.append("isMultiFile", data.isMultiFile ? "true" : "false");

    // Add files based on upload mode
    if (data.isMultiFile && data.files) {
      // Multi-file upload: add all files with "files" key
      data.files.forEach((file) => {
        formData.append("files", file);
      });
    } else if (data.file) {
      // Single-file upload: add single file with "file" key
      formData.append("file", data.file);
    }

    const res = await axios.post("/api/v1/dataset-meta", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return res.data.data;
  } catch (error) {
    console.error("Add XLSX dataset meta error:", error);
    throw error;
  }
};

export const DeleteDatasetMeta = async (collectionName: string) => {
  try {
    const res = await axios.delete(`/api/v1/dataset-meta/${collectionName}`);
    return res.data.data;
  } catch (error) {
    console.error("Delete dataset meta error:", error);
    throw error;
  }
};

// forecast config :
export const getForecastConfigs = async () => {
  const response = await axios.get("/api/v1/forecast-config");
  return response.data.data;
};

export const createForecastConfig = async (data: {
  name: string;
  columns: { collectionName: string; columnName: string }[];
}) => {
  const response = await axios.post("/api/v1/forecast-config", data);
  return response.data;
};

export const triggerForecastRun = async () => {
  try {
    // const res = await axios.post("https://de53213413b6.ngrok-free.app/run-forecast");
    const res = await axios.post("https://api.zonapetik.tech/run-forecast");
    return res.data;
  } catch (error: any) {
    if (error.response?.status === 404) {
      return { message: "Tidak ada config pending" };
    }
    throw error;
  }
};

export interface NasaPowerParams {
  start: string; // format YYYYMMDD
  end: string;
  latitude: number;
  longitude: number;
  parameters: string[];
  community?: string;
}

export const fetchNasaPowerData = async (params: NasaPowerParams) => {
  try {
    const response = await axios.get(`/api/v1/nasa-power`, {
      params: {
        start: params.start,
        end: params.end,
        latitude: params.latitude,
        longitude: params.longitude,
        parameters: params.parameters.join(","),
        community: params.community || "ag",
      },
    });
    return response.data;
  } catch (error) {
    console.error("Error fetching NASA Power data:", error);
    throw error;
  }
};
export const saveNasaPowerData = async (data: {
  name: string;
  collectionName?: string;
  description?: string;
  status?: string;
  source?: string;
  nasaParams: NasaPowerParams;
}) => {
  try {
    const response = await axios.post(`/api/v1/nasa-power/save`, data);
    return response.data;
  } catch (error) {
    console.error("Error saving NASA Power data:", error);
    throw error;
  }
};

export const getNasaPowerRefreshStatus = async (datasetId: string) => {
  try {
    const response = await axios.get(`/api/v1/nasa-power/refreshable`);
    const datasets = (response.data.data || []) as DatasetMetaType[];

    // Find the specified dataset - add proper type annotation
    const dataset = datasets.find((d) => d._id === datasetId);

    if (!dataset) {
      return { canRefresh: false, message: "Dataset not found" };
    }

    return {
      canRefresh: dataset.refreshInfo.canRefresh,
      daysSinceLastRecord: dataset.refreshInfo.daysSinceLastRecord,
      lastRecordDate: dataset.refreshInfo.lastRecordDate,
      message: dataset.refreshInfo.canRefresh
        ? `Dataset can be refreshed with ${dataset.refreshInfo.daysSinceLastRecord} days of new data`
        : "Dataset is already up-to-date",
    };
  } catch (error) {
    console.error("Error checking refresh status:", error);
    // Default to allowing refresh if we can't determine status
    return { canRefresh: true, message: "Unable to check refresh status" };
  }
};

// Modify refreshNasaPowerDataset to check first
export const refreshNasaPowerDataset = async (datasetId: string) => {
  try {
    // First check if refresh is needed
    const refreshStatus = await getNasaPowerRefreshStatus(datasetId);

    // If dataset is already up-to-date
    if (!refreshStatus.canRefresh) {
      return {
        message: "Dataset is already up-to-date",
        data: {
          newRecordsCount: 0,
          lastUpdated: refreshStatus.lastRecordDate,
        },
      };
    }

    // Proceed with refresh if needed
    const response = await axios.post(
      `/api/v1/nasa-power/refresh/${datasetId}`
    );
    return response.data;
  } catch (error) {
    console.error("Error refreshing NASA POWER dataset:", error);
    throw error;
  }
};

// Fetch All NASA datasets
export const refreshAllNasaDatasets = async (): Promise<RefreshAllResponse> => {
  try {
    const response = await axios.post("/api/v1/nasa-power/refresh-all");
    console.log("Full response:", response);
    console.log("response.data:", response.data);
    console.log("response.status:", response.status);
    return response.data;
  } catch (error) {
    console.error("Error refreshing all NASA datasets:", error);
    throw error;
  }
};

export const preprocessNasaDatasetWithStream = (
  collectionName: string,
  onLog: (log: any) => void,
  onProgress: (progress: number, stage: string, message: string) => void,
  onComplete: (result: any) => void,
  onError: (error: string) => void
) => {
  const encodedName = encodeURIComponent(collectionName);
  const eventSource = new EventSource(
    `http://localhost:5001/api/v1/preprocess/nasa/${encodedName}/stream`
  );

  let connectionTimeout: NodeJS.Timeout;
  let hasCompletedSuccessfully = false;
  let hasReceivedData = false;
  let streamClosed = false;

  const resetTimeout = () => {
    if (connectionTimeout) {
      clearTimeout(connectionTimeout);
    }
    connectionTimeout = setTimeout(() => {
      if (!hasCompletedSuccessfully) {
        onError(
          "Connection timeout - preprocessing may still be running in background"
        );
        eventSource.close();
      }
    }, 300000); // 5 minutes timeout
  };

  resetTimeout();

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      hasReceivedData = true;
      resetTimeout();

      switch (data.type) {
        case "connected":
          onLog({
            type: "info",
            message: `Connected to preprocessing stream. Session: ${data.session_id}`,
          });
          break;

        case "log":
          onLog(data);
          break;

        case "progress":
          const progressValue =
            data.percentage ?? data.progress ?? data.percent ?? 0;
          onProgress(
            progressValue,
            data.stage || "Processing",
            data.message || "Processing..."
          );
          break;

        case "complete":
          hasCompletedSuccessfully = true;
          streamClosed = true;

          onLog({
            type: "success",
            message: "🎉 NASA preprocessing completed successfully!",
          });

          const completionResult =
            data.result && typeof data.result === "object"
              ? data.result
              : {
                  recordCount: null,
                  cleanedCollection: `${collectionName}_cleaned`,
                  preprocessing_report: null,
                };

          onComplete(completionResult);
          clearTimeout(connectionTimeout);
          eventSource.close();
          break;

        case "error":
          streamClosed = true;
          onError(data.message || "An error occurred during preprocessing");
          clearTimeout(connectionTimeout);
          eventSource.close();
          break;

        default:
          break;
      }
    } catch (error) {
      console.error("Error parsing SSE data:", error);
    }
  };

  eventSource.onerror = (error) => {
    if (streamClosed) {
      return;
    }

    clearTimeout(connectionTimeout);
    streamClosed = true;

    if (hasCompletedSuccessfully) {
      return; // Don't call onError for successful completion
    }

    if (hasReceivedData) {
      onLog({
        type: "success",
        message: "✅ Preprocessing completed successfully!",
      });

      onComplete({
        recordCount: null,
        cleanedCollection: `${collectionName}_cleaned`,
        preprocessing_report: null,
      });
      return;
    }

    onError("❌ Connection failed - no preprocessing data received");
    eventSource.close();
  };

  return {
    eventSource,
    cleanup: () => {
      streamClosed = true;
      clearTimeout(connectionTimeout);
      eventSource.close();
    },
  };
};

// Trigger NASA POWER Preprocessing with stream
// export const preprocessNasaDatasetWithStream = (
//   collectionName: string,
//   onLog: (log: any) => void,
//   onProgress: (progress: number, stage: string, message: string) => void,
//   onComplete: (result: any) => void,
//   onError: (error: string) => void
// ) => {
//   const encodedName = encodeURIComponent(collectionName);
//   const eventSource = new EventSource(
//     `http://localhost:5001/api/v1/preprocess/nasa/${encodedName}/stream`
//   );

//   // Add Connection timeout
//   let connectionTimeout: NodeJS.Timeout;
//   let hasCompletedSuccessfully = false; // ✅ Track completion status

//   const resetTimeout = () => {
//     if (connectionTimeout) {
//       clearTimeout(connectionTimeout);
//     }
//     connectionTimeout = setTimeout(() => {
//       if (!hasCompletedSuccessfully) {
//         onError(
//           "Connection timeout - preprocessing may still be running in background"
//         );
//         eventSource.close();
//       }
//     }, 300000); // 5 minutes timeout
//   };

//   resetTimeout();

//   eventSource.onmessage = (event) => {
//     try {
//       const data = JSON.parse(event.data);

//       resetTimeout();

//       switch (data.type) {
//         case "connected":
//           onLog({
//             type: "info",
//             message: `Connected to preprocessing stream. Session: ${data.session_id}`,
//           });
//           break;

//         case "log":
//           onLog(data);
//           break;

//         case "progress":
//           // Handle multiple possible field names for progress percentage
//           const progressValue =
//             data.percentage ?? data.progress ?? data.percent ?? 0;

//           onProgress(
//             progressValue,
//             data.stage || "Processing",
//             data.message || "Processing..."
//           );
//           break;

//         case "complete":
//           hasCompletedSuccessfully = true;

//           onLog({
//             type: "success",
//             message: "NASA preprocessing completed successfully!",
//           });

//           // Trigger completion callback with result
//           onComplete(data.result);

//           // Note: Don't close yet, wait for stream_complete
//           break;

//         case "stream_complete":
//           if (!hasCompletedSuccessfully) {
//             onLog({
//               type: "info",
//               message: "Stream closed - preprocessing may have completed",
//             });

//             // Fallback: If we got stream_complete without complete,
//             // assume success but with no result data
//             onComplete({
//               recordCount: null,
//               cleanedCollection: null,
//               preprocessing_report: null,
//             });
//           } else {
//             onLog({
//               type: "info",
//               message: "Preprocessing stream closed successfully",
//             });
//           }

//           clearTimeout(connectionTimeout);
//           eventSource.close();
//           break;

//         case "error":
//           onError(data.message || "An error occurred during preprocessing");
//           clearTimeout(connectionTimeout);
//           eventSource.close();
//           break;

//         default:
//           console.log("Unknown SSE message type:", data.type, data);
//           break;
//       }
//     } catch (error) {
//       console.error("Error parsing SSE data:", error);
//       console.log("Raw event data:", event.data);
//     }
//   };

//   eventSource.onerror = (error) => {
//     console.error("SSE error:", error);
//     clearTimeout(connectionTimeout);

//     if (eventSource.readyState === EventSource.CLOSED) {
//       // ✅ FIXED: Better handling of connection close
//       if (!hasCompletedSuccessfully) {
//         onLog({
//           type: "info",
//           message: "Connection closed - check if preprocessing completed",
//         });
//       } else {
//         onLog({
//           type: "info",
//           message: "Preprocessing completed, connection closed",
//         });
//       }
//     } else {
//       onError("Connection error occurred with the preprocessing stream.");
//     }
//     eventSource.close();
//   };

//   return {
//     eventSource,
//     cleanup: () => {
//       clearTimeout(connectionTimeout);
//       eventSource.close();
//     },
//   };
// };

// Trigger BMKG Preprocessing with stream
export const preprocessBmkgDatasetWithStream = (
  collectionName: string,
  onLog: (log: any) => void,
  onProgress: (progress: number, stage: string, message: string) => void,
  onComplete: (result: any) => void,
  onError: (error: string) => void
) => {
  const encodedName = encodeURIComponent(collectionName);
  const eventSource = new EventSource(
    `http://localhost:5001/api/v1/preprocess/bmkg/${encodedName}/stream`
  );

  let connectionTimeout: NodeJS.Timeout;

  const resetTimeout = () => {
    if (connectionTimeout) {
      clearTimeout(connectionTimeout);
    }
    connectionTimeout = setTimeout(() => {
      onError(
        "Connection timeout - preprocessing may still be running in background"
      );
      eventSource.close();
    }, 300000); // 5 minutes timeout
  };
  resetTimeout();

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      resetTimeout();

      switch (data.type) {
        case "connected":
          onLog({
            type: "info",
            message: `Connected to BMKG preprocessing stream. Session: ${data.session_id}`,
          });
          break;

        case "log":
          onLog(data);
          break;

        case "progress":
          // Handle multiple possible field names for progress percentage
          const progressValue =
            data.percentage ?? data.progress ?? data.percent ?? 0;

          onProgress(
            progressValue,
            data.stage || "Processing",
            data.message || "Processing..."
          );
          break;

        case "complete":
          onComplete(data.result);
          eventSource.close();
          break;

        case "stream_complete":
          onLog({
            type: "info",
            message: "BMKG preprocessing stream completed successfully",
          });
          clearTimeout(connectionTimeout);
          eventSource.close();
          break;

        case "error":
          onError(
            data.message || "An error occurred during BMKG preprocessing"
          );
          eventSource.close();
          break;

        default:
          console.log("Unknown SSE message type:", data.type, data);
          break;
      }
    } catch (error) {
      console.error("Error parsing SSE data:", error);
      console.log("Raw event data:", event.data);
    }
  };

  eventSource.onerror = (error) => {
    console.error("SSE error:", error);
    clearTimeout(connectionTimeout);

    if (eventSource.readyState === EventSource.CLOSED) {
      onLog({
        type: "info",
        message: "Preprocessing stream closed - check results in dashboard",
      });
    } else {
      onError("Connection error occurred with the BMKG preprocessing stream.");
    }
    eventSource.close();
  };

  return {
    eventSource,
    cleanup: () => {
      clearTimeout(connectionTimeout);
      eventSource.close();
    },
  };
};

// Trigger BMKG Preprocessing without stream (standard)
export const preprocessBmkgDataset = async (
  collectionName: string,
  options?: {
    smoothing_method?: "exponential" | "moving_average";
    window_size?: number;
    exponential_alpha?: number;
    drop_outliers?: boolean;
    outlier_methods?: string[];
    iqr_multiplier?: number;
    zscore_threshold?: number;
    outlier_treatment?: "interpolate" | "cap" | "remove";
    fill_missing?: boolean;
    detect_gaps?: boolean;
    max_gap_interpolate?: number;
    columns_to_process?: string[];
    parameter_configs?: Record<string, any>;
  }
) => {
  try {
    const response = await axios.post(
      `/api/v1/preprocess/bmkg/${collectionName}`,
      options || {}
    );
    return response.data;
  } catch (error) {
    console.error("Error preprocessing BMKG dataset:", error);
    throw error;
  }
};

// Get BMKG preprocessing job status
export const getBmkgPreprocessingStatus = async (jobId: string) => {
  try {
    const response = await axios.get(`/api/v1/preprocess/status/${jobId}`);
    return response.data;
  } catch (error) {
    console.error("Error getting BMKG preprocessing status:", error);
    throw error;
  }
};

// Get all BMKG preprocessing jobs
export const getAllBmkgPreprocessingJobs = async (page = 1, pageSize = 10) => {
  try {
    const response = await axios.get("/api/v1/preprocess/jobs", {
      params: {
        page,
        pageSize,
        type: "bmkg", // Filter for BMKG jobs only
      },
    });
    return response.data;
  } catch (error) {
    console.error("Error getting BMKG preprocessing jobs:", error);
    throw error;
  }
};

// Check if BMKG dataset can be preprocessed
export const checkBmkgDatasetForPreprocessing = async (
  collectionName: string
) => {
  try {
    const response = await axios.get(
      `/api/v1/preprocess/bmkg/${collectionName}/validate`
    );
    return response.data;
  } catch (error) {
    console.error("Error validating BMKG dataset:", error);
    throw error;
  }
};
