import axios from "axios";

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

export async function exportDatasetCsv(
  collectionName: string,
  sortBy = "Date",
  sortOrder = "desc"
) {
  try {
    const response = await axios.get("/api/v1/export-csv/dataset-meta", {
      params: { category: collectionName, sortBy, sortOrder }, // category → collectionName
      responseType: "blob",
    });

    if (response.status === 200) {
      const blob = new Blob([response.data], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");

      const contentDisposition = response.headers["content-disposition"];
      let filename = `${collectionName}_data_${
        new Date().toISOString().split("T")[0]
      }.csv`;

      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="(.+)"/);
        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      }

      link.href = url;
      link.download = filename;
      link.style.display = "none";

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      return { success: true, message: "File downloaded successfully" };
    }
  } catch (error) {
    console.error("Error exporting CSV:", error);
    return { success: false, message: "Export failed" };
  }
}
export async function exportHoltWinterCsv(
  sortBy = "forecast_date",
  sortOrder = "desc"
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
    const res = await axios.post("http://localhost:5001/run-forecast"); // ganti host jika deploy
    return res.data;
  } catch (error) {
    console.error("Trigger forecast run failed:", error);
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
