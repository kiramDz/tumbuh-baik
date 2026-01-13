import axios from "axios";
export interface DatasetMetaType {
  name: string;
  source: string;
  collectionName: string;
  description?: string;
  status: string;
  uploadDate: string;
  deletedAt?: string;
}

const getBaseUrl = () => {
  if (typeof window !== "undefined") {
    return "";
  }

  if (process.env.NODE_ENV === "production") {
    return "https://zonapetik.tech";
  }

  // Local development
  return "http://localhost:3000";
};

export async function exportDatasetCsv(collectionName: string, sortBy = "Date", sortOrder = "desc") {
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
      let filename = `${collectionName}_data_${new Date().toISOString().split("T")[0]}.csv`;

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
export async function exportHoltWinterCsv(sortBy = "forecast_date", sortOrder = "desc") {
  try {
    const response = await axios.get("/api/v1/export-csv/hw-daily", {
      params: { sortBy, sortOrder },
      responseType: "blob",
    });

    if (response.status === 200) {
      const blob = new Blob([response.data], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");

      const filename = `holt_winter_daily_${new Date().toISOString().split("T")[0]}.csv`;
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
    console.error("Export HW CSV failed:", error?.response?.data || error.message);
    return { success: false, message: "Export failed" };
  }
}





export async function exportLSTMForecastCsv(sortBy = "forecast_date", sortOrder = "desc") {
  try {
    const response = await axios.get("/api/v1/export-csv/lstm/daily", {
      params: { sortBy, sortOrder },
      responseType: "blob",
    });

    if (response.status === 200) {
      const blob = new Blob([response.data], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");

      const filename = `lstm_daily_${new Date().toISOString().split("T")[0]}.csv`;
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
    console.error("Export LSTM Forecast CSV failed:", error);
    return { success: false, message: "Export failed" };
  }
}

export const getBmkgLive = async () => {
  try {
    const response = await axios.get("/api/v1/bmkg-live/all");
    return response.data;
  } catch (error: any) {
    console.error("Error fetching BMKG live data:", error);
    throw new Error(error?.response?.data?.description || "Failed to fetch BMKG live data");
  }
};

//bmkg api
export const getBmkgApi = async () => {
  try {
    const response = await axios.get("/api/v1/bmkg-api/all");
    return response.data;
  } catch (error: any) {
    console.error("Error fetching BMKG API data:", error);
    throw new Error(error?.response?.data?.description || "Failed to fetch BMKG data");
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

export const getLSTMDaily = async (page = 1, pageSize = 10) => {
  try {
    const res = await axios.get("/api/v1/lstm/daily", {
      params: { page, pageSize },
    });
    if (res.status === 200) {
      console.log("🟢 Retrieved documents:", res.data.length);
      console.log("✅ LSTM API response:", res.data.data);
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
    console.error("Error fetching LSTM Daily:", error);
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

export const updateSeed = async (id: string, data: { name: string; duration: number }) => {
  try {
    const res = await axios.put(`/api/v1/seeds/${id}`, data);
    return res.data.data;
  } catch (error) {
    console.error("Update seed error:", error);
    throw error;
  }
};

export const deleteSeed = async (id: string) => {
  try {
    const res = await axios.delete(`/api/v1/seeds/${id}`);
    return res.data;
  } catch (error) {
    console.error("Delete seed error:", error);
    throw error;
  }
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

export const getAllKuesionerPetani = async () => {
  try {
    const res = await axios.get("/api/v1/kuesioner/petani");
    console.log("getAllKuesionerPetani", res.data);
    return res.data;
  } catch (error) {
    console.error("Error fetching kuesioner petani:", error);
    throw error;
  }
}

// export const getKuesionerPetaniById = async (id: string) => {
//   try {
//     const res = await axios.get(`/api/v1/kuesioner/petani/${id}`);
//     return res.data;
//   } catch (error) {
//     console.error("Error fetching kuesioner petani by ID:", error);
//     throw error;
//   }
// };

export const getAllKuesionerManajemen = async () => {
  try {
    const res = await axios.get("/api/v1/kuesioner-manajemen/manajemen");
    console.log("getAllKuesionerManajemen", res.data);
    return res.data;
  } catch (error) {
    console.error("Error fetching kuesioner manajemen:", error);
    throw error;
  }
};

export const getAllKuesionerPeriode = async () => {
  try {
    const res = await axios.get("/api/v1/kuesioner-periode/periode");
    console.log("getAllKuesionerPeriode", res.data);
    return res.data;
  } catch (error) {
    console.error("Error fetching kuesioner periode:", error);
    throw error;
  }
};

export const updateUserRole = async (userId: string, role: "user" | "admin") => {
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
    console.log("[GetRecycleBinDatasets] Fetching recycle bin datasets", { page, pageSize });

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
export const GetDatasetBySlug = async (slug: string): Promise<{ meta: any; items: any[] }> => {
  const baseUrl = getBaseUrl();
  const url = `${baseUrl}/api/v1/dataset-meta/${slug}`;

  const res = await fetch(url, {
    cache: "no-store",
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
  const baseUrl = getBaseUrl(); // Gunakan helper yang sama
  const url = `${baseUrl}/api/v1/dataset-meta/${slug}/chart-data`;

  const res = await fetch(url, {
    cache: "no-store",
  });

  if (!res.ok) throw new Error("Failed to fetch chart data");

  const json = await res.json();
  return json.data;
};

export const SoftDeleteDataset = async (collectionName: string) => {
  try {
    const res = await axios.patch(`/api/v1/dataset-meta/${collectionName}/delete`);
    return res.data.data;
  } catch (error) {
    console.error("Soft delete dataset error:", error);
    throw error;
  }
};

// API function - tambah Promise type

export const RestoreDataset = async (collectionName: string) => {
  try {
    const res = await axios.patch(`/api/v1/dataset-meta/${collectionName}/restore`);
    return res.data.data;
  } catch (error) {
    console.error("Restore dataset error:", error);
    throw error;
  }
};

export const PermanentDeleteDataset = async (collectionName: string) => {
  try {
    const res = await axios.delete(`/api/v1/dataset-meta/${collectionName}`);
    return res.data;
  } catch (error) {
    console.error("Permanent delete dataset error:", error);
    throw error;
  }
};

// for dataset table
// lib/fetch/files.fetch.tsx
export async function getDynamicDatasetData(slug: string, page = 1, pageSize = 10, sortBy = "Date", sortOrder: "asc" | "desc" = "desc") {
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

// forecast config :
export const getForecastConfigs = async () => {
  const response = await axios.get("/api/v1/forecast-config");
  return response.data.data;
};

export const createForecastConfig = async (data: {
  name: string;
  columns: { collectionName: string; columnName: string }[];
  startDate: string; // Format: "2025-01-15"
}) => {
  const response = await axios.post("/api/v1/forecast-config", data);
  return response.data;
};

export const updateForecastConfig = async (id: string, data: { name: string; columns: { collectionName: string; columnName: string }[]; startDate: string; }) => {
  const response = await axios.put(`/api/v1/forecast-config/${id}`, data);
  return response.data;
};

export const deleteForecastConfig = async (id: string) => {
  const response = await axios.delete(`/api/v1/forecast-config/${id}`);
  return response.data;
};

export const getLSTMConfigs = async () => {
  const response = await axios.get("/api/v1/lstm-config");
  return response.data.data;
};

export const createLSTMConfig = async (data: { name: string; columns: { collectionName: string; columnName: string }[]; startDate: string; }) => {
  const response = await axios.post("/api/v1/lstm-config", data);
  return response.data;
};

export const updateLSTMConfig = async (id: string, data: { name: string; columns: { collectionName: string; columnName: string }[]; startDate: string; }) => {
  const response = await axios.put(`/api/v1/lstm-config/${id}`, data);
  return response.data;
};

export const deleteLSTMConfig = async (id: string) => {
  const response = await axios.delete(`/api/v1/lstm-config/${id}`);
  return response.data;
};

export const triggerForecastRun = async () => {
  try {
    const res = await axios.post("http://localhost:5001/run-forecast");
    // const res = await axios.post("https://api.zonapetik.tech/run-forecast");
    return res.data;
  } catch (error: any) {
    if (error.response?.status === 404) {
      return { message: "Tidak ada config pending" };
    }
    throw error;
  }
};

export const triggerLSTMForecast = async () => {
  try {
    const res = await axios.post("http://127.0.0.1:5001/run-lstm");
    return res.data;
  } catch (error) {
    console.error("Trigger LSTM forecast failed:", error);
    throw error;
  }
};

// ==================== FARM API FUNCTIONS ====================

// Get all farms (optional filter by userId)
export const getAllFarms = async (userId?: string) => {
  try {
    const params = userId ? { userId } : {};
    const res = await axios.get("/api/v1/farm", { params });
    console.log("getAllFarms", res.data);
    return res.data;
  } catch (error) {
    console.error("Error fetching all farms:", error);
    throw error;
  }
};

// Get farm by ID
export const getFarmById = async (id: string) => {
  try {
    const res = await axios.get(`/api/v1/farm/${id}`);
    console.log("getFarmById", res.data);
    return res.data;
  } catch (error) {
    console.error("Error fetching farm by ID:", error);
    throw error;
  }
};

// Get all farms by user ID
export const getFarmsByUserId = async (userId: string) => {
  try {
    const res = await axios.get(`/api/v1/farm/user/${userId}`);
    console.log("getFarmsByUserId", res.data);
    return res.data;
  } catch (error) {
    console.error("Error fetching farms by user ID:", error);
    throw error;
  }
};

// Create new farm
export const createFarm = async (data: any) => {
  try {
    const res = await axios.post("/api/v1/farm", data);
    console.log("createFarm", res.data);
    return res.data;
  } catch (error) {
    console.error("Create farm error:", error);
    throw error;
  }
};

// Update farm by ID
export const updateFarm = async (id: string, data: any) => {
  try {
    const res = await axios.put(`/api/v1/farm/${id}`, data);
    console.log("updateFarm", res.data);
    return res.data;
  } catch (error) {
    console.error("Update farm error:", error);
    throw error;
  }
};

// Delete farm by ID
export const deleteFarm = async (id: string) => {
  try {
    const res = await axios.delete(`/api/v1/farm/${id}`);
    console.log("deleteFarm", res.data);
    return res.data;
  } catch (error) {
    console.error("Delete farm error:", error);
    throw error;
  }
};

// ==================== DECOMPOSE LSTM API FUNCTIONS ====================

export const getDecomposeLSTM = async () => {
  try {
    const res = await axios.get("/api/v1/decompose-lstm/all");
    console.log("✅ Decompose LSTM API response:", res.data);
    return res.data.data || [];
  } catch (error) {
    console.error("Error fetching Decompose LSTM:", error);
    return [];
  }
};

export const getDecomposeLSTMByDate = async (date: string) => {
  try {
    const res = await axios.get(`/api/v1/decompose-lstm/date/${date}`);
    return res.data.data;
  } catch (error) {
    console.error("Error fetching Decompose LSTM by date:", error);
    throw error;
  }
};

export const getDecomposeLSTMByConfigId = async (configId: string) => {
  try {
    const res = await axios.get(`/api/v1/decompose-lstm/config/${configId}`);
    return res.data.data || [];
  } catch (error) {
    console.error("Error fetching Decompose LSTM by config:", error);
    return [];
  }
};

// Fetch historical data from collection for comparison with forecast
export const fetchHistoricalData = async (collectionName: string, columnName: string) => {
  try {
    const countResponse = await axios.get(`/api/v1/dataset-meta/${collectionName}`, {
      params: { 
        page: 1, 
        pageSize: 10,
        sortBy: "Date",
        sortOrder: "asc"
      }
    });
    
    const total = countResponse.data?.data?.total || 0;
    if (total === 0) return [];

    const response = await axios.get(`/api/v1/dataset-meta/${collectionName}`, {
      params: { 
        page: 1, 
        pageSize: total,
        sortBy: "Date",
        sortOrder: "asc"
      }
    });
    
    const items = response.data?.data?.items || [];
    return items
      .filter((item: any) => item.Date && item[columnName] != null)
      .map((item: any) => ({
        date: item.Date,
        value: item[columnName]
      }));
  } catch (error) {
    console.error(`Error fetching historical data for ${collectionName}:`, error);
    return [];
  }
};
