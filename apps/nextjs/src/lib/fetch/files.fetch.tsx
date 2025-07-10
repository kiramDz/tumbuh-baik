import axios from "axios";

export interface DatasetMetaType {
  name: string;
  source: string;
  collectionName: string;
  description?: string;
  status: string;
  uploadDate: string;
}

export async function exportToCsv(collectionName: string, sortBy = "Date", sortOrder = "desc") {
  try {
    const response = await axios.get("/api/v1/export-csv", {
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

//for monthly calender in public
export const getBmkgSummary = async () => {
  try {
    const response = await axios.get("/api/v1/bmkg-summary/all");
    return response.data.data;
  } catch (error: any) {
    console.error("Error fetching BMKG Summary:", error);
    throw new Error(error?.response?.data?.description || "Failed to fetch BMKG summary data");
  }
};

export const getBmkgDaily = async (page = 1, pageSize = 10) => {
  try {
    const res = await axios.get("/api/v1/bmkg-daily", {
      params: { page, pageSize },
    });
    if (res.status === 200) {
      console.log("✅ BMKG API response:", res.data.data);
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
    console.error("Error fetching BMKG Daily:", error);
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

//for dataset detail page
export const GetDatasetBySlug = async (slug: string): Promise<{ meta: any; items: any[] }> => {
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

export const createForecastConfig = async (data: { name: string; columns: { collectionName: string; columnName: string }[] }) => {
  const response = await axios.post("/api/v1/forecast-config", data);
  return response.data;
};
// export const createForecastConfig = async (payload: { name: string; columns: { collectionName: string; columnName: string }[] }) => {
//   try {
//     console.log("🔍 Sending payload to /api/v1/forecast-config:", payload);

//     const res = await axios.post("/api/v1/forecast-config", payload);

//     console.log("✅ Server response:", res.data);

//     return res.data.data;
//   } catch (error: any) {
//     if (axios.isAxiosError(error)) {
//       console.error("❌ Axios error response:", error.response?.data);
//       console.error("❌ Axios status:", error.response?.status);
//     } else {
//       console.error("❌ Unknown error:", error);
//     }

//     throw error;
//   }
// };

// export const createForecastConfig = async (payload: { name: string; columns: { collectionName: string; columnName: string }[] }) => {
//   try {
//     const res = await axios.post("/api/v1/forecast-config", payload);
//     return res.data.data;
//   } catch (error) {
//     console.error("Create forecast config error:", error);
//     throw error;
//   }
// };
