import axios from "axios";

export interface DatasetMetaType {
  name: string;
  source: string;
  collectionName: string;
  description?: string;
  status: string;
  uploadDate: string;
}

export async function exportToCsv(category: string, sortBy = "Date", sortOrder = "desc") {
  try {
    const response = await axios.get("/api/v1/export-csv", {
      params: { category, sortBy, sortOrder },
      responseType: "blob", // Important untuk file download
    });

    if (response.status === 200) {
      // Create download link
      const blob = new Blob([response.data], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");

      // Extract filename from Content-Disposition header atau buat default
      const contentDisposition = response.headers["content-disposition"];
      let filename = `${category}_data_${new Date().toISOString().split("T")[0]}.csv`;

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

      // Cleanup
      window.URL.revokeObjectURL(url);

      return { success: true, message: "File downloaded successfully" };
    }
  } catch (error) {
    console.error("Error exporting CSV:", error);
    return { success: false, message: "Failed to export CSV" };
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

//display bmkg in dahsboard
export async function getBmkgData(page = 1, pageSize = 10, sortBy = "Date", sortOrder = "desc") {
  try {
    const res = await axios.get("/api/v1/bmkg", {
      params: { page, pageSize, sortBy, sortOrder },
    });

    console.log("Raw API response:", res);
    if (res.status === 200) {
      console.log("✅ BMKG API response:", res.data.data);
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
    console.error("❌ Error fetching BMKG data:", error);
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

export const createBmkgData = async (data: any) => {
  const res = await axios.post("/api/v1/bmkg", data);
  return res.data.data;
};

export async function getBuoysData(page = 1, pageSize = 10, sortBy = "Date", sortOrder = "desc") {
  console.log("[Client] Fetching buoys:", { page, pageSize, sortBy, sortOrder });
  try {
    const res = await axios.get("/api/v1/buoys", {
      params: {
        page,
        pageSize,
        sortBy,
        sortOrder,
      },
    });

    console.log("Raw API response:", res);
    if (res.status === 200) {
      console.log("✅ Buoys API response:", res.data.data);
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
    console.error("❌ Error fetching Buoys data:", error);
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

export const searchFiles = async (search: string) => {
  if (!search) return [];

  const res = await axios.get("/api/v1/files", {
    params: {
      search,
    },
  });

  return res.data.data;
};

//
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

export const GetAllDatasetMeta = async (): Promise<DatasetMetaType[]> => {
  try {
    const res = await axios.get("/api/v1/dataset-meta");
    return res.data.data;
  } catch (error) {
    console.error("Get all dataset meta error:", error);
    throw error;
  }
};

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
