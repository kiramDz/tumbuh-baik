import axios from "axios";

export async function getFiles({ currentPage, category }: { category: string; currentPage: number }) {
  const res = await axios.get(`/api/v1/files/${category}`, {
    params: {
      page: currentPage,
    },
  });

  return res.status === 200 ? res.data.data : { files: [] };
}

export async function getRecentFiles() {
  try {
    const res = await axios.get("/api/v1/files/recent");
    if (res.status === 200) {
      console.log("API Response:", res.data);
      return res.data.data || { files: [] };
    }
  } catch (error) {
    console.error("Error fetching recent files:", error);
    return { files: [] };
  }
}

export const getBmkgApi = async () => {
  try {
    const response = await axios.get("/api/v1/bmkg-api/all");
    return response.data;
  } catch (error: any) {
    console.error("Error fetching BMKG API data:", error);
    throw new Error(error?.response?.data?.description || "Failed to fetch BMKG data");
  }
};

export async function getBmkgData(page = 1) {
  try {
    const res = await axios.get(`/api/v1/bmkg?page=${page}`);
    console.log("Raw API response:", res);
    if (res.status === 200) {
      console.log("BMKG API Response:", res.data);
      console.log("Items specifically:", res.data.data?.items);
      return res.data.data || { items: [], total: 0, currentPage: 1, totalPages: 1 };
    }
  } catch (error) {
    console.error("Error fetching BMKG data:", error);
    return { items: [], total: 0, currentPage: 1, totalPages: 1 };
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
