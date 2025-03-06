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

export const searchFiles = async (search: string) => {
  if (!search) return [];

  const res = await axios.get("/api/v1/files", {
    params: {
      search,
    },
  });

  return res.data.data;
};
