import axios from "axios";

export async function getFiles({ currentPage, category }: { category: string; currentPage: number }) {
  // if (page === "subscription") {
  //   return { files: [] };
  // }

  const res = await axios.get(`/api/v1/files/${category}`, {
    params: {
      page: currentPage,
    },
  });

  return res.status === 200 ? res.data.data : { files: [] };
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
