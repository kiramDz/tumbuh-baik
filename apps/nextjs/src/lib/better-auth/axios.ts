import axios from "axios";

export const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_FLASK_API_URL,
  withCredentials: true,
  headers: {
    "Content-Type": "application/json",
  },
});

api.interceptors.request.use((config) => {
  config.headers["ngrok-skip-browser-warning"] = "69420";

  return config;
});
