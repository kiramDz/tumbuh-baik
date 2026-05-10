import axios from "axios";
import { api } from "../better-auth/axios";
import {
  SchedulerStatus,
  LogsPagination,
  TriggerRequest,
  TriggerResponse,
  DatasetConfig,
} from "@/types/scheduler";

/**
 * Mendapatkan status scheduler (publik, tidak perlu auth)
 */
export const getSchedulerStatus = async (): Promise<SchedulerStatus> => {
  try {
    const res = await api.get("/api/v1/scheduler/status");
    // Flask membungkus data di res.data.data
    return res.data.data || res.data;
  } catch (error) {
    console.error("Error fetching scheduler status:", error);
    throw error;
  }
};

/**
 * Mendapatkan logs scheduler (publik, tidak perlu auth)
 */
export const getSchedulerLogs = async (
  limit = 10,
  offset = 0,
  status?: string,
): Promise<LogsPagination> => {
  try {
    const params: any = { limit, offset };
    if (status) params.status = status;

    const res = await api.get("/api/v1/scheduler/logs", { params });
    return res.data.data || res.data;
  } catch (error) {
    console.error("Error fetching scheduler logs:", error);
    throw error;
  }
};

/**
 * Trigger manual scheduler (membutuhkan auth)
 */
export const triggerScheduler = async (
  request: TriggerRequest,
): Promise<TriggerResponse> => {
  try {
    const payload = {
      mode: request.mode || "quick",
      tasks: request.tasks || [],
      datasets: request.datasets || {},
      async: request.async ?? true,
    };

    const res = await api.post("/api/v1/scheduler/trigger", payload);
    return res.data.data || res.data;
  } catch (error: any) {
    console.error("Error triggering scheduler:", error);
    if (error.response?.data) {
      throw new Error(
        error.response.data.error?.message || "Failed to trigger scheduler",
      );
    }
    throw error;
  }
};

/**
 * Mendapatkan daftar dataset yang tersedia (NASA dan BMKG) dari Next.js internal API
 * Gunakan axios biasa karena bukan ke Flask
 */
export const getAvailableDatasets = async (): Promise<{
  nasa_raw: DatasetConfig[];
  bmkg_raw: DatasetConfig[];
}> => {
  try {
    const baseUrl = process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000";
    // Menggunakan axios biasa (bukan instance api) untuk memanggil API Next.js sendiri
    const [nasaRes, bmkgRes] = await Promise.all([
      axios.get(
        `${baseUrl}/api/v1/dataset-meta?dataType=nasa&status=raw,latest`,
      ),
      axios.get(`${baseUrl}/api/v1/dataset-meta?dataType=bmkg&status=raw`),
    ]);

    return {
      nasa_raw: nasaRes.data.data || [],
      bmkg_raw: bmkgRes.data.data || [],
    };
  } catch (error) {
    console.error("Error fetching available datasets:", error);
    throw error;
  }
};

/**
 * Mendapatkan konfigurasi automation (membutuhkan auth)
 */
export const getAutomationConfig = async (): Promise<any> => {
  try {
    const res = await api.get("/api/v1/scheduler/config");
    return res.data.data;
  } catch (error) {
    console.error("Error getting config:", error);
    throw error;
  }
};

/**
 * Menyimpan konfigurasi automation (membutuhkan auth)
 */
export const saveAutomationConfig = async (configData: any): Promise<any> => {
  try {
    const res = await api.post("/api/v1/scheduler/config", configData);
    return res.data;
  } catch (error) {
    console.error("Error saving config:", error);
    throw error;
  }
};
