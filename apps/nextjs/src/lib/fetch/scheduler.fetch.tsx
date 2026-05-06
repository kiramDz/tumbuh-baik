import axios from "axios";
import { FLASK_API_URL } from "../env";
import {
  SchedulerStatus,
  LogsPagination,
  TriggerRequest,
  TriggerResponse,
  DatasetConfig,
} from "@/types/scheduler";

export const getSchedulerStatus = async (): Promise<SchedulerStatus> => {
  try {
    const res = await axios.get(`${FLASK_API_URL}/api/v1/scheduler/status`, {
      withCredentials: true,
    });
    // Menyesuaikan apabila Flask membungkus data dalam properti "data"
    return res.data.data || res.data;
  } catch (error) {
    console.error("Error fetching scheduler status:", error);
    throw error;
  }
};

export const getSchedulerLogs = async (
  limit = 10,
  offset = 0,
  status?: string,
): Promise<LogsPagination> => {
  try {
    const params: any = { limit, offset };
    if (status) params.status = status;

    const res = await axios.get(`${FLASK_API_URL}/api/v1/scheduler/logs`, {
      params,
      withCredentials: true,
    });
    return res.data.data || res.data;
  } catch (error) {
    console.error("Error fetching scheduler logs:", error);
    throw error;
  }
};

// Diperbarui menerima objek TriggerRequest untuk mengakomodasi mode Custom/Quick
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

    const res = await axios.post(
      `${FLASK_API_URL}/api/v1/scheduler/trigger`,
      payload,
      {
        withCredentials: true,
      },
    );
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
 * Endpoint baru untuk mengambil dataset yang tersedia (NASA dan BMKG)
 * Untuk keperluan Manual Trigger Custom Selection
 */
export const getAvailableDatasets = async (): Promise<{
  nasa_raw: DatasetConfig[];
  bmkg_raw: DatasetConfig[];
}> => {
  try {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3000";

    // Menembak endpoint Next.js secara paralel (Promise.all) agar lebih cepat
    const [nasaRes, bmkgRes] = await Promise.all([
      axios.get(
        `${baseUrl}/api/v1/dataset-meta?dataType=nasa&status=raw,latest`,
        {
          withCredentials: true,
        },
      ),
      axios.get(`${baseUrl}/api/v1/dataset-meta?dataType=bmkg&status=raw`, {
        withCredentials: true,
      }),
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

export const getAutomationConfig = async (): Promise<any> => {
  try {
    const res = await axios.get(`${FLASK_API_URL}/api/v1/scheduler/config`, {
      withCredentials: true,
    });
    return res.data.data;
  } catch (error) {
    console.error("Error getting config:", error);
    throw error;
  }
};

export const saveAutomationConfig = async (configData: any): Promise<any> => {
  try {
    const res = await axios.post(
      `${FLASK_API_URL}/api/v1/scheduler/config`,
      configData,
      {
        withCredentials: true,
      },
    );
    return res.data;
  } catch (error) {
    console.error("Error saving config:", error);
    throw error;
  }
};
