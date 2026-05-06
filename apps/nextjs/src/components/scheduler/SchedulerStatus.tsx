"use client";
import { useEffect, useState, useCallback } from "react";
import { getSchedulerStatus } from "@/lib/fetch/scheduler.fetch";
import { SchedulerStatus as SchedulerStatusType } from "@/types/scheduler";
import { Icons } from "@/app/dashboard/_components/icons";

export default function SchedulerStatus() {
  const [data, setData] = useState<SchedulerStatusType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const result = await getSchedulerStatus();
      setData(result);
    } catch (err: any) {
      setError(err.message || "Failed to load status");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000); // Auto-refresh 30
    return () => clearInterval(interval); // Cleanup
  }, [fetchData]);

  const formatDuration = (seconds?: number | null) => {
    if (seconds == null) return "-";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
  };

  const formatDateTime = (isoString?: string) => {
    if (!isoString) return "-";
    return new Date(isoString).toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      timeZoneName: "short",
    });
  };

  const getCountdown = (isoString?: string) => {
    if (!isoString) return "";
    const diff = new Date(isoString).getTime() - Date.now();
    if (diff <= 0) return "running soon";
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const mins = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    return hours > 0 ? `in ${hours}h ${mins}m` : `in ${mins} minutes`;
  };

  if (loading && !data) {
    return (
      <div className="bg-white border rounded-xl shadow-sm p-6 animate-pulse">
        <div className="h-6 w-1/3 bg-gray-200 rounded mb-4"></div>
        <div className="space-y-4">
          <div className="h-4 bg-gray-200 rounded w-full"></div>
          <div className="h-4 bg-gray-200 rounded w-5/6"></div>
          <div className="h-4 bg-gray-200 rounded w-4/6"></div>
        </div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="bg-white border border-red-200 rounded-xl shadow-sm p-6 text-center">
        <Icons.alert className="mx-auto h-8 w-8 text-red-500 mb-3" />
        <h3 className="text-lg font-medium text-gray-900 mb-1">Status Error</h3>
        <p className="text-gray-500 mb-4">{error}</p>
        <button
          onClick={fetchData}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg inline-flex items-center"
        >
          <Icons.refresh className="w-4 h-4 mr-2" /> Retry
        </button>
      </div>
    );
  }

  if (!data) return null; // Should not happen

  const isRunning = (data.lastRun?.status as string) === "running";
  const lastRunStatusColor =
    data.lastRun?.status === "success"
      ? "text-green-600 bg-green-50"
      : data.lastRun?.status === "failed"
        ? "text-red-600 bg-red-50"
        : (data.lastRun?.status as string) === "running"
          ? "text-blue-600 bg-blue-50"
          : "text-yellow-600 bg-yellow-50";

  return (
    <div className="bg-white border rounded-xl shadow-sm overflow-hidden flex flex-col h-full">
      <div className="p-5 border-b flex justify-between items-center bg-gray-50">
        <h2 className="text-lg font-semibold text-gray-800">
          Scheduler Status
        </h2>
        <div className="flex items-center">
          {isRunning ? (
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800 animate-pulse">
              <span className="w-2 h-2 mr-2 bg-blue-600 rounded-full"></span>{" "}
              Running
            </span>
          ) : data.isActive ? (
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
              <span className="w-2 h-2 mr-2 bg-green-600 rounded-full"></span>{" "}
              Active
            </span>
          ) : (
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-100 text-gray-800">
              <span className="w-2 h-2 mr-2 bg-gray-500 rounded-full"></span>{" "}
              Inactive
            </span>
          )}
        </div>
      </div>

      <div className="p-5 flex-grow grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Last Run */}
        <div className="space-y-2">
          <p className="text-sm font-medium text-gray-500 uppercase flex items-center">
            <Icons.checked className="w-4 h-4 mr-1" /> Last Run
          </p>
          <div className="font-medium text-gray-900">
            {formatDateTime(data.lastRun?.executedAt)}
          </div>
          <div className="text-sm text-gray-600">
            Duration: {formatDuration(data.lastRun?.duration)}
          </div>
          {data.lastRun && (
            <span
              className={`inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-semibold capitalize ${lastRunStatusColor}`}
            >
              {data.lastRun.status} • {data.lastRun.datasetsUpdated}/
              {data.lastRun.totalDatasets} datasets
            </span>
          )}
        </div>

        {/* Next Run */}
        <div className="space-y-2">
          <p className="text-sm font-medium text-gray-500 uppercase flex items-center">
            <Icons.clock className="w-4 h-4 mr-1" /> Next Run
          </p>
          <div className="font-medium text-gray-900">
            {formatDateTime(data.nextRun)}
          </div>
          <div className="text-sm text-blue-600 font-medium">
            ({getCountdown(data.nextRun)})
          </div>
        </div>

        {/* Statistics */}
        <div className="space-y-3">
          <p className="text-sm font-medium text-gray-500 uppercase flex items-center">
            <Icons.activity className="w-4 h-4 mr-1" /> Performance (30 Days)
          </p>
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="font-medium text-gray-700">Success Rate</span>
              <span className="font-bold">{data.statistics.successRate}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`bg-green-500 h-2 rounded-full ${data.statistics.successRate < 80 ? "bg-yellow-500" : ""}`}
                style={{ width: `${data.statistics.successRate}%` }}
              ></div>
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {data.statistics.totalExecutions -
                data.statistics.failedExecutions}{" "}
              / {data.statistics.totalExecutions} successful runs
            </div>
          </div>
          <div className="text-sm text-gray-600">
            Avg Duration:{" "}
            <strong>{formatDuration(data.statistics.avgDuration)}</strong>
          </div>
        </div>
      </div>
    </div>
  );
}
