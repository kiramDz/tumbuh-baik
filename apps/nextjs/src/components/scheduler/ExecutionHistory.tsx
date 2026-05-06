"use client";

import { useEffect, useState, useCallback, Fragment } from "react";
import { getSchedulerLogs } from "@/lib/fetch/scheduler.fetch";
import { LogsPagination, SchedulerLog } from "@/types/scheduler";
import { Icons } from "@/app/dashboard/_components/icons";

export default function ExecutionHistory() {
  const [data, setData] = useState<LogsPagination | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filter and pagination
  const limit = 10;
  const [offset, setOffset] = useState(0);
  const [statusFilter, setStatusFilter] = useState<string | undefined>("");
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());

  const fetchLogs = useCallback(async () => {
    setLoading(true);
    try {
      const result = await getSchedulerLogs(
        limit,
        offset,
        statusFilter || undefined,
      );
      setData(result);
      setError(null);
    } catch (err: any) {
      setError(err.message || "Failed to load logs");
    } finally {
      setLoading(false);
    }
  }, [offset, statusFilter, limit]);

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  const toggleRow = (id: string) => {
    const newDocs = new Set(expandedRows);
    if (newDocs.has(id)) newDocs.delete(id);
    else newDocs.add(id);
    setExpandedRows(newDocs);
  };

  const handleRefresh = () => fetchLogs();

  const handleStatusChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setStatusFilter(e.target.value);
    setOffset(0); // Reset pagination
    setExpandedRows(new Set());
  };

  const formatDuration = (seconds?: number | null) => {
    if (seconds == null) return "-";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
  };

  const formatDateTime = (iso?: string) => {
    if (!iso) return "-";
    return new Date(iso).toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getStatusBadge = (status: string) => {
    const styles: Record<string, string> = {
      success: "bg-green-100 text-green-800",
      failed: "bg-red-100 text-red-800",
      partial: "bg-yellow-100 text-yellow-800",
      running: "bg-blue-100 text-blue-800 animate-pulse",
    };
    return (
      <span
        className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold capitalize ${styles[status] || "bg-gray-100 text-gray-800"}`}
      >
        {status}
      </span>
    );
  };

  return (
    <div className="bg-white border rounded-xl shadow-sm flex flex-col">
      {/* Header & Controls */}
      <div className="p-4 border-b flex flex-wrap gap-4 justify-between items-center bg-gray-50 rounded-t-xl">
        <h2 className="text-lg font-semibold text-gray-800">
          Execution History
        </h2>

        <div className="flex items-center gap-3">
          <select
            value={statusFilter}
            onChange={handleStatusChange}
            className="border-gray-300 rounded-md text-sm shadow-sm focus:ring-blue-500 focus:border-blue-500 bg-white px-3 py-1.5"
          >
            <option value="">All Status</option>
            <option value="success">Success</option>
            <option value="failed">Failed</option>
            <option value="partial">Partial</option>
            <option value="running">Running</option>
          </select>

          <button
            onClick={handleRefresh}
            disabled={loading}
            className="p-1.5 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded-md transition-colors disabled:opacity-50"
            aria-label="Refresh logs"
          >
            <Icons.refresh
              className={`w-5 h-5 ${loading ? "animate-spin" : ""}`}
            />
          </button>
        </div>
      </div>
      {/* Table Area */}
      <div className="overflow-x-auto min-h-[300px]">
        {error ? (
          <div className="flex flex-col items-center justify-center p-12 text-center h-full">
            <Icons.alert className="w-10 h-10 text-red-500 mb-3" />
            <p className="text-gray-800 font-medium">{error}</p>
            <button
              onClick={handleRefresh}
              className="mt-3 text-blue-600 font-medium hover:underline"
            >
              Try Again
            </button>
          </div>
        ) : loading && !data ? (
          <div className="p-6 space-y-4">
            {[...Array(5)].map((_, i) => (
              <div
                key={i}
                className="h-12 bg-gray-100 animate-pulse rounded-lg w-full"
              ></div>
            ))}
          </div>
        ) : data?.logs.length === 0 ? (
          <div className="flex flex-col flex-grow items-center justify-center p-12 text-center h-full text-gray-500">
            <Icons.refresh className="w-12 h-12 mb-4 text-gray-300" />
            <p className="text-lg font-medium text-gray-700">
              No execution history yet
            </p>
            <p className="text-sm mt-1">Run the scheduler to see logs here.</p>
          </div>
        ) : (
          <table className="w-full text-left border-collapse text-sm">
            <thead>
              <tr className="bg-white border-b text-gray-600 uppercase tracking-wider text-xs">
                <th className="p-4 font-medium w-1/4">Date & Time</th>
                <th className="p-4 font-medium w-[15%]">Duration</th>
                <th className="p-4 font-medium w-[15%]">Status</th>
                <th className="p-4 font-medium w-[20%]">Tasks</th>
                <th className="p-4 font-medium w-1/4 text-right">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {data?.logs.map((log: SchedulerLog) => {
                const isExpanded = expandedRows.has(log._id);
                return (
                  <Fragment key={log._id}>
                    <tr
                      className={`hover:bg-gray-50 transition-colors cursor-pointer ${isExpanded ? "bg-blue-50/50" : ""}`}
                      onClick={() => toggleRow(log._id)}
                    >
                      <td className="p-4 whitespace-nowrap text-gray-900 font-medium">
                        {formatDateTime(log.executedAt)}
                      </td>
                      <td className="p-4 whitespace-nowrap text-gray-600">
                        {formatDuration(log.duration)}
                      </td>
                      <td className="p-4 whitespace-nowrap">
                        {getStatusBadge(log.status)}
                      </td>
                      <td className="p-4 whitespace-nowrap text-gray-600">
                        {log.datasetsUpdated}/{log.totalDatasets} updated
                      </td>
                      <td className="p-4 whitespace-nowrap text-right">
                        <button className="text-blue-600 hover:text-blue-800 font-medium inline-flex items-center">
                          {isExpanded ? (
                            <>
                              <Icons.down className="w-4 h-4 mr-1" /> Hide
                            </>
                          ) : (
                            <>
                              <Icons.next className="w-4 h-4 mr-1" /> View
                            </>
                          )}
                        </button>
                      </td>
                    </tr>

                    {/* Expanded Content */}
                    {isExpanded && (
                      <tr className="bg-gray-50 border-b">
                        <td colSpan={5} className="p-0">
                          <div className="px-6 py-4 border-l-4 border-blue-400 m-4 bg-white rounded shadow-sm">
                            <h4 className="font-semibold text-gray-800 mb-3">
                              Task Breakdown:
                            </h4>
                            {!log.tasks || log.tasks.length === 0 ? (
                              <p className="text-gray-500 italic">
                                No tasks executed or recorded yet.
                              </p>
                            ) : (
                              <div className="space-y-4">
                                {log.tasks.map((task, idx) => (
                                  <div key={idx} className="text-sm">
                                    <div className="flex items-center gap-2 font-medium text-gray-700 capitalize">
                                      {getStatusBadge(task.status)}
                                      {task.name.replace("_", " ")}
                                    </div>
                                    <ul className="mt-1 ml-6 list-disc text-gray-600">
                                      {task.recordsUpdated !== undefined && (
                                        <li>
                                          Records updated: {task.recordsUpdated}
                                        </li>
                                      )}
                                      {task.recordsFailed !== undefined &&
                                        task.recordsFailed > 0 && (
                                          <li className="text-red-500">
                                            Records failed: {task.recordsFailed}
                                          </li>
                                        )}
                                      {task.errorMessage && (
                                        <li className="text-red-500 break-words">
                                          Error: {task.errorMessage}
                                        </li>
                                      )}
                                      {!task.recordsUpdated &&
                                        !task.errorMessage && (
                                          <li className="italic text-gray-400">
                                            Execution recorded.
                                          </li>
                                        )}
                                    </ul>
                                  </div>
                                ))}
                              </div>
                            )}

                            {log.errors && log.errors.length > 0 && (
                              <div className="mt-4 pt-3 border-t">
                                <h5 className="font-medium text-red-700">
                                  Critical Errors:
                                </h5>
                                <ul className="list-disc ml-5 mt-1 text-red-600 text-sm space-y-1">
                                  {log.errors.map((e, i) => (
                                    <li key={i}>{e}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination Footer */}
      {!loading && data && data.total > 0 && (
        <div className="p-4 border-t flex flex-wrap gap-4 items-center justify-between bg-white rounded-b-xl">
          <p className="text-sm text-gray-600">
            Showing <span className="font-medium">{offset + 1}</span> to{" "}
            <span className="font-medium">
              {Math.min(offset + limit, data.total)}
            </span>{" "}
            of <span className="font-medium">{data.total}</span> entries
          </p>
          <div className="inline-flex rounded-md shadow-sm">
            <button
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - limit))}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-l-lg hover:bg-gray-50 focus:z-10 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            <button
              disabled={!data.hasMore}
              onClick={() => setOffset(offset + limit)}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-l-0 border-gray-300 rounded-r-lg hover:bg-gray-50 focus:z-10 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
