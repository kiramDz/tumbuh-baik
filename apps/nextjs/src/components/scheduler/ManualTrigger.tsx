"use client";

import { useEffect, useState, useCallback } from "react";
import {
  getAvailableDatasets,
  triggerScheduler,
} from "@/lib/fetch/scheduler.fetch";
import { DatasetConfig } from "@/types/scheduler";
import toast from "react-hot-toast";
import { Icons } from "@/app/dashboard/_components/icons";

type Mode = "quick" | "custom";

interface SelectedDatasets {
  nasa_refresh: string[];
  nasa_preprocess: string[];
  bmkg_preprocess: string[];
}

export default function ManualTrigger({
  isSchedulerRunning = false,
}: {
  isSchedulerRunning?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [mode, setMode] = useState<Mode>("quick");

  // Datasets fetching states
  const [availableDatasets, setAvailableDatasets] = useState<{
    nasa: DatasetConfig[];
    bmkg: DatasetConfig[];
  }>({ nasa: [], bmkg: [] });
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  // Execution states
  const [isTriggering, setIsTriggering] = useState(false);
  const [triggerError, setTriggerError] = useState<string | null>(null);

  // Selection states
  const [selected, setSelected] = useState<SelectedDatasets>({
    nasa_refresh: [],
    nasa_preprocess: [],
    bmkg_preprocess: [],
  });

  // Handle ESC to close
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !isTriggering) setIsOpen(false);
    };
    if (isOpen) window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, [isOpen, isTriggering]);

  const loadDatasets = useCallback(async () => {
    setIsLoadingDatasets(true);
    setFetchError(null);
    try {
      const data = await getAvailableDatasets();
      setAvailableDatasets({ nasa: data.nasa_raw, bmkg: data.bmkg_raw });

      // Auto-select all by default for convenience
      setSelected({
        nasa_refresh: data.nasa_raw.map((d) => d.collectionName),
        nasa_preprocess: data.nasa_raw.map((d) => d.collectionName),
        bmkg_preprocess: data.bmkg_raw.map((d) => d.collectionName),
      });
    } catch (err: any) {
      setFetchError(err.message || "Failed to load datasets");
    } finally {
      setIsLoadingDatasets(false);
    }
  }, []);

  // Fetch when modal opens and custom is selected (or pre-fetch when opened)
  useEffect(() => {
    if (
      isOpen &&
      availableDatasets.nasa.length === 0 &&
      availableDatasets.bmkg.length === 0
    ) {
      loadDatasets();
    }
  }, [isOpen, loadDatasets, availableDatasets]);

  const toggleDataset = (category: keyof SelectedDatasets, name: string) => {
    setSelected((prev) => {
      const current = prev[category];
      if (current.includes(name)) {
        return { ...prev, [category]: current.filter((n) => n !== name) };
      } else {
        return { ...prev, [category]: [...current, name] };
      }
    });
  };

  const selectAllInCategory = (
    category: keyof SelectedDatasets,
    list: string[],
    selectAll: boolean,
  ) => {
    setSelected((prev) => ({
      ...prev,
      [category]: selectAll ? list : [],
    }));
  };

  const getSelectedCount = () => {
    return (
      selected.nasa_refresh.length +
      selected.nasa_preprocess.length +
      selected.bmkg_preprocess.length
    );
  };

  const estimateDuration = (count: number) => {
    if (mode === "quick") return "3-8 minutes";
    if (count === 0) return "0 minutes";
    if (count <= 2) return "1-3 minutes";
    if (count <= 5) return "3-8 minutes";
    return "8-15+ minutes";
  };

  const handleRunNow = async () => {
    setTriggerError(null);
    setIsTriggering(true);

    try {
      const requestPayload =
        mode === "quick"
          ? {
              mode: "quick" as Mode,
              tasks: [
                "nasa_refresh",
                "nasa_preprocess",
                "bmkg_preprocess",
              ] as any,
              async: true,
            }
          : { mode: "custom" as Mode, datasets: selected, async: true };

      const res = await triggerScheduler(requestPayload);

      toast.success(
        <div>
          <strong>Scheduler Started</strong>
          <div className="text-sm">Execution ID: {res.executionId}</div>
        </div>,
        { duration: 5000 },
      );
      setIsOpen(false);
    } catch (err: any) {
      setTriggerError(err.message || "Failed to start scheduler");
    } finally {
      setIsTriggering(false);
    }
  };

  const renderCheckboxList = (
    title: string,
    category: keyof SelectedDatasets,
    availableList: DatasetConfig[],
  ) => {
    const listNames = availableList.map((d) => d.collectionName);
    const isAllSelected =
      listNames.length > 0 && selected[category].length === listNames.length;

    return (
      <div className="mb-4 bg-gray-50 border rounded-lg p-3">
        <div className="flex justify-between items-center mb-2 border-b pb-2">
          <h4 className="font-semibold text-gray-700 text-sm">{title}</h4>
          <div className="space-x-2 text-xs">
            <button
              onClick={() => selectAllInCategory(category, listNames, true)}
              className="text-blue-600 hover:underline"
            >
              Select All
            </button>
            <span className="text-gray-300">|</span>
            <button
              onClick={() => selectAllInCategory(category, listNames, false)}
              className="text-gray-500 hover:underline"
            >
              Deselect All
            </button>
          </div>
        </div>

        {availableList.length === 0 ? (
          <p className="text-xs text-gray-500 italic py-1">
            No datasets available
          </p>
        ) : (
          <div className="space-y-2 max-h-40 overflow-y-auto pr-2 custom-scrollbar">
            {availableList.map((ds) => (
              <label
                key={ds.collectionName}
                className="flex items-center space-x-2 cursor-pointer group"
              >
                <input
                  type="checkbox"
                  checked={selected[category].includes(ds.collectionName)}
                  onChange={() => toggleDataset(category, ds.collectionName)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
                />
                <span className="text-sm text-gray-700 group-hover:text-gray-900 truncate">
                  {ds.collectionName}
                  {ds.status && (
                    <span className="ml-2 text-xs text-gray-400">
                      ({ds.status})
                    </span>
                  )}
                </span>
              </label>
            ))}
          </div>
        )}
      </div>
    );
  };

  const isValid = mode === "quick" || getSelectedCount() > 0;

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        disabled={isSchedulerRunning}
        className="w-full flex justify-center items-center px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        <Icons.play className="w-4 h-4 mr-2" />
        {isSchedulerRunning ? "Scheduler is Running..." : "Run Scheduler Now"}
      </button>

      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-md max-h-[90vh] flex flex-col overflow-hidden animate-in fade-in zoom-in duration-200">
            {/* Header */}
            <div className="flex justify-between items-center px-5 py-4 border-b">
              <h2 className="text-lg font-bold text-gray-900">
                Manual Scheduler Execution
              </h2>
              <button
                onClick={() => setIsOpen(false)}
                disabled={isTriggering}
                className="text-gray-400 hover:text-gray-700 disabled:opacity-50"
              >
                <Icons.closeX className="w-5 h-5" />
              </button>
            </div>

            {/* Body */}
            <div className="p-5 overflow-y-auto flex-grow">
              {/* Mode Toggle */}
              <div className="flex p-1 mb-5 bg-gray-100 rounded-lg space-x-1">
                <button
                  onClick={() => setMode("quick")}
                  className={`flex-1 py-1.5 text-sm font-medium rounded-md transition-colors ${mode === "quick" ? "bg-white shadow text-gray-900" : "text-gray-500 hover:text-gray-700"}`}
                >
                  Quick Run
                </button>
                <button
                  onClick={() => setMode("custom")}
                  className={`flex-1 py-1.5 text-sm font-medium rounded-md transition-colors ${mode === "custom" ? "bg-white shadow text-gray-900" : "text-gray-500 hover:text-gray-700"}`}
                >
                  Custom Selection
                </button>
              </div>

              {triggerError && (
                <div className="mb-4 p-3 bg-red-50 text-red-700 border border-red-200 rounded-lg text-sm">
                  <strong>Failed to start:</strong> {triggerError}
                </div>
              )}

              {/* Quick Mode Content */}
              {mode === "quick" && (
                <div className="text-sm text-gray-600 space-y-4">
                  <div className="p-4 border rounded-lg bg-blue-50/50 border-blue-100">
                    <h3 className="font-semibold text-blue-900 flex items-center mb-2">
                      <Icons.checkSquare className="w-4 h-4 mr-2" /> Run all
                      available tasks:
                    </h3>
                    <ul className="list-disc pl-8 space-y-1 text-blue-800">
                      <li>Refresh all NASA datasets</li>
                      <li>Preprocess all NASA datasets</li>
                      <li>Preprocess all pending BMKG datasets</li>
                    </ul>
                  </div>
                </div>
              )}

              {/* Custom Mode Content */}
              {mode === "custom" && (
                <div>
                  {isLoadingDatasets ? (
                    <div className="flex flex-col items-center justify-center py-10 text-gray-500">
                      <Icons.spinner className="w-8 h-8 animate-spin mb-2" />
                      <p className="text-sm">Loading available datasets...</p>
                    </div>
                  ) : fetchError ? (
                    <div className="text-center py-4 text-red-500 text-sm">
                      <p>{fetchError}</p>
                      <button
                        onClick={loadDatasets}
                        className="mt-2 text-blue-600 underline"
                      >
                        Try Again
                      </button>
                    </div>
                  ) : (
                    <>
                      {renderCheckboxList(
                        "NASA Datasets to Refresh",
                        "nasa_refresh",
                        availableDatasets.nasa,
                      )}
                      {renderCheckboxList(
                        "NASA Datasets to Preprocess",
                        "nasa_preprocess",
                        availableDatasets.nasa,
                      )}
                      {renderCheckboxList(
                        "BMKG Datasets to Preprocess",
                        "bmkg_preprocess",
                        availableDatasets.bmkg,
                      )}
                    </>
                  )}
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="px-5 py-4 border-t bg-gray-50 flex flex-col sm:flex-row justify-between items-center gap-3">
              <div className="text-xs text-gray-500 font-medium">
                {mode === "custom" && (
                  <span>Selected: {getSelectedCount()} datasets. </span>
                )}
                Estimated: {estimateDuration(getSelectedCount())}
              </div>
              <div className="flex gap-2 w-full sm:w-auto">
                <button
                  onClick={() => setIsOpen(false)}
                  disabled={isTriggering}
                  className="px-4 py-2 w-full sm:w-auto text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  onClick={handleRunNow}
                  disabled={isTriggering || !isValid}
                  className="px-4 py-2 w-full sm:w-auto flex items-center justify-center text-sm font-medium text-white bg-blue-600 border border-transparent rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:bg-blue-400"
                >
                  {isTriggering && (
                    <Icons.spinner className="w-4 h-4 mr-2 animate-spin" />
                  )}
                  {isTriggering ? "Starting..." : "Run Now"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Basic responsive custom scrollbar styles injected */}
      <style
        dangerouslySetInnerHTML={{
          __html: `
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }
      `,
        }}
      />
    </>
  );
}
