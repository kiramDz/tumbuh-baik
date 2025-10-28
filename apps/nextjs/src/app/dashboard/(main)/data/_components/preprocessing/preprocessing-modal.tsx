import React, { useState, useEffect, useRef, useCallback } from "react";
import { Icons } from "@/app/dashboard/_components/icons";
import { preprocessNasaDatasetWithStream } from "@/lib/fetch/files.fetch";

interface LogEntry {
  type: "log" | "progress" | "error" | "complete" | "info" | "success";
  level?: string;
  message: string;
  timestamp?: number;
  percentage?: number;
  stage?: string;
}
interface PreprocessingResult {
  recordCount?: number;
  cleanedCollection?: string;
  preprocessing_report?: {
    outliers?: {
      total_outliers: number;
    };
    quality_metrics?: {
      completeness_percentage: number;
    };
  };
}

interface PreprocessingModalProps {
  collectionName: string;
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: (result: any) => void;
}

// Constants
const MAX_LOG_ENTRIES = 1000;
const CLOSE_DELAY = 300;

export default function PreprocessingModal({
  collectionName,
  isOpen,
  onClose,
  onSuccess,
}: PreprocessingModalProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [progress, setProgress] = useState<number>(0);
  const [currentStage, setCurrentStage] = useState("");
  const [status, setStatus] = useState<
    "processing" | "success" | "error" | "idle"
  >("idle");
  const [result, setResult] = useState<PreprocessingResult | null>(null);
  const [isAutoScroll, setIsAutoScroll] = useState(true);
  const [isClosing, setIsClosing] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const logsEndRef = useRef<HTMLDivElement | null>(null);
  const logsContainerRef = useRef<HTMLDivElement | null>(null);
  const closeTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup function
  const cleanup = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    if (closeTimeoutRef.current) {
      clearTimeout(closeTimeoutRef.current);
      closeTimeoutRef.current = null;
    }
  }, []);

  // Auto-scroll with user control
  const scrollToBottom = useCallback(() => {
    if (isAutoScroll) {
      logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [isAutoScroll]);

  useEffect(() => {
    scrollToBottom();
  }, [logs, scrollToBottom]);

  // Start preprocessing when modal opens
  useEffect(() => {
    if (isOpen && status === "idle" && !isClosing) {
      startPreprocessing();
    }

    // Cleanup on unmount only
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, [isOpen]); // ← Only depend on isOpen

  // Detect manual scroll - disable auto-scroll
  const handleLogScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const element = e.currentTarget;
    const isAtBottom =
      element.scrollHeight - element.scrollTop - element.clientHeight < 50;
    setIsAutoScroll(isAtBottom);
  }, []);

  const addLog = useCallback(
    (type: LogEntry["type"], message: string, level: string) => {
      const newLog: LogEntry = {
        type,
        message,
        level,
        timestamp: Date.now(),
      };

      setLogs((prev) => {
        const updated = [...prev, newLog];
        // Limit log entries to prevent memory issues
        if (updated.length > MAX_LOG_ENTRIES) {
          return updated.slice(-MAX_LOG_ENTRIES);
        }
        return updated;
      });
    },
    []
  );
  const startPreprocessing = () => {
    setStatus("processing");
    setLogs([]);
    setProgress(0);
    setCurrentStage("Connecting...");
    setIsAutoScroll(true);
    setResult(null);

    try {
      // Use the existing function from files.fetch.tsx
      const eventSource = preprocessNasaDatasetWithStream(
        collectionName,
        // onLog callback
        (logData: any) => {
          if (logData.type === "info") {
            addLog("info", logData.message, "INFO");
          } else {
            addLog("log", logData.message, logData.level || "INFO");
          }
        },
        // onProgress callback
        (progressPercent: number, stage: string, message: string) => {
          const safeProgress = Math.max(0, Math.min(100, progressPercent || 0));
          setProgress(safeProgress);
          setCurrentStage(message || stage || "Processing...");
          addLog("progress", `[${safeProgress}%] ${message}`, "INFO");
        },
        // onComplete callback
        (completionResult: any) => {
          const typedResult: PreprocessingResult = {
            recordCount: completionResult?.recordCount,
            cleanedCollection: completionResult?.cleanedCollection,
            preprocessing_report: completionResult?.preprocessing_report,
          };

          setStatus("success");
          setResult(typedResult);
          addLog(
            "success",
            "✅ Preprocessing completed successfully!",
            "SUCCESS"
          );

          if (onSuccess) {
            setTimeout(() => onSuccess(typedResult), 1000);
          }
        },
        // onError callback
        (errorMessage: string) => {
          setStatus("error");
          const safeErrorMessage = errorMessage || "Unknown error occurred";
          addLog("error", `❌ Error: ${safeErrorMessage}`, "ERROR");
        }
      );

      eventSourceRef.current = eventSource;
      addLog("info", "Connecting to preprocessing server...", "INFO");
    } catch (error) {
      console.error("Failed to start preprocessing:", error);
      setStatus("error");
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Failed to start preprocessing";
      addLog("error", `❌ ${errorMessage}`, "ERROR");
    }
  };

  const handleClose = useCallback(() => {
    if (status === "processing") {
      const confirmClose = window.confirm(
        "Preprocessing is still running. Are you sure you want to close? This will stop the process."
      );
      if (!confirmClose) return;
    }

    setIsClosing(true);
    cleanup();
    onClose();

    // Reset state after close animation with proper cleanup
    closeTimeoutRef.current = setTimeout(() => {
      if (!isClosing) return; // Prevent race condition

      setStatus("idle");
      setLogs([]);
      setProgress(0);
      setCurrentStage("");
      setResult(null);
      setIsAutoScroll(true);
      setIsClosing(false);
    }, CLOSE_DELAY);
  }, [status, cleanup, onClose, isClosing]);

  const getLogColor = (level?: string) => {
    switch (level) {
      case "ERROR":
        return "text-red-600";
      case "WARNING":
        return "text-yellow-600";
      case "SUCCESS":
        return "text-green-600";
      default:
        return "text-gray-700";
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case "processing":
        return <Icons.spinner className="h-6 w-6 animate-spin text-blue-500" />;
      case "success":
        return <Icons.checked className="h-6 w-6 text-green-500" />;
      case "error":
        return <Icons.closeX className="h-6 w-6 text-red-500" />;
      default:
        return <Icons.alertCircle className="h-6 w-6 text-gray-400" />;
    }
  };

  if (!isOpen) return null;
  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="preprocessing-title"
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          handleClose();
        }
      }}
    >
      <div
        className="bg-white rounded-lg shadow-2xl w-full max-w-3xl max-h-[90vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div className="flex items-center gap-3">
            {getStatusIcon()}
            <div>
              <h2
                id="preprocessing-title"
                className="text-xl font-semibold text-gray-900"
              >
                Preprocessing NASA POWER Data
              </h2>
              <p className="text-sm text-gray-500 mt-1">{collectionName}</p>
            </div>
          </div>
          <button
            onClick={handleClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
            aria-label={status === "processing" ? "Stop and close" : "Close"}
            title={status === "processing" ? "Stop and close" : "Close"}
          >
            <Icons.closeX className="h-6 w-6" />
          </button>
        </div>

        {/* Progress Bar */}
        {status === "processing" && (
          <div className="px-6 pt-4">
            <div className="flex justify-between text-sm mb-2">
              <span className="text-gray-600">{currentStage}</span>
              <span className="font-medium text-blue-600">{progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
                role="progressbar"
                aria-valuenow={progress}
                aria-valuemin={0}
                aria-valuemax={100}
              />
            </div>
          </div>
        )}

        {/* Live Logs with Enhanced Controls */}
        <div className="flex-1 overflow-hidden p-6 min-h-0">
          <div className="h-full flex flex-col">
            {/* Scroll controls */}
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600 font-medium">
                Live Logs ({logs.length})
              </span>
              <div className="flex items-center gap-2">
                {!isAutoScroll && (
                  <button
                    onClick={() => {
                      setIsAutoScroll(true);
                      scrollToBottom();
                    }}
                    className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors flex items-center gap-1"
                  >
                    <Icons.down className="h-3 w-3" />
                    Scroll to bottom
                  </button>
                )}
                <div className="flex items-center gap-1">
                  <div
                    className={`w-2 h-2 rounded-full ${
                      isAutoScroll ? "bg-green-500" : "bg-gray-400"
                    }`}
                  />
                  <span className="text-xs text-gray-500">
                    {isAutoScroll ? "Auto-scroll ON" : "Auto-scroll OFF"}
                  </span>
                </div>
              </div>
            </div>

            {/* Logs container */}
            <div
              ref={logsContainerRef}
              onScroll={handleLogScroll}
              className="bg-gray-900 rounded-lg flex-1 overflow-y-auto p-4 font-mono text-sm"
              role="log"
              aria-live="polite"
              aria-label="Preprocessing logs"
            >
              {logs.length === 0 ? (
                <div className="text-gray-400 text-center py-8">
                  Waiting for logs...
                </div>
              ) : (
                <div className="space-y-1">
                  {logs.map((log, index) => (
                    <div
                      key={`${log.timestamp}-${index}`}
                      className={`${
                        log.level === "ERROR"
                          ? "text-red-400"
                          : log.level === "WARNING"
                          ? "text-yellow-400"
                          : log.level === "SUCCESS"
                          ? "text-green-400"
                          : "text-gray-300"
                      }`}
                    >
                      <span className="text-gray-500 mr-2">
                        {new Date(
                          log.timestamp || Date.now()
                        ).toLocaleTimeString()}
                      </span>
                      {log.message}
                    </div>
                  ))}
                  <div ref={logsEndRef} />
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Result Summary - Enhanced */}
        {status === "success" && result && (
          <div className="px-6 pb-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h3 className="font-semibold text-green-900 mb-2">
                Preprocessing Complete!
              </h3>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-gray-600">Records Processed:</span>
                  <span className="ml-2 font-medium text-gray-900">
                    {result.recordCount?.toLocaleString() ?? "N/A"}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Cleaned Collection:</span>
                  <span className="ml-2 font-medium text-gray-900 text-xs break-all">
                    {result.cleanedCollection ?? "N/A"}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Outliers Removed:</span>
                  <span className="ml-2 font-medium text-gray-900">
                    {result.preprocessing_report?.outliers?.total_outliers ?? 0}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Completeness:</span>
                  <span className="ml-2 font-medium text-gray-900">
                    {result.preprocessing_report?.quality_metrics
                      ?.completeness_percentage ?? "N/A"}
                    %
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Error Message */}
        {status === "error" && (
          <div className="px-6 pb-4">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <h3 className="font-semibold text-red-900 mb-2">
                Preprocessing Failed
              </h3>
              <p className="text-sm text-red-700">
                An error occurred during preprocessing. Check the logs above for
                details.
              </p>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="flex justify-end gap-3 p-6 border-t bg-gray-50">
          <button
            onClick={handleClose}
            disabled={isClosing}
            className={`px-4 py-2 rounded-lg transition-colors disabled:opacity-50 ${
              status === "processing"
                ? "bg-red-600 text-white hover:bg-red-700"
                : "bg-blue-600 text-white hover:bg-blue-700"
            }`}
          >
            {isClosing
              ? "Closing..."
              : status === "processing"
              ? "Stop & Close"
              : status === "success"
              ? "Done"
              : "Close"}
          </button>
        </div>
      </div>
    </div>
  );
}
