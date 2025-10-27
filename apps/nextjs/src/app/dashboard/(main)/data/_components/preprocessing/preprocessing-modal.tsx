import React, { useState, useEffect, useRef } from "react";
import { Icons } from "@/app/dashboard/_components/icons";
import { preprocessNasaDatasetWithStream } from "@/lib/fetch/files.fetch";

interface LogEntry {
  type: "log" | "progress" | "error" | "complete";
  level?: string;
  message: string;
  timestamp?: number;
  percentage?: number;
  stage?: string;
}
interface PreprocessingModalProps {
  collectionName: string;
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: (result: any) => void;
}
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
  const [result, setResult] = useState<any>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const logsEndRef = useRef<HTMLDivElement | null>(null);

  // AutoScroll logs
  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(() => {
    scrollToBottom();
  }, [logs]);
  // Start preprocessing when modal opens
  useEffect(() => {
    if (isOpen && status === "idle") {
      startPreprocessing();
    }

    // Cleanup on unmount
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [isOpen]);

  const startPreprocessing = () => {
    setStatus("processing");
    setLogs([]);
    setProgress(0);
    setCurrentStage("Connecting...");

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
          setProgress(progressPercent || 0);
          setCurrentStage(message || stage || "Processing...");
          addLog("progress", `[${progressPercent}%] ${message}`, "INFO");
        },
        // onComplete callback
        (completionResult: any) => {
          setStatus("success");
          setResult(completionResult);
          addLog(
            "success",
            "✅ Preprocessing completed successfully!",
            "SUCCESS"
          );

          if (onSuccess) {
            setTimeout(() => onSuccess(completionResult), 1000);
          }
        },
        // onError callback
        (errorMessage: string) => {
          setStatus("error");
          addLog("error", `❌ Error: ${errorMessage}`, "ERROR");
        }
      );

      eventSourceRef.current = eventSource;

      // Add connection success log
      addLog("info", "Connecting to preprocessing server...", "INFO");
    } catch (error) {
      console.error("Failed to start preprocessing:", error);
      setStatus("error");
      addLog("error", "Failed to start preprocessing", "ERROR");
    }
  };

  const addLog = (type: string, message: string, level: string) => {
    setLogs((prev) => [
      ...prev,
      {
        type: type as any,
        message,
        level,
        timestamp: Date.now(),
      },
    ]);
  };

  const handleClose = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    onClose();
    // Reset state after close animation
    setTimeout(() => {
      setStatus("idle");
      setLogs([]);
      setProgress(0);
    }, 300);
  };

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
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-2xl w-full max-w-3xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div className="flex items-center gap-3">
            {getStatusIcon()}
            <div>
              <h2 className="text-xl font-semibold text-gray-900">
                Preprocessing NASA POWER Data
              </h2>
              <p className="text-sm text-gray-500 mt-1">{collectionName}</p>
            </div>
          </div>
          <button
            onClick={handleClose}
            disabled={status === "processing"}
            className="text-gray-400 hover:text-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
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
              />
            </div>
          </div>
        )}

        {/* Live Logs */}
        <div className="flex-1 overflow-hidden p-6">
          <div className="bg-gray-900 rounded-lg h-full overflow-y-auto p-4 font-mono text-sm">
            {logs.length === 0 ? (
              <div className="text-gray-400 text-center py-8">
                Waiting for logs...
              </div>
            ) : (
              <div className="space-y-1">
                {logs.map((log, index) => (
                  <div
                    key={index}
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

        {/* Result Summary */}
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
                    {result.recordCount?.toLocaleString()}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Cleaned Collection:</span>
                  <span className="ml-2 font-medium text-gray-900 text-xs">
                    {result.cleanedCollection}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Outliers Removed:</span>
                  <span className="ml-2 font-medium text-gray-900">
                    {result.preprocessing_report?.outliers?.total_outliers || 0}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Completeness:</span>
                  <span className="ml-2 font-medium text-gray-900">
                    {
                      result.preprocessing_report?.quality_metrics
                        ?.completeness_percentage
                    }
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
          {status === "processing" ? (
            <button
              disabled
              className="px-4 py-2 bg-gray-300 text-gray-500 rounded-lg cursor-not-allowed"
            >
              Processing...
            </button>
          ) : (
            <button
              onClick={handleClose}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              {status === "success" ? "Done" : "Close"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
