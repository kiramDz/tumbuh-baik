import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useLayoutEffect,
} from "react";
import { Icons } from "@/app/dashboard/_components/icons";
import {
  preprocessNasaDatasetWithStream,
  preprocessBmkgDatasetWithStream,
} from "@/lib/fetch/files.fetch";
import { useRouter } from "next/navigation";
import { ScrollArea } from "@/components/ui/scroll-area";

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
  originalRecordCount?: number;
  cleanedCollection?: string;
  collection?: string;
  message?: string;
  preprocessedCollections?: string[];
  preprocessing_report?: {
    missing_data?: any;
    outliers?: {
      total_outliers: number;
      by_parameter: Record<string, number>;
      methods_used: string[];
      treatment: string;
    };
    smoothing?: any;
    gaps?: any;
    r2_validation?: any;
    model_coverage?: any;
    quality_metrics?: {
      original_records: number;
      processed_records: number;
      records_removed: number;
      completeness_percentage: number;
      data_quality: string;
    };
    warnings?: string[];
  };
}

interface PreprocessingModalProps {
  collectionName: string;
  isNasaDataset?: boolean;
  isBmkgDataset?: boolean;
  isAPI?: boolean;
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: (result: any) => void;
}

// Constants
const MAX_LOG_ENTRIES = 1000;
const CLOSE_DELAY = 300;

export default function PreprocessingModal({
  collectionName,
  isNasaDataset = false,
  isBmkgDataset = false,
  isAPI = false,
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
  const [isClosing, setIsClosing] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const logsEndRef = useRef<HTMLDivElement | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement | null>(null);
  const closeTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [isUserScrolling, setIsUserScrolling] = useState(false);
  const userScrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const router = useRouter();

  useEffect(() => {
    if (!isOpen) return;

    // Save original styles
    const originalOverflow = document.body.style.overflow;
    const originalPaddingRight = document.body.style.paddingRight;

    // Calculate scrollbar width to prevent layout shift
    const scrollbarWidth =
      window.innerWidth - document.documentElement.clientWidth;

    // Lock scroll
    document.body.style.overflow = "hidden";
    document.body.style.paddingRight = `${scrollbarWidth}px`;

    // Cleanup when modal closes
    return () => {
      document.body.style.overflow = originalOverflow;
      document.body.style.paddingRight = originalPaddingRight;
    };
  }, [isOpen]);

  // Determine preprocessing type
  const preprocessingType = isNasaDataset
    ? "NASA POWER"
    : isBmkgDataset
    ? "BMKG"
    : "Unknown";

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
    if (userScrollTimeoutRef.current) {
      clearTimeout(userScrollTimeoutRef.current);
      userScrollTimeoutRef.current = null;
    }
  }, []);

  const scrollToBottom = useCallback(() => {
    if (status !== "processing") return; // ← Add this guard

    requestAnimationFrame(() => {
      const viewport = scrollAreaRef.current?.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (viewport) {
        viewport.scrollTo({
          top: viewport.scrollHeight,
          behavior: "smooth",
        });
      }
    });
  }, [status]);

  useEffect(() => {
    const viewport = scrollAreaRef.current?.querySelector(
      "[data-radix-scroll-area-viewport]"
    );
    if (!viewport) return;

    const handleScroll = () => {
      setIsUserScrolling(true);

      // Clear existing timeout
      if (userScrollTimeoutRef.current) {
        clearTimeout(userScrollTimeoutRef.current);
      }

      // Resume auto-scroll after 3 seconds of no manual scroll
      userScrollTimeoutRef.current = setTimeout(() => {
        setIsUserScrolling(false);
      }, 3000);
    };

    viewport.addEventListener("scroll", handleScroll, { passive: true });
    return () => {
      viewport.removeEventListener("scroll", handleScroll);
      if (userScrollTimeoutRef.current) {
        clearTimeout(userScrollTimeoutRef.current);
      }
    };
  }, []);

  useLayoutEffect(() => {
    if (status === "processing" && logs.length > 0) {
      scrollToBottom();
    }
  }, [logs.length, status, scrollToBottom]);

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

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
    setResult(null);

    try {
      let streamResult: { eventSource: EventSource; cleanup: () => void };

      // Callback functions reusable for both types
      const onLog = (logData: any) => {
        if (logData.type === "info") {
          addLog("info", logData.message, "INFO");
        } else {
          addLog("log", logData.message, logData.level || "INFO");
        }
      };
      const onProgress = (
        progressPercent: number,
        stage: string,
        message: string
      ) => {
        const safeProgress = Math.max(0, Math.min(100, progressPercent || 0));
        setProgress(safeProgress);
        setCurrentStage(message || stage || "Processing...");
        addLog("progress", `[${safeProgress}%] ${message}`, "INFO");
      };

      const onComplete = (completionResult: any) => {
        const typedResult: PreprocessingResult = {
          recordCount: completionResult?.recordCount,
          originalRecordCount: completionResult?.originalRecordCount,
          cleanedCollection: completionResult?.cleanedCollection,
          collection: completionResult?.collection,
          message: completionResult?.message,
          preprocessedCollections: completionResult?.preprocessedCollections,
          preprocessing_report: completionResult?.preprocessing_report,
        };

        setStatus("success");
        setResult(typedResult);

        addLog(
          "success",
          `✅ ${preprocessingType} preprocessing completed successfully!`,
          "SUCCESS"
        );
      };

      const onError = (errorMessage: string) => {
        setStatus("error");
        const safeErrorMessage = errorMessage || "Unknown error occurred";
        addLog("error", `❌ Error: ${safeErrorMessage}`, "ERROR");
      };

      // Choose preprocessing type
      if (isNasaDataset) {
        addLog("info", "Starting NASA POWER preprocessing...", "INFO");
        streamResult = preprocessNasaDatasetWithStream(
          collectionName,
          onLog,
          onProgress,
          onComplete,
          onError
        );
      } else if (isBmkgDataset) {
        addLog("info", "Starting BMKG preprocessing...", "INFO");
        streamResult = preprocessBmkgDatasetWithStream(
          collectionName,
          onLog,
          onProgress,
          onComplete,
          onError
        );
      } else {
        throw new Error(
          "Unknown dataset type. Please specify isNasaDataset or isBmkgDataset."
        );
      }
      eventSourceRef.current = streamResult.eventSource;
      addLog(
        "info",
        `Connecting to ${preprocessingType} preprocessing server...`,
        "INFO"
      );
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

    // Clear any existing timeout first
    if (closeTimeoutRef.current) {
      clearTimeout(closeTimeoutRef.current);
      closeTimeoutRef.current = null;
    }

    setIsClosing(true);
    cleanup();

    // Reset state immediately, then close
    setStatus("idle");
    setLogs([]);
    setProgress(0);
    setCurrentStage("");
    setResult(null);
    setIsUserScrolling(false);

    // Close after short delay for animation
    closeTimeoutRef.current = setTimeout(() => {
      onClose();
      setIsClosing(false);
    }, CLOSE_DELAY);
  }, [status, cleanup, onClose]);

  const handleDoneClick = useCallback(() => {
    if (status === "success" && result?.cleanedCollection) {
      // Clear any existing timeout
      if (closeTimeoutRef.current) {
        clearTimeout(closeTimeoutRef.current);
        closeTimeoutRef.current = null;
      }

      // Trigger success callback first
      if (onSuccess) {
        onSuccess(result);
      }

      setIsClosing(true);
      cleanup();

      // Reset state immediately
      setStatus("idle");
      setLogs([]);
      setProgress(0);
      setCurrentStage("");
      setResult(null);
      setIsUserScrolling(false);

      // Redirect to cleaned dataset
      const cleanedCollection = result.cleanedCollection;
      router.push(`/dashboard/data/${encodeURIComponent(cleanedCollection)}`);

      // Close modal after animation
      closeTimeoutRef.current = setTimeout(() => {
        onClose();
        setIsClosing(false);
      }, CLOSE_DELAY);
    } else {
      // Regular close for non-success states
      handleClose();
    }
  }, [status, result, onSuccess, router, cleanup, onClose, handleClose]);

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
      onClick={handleClose}
    >
      <div
        className="bg-white rounded-lg shadow-2xl w-full max-w-3xl max-h-[90vh] flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* HEADER */}
        <div className="flex items-center justify-between p-6 border-b">
          <div className="flex items-center gap-3">
            {getStatusIcon()}
            <div>
              <h2
                id="preprocessing-title"
                className="text-xl font-semibold text-gray-900"
              >
                Preprocessing {preprocessingType} Data
              </h2>
              <p className="text-sm text-gray-500 mt-1">{collectionName}</p>
            </div>
          </div>

          <button
            onClick={handleClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
            disabled={status === "processing"}
          >
            <Icons.closeX className="h-6 w-6" />
          </button>
        </div>

        {/* PROGRESS BAR */}
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

        {/* LOG SECTION */}
        <div className="p-6 flex flex-col flex-1 min-h-0 w-full">
          {/* Log Header */}
          <div className="flex items-center justify-between mb-3 w-full">
            <span className="text-sm text-gray-600 font-medium">
              Live Logs ({logs.length})
            </span>

            {status === "success" && (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    const vp = scrollAreaRef.current?.querySelector(
                      "[data-radix-scroll-area-viewport]"
                    );
                    vp?.scrollTo({ top: 0, behavior: "smooth" });
                  }}
                  className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded flex items-center gap-1"
                >
                  <Icons.up className="h-3 w-3" /> Top
                </button>

                <button
                  onClick={() => {
                    const vp = scrollAreaRef.current?.querySelector(
                      "[data-radix-scroll-area-viewport]"
                    );
                    vp?.scrollTo({
                      top: vp.scrollHeight,
                      behavior: "smooth",
                    });
                  }}
                  className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded flex items-center gap-1"
                >
                  <Icons.down className="h-3 w-3" /> Bottom
                </button>

                <span className="text-xs text-gray-500 flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                  Free scroll
                </span>
              </div>
            )}

            {status === "processing" && (
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-xs text-gray-500">Auto-scrolling</span>
              </div>
            )}
          </div>

          {/* Scroll Area — Already Full Width */}
          <ScrollArea
            ref={scrollAreaRef}
            className="h-[400px] border rounded-lg w-full"
          >
            <div className="bg-gray-900 p-4">
              {logs.length === 0 ? (
                <div className="text-gray-400 text-center py-8 font-mono text-sm">
                  Waiting for logs...
                </div>
              ) : (
                <div className="space-y-1 font-mono text-sm">
                  {logs.map((log, index) => (
                    <div
                      key={`${log.timestamp}-${index}`}
                      className={
                        log.level === "ERROR"
                          ? "text-red-400"
                          : log.level === "WARNING"
                          ? "text-yellow-400"
                          : log.level === "SUCCESS"
                          ? "text-green-400"
                          : "text-gray-300"
                      }
                    >
                      <span className="text-gray-500 mr-2">
                        {new Date(log.timestamp || Date.now()).toLocaleString()}
                      </span>
                      {log.message}
                    </div>
                  ))}
                  <div ref={logsEndRef} />
                </div>
              )}
            </div>
          </ScrollArea>
        </div>

        {/* RESULT SUMMARY */}
        {status === "success" && result && (
          <div className="px-6 pb-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h3 className="font-semibold text-green-900 mb-2">
                Preprocessing Complete!
              </h3>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-gray-600">Original Records:</span>
                  <span className="ml-2 font-medium">
                    {result.originalRecordCount?.toLocaleString()}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Records Processed:</span>
                  <span className="ml-2 font-medium">
                    {result.recordCount?.toLocaleString()}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Cleaned Collection:</span>
                  <span className="ml-2 font-medium text-xs break-all">
                    {result.cleanedCollection}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Outliers Removed:</span>
                  <span className="ml-2 font-medium">
                    {result.preprocessing_report?.outliers?.total_outliers ?? 0}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Completeness:</span>
                  <span className="ml-2 font-medium">
                    {result.preprocessing_report?.quality_metrics
                      ?.completeness_percentage ?? "N/A"}
                    %
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* FOOTER */}
        <div className="flex justify-between items-center gap-3 p-6 border-t bg-gray-50">
          {status === "success" && (
            <div className="flex items-center gap-2 text-green-600 text-sm">
              <Icons.checked className="h-4 w-4" />
              Processing completed successfully!
            </div>
          )}

          {status === "processing" && (
            <div className="flex items-center gap-2 text-gray-600 text-sm">
              <Icons.spinner className="h-4 w-4 animate-spin" />
              Processing...
            </div>
          )}

          {status === "error" && (
            <div className="flex items-center gap-2 text-red-600 text-sm">
              <Icons.closeX className="h-4 w-4" />
              Processing failed
            </div>
          )}

          <div className="flex-1" />

          <button
            onClick={status === "success" ? handleDoneClick : handleClose}
            disabled={isClosing}
            className={`px-4 py-2 rounded-lg text-white transition-colors ${
              status === "processing"
                ? "bg-red-600 hover:bg-red-700"
                : status === "success"
                ? "bg-green-600 hover:bg-green-700"
                : "bg-gray-600 hover:bg-gray-700"
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
