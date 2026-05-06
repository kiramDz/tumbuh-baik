export interface TaskResult {
  name: "nasa_refresh" | "nasa_preprocess" | "bmkg_preprocess";
  status: "success" | "failed" | "skipped" | "partial";
  startedAt?: string;
  completedAt?: string;
  recordsUpdated?: number;
  recordsFailed?: number;
  errorMessage?: string;
}

export interface SchedulerLog {
  _id: string;
  executedAt: string;
  completedAt: string | null;
  status: "running" | "success" | "failed" | "partial";
  duration: number | null;
  triggeredBy: "cron" | "manual";
  tasks: TaskResult[];
  totalDatasets: number;
  datasetsUpdated: number;
  errors: string[];
  metadata: {
    nasaApiVersion: string;
    pythonVersion: string;
    system: string;
  };
  createdAt: string;
  updatedAt: string;
}

export interface SchedulerStatus {
  lastRun: {
    executedAt: string;
    status: "success" | "failed" | "partial";
    duration: number;
    datasetsUpdated: number;
    totalDatasets: number;
  } | null;
  nextRun: string;
  isActive: boolean;
  statistics: {
    successRate: number;
    avgDuration: number;
    totalExecutions: number;
    failedExecutions: number;
  };
}

export interface LogsPagination {
  logs: SchedulerLog[];
  total: number;
  limit: number;
  offset: number;
  hasMore: boolean;
  nextOffset: number | null;
}

export interface TriggerRequest {
  mode?: "quick" | "custom";
  tasks?: ("nasa_refresh" | "nasa_preprocess" | "bmkg_preprocess")[];
  datasets?: {
    nasa_refresh?: string[];
    nasa_preprocess?: string[];
    bmkg_preprocess?: string[];
  };
  async?: boolean;
}

export interface TriggerResponse {
  executionId: string;
  status: "running" | "queued";
  startedAt: string;
  estimatedDuration: number;
  pollUrl: string;
}

export interface DatasetConfig {
  _id: string;
  collectionName: string;
  name: string;
  dataType?: "nasa" | "bmkg";
  source: string;
  status: "raw" | "latest" | "preprocessed";
  selected?: boolean;
}

export interface AutomationConfig {
  enabled: boolean;
  frequency: "weekly" | "biweekly" | "monthly";
  executionTime: string;
  dayOfWeek?: number;
  dayOfMonth?: number[];
  selectedDatasets: {
    nasa: string[];
    bmkg: string[];
  };
}
