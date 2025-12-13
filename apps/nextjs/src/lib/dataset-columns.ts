// lib/dataset-columns.ts

export const DATASET_COLUMNS = {
  rain: ["RR", "RR_imputed", "PRECTOTCORR"],
  temperature: ["TAVG", "TMAX", "TMIN", "T2M"],
  humidity: ["RH_AVG", "RH_AVG_preprocessed", "RH2M"],
  radiation: ["ALLSKY_SFC_SW_DWN", "SRAD", "GHI"],
};

export function getForecastValue(parameters: Record<string, any> | undefined, keys: string[], fallback = 0): number {
  if (!parameters) return fallback;

  for (const key of keys) {
    const value = parameters[key]?.forecast_value;
    if (value !== undefined && value !== null) {
      return value;
    }
  }

  return fallback;
}
