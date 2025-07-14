export type ForecastItem = {
  forecast_date: Date | string;
  timestamp: Date | string;
  config_id: string;
  parameters: Record<string, any>;
  [key: string]: any;
};

export type FlattenedForecast = {
  forecast_date: string;
  [key: string]: any;
};

export function flattenForecastData(data: ForecastItem[]): FlattenedForecast[] {
  return data.map((item) => {
    const flatItem: FlattenedForecast = {
      forecast_date: item.forecast_date instanceof Date ? item.forecast_date.toISOString().split("T")[0] : typeof item.forecast_date === "string" ? item.forecast_date.split("T")[0] : "-",
    };

    if (item.parameters && typeof item.parameters === "object") {
      Object.entries(item.parameters).forEach(([paramName, paramValue]) => {
        if (paramValue && typeof paramValue === "object") {
          flatItem[paramName] = paramValue.forecast_value ?? "-";
        }
      });
    }

    return flatItem;
  });
}
