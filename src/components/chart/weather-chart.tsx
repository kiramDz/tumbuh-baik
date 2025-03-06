'use client'
import React, { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

const WeatherChart = () => {
  const [weatherData, setWeatherData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch("/api/v1/visualization/daily-weather");

        if (!response.ok) {
          throw new Error("Failed to fetch data");
        }

        const result = await response.json();

        // Format data untuk chart
        const formattedData = result.data.map((item) => ({
          timestamp: new Date(item.timestamp).toLocaleString(),
          temperature: parseFloat(item.temperature),
        }));

        setWeatherData(formattedData);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Custom formatter untuk tooltip
  const formatTooltip = (value, name) => {
    if (name === "temperature") {
      return [`${value}°C`, "Suhu"];
    }
    return [value, name];
  };

  // Custom formatter untuk X axis
  const formatXAxis = (tickItem) => {
    const date = new Date(tickItem);
    return `${date.getHours()}:${date.getMinutes().toString().padStart(2, "0")}`;
  };

  if (loading) return <div className="flex justify-center items-center h-64">Loading data...</div>;
  if (error) return <div className="text-red-500">Error: {error}</div>;
  if (!weatherData.length) return <div>No weather data available</div>;

  return (
    <div className="w-full p-4 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Grafik Suhu (30 Menit Interval)</h2>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={weatherData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={formatXAxis}
            interval={6} // Tampilkan setiap 6 titik data (sekitar 3 jam)
          />
          <YAxis domain={["dataMin - 2", "dataMax + 2"]} label={{ value: "Suhu (°C)", angle: -90, position: "insideLeft" }} />
          <Tooltip formatter={formatTooltip} labelFormatter={(value) => `Waktu: ${value}`} />
          <Legend />
          <Line
            type="monotone"
            dataKey="temperature"
            stroke="#ff7300"
            dot={false} // Hilangkan dot untuk data yang banyak
            activeDot={{ r: 5 }} // Dot aktif saat hover
            name="Suhu"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default WeatherChart;
