interface ForecastDay {
  day: string;
  condition: string;
  temperature: number;
  icon: string;
}

interface WeatherForecastProps {
  forecast?: ForecastDay[];
}

const defaultForecast: ForecastDay[] = [
  { day: "Friday", condition: "Cloudy with rain", temperature: 18, icon: "🌧️" },
  { day: "Saturday", condition: "Sunny", temperature: 26, icon: "☀️" },
  { day: "Sunday", condition: "Rain", temperature: 22, icon: "🌧️" },
  { day: "Monday", condition: "Rain with thunder", temperature: 17, icon: "⛈️" },
  { day: "Tuesday", condition: "Snow", temperature: 9, icon: "❄️" },
  { day: "Wednesday", condition: "Cloudy with rain", temperature: 12, icon: "🌧️" },
];

export default function WeatherForecast({ forecast = defaultForecast }: WeatherForecastProps) {
  return (
    <div className="bg-black/20 backdrop-blur-sm rounded-2xl p-6 border border-gray-700/50">
      <div className="space-y-4">
        {forecast.map((day, index) => (
          <div key={index} className="flex items-center justify-between text-white">
            <div className="flex items-center gap-4">
              <span className="text-2xl">{day.icon}</span>
              <div>
                <div className="font-medium text-gray-200">{day.day}</div>
                <div className="text-sm text-gray-400">{day.condition}</div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xl font-light">{day.temperature}°</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
