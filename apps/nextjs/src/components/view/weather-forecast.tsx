import { Card, CardContent } from "../ui/card";

interface ForecastDay {
  day: string;
  condition: string;
  temperature: number;
  icon: string;
}

interface WeatherForecastProps {
  forecast?: ForecastDay[];
}

const WeatherForecast: React.FC<WeatherForecastProps> = ({ forecast = [] }) => {
  return (
    <Card className="rounded-2xl border h-fit ">
      <CardContent className="p-6">
        <div className="space-y-4">
          {forecast.map((day, index) => (
            <div key={index} className="flex items-center justify-between text-white gap-6">
              <div className="flex items-center gap-2">
                <span className="text-2xl">{day.icon}</span>
                <div>
                  <div className="font-medium text-black">{day.day}</div>
                  <div className="text-sm text-black">{day.condition}</div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-xl text-black font-light">{day.temperature}°</div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default WeatherForecast;
