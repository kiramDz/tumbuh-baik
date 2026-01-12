import { Icons } from "@/components/icons";

interface WeatherIconProps {
  description: string;
}

export default function WeatherIcon({ description }: WeatherIconProps) {
  const getIcon = () => {
    if (description.includes("Hujan")) return <Icons.rainy />;
    if (description.includes("Berawan")) return <Icons.cloudyy />;
    if (description.includes("Cerah")) return <Icons.sunni />;
    return <Icons.clear />; // default
  };

  return <div className="flex items-center justify-center">{getIcon()}</div>;
}
