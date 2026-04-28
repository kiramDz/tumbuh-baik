export type SpeedBin = "0.5-2.1" | "2.1-3.6" | "3.6-5.7" | "5.7-8.8" | ">8.8";

export interface WindRoseData {
  direction: string;
  "0.5-2.1": number;
  "2.1-3.6": number;
  "3.6-5.7": number;
  "5.7-8.8": number;
  ">8.8": number;
  total: number;
}

export interface WindRoseProcessed {
  roseData: WindRoseData[];
  calmCount: number;
  calmPercentage: number;
  totalValid: number;
}

// Constants
export const DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];

export const SPEED_BINS: SpeedBin[] = [
  "0.5-2.1",
  "2.1-3.6",
  "3.6-5.7",
  "5.7-8.8",
  ">8.8",
];

export const SPEED_LABELS: Record<SpeedBin, string> = {
  "0.5-2.1": "Angin tenang/sepoi",
  "2.1-3.6": "Angin sepoi lemah",
  "3.6-5.7": "Angin sepoi sedang",
  "5.7-8.8": "Angin kencang",
  ">8.8": "Angin kencang/berbahaya",
};

export const SPEED_COLORS = {
  "0.5-2.1": "url(#color_1)",
  "2.1-3.6": "url(#color_2)",
  "3.6-5.7": "url(#color_3)",
  "5.7-8.8": "url(#color_4)",
  ">8.8": "url(#color_5)",
};

export const SPEED_GRADIENTS = {
  "0.5-2.1": { start: "#60a5fa", end: "#3b82f6" },
  "2.1-3.6": { start: "#4ade80", end: "#22c55e" },
  "3.6-5.7": { start: "#fbbf24", end: "#eab308" },
  "5.7-8.8": { start: "#fb923c", end: "#f97316" },
  ">8.8": { start: "#f87171", end: "#ef4444" },
};
// Utility functions
export function isInvalidValue(val: any): boolean {
  return (
    val === null ||
    val === undefined ||
    val === 8888 ||
    val === 9999 ||
    Number.isNaN(Number(val))
  );
}

export function getSpeedBin(speed: number): SpeedBin {
  if (speed <= 2.1) return "0.5-2.1";
  if (speed <= 3.6) return "2.1-3.6";
  if (speed <= 5.7) return "3.6-5.7";
  if (speed <= 8.8) return "5.7-8.8";
  return ">8.8";
}
export function degreeToDirection(degree: number): string {
  const normalized = ((degree % 360) + 360) % 360;
  const index = Math.round(normalized / 45) % 8;
  return DIRECTIONS[index];
}

// Convert data to cumulative stacks for Recharts Radar
export function createCumulativeStacks(data: WindRoseData[]) {
  return data.map((d) => ({
    direction: d.direction,
    "0.5-2.1": d["0.5-2.1"],
    "2.1-3.6": d["2.1-3.6"],
    "3.6-5.7": d["3.6-5.7"],
    "5.7-8.8": d["5.7-8.8"],
    ">8.8": d[">8.8"],
    total: d.total,
    stack_5:
      d["0.5-2.1"] + d["2.1-3.6"] + d["3.6-5.7"] + d["5.7-8.8"] + d[">8.8"],
    stack_4: d["0.5-2.1"] + d["2.1-3.6"] + d["3.6-5.7"] + d["5.7-8.8"],
    stack_3: d["0.5-2.1"] + d["2.1-3.6"] + d["3.6-5.7"],
    stack_2: d["0.5-2.1"] + d["2.1-3.6"],
    stack_1: d["0.5-2.1"],
  }));
}
export function processWindRoseData(
  items: any[],
  dirCol: string,
  speedCol: string,
  asPercentage: boolean = false,
): WindRoseProcessed {
  let calmCount = 0;
  let totalValid = 0;

  const dataMap: Record<string, Omit<WindRoseData, "direction">> = {};
  DIRECTIONS.forEach((dir) => {
    dataMap[dir] = {
      "0.5-2.1": 0,
      "2.1-3.6": 0,
      "3.6-5.7": 0,
      "5.7-8.8": 0,
      ">8.8": 0,
      total: 0,
    };
  });

  items.forEach((item) => {
    const rawDir = item[dirCol];
    const rawSpeed = item[speedCol];

    if (isInvalidValue(rawSpeed)) return;

    let dirString = "";
    if (typeof rawDir === "string" && Number.isNaN(Number(rawDir))) {
      dirString = rawDir.trim().toUpperCase();
    } else {
      if (isInvalidValue(rawDir)) return;
      dirString = degreeToDirection(Number(rawDir));
    }

    const speed = Number(rawSpeed);

    if (dirString === "C" || speed === 0) {
      calmCount++;
      totalValid++;
      return;
    }

    if (!dataMap[dirString]) {
      dataMap[dirString] = {
        "0.5-2.1": 0,
        "2.1-3.6": 0,
        "3.6-5.7": 0,
        "5.7-8.8": 0,
        ">8.8": 0,
        total: 0,
      };
    }

    const bin = getSpeedBin(speed);
    dataMap[dirString][bin]++;
    dataMap[dirString].total++;
    totalValid++;
  });

  const roseData = Object.entries(dataMap).map(([direction, bins]) => {
    if (asPercentage && totalValid > 0) {
      return {
        direction,
        "0.5-2.1": (bins["0.5-2.1"] / totalValid) * 100,
        "2.1-3.6": (bins["2.1-3.6"] / totalValid) * 100,
        "3.6-5.7": (bins["3.6-5.7"] / totalValid) * 100,
        "5.7-8.8": (bins["5.7-8.8"] / totalValid) * 100,
        ">8.8": (bins[">8.8"] / totalValid) * 100,
        total: (bins.total / totalValid) * 100,
      } as WindRoseData;
    }
    return { direction, ...bins } as WindRoseData;
  });

  return {
    roseData,
    calmCount,
    calmPercentage: totalValid > 0 ? (calmCount / totalValid) * 100 : 0,
    totalValid,
  };
}
