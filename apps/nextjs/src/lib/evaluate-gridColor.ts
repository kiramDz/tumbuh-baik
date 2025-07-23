import { thresholds, ThresholdKey } from "@/config/tresholds";

export function EvaluateGridColor(parameters: Record<ThresholdKey, { value: number }>): "gray" | "green" | "red" {
  const total = Object.keys(thresholds).length;
  let validCount = 0;
  let hasExtreme = false;

  for (const key of Object.keys(thresholds) as ThresholdKey[]) {
    const param = parameters[key];
    if (!param) continue;

    const value = param.value;
    const { min, max } = thresholds[key];

    if (value < min) {
      continue; // tetap abu-abu
    } else if (value > max) {
      hasExtreme = true;
    } else {
      validCount++;
    }
  }

  if (validCount === 0) return "gray";
  if (hasExtreme) return "red";
  if (validCount >= Math.ceil(total / 2)) return "green";
  return "gray";
}
