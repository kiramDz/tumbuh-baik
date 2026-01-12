export const thresholds = {
  RR_imputed: {
    min: 0,
    max: 100,
  },
  temperature: {
    min: 20,
    max: 35,
  },
  humidity: {
    min: 60,
    max: 90,
  },
  // Tambah parameter lain bila perlu
};

export type ThresholdKey = keyof typeof thresholds;
