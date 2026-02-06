// 10MWT Normative Data and Clinical Thresholds
// Based on published research and clinical guidelines

export interface NormativeRange {
  ageGroup: string;
  minAge: number;
  maxAge: number;
  gender: "male" | "female" | "both";
  comfortableSpeed: { mean: number; sd: number; min: number; max: number };
  fastSpeed: { mean: number; sd: number; min: number; max: number };
}

// Healthy adult normative values (m/s)
export const healthyNormativeData: NormativeRange[] = [
  {
    ageGroup: "20-29",
    minAge: 20,
    maxAge: 29,
    gender: "both",
    comfortableSpeed: { mean: 1.36, sd: 0.16, min: 1.04, max: 1.68 },
    fastSpeed: { mean: 2.05, sd: 0.26, min: 1.53, max: 2.57 },
  },
  {
    ageGroup: "30-39",
    minAge: 30,
    maxAge: 39,
    gender: "both",
    comfortableSpeed: { mean: 1.34, sd: 0.17, min: 1.00, max: 1.68 },
    fastSpeed: { mean: 1.97, sd: 0.27, min: 1.43, max: 2.51 },
  },
  {
    ageGroup: "40-49",
    minAge: 40,
    maxAge: 49,
    gender: "both",
    comfortableSpeed: { mean: 1.34, sd: 0.17, min: 1.00, max: 1.68 },
    fastSpeed: { mean: 1.93, sd: 0.25, min: 1.43, max: 2.43 },
  },
  {
    ageGroup: "50-59",
    minAge: 50,
    maxAge: 59,
    gender: "both",
    comfortableSpeed: { mean: 1.31, sd: 0.18, min: 0.95, max: 1.67 },
    fastSpeed: { mean: 1.84, sd: 0.26, min: 1.32, max: 2.36 },
  },
  {
    ageGroup: "60-69",
    minAge: 60,
    maxAge: 69,
    gender: "both",
    comfortableSpeed: { mean: 1.24, sd: 0.18, min: 0.88, max: 1.60 },
    fastSpeed: { mean: 1.69, sd: 0.26, min: 1.17, max: 2.21 },
  },
  {
    ageGroup: "70-79",
    minAge: 70,
    maxAge: 79,
    gender: "both",
    comfortableSpeed: { mean: 1.13, sd: 0.20, min: 0.73, max: 1.53 },
    fastSpeed: { mean: 1.51, sd: 0.26, min: 0.99, max: 2.03 },
  },
  {
    ageGroup: "80+",
    minAge: 80,
    maxAge: 120,
    gender: "both",
    comfortableSpeed: { mean: 0.94, sd: 0.23, min: 0.48, max: 1.40 },
    fastSpeed: { mean: 1.27, sd: 0.28, min: 0.71, max: 1.83 },
  },
];

// Clinical thresholds for specific conditions
export const clinicalThresholds = {
  // Fall risk thresholds
  fallRisk: {
    highRisk: 0.6, // < 0.6 m/s = high fall risk
    moderateRisk: 0.8, // 0.6-0.8 m/s = moderate fall risk
    lowRisk: 1.0, // > 1.0 m/s = low fall risk
  },

  // Community ambulation
  communityAmbulation: {
    limitedCommunity: 0.4, // < 0.4 m/s = household ambulator
    community: 0.8, // 0.4-0.8 m/s = limited community ambulator
    fullCommunity: 1.2, // > 0.8 m/s = full community ambulator
  },

  // Minimal Detectable Change (MDC) values
  mdc: {
    stroke: 0.16, // m/s for stroke patients
    parkinsons: 0.18, // m/s for Parkinson's patients
    spinalCord: 0.13, // m/s for SCI patients
    elderly: 0.10, // m/s for elderly
  },

  // Minimal Clinically Important Difference (MCID)
  mcid: {
    stroke: 0.14, // m/s for stroke patients
    general: 0.10, // m/s general
  },
};

// Diagnosis-specific reference data
export const diagnosisReferenceData: Record<string, { comfortableSpeed: number; fastSpeed: number }> = {
  stroke_acute: { comfortableSpeed: 0.32, fastSpeed: 0.45 },
  stroke_subacute: { comfortableSpeed: 0.56, fastSpeed: 0.78 },
  stroke_chronic: { comfortableSpeed: 0.72, fastSpeed: 0.95 },
  parkinsons_early: { comfortableSpeed: 1.0, fastSpeed: 1.3 },
  parkinsons_moderate: { comfortableSpeed: 0.75, fastSpeed: 1.0 },
  parkinsons_advanced: { comfortableSpeed: 0.5, fastSpeed: 0.7 },
  sci_asia_d: { comfortableSpeed: 0.8, fastSpeed: 1.1 },
  sci_asia_c: { comfortableSpeed: 0.4, fastSpeed: 0.6 },
  tha_postop_4wk: { comfortableSpeed: 0.6, fastSpeed: 0.8 },
  tha_postop_12wk: { comfortableSpeed: 0.9, fastSpeed: 1.2 },
};

// Get normative range for age
export function getNormativeRange(age: number): NormativeRange | undefined {
  return healthyNormativeData.find((range) => age >= range.minAge && age <= range.maxAge);
}

// Calculate percentile
export function calculatePercentile(value: number, mean: number, sd: number): number {
  const zScore = (value - mean) / sd;
  // Approximate normal CDF
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = zScore < 0 ? -1 : 1;
  const z = Math.abs(zScore) / Math.sqrt(2);
  const t = 1.0 / (1.0 + p * z);
  const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-z * z);

  return Math.round((0.5 * (1.0 + sign * y)) * 100);
}
