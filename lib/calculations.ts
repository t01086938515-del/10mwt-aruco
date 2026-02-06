import { TrialResult } from "@/store/slices/testSessionSlice";
import { clinicalThresholds, getNormativeRange, calculatePercentile } from "./normativeData";

// Calculate gait speed (m/s) from time and distance
export function calculateSpeed(time: number, distance: number = 10): number {
  if (time <= 0) return 0;
  return distance / time;
}

// Calculate cadence (steps/min) from step count and time
export function calculateCadence(stepCount: number, time: number): number {
  if (time <= 0) return 0;
  return (stepCount / time) * 60;
}

// Calculate stride length (m) from speed and cadence
export function calculateStrideLength(speed: number, cadence: number): number {
  if (cadence <= 0) return 0;
  return (speed * 60) / (cadence / 2); // Stride = 2 steps
}

// Get average from valid trials
export function getAverageFromTrials(trials: TrialResult[], key: keyof TrialResult): number {
  const validTrials = trials.filter((t) => t.isValid);
  if (validTrials.length === 0) return 0;

  const sum = validTrials.reduce((acc, trial) => {
    const value = trial[key];
    return acc + (typeof value === "number" ? value : 0);
  }, 0);

  return sum / validTrials.length;
}

// Get best (fastest) trial
export function getBestTrial(trials: TrialResult[]): TrialResult | null {
  const validTrials = trials.filter((t) => t.isValid);
  if (validTrials.length === 0) return null;

  return validTrials.reduce((best, trial) => (trial.speed > best.speed ? trial : best));
}

// Calculate coefficient of variation
export function calculateCV(trials: TrialResult[], key: "time" | "speed"): number {
  const validTrials = trials.filter((t) => t.isValid);
  if (validTrials.length < 2) return 0;

  const values = validTrials.map((t) => t[key]);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const squaredDiffs = values.map((v) => Math.pow(v - mean, 2));
  const variance = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
  const sd = Math.sqrt(variance);

  return (sd / mean) * 100;
}

// Assess fall risk based on gait speed
export function assessFallRisk(speed: number): {
  level: "high" | "moderate" | "low";
  description: string;
  color: string;
} {
  const { fallRisk } = clinicalThresholds;

  if (speed < fallRisk.highRisk) {
    return {
      level: "high",
      description: "높은 낙상 위험",
      color: "destructive",
    };
  } else if (speed < fallRisk.moderateRisk) {
    return {
      level: "moderate",
      description: "중등도 낙상 위험",
      color: "warning",
    };
  } else {
    return {
      level: "low",
      description: "낮은 낙상 위험",
      color: "success",
    };
  }
}

// Assess community ambulation level
export function assessCommunityAmbulation(speed: number): {
  level: "household" | "limited" | "full";
  description: string;
  color: string;
} {
  const { communityAmbulation } = clinicalThresholds;

  if (speed < communityAmbulation.limitedCommunity) {
    return {
      level: "household",
      description: "가정 내 보행자",
      color: "destructive",
    };
  } else if (speed < communityAmbulation.community) {
    return {
      level: "limited",
      description: "제한적 지역사회 보행자",
      color: "warning",
    };
  } else {
    return {
      level: "full",
      description: "완전한 지역사회 보행자",
      color: "success",
    };
  }
}

// Generate clinical interpretation
export function generateClinicalInterpretation(
  speed: number,
  age: number,
  diagnosis?: string
): {
  percentile: number | null;
  fallRisk: ReturnType<typeof assessFallRisk>;
  communityAmbulation: ReturnType<typeof assessCommunityAmbulation>;
  comparison: string;
  recommendations: string[];
} {
  const fallRisk = assessFallRisk(speed);
  const communityAmbulation = assessCommunityAmbulation(speed);

  // Get normative data comparison
  const normativeRange = getNormativeRange(age);
  let percentile: number | null = null;
  let comparison = "";

  if (normativeRange) {
    percentile = calculatePercentile(
      speed,
      normativeRange.comfortableSpeed.mean,
      normativeRange.comfortableSpeed.sd
    );

    if (percentile < 5) {
      comparison = "동일 연령대 평균보다 현저히 낮음";
    } else if (percentile < 25) {
      comparison = "동일 연령대 평균보다 낮음";
    } else if (percentile < 75) {
      comparison = "동일 연령대 평균 범위 내";
    } else if (percentile < 95) {
      comparison = "동일 연령대 평균보다 높음";
    } else {
      comparison = "동일 연령대 평균보다 현저히 높음";
    }
  }

  // Generate recommendations
  const recommendations: string[] = [];

  if (fallRisk.level === "high") {
    recommendations.push("낙상 예방 프로그램 적용 권장");
    recommendations.push("보조기구 사용 평가 필요");
    recommendations.push("가정 환경 안전 평가 권장");
  } else if (fallRisk.level === "moderate") {
    recommendations.push("균형 훈련 강화 권장");
    recommendations.push("정기적인 낙상 위험 재평가 필요");
  }

  if (communityAmbulation.level === "household") {
    recommendations.push("지역사회 보행 능력 향상 훈련 필요");
    recommendations.push("실외 보행 시 보조 필요");
  } else if (communityAmbulation.level === "limited") {
    recommendations.push("지역사회 보행 훈련 지속 권장");
  }

  if (recommendations.length === 0) {
    recommendations.push("현재 보행 능력 유지 프로그램 권장");
    recommendations.push("정기적인 추적 검사 권장");
  }

  return {
    percentile,
    fallRisk,
    communityAmbulation,
    comparison,
    recommendations,
  };
}

// Calculate improvement from baseline
export function calculateImprovement(
  currentSpeed: number,
  baselineSpeed: number,
  diagnosis?: string
): {
  absoluteChange: number;
  percentChange: number;
  isClinicallyMeaningful: boolean;
  isMinimallyDetectable: boolean;
} {
  const absoluteChange = currentSpeed - baselineSpeed;
  const percentChange = baselineSpeed > 0 ? ((absoluteChange / baselineSpeed) * 100) : 0;

  // Get appropriate thresholds based on diagnosis
  const mdc = diagnosis?.includes("stroke")
    ? clinicalThresholds.mdc.stroke
    : diagnosis?.includes("parkinson")
    ? clinicalThresholds.mdc.parkinsons
    : clinicalThresholds.mdc.elderly;

  const mcid = diagnosis?.includes("stroke")
    ? clinicalThresholds.mcid.stroke
    : clinicalThresholds.mcid.general;

  return {
    absoluteChange,
    percentChange,
    isClinicallyMeaningful: Math.abs(absoluteChange) >= mcid,
    isMinimallyDetectable: Math.abs(absoluteChange) >= mdc,
  };
}

// Format time display (mm:ss.ms)
export function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  const wholeSecs = Math.floor(secs);
  const ms = Math.round((secs - wholeSecs) * 100);

  if (mins > 0) {
    return `${mins}:${wholeSecs.toString().padStart(2, "0")}.${ms.toString().padStart(2, "0")}`;
  }
  return `${wholeSecs}.${ms.toString().padStart(2, "0")}`;
}

// Format speed display
export function formatSpeed(speed: number): string {
  return speed.toFixed(2);
}
