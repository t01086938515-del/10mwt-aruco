import { createSlice, PayloadAction } from "@reduxjs/toolkit";

export type TestMode = "comfortable" | "fast" | "both";
export type TestStatus = "idle" | "setup" | "ready" | "running" | "rest" | "completed";
export type MeasurementMethod = "ai";
export type CameraAngle = "lateral" | "posterior"; // 측면 | 후면

export interface TrialResult {
  trialNumber: number;
  mode: "comfortable" | "fast";
  time: number;
  speed: number;
  stepCount?: number;
  cadence?: number;
  strideLength?: number;
  isValid: boolean;
  invalidReason?: string;
}

export interface TestConfig {
  patientId: string;
  patientName: string;
  mode: TestMode;
  trialsPerMode: number;
  restDuration: number;
  useCamera: boolean;
  useStepCounter: boolean;
  distance: number;
  // AI 분석 관련
  measurementMethod: MeasurementMethod;
  cameraAngle: CameraAngle; // 촬영 각도: 측면 | 후면
  patientHeight?: number; // 환자 키 (cm), 보정용
}

export interface TestSession {
  id: string;
  config: TestConfig;
  status: TestStatus;
  trials: TrialResult[];
  currentTrialIndex: number;
  currentMode: "comfortable" | "fast";
  startTime?: string;
  endTime?: string;
}

interface TestSessionState {
  currentSession: TestSession | null;
  config: TestConfig | null;
  status: TestStatus;
  trials: TrialResult[];
  currentTrialIndex: number;
  currentMode: "comfortable" | "fast";
  timerValue: number;
  isTimerRunning: boolean;
}

const initialState: TestSessionState = {
  currentSession: null,
  config: null,
  status: "idle",
  trials: [],
  currentTrialIndex: 0,
  currentMode: "comfortable",
  timerValue: 0,
  isTimerRunning: false,
};

const testSessionSlice = createSlice({
  name: "testSession",
  initialState,
  reducers: {
    setConfig: (state, action: PayloadAction<TestConfig>) => {
      state.config = action.payload;
      state.status = "setup";
      state.trials = [];
      state.currentTrialIndex = 0;
      state.currentMode = "comfortable";
    },
    setStatus: (state, action: PayloadAction<TestStatus>) => {
      state.status = action.payload;
    },
    startTest: (state) => {
      state.status = "running";
      state.isTimerRunning = true;
      state.timerValue = 0;
    },
    stopTest: (state) => {
      state.isTimerRunning = false;
    },
    updateTimer: (state, action: PayloadAction<number>) => {
      state.timerValue = action.payload;
    },
    addTrialResult: (state, action: PayloadAction<TrialResult>) => {
      state.trials.push(action.payload);
      state.currentTrialIndex += 1;
      state.isTimerRunning = false;
      state.timerValue = 0;
    },
    setCurrentMode: (state, action: PayloadAction<"comfortable" | "fast">) => {
      state.currentMode = action.payload;
    },
    startRest: (state) => {
      state.status = "rest";
    },
    completeTest: (state) => {
      state.status = "completed";
      state.isTimerRunning = false;
    },
    resetSession: (state) => {
      state.currentSession = null;
      state.config = null;
      state.status = "idle";
      state.trials = [];
      state.currentTrialIndex = 0;
      state.currentMode = "comfortable";
      state.timerValue = 0;
      state.isTimerRunning = false;
    },
    invalidateTrial: (state, action: PayloadAction<{ index: number; reason: string }>) => {
      if (state.trials[action.payload.index]) {
        state.trials[action.payload.index].isValid = false;
        state.trials[action.payload.index].invalidReason = action.payload.reason;
      }
    },
  },
});

export const {
  setConfig,
  setStatus,
  startTest,
  stopTest,
  updateTimer,
  addTrialResult,
  setCurrentMode,
  startRest,
  completeTest,
  resetSession,
  invalidateTrial,
} = testSessionSlice.actions;
export default testSessionSlice.reducer;
