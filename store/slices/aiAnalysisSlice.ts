import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import type {
  CalibrationData,
  FrameData,
  CrossingEvent,
  GaitMetrics,
  AnalysisResult,
} from "@/lib/websocket";

export type AIAnalysisStatus =
  | "idle"           // 초기 상태
  | "uploading"      // 영상 업로드 중
  | "connecting"     // WebSocket 연결 중
  | "calibrating"    // ArUco 마커 보정 중
  | "analyzing"      // 분석 진행 중
  | "completed"      // 분석 완료
  | "error";         // 에러 발생

export interface AIAnalysisState {
  // 상태
  status: AIAnalysisStatus;
  error: string | null;

  // 연결 정보
  videoId: string | null;
  fileName: string | null;
  isConnected: boolean;

  // 보정 데이터
  calibration: CalibrationData | null;

  // 실시간 분석 데이터
  currentFrame: FrameData | null;
  frameHistory: FrameData[];
  crossingEvents: CrossingEvent[];

  // 최종 결과
  result: AnalysisResult | null;

  // 진행률
  progress: {
    currentFrame: number;
    totalFrames: number;
    percentage: number;
  };
}

const initialState: AIAnalysisState = {
  status: "idle",
  error: null,
  videoId: null,
  fileName: null,
  isConnected: false,
  calibration: null,
  currentFrame: null,
  frameHistory: [],
  crossingEvents: [],
  result: null,
  progress: {
    currentFrame: 0,
    totalFrames: 0,
    percentage: 0,
  },
};

const aiAnalysisSlice = createSlice({
  name: "aiAnalysis",
  initialState,
  reducers: {
    // 상태 변경
    setStatus: (state, action: PayloadAction<AIAnalysisStatus>) => {
      state.status = action.payload;
      if (action.payload !== "error") {
        state.error = null;
      }
    },

    setError: (state, action: PayloadAction<string>) => {
      state.status = "error";
      state.error = action.payload;
    },

    // 연결 관리
    setVideoId: (state, action: PayloadAction<string>) => {
      state.videoId = action.payload;
    },

    setFileName: (state, action: PayloadAction<string>) => {
      state.fileName = action.payload;
    },

    setConnected: (state, action: PayloadAction<boolean>) => {
      state.isConnected = action.payload;
    },

    // 보정 데이터
    setCalibration: (state, action: PayloadAction<CalibrationData>) => {
      state.calibration = action.payload;
      state.status = "analyzing";
    },

    // 프레임 데이터
    updateFrame: (state, action: PayloadAction<FrameData>) => {
      state.currentFrame = action.payload;
      // 히스토리에는 이미지 제외 (메모리 최적화)
      const { frame_image, ...frameWithoutImage } = action.payload;
      if (state.frameHistory.length >= 100) {
        state.frameHistory.shift();
      }
      state.frameHistory.push(frameWithoutImage as FrameData);
    },

    // 라인 통과 이벤트
    addCrossingEvent: (state, action: PayloadAction<CrossingEvent>) => {
      state.crossingEvents.push(action.payload);
    },

    // 진행률 업데이트
    updateProgress: (
      state,
      action: PayloadAction<{ currentFrame: number; totalFrames: number; percentage?: number }>
    ) => {
      state.progress.currentFrame = action.payload.currentFrame;
      state.progress.totalFrames = action.payload.totalFrames;
      // 백엔드에서 percentage를 직접 보내면 사용, 아니면 계산
      state.progress.percentage = action.payload.percentage ?? (
        action.payload.totalFrames > 0
          ? Math.round(
              (action.payload.currentFrame / action.payload.totalFrames) * 100
            )
          : 0
      );
    },

    // 분석 완료
    setAnalysisResult: (state, action: PayloadAction<AnalysisResult>) => {
      state.result = action.payload;
      state.status = "completed";
    },

    // 리셋
    resetAnalysis: (state) => {
      return { ...initialState };
    },

    // 부분 리셋 (새 분석 시작 시)
    prepareNewAnalysis: (state) => {
      state.status = "idle";
      state.error = null;
      state.calibration = null;
      state.currentFrame = null;
      state.frameHistory = [];
      state.crossingEvents = [];
      state.result = null;
      state.progress = { currentFrame: 0, totalFrames: 0, percentage: 0 };
    },
  },
});

export const {
  setStatus,
  setError,
  setVideoId,
  setFileName,
  setConnected,
  setCalibration,
  updateFrame,
  addCrossingEvent,
  updateProgress,
  setAnalysisResult,
  resetAnalysis,
  prepareNewAnalysis,
} = aiAnalysisSlice.actions;

export default aiAnalysisSlice.reducer;

// Selectors
export const selectAIAnalysisStatus = (state: { aiAnalysis: AIAnalysisState }) =>
  state.aiAnalysis.status;

export const selectCalibration = (state: { aiAnalysis: AIAnalysisState }) =>
  state.aiAnalysis.calibration;

export const selectCurrentFrame = (state: { aiAnalysis: AIAnalysisState }) =>
  state.aiAnalysis.currentFrame;

export const selectAnalysisResult = (state: { aiAnalysis: AIAnalysisState }) =>
  state.aiAnalysis.result;

export const selectProgress = (state: { aiAnalysis: AIAnalysisState }) =>
  state.aiAnalysis.progress;

export const selectGaitMetrics = (state: { aiAnalysis: AIAnalysisState }) =>
  state.aiAnalysis.result?.results ?? null;
