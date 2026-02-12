"use client";

import { useCallback, useRef, useEffect } from "react";
import { useAppDispatch, useAppSelector } from "@/store/hooks";
import {
  setStatus,
  setError,
  setVideoId,
  setConnected,
  setCalibration,
  updateFrame,
  addCrossingEvent,
  updateProgress,
  setAnalysisResult,
  resetAnalysis,
  prepareNewAnalysis,
} from "@/store/slices/aiAnalysisSlice";
import {
  GaitAnalysisWebSocket,
  getWebSocketClient,
  resetWebSocketClient,
  WSMessage,
  CalibrationData,
  FrameData,
  CrossingEvent,
  AnalysisResult,
} from "./websocket";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "ws://localhost:8000";

export function useAIAnalysis() {
  const dispatch = useAppDispatch();
  const wsRef = useRef<GaitAnalysisWebSocket | null>(null);

  const {
    status,
    error,
    videoId,
    isConnected,
    calibration,
    currentFrame,
    crossingEvents,
    result,
    progress,
  } = useAppSelector((state) => state.aiAnalysis);

  // WebSocket 이벤트 핸들러 설정
  // 백엔드는 {type, ...data} 형식으로 보내므로 msg 자체가 데이터
  const setupEventHandlers = useCallback(
    (ws: GaitAnalysisWebSocket) => {
      // 보정 완료
      ws.on("calibration", (msg: WSMessage) => {
        dispatch(setCalibration(msg as unknown as CalibrationData));
      });

      // 프레임 데이터
      ws.on("frame_data", (msg: WSMessage) => {
        const data = msg as unknown as FrameData & { type: string };
        dispatch(updateFrame(data));
        if (data.frame_idx !== undefined) {
          dispatch(
            updateProgress({
              currentFrame: data.frame_idx,
              totalFrames: progress.totalFrames || data.frame_idx + 100,
            })
          );
        }
      });

      // 라인 통과 이벤트
      ws.on("crossing_event", (msg: WSMessage) => {
        const data = msg as unknown as CrossingEvent & { type: string };
        dispatch(addCrossingEvent(data));
      });

      // 분석 완료
      ws.on("analysis_complete", (msg: WSMessage) => {
        const data = msg as unknown as AnalysisResult & { type: string };
        dispatch(setAnalysisResult(data));
      });

      // 진행률
      ws.on("progress", (msg: WSMessage) => {
        const data = msg as unknown as { percent: number; frame_idx: number; total_frames: number };
        dispatch(
          updateProgress({
            currentFrame: data.frame_idx,
            totalFrames: data.total_frames,
            percentage: data.percent,
          })
        );
        // 보정 완료 후 분석 중으로 상태 변경
        if (data.percent > 50) {
          dispatch(setStatus("analyzing"));
        }
      });

      // 에러
      ws.on("error", (msg: WSMessage) => {
        const data = msg as unknown as { message: string };
        dispatch(setError(data.message));
      });
    },
    [dispatch, progress.totalFrames]
  );

  // 영상 업로드 및 분석 시작
  const startAnalysis = useCallback(
    async (file: File, config?: { walkDistance?: number; markerSize?: number }) => {
      try {
        dispatch(prepareNewAnalysis());
        dispatch(setStatus("uploading"));

        const ws = getWebSocketClient(BACKEND_URL);
        wsRef.current = ws;

        // 영상 업로드
        const uploadedVideoId = await ws.uploadVideo(file);
        dispatch(setVideoId(uploadedVideoId));

        // WebSocket 연결
        dispatch(setStatus("connecting"));
        await ws.connect(uploadedVideoId);
        dispatch(setConnected(true));

        // 이벤트 핸들러 설정
        setupEventHandlers(ws);

        // 설정 전송
        ws.configure({
          walk_distance: config?.walkDistance || 10,
          marker_size: config?.markerSize || 0.15,
          direction: "auto",
        });

        // 분석 시작
        dispatch(setStatus("calibrating"));
        ws.startAnalysis();
      } catch (err) {
        const message = err instanceof Error ? err.message : "분석 시작 실패";
        dispatch(setError(message));
      }
    },
    [dispatch, setupEventHandlers]
  );

  // 분석 일시정지
  const pauseAnalysis = useCallback(() => {
    wsRef.current?.pauseAnalysis();
  }, []);

  // 분석 재개
  const resumeAnalysis = useCallback(() => {
    wsRef.current?.startAnalysis();
  }, []);

  // 분석 중단 및 리셋
  const cancelAnalysis = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.disconnect();
      wsRef.current = null;
    }
    resetWebSocketClient();
    dispatch(resetAnalysis());
  }, [dispatch]);

  // 페이지 전환 시에도 WebSocket 연결 및 이벤트 핸들러 유지
  // ai-analyze 페이지로 이동했을 때 이벤트 핸들러 재설정
  useEffect(() => {
    // 이미 분석 중이고 WebSocket 클라이언트가 있으면 이벤트 핸들러 재설정
    if ((status === "calibrating" || status === "analyzing") && !wsRef.current) {
      const ws = getWebSocketClient(BACKEND_URL);
      if (ws.isConnected()) {
        wsRef.current = ws;
        setupEventHandlers(ws);
      }
    }

    return () => {
      // 분석 중이 아닐 때만 연결 종료
      if (wsRef.current && status !== "analyzing" && status !== "calibrating" && status !== "uploading" && status !== "connecting") {
        wsRef.current.disconnect();
        wsRef.current = null;
      }
    };
  }, [status, setupEventHandlers]);

  return {
    // 상태
    status,
    error,
    videoId,
    isConnected,
    calibration,
    currentFrame,
    crossingEvents,
    result,
    progress,

    // 계산된 값
    isAnalyzing: status === "analyzing" || status === "calibrating",
    isCompleted: status === "completed",
    hasError: status === "error",

    // 메트릭스 (편의 접근) - result.results가 실제 GaitMetrics
    metrics: result?.results ?? null,
    clinicalInterpretation: result?.results?.clinical ?? null,

    // 액션
    startAnalysis,
    pauseAnalysis,
    resumeAnalysis,
    cancelAnalysis,
    reset: () => dispatch(resetAnalysis()),
  };
}
