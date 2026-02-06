"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { useAppSelector } from "@/store/hooks";
import { useAIAnalysis } from "@/lib/useAIAnalysis";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  ArrowLeft,
  Pause,
  Play,
  X,
  Activity,
  Target,
  CheckCircle,
  AlertCircle,
  Loader2,
  Timer,
  Video,
} from "lucide-react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function AIAnalyzePage() {
  const router = useRouter();
  const { config } = useAppSelector((state) => state.testSession);
  const {
    status,
    error,
    videoId,
    calibration,
    currentFrame,
    crossingEvents,
    progress,
    result,
    isAnalyzing,
    isCompleted,
    pauseAnalysis,
    resumeAnalysis,
    cancelAnalysis,
  } = useAIAnalysis();

  // 타이머 표시용
  const [elapsedTime, setElapsedTime] = useState<number | null>(null);

  // 분석 프레임 이미지 표시용
  const [frameImage, setFrameImage] = useState<string | null>(null);

  // 설정이 없거나 분석이 완료되면 리다이렉트
  useEffect(() => {
    if (!config) {
      router.push("/test/setup");
    }
  }, [config, router]);

  useEffect(() => {
    if (isCompleted) {
      router.push("/test/ai-result");
    }
  }, [isCompleted, router]);

  if (!config) {
    return null;
  }

  const handleCancel = () => {
    if (confirm("분석을 취소하시겠습니까?")) {
      cancelAnalysis();
      router.push("/test/setup");
    }
  };

  const getStatusBadge = () => {
    switch (status) {
      case "calibrating":
        return <Badge className="bg-yellow-500">마커 보정 중</Badge>;
      case "analyzing":
        return <Badge className="bg-blue-500">분석 중</Badge>;
      case "error":
        return <Badge variant="destructive">오류 발생</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  // 백엔드에서 보내는 crossing_event 형식: {line: 'start'|'finish', timestamp_s, ...}
  const startEvent = crossingEvents.find((e: any) => e.line === "start" || e.type === "start");
  const finishEvent = crossingEvents.find((e: any) => e.line === "finish" || e.type === "end");

  // 타이머 계산
  useEffect(() => {
    if (startEvent && finishEvent) {
      const startTime = startEvent.timestamp_s ?? startEvent.timestamp ?? 0;
      const finishTime = finishEvent.timestamp_s ?? finishEvent.timestamp ?? 0;
      setElapsedTime(finishTime - startTime);
    } else if (startEvent && currentFrame?.timestamp_s) {
      const startTime = startEvent.timestamp_s ?? startEvent.timestamp ?? 0;
      setElapsedTime(currentFrame.timestamp_s - startTime);
    }
  }, [startEvent, finishEvent, currentFrame]);

  // 분석 프레임 이미지 업데이트
  useEffect(() => {
    if (currentFrame?.frame_image) {
      setFrameImage(currentFrame.frame_image);
    }
  }, [currentFrame?.frame_image]);

  // 첫 프레임 이미지 (poster)
  const posterUrl = videoId ? `${BACKEND_URL}/api/frame/${videoId}` : null;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-gray-200 bg-gray-50/95 backdrop-blur">
        <div className="flex h-16 items-center justify-between px-4">
          <Button variant="ghost" onClick={handleCancel}>
            <X className="mr-2 h-4 w-4" />
            취소
          </Button>
          <div className="text-center">
            <p className="text-sm font-medium">{config.patientName}</p>
            {getStatusBadge()}
          </div>
          <div className="w-16" />
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-2xl p-4 md:p-6">
        {/* 영상 미리보기 */}
        {videoId && (
          <Card className="mb-6">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-lg">
                <Video className="h-5 w-5" />
                분석 영상
                {!frameImage && (
                  <Loader2 className="h-4 w-4 animate-spin ml-2" />
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4">
              <div className="relative overflow-hidden rounded-xl bg-black aspect-video">
                {/* 분석 프레임 이미지 (실시간 업데이트) */}
                <img
                  src={frameImage ? `data:image/jpeg;base64,${frameImage}` : (posterUrl || undefined)}
                  alt="분석 프레임"
                  className="w-full h-full object-contain"
                />
                {/* 타이머 오버레이 */}
                {startEvent && (
                  <div className="absolute bottom-4 right-4 flex items-center gap-2 rounded-lg bg-black/70 px-3 py-2 text-white">
                    <Timer className="h-5 w-5" />
                    <span className="font-mono text-xl font-bold">
                      {elapsedTime !== null ? elapsedTime.toFixed(2) : "0.00"}s
                    </span>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* 진행률 */}
        <Card className="mb-6">
          <CardContent className="p-6">
            <div className="mb-4 flex items-center justify-between">
              <span className="text-sm font-medium">분석 진행률</span>
              <span className="text-2xl font-bold">{progress.percentage}%</span>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-gray-200">
              <div
                className="h-full rounded-full bg-[hsl(var(--primary))] transition-all duration-300"
                style={{ width: `${progress.percentage}%` }}
              />
            </div>
            <p className="mt-2 text-sm text-gray-500">
              {progress.currentFrame} / {progress.totalFrames || "?"} 프레임
            </p>
          </CardContent>
        </Card>

        {/* 보정 상태 */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Target className="h-5 w-5" />
              ArUco 마커 보정
            </CardTitle>
          </CardHeader>
          <CardContent>
            {calibration && calibration.calibrated ? (
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-green-600">
                  <CheckCircle className="h-5 w-5" />
                  <span className="font-medium">보정 완료</span>
                </div>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="rounded-lg bg-gray-100 p-3">
                    <p className="text-gray-500">감지된 마커</p>
                    <p className="text-lg font-bold">{calibration.stored_markers?.length || 0}개</p>
                  </div>
                  <div className="rounded-lg bg-gray-100 p-3">
                    <p className="text-gray-500">측정 거리</p>
                    <p className="text-lg font-bold">{calibration.marker_distance_m?.toFixed(2) || "-"}m</p>
                  </div>
                  <div className="rounded-lg bg-gray-100 p-3">
                    <p className="text-gray-500">해상도</p>
                    <p className="text-lg font-bold">
                      {calibration.pixels_per_meter?.toFixed(0) || "-"} px/m
                    </p>
                  </div>
                  <div className="rounded-lg bg-gray-100 p-3">
                    <p className="text-gray-500">측정 구간</p>
                    <p className="text-lg font-bold">
                      {calibration.start_x ? Math.round(calibration.start_x) : "-"} - {calibration.finish_x ? Math.round(calibration.finish_x) : "-"} px
                    </p>
                  </div>
                </div>
              </div>
            ) : status === "calibrating" ? (
              <div className="flex items-center gap-3 text-yellow-600">
                <Loader2 className="h-5 w-5 animate-spin" />
                <span>마커를 찾는 중...</span>
              </div>
            ) : (
              <div className="flex items-center gap-3 text-gray-500">
                <AlertCircle className="h-5 w-5" />
                <span>대기 중</span>
              </div>
            )}
          </CardContent>
        </Card>

        {/* 실시간 분석 데이터 */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Activity className="h-5 w-5" />
              실시간 분석
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4">
              <div className="rounded-lg bg-gray-100 p-3 text-center">
                <p className="text-xs text-gray-500">현재 프레임</p>
                <p className="text-xl font-bold">
                  {currentFrame?.frame_number || "-"}
                </p>
              </div>
              <div className="rounded-lg bg-gray-100 p-3 text-center">
                <p className="text-xs text-gray-500">측정 구간 내</p>
                <p className="text-xl font-bold">
                  {currentFrame?.in_zone ? (
                    <span className="text-green-600">예</span>
                  ) : (
                    <span className="text-gray-400">아니오</span>
                  )}
                </p>
              </div>
              <div className="rounded-lg bg-gray-100 p-3 text-center">
                <p className="text-xs text-gray-500">중심 위치</p>
                <p className="text-xl font-bold">
                  {currentFrame?.center_x
                    ? `${Math.round(currentFrame.center_x)}px`
                    : "-"}
                </p>
              </div>
            </div>

            {/* 라인 통과 이벤트 */}
            {crossingEvents.length > 0 && (
              <div className="mt-4 space-y-2">
                <p className="text-sm font-medium">라인 통과 이벤트</p>
                <div className="space-y-1">
                  {startEvent && (
                    <div className="flex items-center justify-between rounded bg-green-50 px-3 py-2 text-sm">
                      <span className="text-green-700">시작선 통과</span>
                      <span className="font-mono text-green-600">
                        {((startEvent as any).timestamp_s ?? (startEvent as any).timestamp ?? 0).toFixed(2)}s
                      </span>
                    </div>
                  )}
                  {finishEvent && (
                    <div className="flex items-center justify-between rounded bg-blue-50 px-3 py-2 text-sm">
                      <span className="text-blue-700">종료선 통과</span>
                      <span className="font-mono text-blue-600">
                        {((finishEvent as any).timestamp_s ?? (finishEvent as any).timestamp ?? 0).toFixed(2)}s
                      </span>
                    </div>
                  )}
                  {finishEvent && elapsedTime !== null && (
                    <div className="flex items-center justify-between rounded bg-purple-50 px-3 py-2 text-sm">
                      <span className="text-purple-700 font-medium">소요 시간</span>
                      <span className="font-mono text-purple-600 text-lg font-bold">
                        {elapsedTime.toFixed(3)}s
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* 에러 표시 */}
        {error && (
          <Card className="mb-6 border-red-200 bg-red-50">
            <CardContent className="flex items-center gap-3 p-4">
              <AlertCircle className="h-5 w-5 text-red-500" />
              <p className="text-red-700">{error}</p>
            </CardContent>
          </Card>
        )}

        {/* 컨트롤 버튼 */}
        <div className="flex gap-3">
          {isAnalyzing ? (
            <Button variant="outline" className="flex-1" onClick={pauseAnalysis}>
              <Pause className="mr-2 h-4 w-4" />
              일시정지
            </Button>
          ) : status !== "completed" && status !== "error" ? (
            <Button variant="outline" className="flex-1" onClick={resumeAnalysis}>
              <Play className="mr-2 h-4 w-4" />
              재개
            </Button>
          ) : null}
          <Button variant="destructive" className="flex-1" onClick={handleCancel}>
            <X className="mr-2 h-4 w-4" />
            분석 취소
          </Button>
        </div>
      </main>
    </div>
  );
}
