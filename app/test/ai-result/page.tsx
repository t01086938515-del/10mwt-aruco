"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useAppSelector, useAppDispatch } from "@/store/hooks";
import { useAIAnalysis } from "@/lib/useAIAnalysis";
import { resetSession } from "@/store/slices/testSessionSlice";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { JudgmentVariable, GaitJudgment, GaitMetrics } from "@/lib/websocket";
import {
  ArrowLeft,
  Save,
  Share2,
  Home,
  RefreshCw,
  FileText,
  CheckCircle,
  TrendingUp,
  Clock,
  Footprints,
  Activity,
  BarChart3,
  AlertTriangle,
  Loader2,
  Info,
  Zap,
  Play,
  Pause,
  X,
  Rewind,
  SkipBack,
  SkipForward,
  Gauge,
} from "lucide-react";

// 판정 색상 → Tailwind 클래스 매핑
const COLOR_MAP: Record<string, { bg: string; text: string; border: string }> = {
  green:  { bg: "bg-green-50",  text: "text-green-700",  border: "border-green-200" },
  yellow: { bg: "bg-yellow-50", text: "text-yellow-700", border: "border-yellow-300" },
  orange: { bg: "bg-orange-50", text: "text-orange-700", border: "border-orange-200" },
  red:    { bg: "bg-red-50",    text: "text-red-700",    border: "border-red-200" },
  gray:   { bg: "bg-gray-50",   text: "text-gray-500",   border: "border-gray-200" },
};

// 변수 그룹 정의 (UI 표시 순서)
const VARIABLE_GROUPS = [
  {
    label: "보행 지표",
    keys: ["gait_velocity_ms", "cadence_spm"],
  },
  {
    label: "거리 변수",
    keys: ["left_step_length_cm", "right_step_length_cm", "left_stride_length_cm", "right_stride_length_cm"],
  },
  {
    label: "시간 변수",
    keys: ["step_time_s", "stride_time_s"],
  },
  {
    label: "비율 변수",
    keys: ["stance_ratio_pct", "swing_ratio_pct"],
  },
];

export default function AIResultPage() {
  const router = useRouter();
  const dispatch = useAppDispatch();
  const { config } = useAppSelector((state) => state.testSession);
  const { result, metrics, videoId, reset, status } = useAIAnalysis();
  const [waitCount, setWaitCount] = useState(0);
  const [clipModal, setClipModal] = useState<{
    open: boolean;
    label: string;
    startS: number;
    endS: number;
    variableName: string;
    foot: 'left' | 'right' | null;
  }>({ open: false, label: "", startS: 0, endS: 0, variableName: "", foot: null });
  const videoRef = useRef<HTMLVideoElement>(null);
  const clipCanvasRef = useRef<HTMLCanvasElement>(null);
  const clipAnimRef = useRef<number>(0);

  const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

  // Analysis Player state
  const playerVideoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [playerTime, setPlayerTime] = useState(0);
  const [showPlayer, setShowPlayer] = useState(true);

  // evidence_clips from analysis results
  const evidenceClips: Record<string, { start_s: number; end_s: number; label: string }> =
    (result as any)?.results?.evidence_clips || {};

  const openClip = useCallback((variableName: string) => {
    const clip = evidenceClips[variableName];
    if (!clip) return;
    const foot = variableName.startsWith('left_') ? 'left' as const
      : variableName.startsWith('right_') ? 'right' as const : null;
    setClipModal({
      open: true, label: clip.label,
      startS: clip.start_s, endS: clip.end_s,
      variableName, foot,
    });
  }, [evidenceClips]);

  // Timeline data from analysis results
  const metricsData = (result as any)?.results as GaitMetrics | undefined;
  const timeline = metricsData?.analysis_timeline || [];
  const timelineMeta = metricsData?.timeline_meta;
  const stepEventsAll = metricsData?.step_events || [];

  // Find the closest timeline entry for a given timestamp
  const getTimelineEntry = useCallback((t: number) => {
    if (timeline.length === 0) return null;
    // Binary search for closest entry
    let lo = 0, hi = timeline.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (timeline[mid].t < t) lo = mid + 1;
      else hi = mid;
    }
    // Check neighbors for closest
    const idx = lo;
    if (idx > 0 && Math.abs(timeline[idx - 1].t - t) < Math.abs(timeline[idx].t - t)) {
      return timeline[idx - 1];
    }
    return timeline[idx];
  }, [timeline]);

  // Canvas overlay rendering
  const renderOverlay = useCallback(() => {
    const video = playerVideoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !timelineMeta) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const currentTime = video.currentTime;
    setPlayerTime(currentTime);

    // Scale canvas to video display size
    const rect = video.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Scale factors: video natural size → canvas display size
    const scaleX = canvas.width / (video.videoWidth || 1);
    const scaleY = canvas.height / (video.videoHeight || 1);

    const entry = getTimelineEntry(currentTime);
    if (!entry) return;

    // Only render if within analysis time range
    if (currentTime < timelineMeta.start_t - 0.5 || currentTime > timelineMeta.finish_t + 0.5) return;

    // Find recent step events (within ±0.5s of current time)
    const recentSteps = stepEventsAll.filter(
      (s) => Math.abs(s.time - currentTime) < 0.8
    );

    // ── Draw step length lines between feet ──
    recentSteps.forEach((step, idx) => {
      const lx = step.left_x * scaleX;
      const rx = step.right_x * scaleX;
      const py = (step.peak_y || entry.ly) * scaleY;

      // Line between feet
      ctx.beginPath();
      ctx.moveTo(lx, py);
      ctx.lineTo(rx, py);
      ctx.strokeStyle = step.leading_foot === 'left' ? 'rgba(59, 130, 246, 0.8)' : 'rgba(239, 68, 68, 0.8)';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Step number label
      const stepNum = stepEventsAll.indexOf(step) + 1;
      const midX = (lx + rx) / 2;
      ctx.font = `bold ${Math.max(12, canvas.width * 0.022)}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      ctx.fillRect(midX - 18, py - 22, 36, 18);
      ctx.fillStyle = step.leading_foot === 'left' ? '#2563eb' : '#dc2626';
      ctx.fillText(`#${stepNum}`, midX, py - 8);
    });

    // ── Draw ankle dots ──
    // Left ankle (blue)
    ctx.beginPath();
    ctx.arc(entry.lx * scaleX, entry.ly * scaleY, 6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(59, 130, 246, 0.9)';
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Right ankle (red)
    ctx.beginPath();
    ctx.arc(entry.rx * scaleX, entry.ry * scaleY, 6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();

    // ── HUD overlay (top-left) ──
    const hudX = 12;
    const hudY = 12;
    const lineH = Math.max(22, canvas.height * 0.04);
    const fontSize = Math.max(13, canvas.width * 0.025);

    // Semi-transparent background
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    const hudW = Math.max(180, canvas.width * 0.32);
    const hudH = lineH * 4.5 + 10;
    ctx.beginPath();
    ctx.moveTo(hudX + 8, hudY);
    ctx.lineTo(hudX + hudW - 8, hudY);
    ctx.quadraticCurveTo(hudX + hudW, hudY, hudX + hudW, hudY + 8);
    ctx.lineTo(hudX + hudW, hudY + hudH - 8);
    ctx.quadraticCurveTo(hudX + hudW, hudY + hudH, hudX + hudW - 8, hudY + hudH);
    ctx.lineTo(hudX + 8, hudY + hudH);
    ctx.quadraticCurveTo(hudX, hudY + hudH, hudX, hudY + hudH - 8);
    ctx.lineTo(hudX, hudY + 8);
    ctx.quadraticCurveTo(hudX, hudY, hudX + 8, hudY);
    ctx.closePath();
    ctx.fill();

    ctx.font = `bold ${fontSize}px sans-serif`;
    ctx.textAlign = 'left';

    // Speed
    ctx.fillStyle = '#22d3ee';
    ctx.fillText(`Speed: ${entry.spd.toFixed(2)} m/s`, hudX + 10, hudY + lineH);

    // Cadence
    ctx.fillStyle = '#a78bfa';
    ctx.fillText(`Cadence: ${entry.cad.toFixed(0)} steps/min`, hudX + 10, hudY + lineH * 2);

    // Step count
    ctx.fillStyle = '#34d399';
    ctx.fillText(`Steps: ${entry.cs} / ${timelineMeta.total_steps}`, hudX + 10, hudY + lineH * 3);

    // Elapsed time
    const elapsed = currentTime - timelineMeta.start_t;
    ctx.fillStyle = '#fbbf24';
    ctx.fillText(`Time: ${elapsed > 0 ? elapsed.toFixed(1) : '0.0'}s`, hudX + 10, hudY + lineH * 4);

    // Continue animation loop
    if (!video.paused) {
      animFrameRef.current = requestAnimationFrame(renderOverlay);
    }
  }, [getTimelineEntry, timelineMeta, stepEventsAll]);

  // Start/stop animation loop when playing
  useEffect(() => {
    if (isPlaying && showPlayer) {
      animFrameRef.current = requestAnimationFrame(renderOverlay);
    }
    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [isPlaying, showPlayer, renderOverlay]);

  // Player controls
  const togglePlay = useCallback(() => {
    const video = playerVideoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play().catch(() => {});
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
      // Render one more frame for current position
      renderOverlay();
    }
  }, [renderOverlay]);

  const seekToStart = useCallback(() => {
    const video = playerVideoRef.current;
    if (!video || !timelineMeta) return;
    video.currentTime = timelineMeta.start_t;
    renderOverlay();
  }, [timelineMeta, renderOverlay]);

  const stepFrame = useCallback((direction: number) => {
    const video = playerVideoRef.current;
    if (!video || !timelineMeta) return;
    video.currentTime += direction / timelineMeta.fps;
    renderOverlay();
  }, [timelineMeta, renderOverlay]);

  const changeSpeed = useCallback((rate: number) => {
    const video = playerVideoRef.current;
    if (!video) return;
    video.playbackRate = rate;
    setPlaybackRate(rate);
  }, []);

  // ═══ Clip Modal: Canvas overlay for step visualization ═══
  const renderClipOverlay = useCallback(() => {
    const video = videoRef.current;
    const canvas = clipCanvasRef.current;
    if (!video || !canvas || !timelineMeta) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const currentTime = video.currentTime;
    const rect = video.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / (video.videoWidth || 1);
    const scaleY = canvas.height / (video.videoHeight || 1);

    const entry = getTimelineEntry(currentTime);
    if (!entry) return;

    const isStepVar = clipModal.variableName.includes('step_length') || clipModal.variableName.includes('stride_length');
    const foot = clipModal.foot;

    // Find the step event closest to this clip's time range
    const clipSteps = stepEventsAll.filter(
      (s) => s.time >= clipModal.startS - 0.2 && s.time <= clipModal.endS + 0.2
    );
    // Find the specific step event for the foot
    const targetStep = foot
      ? clipSteps.find(s => s.leading_foot === foot) || clipSteps[0]
      : clipSteps[Math.floor(clipSteps.length / 2)];

    // ── Draw ankle positions (always) ──
    // Left ankle (blue)
    ctx.beginPath();
    ctx.arc(entry.lx * scaleX, entry.ly * scaleY, 8, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(59, 130, 246, 0.9)';
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Right ankle (red)
    ctx.beginPath();
    ctx.arc(entry.rx * scaleX, entry.ry * scaleY, 8, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Foot labels
    const labelSize = Math.max(11, canvas.width * 0.02);
    ctx.font = `bold ${labelSize}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillStyle = '#3b82f6';
    ctx.fillText('L', entry.lx * scaleX, entry.ly * scaleY - 14);
    ctx.fillStyle = '#ef4444';
    ctx.fillText('R', entry.rx * scaleX, entry.ry * scaleY - 14);

    // ── Step/Stride length visualization ──
    if (isStepVar && targetStep) {
      const stepTime = targetStep.time;
      // Find previous step event (the stance foot position at start of this step)
      const stepIdx = stepEventsAll.indexOf(targetStep);
      const prevStep = stepIdx > 0 ? stepEventsAll[stepIdx - 1] : null;

      if (prevStep) {
        // Anchor point = stance foot position (previous step's leading foot position)
        const anchorFoot = prevStep.leading_foot;
        const anchorX = anchorFoot === 'left'
          ? prevStep.left_x * scaleX
          : prevStep.right_x * scaleX;
        const anchorY = (prevStep.peak_y || entry.ly) * scaleY;

        // Target point = leading foot's current position (animates over time)
        const leadingFoot = targetStep.leading_foot;
        const targetX = leadingFoot === 'left'
          ? targetStep.left_x * scaleX
          : targetStep.right_x * scaleX;

        // Current leading foot position (from timeline entry)
        const currentX = leadingFoot === 'left'
          ? entry.lx * scaleX
          : entry.rx * scaleX;
        const currentY = leadingFoot === 'left'
          ? entry.ly * scaleY
          : entry.ry * scaleY;

        // Progress: how far through the step are we?
        const stepDuration = stepTime - prevStep.time;
        const elapsed = currentTime - prevStep.time;
        const progress = Math.min(1, Math.max(0, stepDuration > 0 ? elapsed / stepDuration : 0));

        // Animated endpoint: lerp from anchor to current foot position
        const lineEndX = currentX;
        const lineEndY = currentY;

        // ── Growing measurement line ──
        // Thick colored line from anchor to current foot
        const lineColor = leadingFoot === 'left' ? '#3b82f6' : '#ef4444';
        const lineColorBg = leadingFoot === 'left' ? 'rgba(59,130,246,0.15)' : 'rgba(239,68,68,0.15)';

        // Background glow
        ctx.beginPath();
        ctx.moveTo(anchorX, anchorY);
        ctx.lineTo(lineEndX, lineEndY);
        ctx.strokeStyle = lineColorBg;
        ctx.lineWidth = 20;
        ctx.stroke();

        // Main line
        ctx.beginPath();
        ctx.moveTo(anchorX, anchorY);
        ctx.lineTo(lineEndX, lineEndY);
        ctx.strokeStyle = lineColor;
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Anchor dot (stance foot)
        ctx.beginPath();
        ctx.arc(anchorX, anchorY, 10, 0, Math.PI * 2);
        ctx.fillStyle = anchorFoot === 'left' ? 'rgba(59,130,246,0.7)' : 'rgba(239,68,68,0.7)';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Leading foot dot (moving)
        ctx.beginPath();
        ctx.arc(lineEndX, lineEndY, 10, 0, Math.PI * 2);
        ctx.fillStyle = lineColor;
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();

        // ── Distance label on the line ──
        const ppm = timelineMeta.ppm || 100;
        const distPx = Math.abs(lineEndX / scaleX - anchorX / scaleX);
        const distCm = (distPx / ppm) * 100;
        const midX = (anchorX + lineEndX) / 2;
        const midY = Math.min(anchorY, lineEndY) - 20;

        // Label background
        const fontSize = Math.max(14, canvas.width * 0.03);
        ctx.font = `bold ${fontSize}px sans-serif`;
        const labelText = `${distCm.toFixed(1)} cm`;
        const textWidth = ctx.measureText(labelText).width;
        ctx.fillStyle = 'rgba(0,0,0,0.75)';
        ctx.beginPath();
        const lbx = midX - textWidth / 2 - 8;
        const lby = midY - fontSize - 4;
        const lbw = textWidth + 16;
        const lbh = fontSize + 10;
        ctx.moveTo(lbx + 6, lby);
        ctx.lineTo(lbx + lbw - 6, lby);
        ctx.quadraticCurveTo(lbx + lbw, lby, lbx + lbw, lby + 6);
        ctx.lineTo(lbx + lbw, lby + lbh - 6);
        ctx.quadraticCurveTo(lbx + lbw, lby + lbh, lbx + lbw - 6, lby + lbh);
        ctx.lineTo(lbx + 6, lby + lbh);
        ctx.quadraticCurveTo(lbx, lby + lbh, lbx, lby + lbh - 6);
        ctx.lineTo(lbx, lby + 6);
        ctx.quadraticCurveTo(lbx, lby, lbx + 6, lby);
        ctx.closePath();
        ctx.fill();

        ctx.fillStyle = lineColor;
        ctx.textAlign = 'center';
        ctx.fillText(labelText, midX, midY);

        // ── Phase indicator (bottom) ──
        const phaseH = 30;
        const phaseY = canvas.height - phaseH - 10;
        const phaseW = canvas.width - 40;
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(20, phaseY, phaseW, phaseH);

        // Progress bar
        ctx.fillStyle = lineColor;
        ctx.fillRect(20, phaseY, phaseW * progress, phaseH);

        // Phase text
        ctx.font = `bold ${Math.max(11, canvas.width * 0.02)}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.fillStyle = 'white';
        const phaseText = progress < 0.3 ? 'Initial Contact → Loading'
          : progress < 0.7 ? 'Mid Stance → Terminal Stance'
          : 'Pre-Swing → Swing';
        ctx.fillText(
          `${(leadingFoot === 'left' ? 'L' : 'R')} Step  |  ${phaseText}  |  ${(progress * 100).toFixed(0)}%`,
          canvas.width / 2, phaseY + phaseH / 2 + 4
        );
      }
    }

    // ── HUD (top-right): speed, time ──
    const hudSize = Math.max(12, canvas.width * 0.022);
    ctx.font = `bold ${hudSize}px sans-serif`;
    ctx.textAlign = 'right';
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(canvas.width - 150, 8, 142, hudSize * 2.5 + 8);
    ctx.fillStyle = '#22d3ee';
    ctx.fillText(`Speed: ${entry.spd.toFixed(2)} m/s`, canvas.width - 16, 8 + hudSize + 2);
    ctx.fillStyle = '#fbbf24';
    const elapsedT = currentTime - timelineMeta.start_t;
    ctx.fillText(`Time: ${elapsedT > 0 ? elapsedT.toFixed(2) : '0.00'}s`, canvas.width - 16, 8 + hudSize * 2.2 + 2);

    if (!video.paused) {
      clipAnimRef.current = requestAnimationFrame(renderClipOverlay);
    }
  }, [clipModal, getTimelineEntry, timelineMeta, stepEventsAll]);

  // 모달 열릴 때 영상 시간 설정 + overlay 시작
  useEffect(() => {
    if (clipModal.open && videoRef.current) {
      const video = videoRef.current;
      // 슬로우모션 기본: 거리→0.25x, 시간/비율→0.1x, 기타→0.5x
      const isDistVar = clipModal.variableName.includes('step_length') || clipModal.variableName.includes('stride_length');
      const isTimeVar = clipModal.variableName.includes('time_s') || clipModal.variableName.includes('swing') || clipModal.variableName.includes('stance');
      video.playbackRate = isTimeVar ? 0.1 : isDistVar ? 0.25 : 0.5;
      video.currentTime = clipModal.startS;
      video.play().catch(() => {});

      // 종료 시간 도달 시 처음으로 루프
      const checkEnd = () => {
        if (video.currentTime >= clipModal.endS) {
          video.pause();
          video.currentTime = clipModal.startS;
        }
      };
      video.addEventListener("timeupdate", checkEnd);

      // overlay animation
      const startOverlay = () => {
        clipAnimRef.current = requestAnimationFrame(renderClipOverlay);
      };
      const stopOverlay = () => {
        renderClipOverlay(); // render one last frame
      };
      video.addEventListener("play", startOverlay);
      video.addEventListener("pause", stopOverlay);
      video.addEventListener("seeked", () => renderClipOverlay());

      return () => {
        video.removeEventListener("timeupdate", checkEnd);
        video.removeEventListener("play", startOverlay);
        video.removeEventListener("pause", stopOverlay);
        if (clipAnimRef.current) cancelAnimationFrame(clipAnimRef.current);
      };
    }
  }, [clipModal.open, clipModal.startS, clipModal.endS, clipModal.variableName, renderClipOverlay]);

  useEffect(() => {
    if (!config) {
      const timer = setTimeout(() => {
        router.push("/test/setup");
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [config, router]);

  useEffect(() => {
    if (!result && config && waitCount < 10) {
      const timer = setTimeout(() => {
        setWaitCount((prev) => prev + 1);
      }, 500);
      return () => clearTimeout(timer);
    } else if (!result && waitCount >= 10) {
      router.push("/test/setup");
    }
  }, [result, config, waitCount, router]);

  if (!config || !result || !metrics) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Card className="w-80">
          <CardContent className="p-8 text-center">
            <Loader2 className="h-12 w-12 animate-spin mx-auto mb-4 text-blue-500" />
            <p className="text-lg font-medium">결과 로딩 중...</p>
            <p className="text-sm text-gray-500 mt-2">
              분석 결과를 불러오고 있습니다
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const handleNewTest = () => {
    reset();
    dispatch(resetSession());
    router.push("/test/setup");
  };

  const handleGoHome = () => {
    reset();
    dispatch(resetSession());
    router.push("/");
  };

  const getSymmetryStatus = (value: number | null) => {
    if (value === null) return { label: "-", color: "text-gray-400" };
    if (value <= 5) return { label: "우수", color: "text-green-600" };
    if (value <= 10) return { label: "양호", color: "text-blue-600" };
    if (value <= 15) return { label: "주의", color: "text-yellow-600" };
    return { label: "비대칭", color: "text-red-600" };
  };

  const judgment = metrics.judgment as GaitJudgment | undefined;
  const velocity = judgment?.velocity;
  const clinical = metrics.clinical;

  // 변수 판정 결과를 키로 빠르게 조회
  const varMap = new Map<string, JudgmentVariable>();
  if (judgment?.variables) {
    for (const v of judgment.variables) {
      varMap.set(v.variable_name, v);
    }
  }

  // Perry 분류 색상 매핑
  const velocityColors = velocity
    ? COLOR_MAP[velocity.color] || COLOR_MAP.gray
    : COLOR_MAP.gray;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-gray-200 bg-gray-50/95 backdrop-blur">
        <div className="flex h-16 items-center justify-between px-4">
          <Button variant="ghost" onClick={handleGoHome}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            홈
          </Button>
          <span className="font-semibold">AI 분석 결과</span>
          <div className="w-16" />
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-2xl p-4 md:p-6">
        {/* 성공 배너 */}
        <Card className="mb-6 border-green-500 bg-green-50">
          <CardContent className="flex items-center gap-4 p-4">
            <div className="rounded-full bg-green-500 p-2">
              <CheckCircle className="h-6 w-6 text-white" />
            </div>
            <div>
              <p className="font-semibold text-green-700">AI 분석 완료</p>
              <p className="text-sm text-green-600">
                {result.total_frames_analyzed}프레임 분석 완료
              </p>
            </div>
          </CardContent>
        </Card>

        {/* ═══ 분석 플레이어 ═══ */}
        {videoId && timeline.length > 0 && (
          <Card className="mb-6 overflow-hidden">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Activity className="h-5 w-5" />
                  보행 분석 영상
                </CardTitle>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowPlayer(!showPlayer)}
                >
                  {showPlayer ? "접기" : "펼치기"}
                </Button>
              </div>
            </CardHeader>
            {showPlayer && (
              <CardContent className="p-0">
                {/* Video + Canvas overlay */}
                <div className="relative bg-black">
                  <video
                    ref={playerVideoRef}
                    src={`${BACKEND_URL}/api/video/${videoId}`}
                    className="w-full"
                    playsInline
                    onPlay={() => {
                      setIsPlaying(true);
                      animFrameRef.current = requestAnimationFrame(renderOverlay);
                    }}
                    onPause={() => {
                      setIsPlaying(false);
                      renderOverlay();
                    }}
                    onSeeked={() => renderOverlay()}
                    onLoadedMetadata={() => {
                      if (playerVideoRef.current && timelineMeta) {
                        playerVideoRef.current.currentTime = timelineMeta.start_t;
                      }
                    }}
                  />
                  <canvas
                    ref={canvasRef}
                    className="absolute inset-0 w-full h-full pointer-events-none"
                  />
                </div>

                {/* Timeline scrubber */}
                {timelineMeta && (
                  <div className="px-4 pt-3 pb-1">
                    <input
                      type="range"
                      min={timelineMeta.start_t}
                      max={timelineMeta.finish_t}
                      step={0.03}
                      value={playerTime}
                      onChange={(e) => {
                        const t = parseFloat(e.target.value);
                        if (playerVideoRef.current) {
                          playerVideoRef.current.currentTime = t;
                          renderOverlay();
                        }
                      }}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
                    />
                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                      <span>{Math.max(0, playerTime - (timelineMeta?.start_t || 0)).toFixed(1)}s</span>
                      <span>{((timelineMeta?.finish_t || 0) - (timelineMeta?.start_t || 0)).toFixed(1)}s</span>
                    </div>
                  </div>
                )}

                {/* Controls */}
                <div className="px-4 pb-4">
                  <div className="flex items-center justify-between">
                    {/* Playback controls */}
                    <div className="flex items-center gap-1">
                      <button
                        onClick={seekToStart}
                        className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
                        title="처음으로"
                      >
                        <Rewind className="h-4 w-4 text-gray-600" />
                      </button>
                      <button
                        onClick={() => stepFrame(-1)}
                        className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
                        title="이전 프레임"
                      >
                        <SkipBack className="h-4 w-4 text-gray-600" />
                      </button>
                      <button
                        onClick={togglePlay}
                        className="p-3 rounded-full bg-blue-500 hover:bg-blue-600 transition-colors text-white"
                      >
                        {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
                      </button>
                      <button
                        onClick={() => stepFrame(1)}
                        className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
                        title="다음 프레임"
                      >
                        <SkipForward className="h-4 w-4 text-gray-600" />
                      </button>
                    </div>

                    {/* Speed controls */}
                    <div className="flex items-center gap-1">
                      <Gauge className="h-4 w-4 text-gray-400" />
                      {[0.25, 0.5, 1.0, 2.0].map((rate) => (
                        <button
                          key={rate}
                          onClick={() => changeSpeed(rate)}
                          className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                            playbackRate === rate
                              ? "bg-blue-500 text-white"
                              : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                          }`}
                        >
                          {rate}x
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Current stats bar */}
                  {(() => {
                    const entry = getTimelineEntry(playerTime);
                    if (!entry) return null;
                    return (
                      <div className="mt-3 grid grid-cols-4 gap-2 text-center">
                        <div className="rounded-lg bg-cyan-50 px-2 py-1.5">
                          <p className="text-xs text-cyan-600">Speed</p>
                          <p className="text-sm font-bold text-cyan-700">{entry.spd.toFixed(2)} m/s</p>
                        </div>
                        <div className="rounded-lg bg-purple-50 px-2 py-1.5">
                          <p className="text-xs text-purple-600">Cadence</p>
                          <p className="text-sm font-bold text-purple-700">{entry.cad.toFixed(0)} spm</p>
                        </div>
                        <div className="rounded-lg bg-emerald-50 px-2 py-1.5">
                          <p className="text-xs text-emerald-600">Steps</p>
                          <p className="text-sm font-bold text-emerald-700">{entry.cs}</p>
                        </div>
                        <div className="rounded-lg bg-amber-50 px-2 py-1.5">
                          <p className="text-xs text-amber-600">Time</p>
                          <p className="text-sm font-bold text-amber-700">
                            {Math.max(0, playerTime - (timelineMeta?.start_t || 0)).toFixed(1)}s
                          </p>
                        </div>
                      </div>
                    );
                  })()}
                </div>
              </CardContent>
            )}
          </Card>
        )}

        {/* 환자 정보 */}
        <Card className="mb-6">
          <CardContent className="p-4">
            <div className="flex items-center gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-[hsl(var(--primary))]/10 text-xl font-bold text-[hsl(var(--primary))]">
                {config.patientName.charAt(0)}
              </div>
              <div>
                <p className="font-medium">{config.patientName}</p>
                <p className="text-sm text-gray-500">
                  측정 거리: {metrics.distance_m?.toFixed(2) || "-"}m
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* ═══ 1. 보행속도 Perry 분류 카드 ═══ */}
        {velocity && (
          <Card className={`mb-6 border-2 ${velocityColors.border}`}>
            <CardContent className={`p-5 ${velocityColors.bg}`}>
              <div className="flex items-center gap-3 mb-3">
                <Zap className={`h-6 w-6 ${velocityColors.text}`} />
                <p className={`font-bold text-lg ${velocityColors.text}`}>
                  보행속도 Perry 분류
                </p>
              </div>
              <div className="flex items-end justify-between">
                <div>
                  <p className={`text-4xl font-bold ${velocityColors.text}`}>
                    {velocity.speed_mps} m/s
                  </p>
                  <p className={`text-sm mt-1 ${velocityColors.text} opacity-80`}>
                    정상 범위: 1.0 ~ 1.4 m/s
                  </p>
                </div>
                <Badge
                  className={`text-base px-4 py-1.5 ${velocityColors.bg} ${velocityColors.text} ${velocityColors.border} border`}
                >
                  {velocity.classification}
                </Badge>
              </div>
              {/* 속도 경고 */}
              {velocity.alerts.length > 0 && (
                <div className="mt-3 space-y-1">
                  {velocity.alerts.map((alert, idx) => (
                    <p key={idx} className={`text-sm ${velocityColors.text}`}>
                      {alert}
                    </p>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* ═══ 2. 핵심 지표 4개 ═══ */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <TrendingUp className="h-5 w-5" />
              핵심 지표
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div className="relative rounded-xl bg-[hsl(var(--primary))]/10 p-4 text-center">
                <TrendingUp className="mx-auto mb-2 h-6 w-6 text-[hsl(var(--primary))]" />
                <p className="text-3xl font-bold text-[hsl(var(--primary))]">
                  {metrics.speed_mps?.toFixed(2) || "-"}
                </p>
                <p className="text-sm text-gray-600">보행 속도 (m/s)</p>
                {evidenceClips['gait_velocity_ms'] && (
                  <button onClick={() => openClip('gait_velocity_ms')}
                    className="absolute top-2 right-2 p-1.5 rounded-full bg-white/80 hover:bg-white shadow-sm transition-colors" title="전체 구간 재생">
                    <Play className="h-3.5 w-3.5 text-[hsl(var(--primary))]" />
                  </button>
                )}
              </div>
              <div className="rounded-xl bg-gray-100 p-4 text-center">
                <Clock className="mx-auto mb-2 h-6 w-6 text-gray-600" />
                <p className="text-3xl font-bold">{metrics.elapsed_time_s?.toFixed(2) || "-"}</p>
                <p className="text-sm text-gray-600">소요 시간 (초)</p>
              </div>
              <div className="relative rounded-xl bg-gray-100 p-4 text-center">
                <Activity className="mx-auto mb-2 h-6 w-6 text-gray-600" />
                <p className="text-3xl font-bold">{metrics.cadence_spm?.toFixed(0) || "-"}</p>
                <p className="text-sm text-gray-600">케이던스 (steps/min)</p>
                {evidenceClips['cadence_spm'] && (
                  <button onClick={() => openClip('cadence_spm')}
                    className="absolute top-2 right-2 p-1.5 rounded-full bg-white/80 hover:bg-white shadow-sm transition-colors" title="케이던스 재생">
                    <Play className="h-3.5 w-3.5 text-gray-600" />
                  </button>
                )}
              </div>
              <div className="rounded-xl bg-gray-100 p-4 text-center">
                <Footprints className="mx-auto mb-2 h-6 w-6 text-gray-600" />
                <p className="text-3xl font-bold">{metrics.step_count || "-"}</p>
                <p className="text-sm text-gray-600">총 걸음 수</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* ═══ 3. 변수별 판정표 ═══ */}
        {judgment && judgment.variables.length > 0 && (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <BarChart3 className="h-5 w-5" />
                변수별 판정
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {VARIABLE_GROUPS.map((group) => {
                const groupVars = group.keys
                  .map((k) => varMap.get(k))
                  .filter((v): v is JudgmentVariable => v !== undefined);

                if (groupVars.length === 0) return null;

                return (
                  <div key={group.label}>
                    {/* 그룹 헤더 */}
                    <div className="bg-gray-100 px-4 py-2 border-b border-t">
                      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                        {group.label}
                      </p>
                    </div>
                    {/* 테이블 헤더 */}
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b bg-gray-50/50">
                            <th className="text-left py-2 pl-4 pr-2 font-medium text-gray-500 w-[35%]">항목</th>
                            <th className="text-center py-2 px-1 font-medium text-gray-500 w-[15%]">측정값</th>
                            <th className="text-center py-2 px-1 font-medium text-gray-500 w-[18%]">정상범위</th>
                            <th className="text-center py-2 px-1 font-medium text-gray-500 w-[12%]">판정</th>
                            <th className="text-left py-2 pl-2 pr-4 font-medium text-gray-500 w-[20%]">코멘트</th>
                          </tr>
                        </thead>
                        <tbody>
                          {groupVars.map((v) => {
                            const colors = COLOR_MAP[v.color] || COLOR_MAP.gray;
                            const isNA = v.status === "N/A";
                            const hasClip = !!evidenceClips[v.variable_name];
                            return (
                              <tr
                                key={v.variable_name}
                                className={`border-b last:border-b-0 ${hasClip ? "cursor-pointer hover:bg-blue-50 active:bg-blue-100 transition-colors" : ""}`}
                                onClick={() => hasClip && openClip(v.variable_name)}
                              >
                                <td className="py-2.5 pl-4 pr-2 text-gray-700 font-medium text-xs">
                                  <span className="flex items-center gap-1.5">
                                    {v.display_name}
                                    {hasClip && <Play className="h-3 w-3 text-blue-400" />}
                                  </span>
                                </td>
                                <td className="text-center py-2.5 px-1 font-mono font-bold">
                                  {isNA ? (
                                    <span className="text-gray-400">N/A</span>
                                  ) : (
                                    <span>{v.measured_value}{v.unit && <span className="text-xs text-gray-400 ml-0.5">{v.unit}</span>}</span>
                                  )}
                                </td>
                                <td className="text-center py-2.5 px-1 text-xs text-gray-500">
                                  {v.normal_range}
                                </td>
                                <td className="text-center py-2.5 px-1">
                                  {isNA ? (
                                    <span className="text-gray-400 text-xs">-</span>
                                  ) : (
                                    <span className={`inline-flex items-center gap-0.5 text-xs font-bold ${colors.text}`}>
                                      {v.status === "정상" ? "정상" : (
                                        <>{v.direction} {v.deviation}</>
                                      )}
                                    </span>
                                  )}
                                </td>
                                <td className="py-2.5 pl-2 pr-4 text-xs text-gray-500">
                                  {v.clinical_comment || "-"}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                );
              })}
            </CardContent>
          </Card>
        )}

        {/* ═══ 4. 좌우 대칭성 - 거리 ═══ */}
        <Card className="mb-6">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Footprints className="h-5 w-5" />
              거리 대칭성
            </CardTitle>
          </CardHeader>
          <CardContent>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2 font-medium text-gray-600">항목</th>
                  <th className="text-center py-2 font-medium text-blue-600 w-16">L</th>
                  <th className="text-center py-2 font-medium text-red-600 w-16">R</th>
                  <th className="text-center py-2 font-medium text-gray-600 w-14">SI%</th>
                  <th className="text-center py-2 font-medium text-gray-500 w-20"></th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="py-3">보폭 Step (cm)</td>
                  <td className="text-center font-mono font-bold">
                    {metrics.left_step_length_m ? (metrics.left_step_length_m * 100).toFixed(0) : "-"}
                  </td>
                  <td className="text-center font-mono font-bold">
                    {metrics.right_step_length_m ? (metrics.right_step_length_m * 100).toFixed(0) : "-"}
                  </td>
                  <td className={`text-center font-mono font-bold ${getSymmetryStatus(metrics.step_length_si).color}`}>
                    {metrics.step_length_si?.toFixed(0) || "-"}
                  </td>
                  <td className="text-center py-2">
                    <div className="flex justify-center gap-1">
                      {evidenceClips['left_step_length_cm'] && (
                        <button onClick={() => openClip('left_step_length_cm')}
                          className="p-1 rounded bg-blue-50 hover:bg-blue-100 transition-colors" title="L Step 재생">
                          <Play className="h-3.5 w-3.5 text-blue-500" />
                        </button>
                      )}
                      {evidenceClips['right_step_length_cm'] && (
                        <button onClick={() => openClip('right_step_length_cm')}
                          className="p-1 rounded bg-red-50 hover:bg-red-100 transition-colors" title="R Step 재생">
                          <Play className="h-3.5 w-3.5 text-red-500" />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="py-3">활보장 Stride (cm)</td>
                  <td className="text-center font-mono font-bold">
                    {metrics.left_stride_length_m ? (metrics.left_stride_length_m * 100).toFixed(0) : "-"}
                  </td>
                  <td className="text-center font-mono font-bold">
                    {metrics.right_stride_length_m ? (metrics.right_stride_length_m * 100).toFixed(0) : "-"}
                  </td>
                  <td className={`text-center font-mono font-bold ${getSymmetryStatus(metrics.stride_length_si).color}`}>
                    {metrics.stride_length_si?.toFixed(0) || "-"}
                  </td>
                  <td className="text-center py-2">
                    <div className="flex justify-center gap-1">
                      {evidenceClips['left_stride_length_cm'] && (
                        <button onClick={() => openClip('left_stride_length_cm')}
                          className="p-1 rounded bg-blue-50 hover:bg-blue-100 transition-colors" title="L Stride 재생">
                          <Play className="h-3.5 w-3.5 text-blue-500" />
                        </button>
                      )}
                      {evidenceClips['right_stride_length_cm'] && (
                        <button onClick={() => openClip('right_stride_length_cm')}
                          className="p-1 rounded bg-red-50 hover:bg-red-100 transition-colors" title="R Stride 재생">
                          <Play className="h-3.5 w-3.5 text-red-500" />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </CardContent>
        </Card>

        {/* ═══ 개별 스텝 분석 ═══ */}
        {stepEventsAll.length > 1 && timelineMeta && (
          <Card className="mb-6">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-lg">
                <Footprints className="h-5 w-5" />
                개별 스텝 분석
              </CardTitle>
              <p className="text-xs text-gray-500 mt-1">각 스텝의 보폭/활보장을 개별 확인</p>
            </CardHeader>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-gray-50">
                      <th className="text-center py-2 px-2 font-medium text-gray-500 w-10">#</th>
                      <th className="text-center py-2 px-1 font-medium text-gray-500 w-10">발</th>
                      <th className="text-center py-2 px-1 font-medium text-gray-500">Step (cm)</th>
                      <th className="text-center py-2 px-1 font-medium text-gray-500">Stride (cm)</th>
                      <th className="text-center py-2 px-1 font-medium text-gray-500">Time (s)</th>
                      <th className="text-center py-2 px-1 font-medium text-gray-500 w-10"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {stepEventsAll.map((step, idx) => {
                      const isLeft = step.leading_foot === 'left';
                      const footColor = isLeft ? 'text-blue-600' : 'text-red-600';
                      const bgColor = isLeft ? 'bg-blue-50/50' : 'bg-red-50/50';
                      // Highlight outliers (>20% from mean)
                      const allStepCms = stepEventsAll
                        .filter(s => s.step_length_cm != null)
                        .map(s => s.step_length_cm!);
                      const meanStep = allStepCms.length > 0
                        ? allStepCms.reduce((a, b) => a + b, 0) / allStepCms.length : 0;
                      const isOutlier = step.step_length_cm != null && meanStep > 0
                        && Math.abs(step.step_length_cm - meanStep) / meanStep > 0.2;

                      return (
                        <tr key={idx} className={`border-b last:border-b-0 ${bgColor}`}>
                          <td className="text-center py-2 px-2 text-xs text-gray-400 font-mono">
                            {step.step_num}
                          </td>
                          <td className={`text-center py-2 px-1 font-bold text-xs ${footColor}`}>
                            {isLeft ? 'L' : 'R'}
                          </td>
                          <td className={`text-center py-2 px-1 font-mono font-bold ${isOutlier ? 'text-orange-600' : ''}`}>
                            {step.step_length_cm != null ? (
                              <span className="flex items-center justify-center gap-1">
                                {step.step_length_cm.toFixed(1)}
                                {isOutlier && <AlertTriangle className="h-3 w-3 text-orange-500" />}
                              </span>
                            ) : '-'}
                          </td>
                          <td className="text-center py-2 px-1 font-mono">
                            {step.stride_length_cm != null ? step.stride_length_cm.toFixed(1) : '-'}
                          </td>
                          <td className="text-center py-2 px-1 font-mono text-gray-600">
                            {step.step_time_s != null ? step.step_time_s.toFixed(3) : '-'}
                          </td>
                          <td className="text-center py-2 px-1">
                            <button
                              onClick={() => {
                                const clipStart = Math.max(timelineMeta?.start_t || 0, step.time - 0.5);
                                const clipEnd = step.time + 0.5;
                                setClipModal({
                                  open: true,
                                  label: `Step #${step.step_num} (${isLeft ? 'L' : 'R'})`,
                                  startS: clipStart,
                                  endS: clipEnd,
                                  variableName: isLeft ? 'left_step_length_cm' : 'right_step_length_cm',
                                  foot: step.leading_foot,
                                });
                              }}
                              className={`p-1 rounded transition-colors ${isLeft ? 'bg-blue-100 hover:bg-blue-200' : 'bg-red-100 hover:bg-red-200'}`}
                              title={`Step #${step.step_num} 재생`}
                            >
                              <Play className={`h-3 w-3 ${footColor}`} />
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {/* Summary bar */}
              {(() => {
                const leftSteps = stepEventsAll.filter(s => s.leading_foot === 'left' && s.step_length_cm != null);
                const rightSteps = stepEventsAll.filter(s => s.leading_foot === 'right' && s.step_length_cm != null);
                const leftAvg = leftSteps.length > 0
                  ? leftSteps.reduce((a, b) => a + b.step_length_cm!, 0) / leftSteps.length : 0;
                const rightAvg = rightSteps.length > 0
                  ? rightSteps.reduce((a, b) => a + b.step_length_cm!, 0) / rightSteps.length : 0;
                const leftMin = leftSteps.length > 0 ? Math.min(...leftSteps.map(s => s.step_length_cm!)) : 0;
                const leftMax = leftSteps.length > 0 ? Math.max(...leftSteps.map(s => s.step_length_cm!)) : 0;
                const rightMin = rightSteps.length > 0 ? Math.min(...rightSteps.map(s => s.step_length_cm!)) : 0;
                const rightMax = rightSteps.length > 0 ? Math.max(...rightSteps.map(s => s.step_length_cm!)) : 0;
                return (
                  <div className="px-4 py-3 bg-gray-50 border-t grid grid-cols-2 gap-3 text-xs">
                    <div className="rounded-lg bg-blue-50 px-3 py-2 border border-blue-100">
                      <p className="font-bold text-blue-700 mb-1">L 평균: {leftAvg.toFixed(1)}cm</p>
                      <p className="text-blue-500">범위: {leftMin.toFixed(0)}~{leftMax.toFixed(0)}cm ({leftSteps.length}회)</p>
                    </div>
                    <div className="rounded-lg bg-red-50 px-3 py-2 border border-red-100">
                      <p className="font-bold text-red-700 mb-1">R 평균: {rightAvg.toFixed(1)}cm</p>
                      <p className="text-red-500">범위: {rightMin.toFixed(0)}~{rightMax.toFixed(0)}cm ({rightSteps.length}회)</p>
                    </div>
                  </div>
                );
              })()}
            </CardContent>
          </Card>
        )}

        {/* ═══ 4. 좌우 대칭성 - 시간 ═══ */}
        <Card className="mb-6">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Clock className="h-5 w-5" />
              시간 대칭성
            </CardTitle>
          </CardHeader>
          <CardContent>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2 font-medium text-gray-600">항목</th>
                  <th className="text-center py-2 font-medium text-blue-600 w-16">L</th>
                  <th className="text-center py-2 font-medium text-red-600 w-16">R</th>
                  <th className="text-center py-2 font-medium text-gray-600 w-14">SI%</th>
                  <th className="text-center py-2 font-medium text-gray-500 w-14"></th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="py-3">Step Time (s)</td>
                  <td className="text-center font-mono font-bold">
                    {metrics.left_step_time_s?.toFixed(2) || "-"}
                  </td>
                  <td className="text-center font-mono font-bold">
                    {metrics.right_step_time_s?.toFixed(2) || "-"}
                  </td>
                  <td className={`text-center font-mono font-bold ${getSymmetryStatus(metrics.step_time_si).color}`}>
                    {metrics.step_time_si?.toFixed(0) || "-"}
                  </td>
                  <td className="text-center py-2">
                    {evidenceClips['step_time_s'] && (
                      <button onClick={() => openClip('step_time_s')}
                        className="p-1 rounded bg-gray-50 hover:bg-gray-100 transition-colors" title="Step Time 재생">
                        <Play className="h-3.5 w-3.5 text-gray-500" />
                      </button>
                    )}
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="py-3">Stride Time (s)</td>
                  <td className="text-center font-mono font-bold">
                    {metrics.left_stride_time_s?.toFixed(2) || "-"}
                  </td>
                  <td className="text-center font-mono font-bold">
                    {metrics.right_stride_time_s?.toFixed(2) || "-"}
                  </td>
                  <td className={`text-center font-mono font-bold ${getSymmetryStatus(metrics.stride_time_si).color}`}>
                    {metrics.stride_time_si?.toFixed(0) || "-"}
                  </td>
                  <td className="text-center py-2">
                    {evidenceClips['stride_time_s'] && (
                      <button onClick={() => openClip('stride_time_s')}
                        className="p-1 rounded bg-gray-50 hover:bg-gray-100 transition-colors" title="Stride Time 재생">
                        <Play className="h-3.5 w-3.5 text-gray-500" />
                      </button>
                    )}
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="py-3">Swing (s)</td>
                  <td className="text-center font-mono font-bold">
                    {metrics.left_swing_time_s?.toFixed(2) || "-"}
                  </td>
                  <td className="text-center font-mono font-bold">
                    {metrics.right_swing_time_s?.toFixed(2) || "-"}
                  </td>
                  <td className={`text-center font-mono font-bold ${getSymmetryStatus(metrics.swing_time_si).color}`}>
                    {metrics.swing_time_si?.toFixed(0) || "-"}
                  </td>
                  <td className="text-center py-2">
                    {evidenceClips['swing_ratio_pct'] && (
                      <button onClick={() => openClip('swing_ratio_pct')}
                        className="p-1 rounded bg-gray-50 hover:bg-gray-100 transition-colors" title="Swing 재생">
                        <Play className="h-3.5 w-3.5 text-gray-500" />
                      </button>
                    )}
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="py-3">Stance (s)</td>
                  <td className="text-center font-mono font-bold">
                    {metrics.left_stance_time_s?.toFixed(2) || "-"}
                  </td>
                  <td className="text-center font-mono font-bold">
                    {metrics.right_stance_time_s?.toFixed(2) || "-"}
                  </td>
                  <td className={`text-center font-mono font-bold ${getSymmetryStatus(metrics.stance_time_si).color}`}>
                    {metrics.stance_time_si?.toFixed(0) || "-"}
                  </td>
                  <td className="text-center py-2">
                    {evidenceClips['stance_ratio_pct'] && (
                      <button onClick={() => openClip('stance_ratio_pct')}
                        className="p-1 rounded bg-gray-50 hover:bg-gray-100 transition-colors" title="Stance 재생">
                        <Play className="h-3.5 w-3.5 text-gray-500" />
                      </button>
                    )}
                  </td>
                </tr>
              </tbody>
            </table>
          </CardContent>
        </Card>

        {/* ═══ 4. 좌우 대칭성 - 비율 ═══ */}
        <Card className="mb-6">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <BarChart3 className="h-5 w-5" />
              비율 대칭성
            </CardTitle>
            <p className="text-xs text-gray-500 mt-1">정상: Swing 40% / Stance 60%</p>
          </CardHeader>
          <CardContent>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2 font-medium text-gray-600">항목</th>
                  <th className="text-center py-2 font-medium text-blue-600 w-16">L</th>
                  <th className="text-center py-2 font-medium text-red-600 w-16">R</th>
                  <th className="text-center py-2 font-medium text-gray-600 w-14">SI%</th>
                  <th className="text-center py-2 font-medium text-gray-500 w-14"></th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="py-3">Swing %</td>
                  <td className="text-center font-mono font-bold">
                    {metrics.left_swing_pct?.toFixed(0) || "-"}
                  </td>
                  <td className="text-center font-mono font-bold">
                    {metrics.right_swing_pct?.toFixed(0) || "-"}
                  </td>
                  <td className={`text-center font-mono font-bold ${getSymmetryStatus(metrics.swing_stance_si).color}`}>
                    {metrics.swing_stance_si?.toFixed(0) || "-"}
                  </td>
                  <td className="text-center py-2">
                    {evidenceClips['swing_ratio_pct'] && (
                      <button onClick={() => openClip('swing_ratio_pct')}
                        className="p-1 rounded bg-gray-50 hover:bg-gray-100 transition-colors" title="Swing 재생">
                        <Play className="h-3.5 w-3.5 text-gray-500" />
                      </button>
                    )}
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="py-3">Stance %</td>
                  <td className="text-center font-mono font-bold">
                    {metrics.left_stance_pct?.toFixed(0) || "-"}
                  </td>
                  <td className="text-center font-mono font-bold">
                    {metrics.right_stance_pct?.toFixed(0) || "-"}
                  </td>
                  <td className="text-center font-mono font-bold text-gray-400">-</td>
                  <td className="text-center py-2">
                    {evidenceClips['stance_ratio_pct'] && (
                      <button onClick={() => openClip('stance_ratio_pct')}
                        className="p-1 rounded bg-gray-50 hover:bg-gray-100 transition-colors" title="Stance 재생">
                        <Play className="h-3.5 w-3.5 text-gray-500" />
                      </button>
                    )}
                  </td>
                </tr>
              </tbody>
            </table>

            {/* 종합 대칭성 지수 */}
            {metrics.overall_symmetry_index !== null && (
              <div className="mt-4 flex items-center justify-between rounded-lg bg-blue-50 px-4 py-3 border border-blue-200">
                <span className="text-sm font-medium text-blue-700">종합 대칭성 지수 (Overall SI)</span>
                <div className="flex items-center gap-3">
                  <span className="font-mono text-xl font-bold text-blue-700">
                    {metrics.overall_symmetry_index?.toFixed(1)}%
                  </span>
                  <span className={`text-sm font-medium ${getSymmetryStatus(metrics.overall_symmetry_index).color}`}>
                    {getSymmetryStatus(metrics.overall_symmetry_index).label}
                  </span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* ═══ 5. 패턴 분석 ═══ */}
        {judgment && judgment.patterns.length > 0 && (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <AlertTriangle className="h-5 w-5" />
                패턴 분석
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {judgment.patterns.map((pattern, idx) => {
                const pColors = COLOR_MAP[pattern.color] || COLOR_MAP.gray;
                return (
                  <div
                    key={idx}
                    className={`rounded-lg border p-4 ${pColors.bg} ${pColors.border}`}
                  >
                    <p className={`font-bold text-sm mb-1 ${pColors.text}`}>
                      {pattern.label}
                    </p>
                    <p className={`text-sm ${pColors.text} opacity-90`}>
                      {pattern.comment}
                    </p>
                  </div>
                );
              })}
            </CardContent>
          </Card>
        )}

        {/* ═══ 임상 해석 (기존) ═══ */}
        {clinical && (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <Activity className="h-5 w-5" />
                임상 해석
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className={`rounded-lg border p-4 ${
                clinical.fall_risk === "high"
                  ? "bg-red-100 text-red-700 border-red-200"
                  : clinical.fall_risk === "moderate"
                  ? "bg-yellow-100 text-yellow-700 border-yellow-200"
                  : "bg-green-100 text-green-700 border-green-200"
              }`}>
                <p className="text-sm font-medium mb-1">낙상 위험도</p>
                <p className="text-lg font-bold">
                  {clinical.fall_risk === "high"
                    ? "높음 - 낙상 예방 조치 필요"
                    : clinical.fall_risk === "moderate"
                    ? "중간 - 주의 관찰 필요"
                    : clinical.fall_risk === "low"
                    ? "낮음 - 양호한 상태"
                    : "평가 불가"}
                </p>
              </div>

              <div className="rounded-lg border bg-gray-50 p-4">
                <p className="text-sm font-medium text-gray-600 mb-1">지역사회 보행 수준</p>
                <p className="text-lg font-bold">
                  {clinical.community_ambulation === "full"
                    ? "완전한 지역사회 보행 가능"
                    : clinical.community_ambulation === "limited"
                    ? "제한적 지역사회 보행"
                    : clinical.community_ambulation === "household"
                    ? "실내 보행만 가능"
                    : "평가 불가"}
                </p>
                <Badge
                  className="mt-2"
                  variant={
                    clinical.community_ambulation === "full"
                      ? "default"
                      : "secondary"
                  }
                >
                  {(metrics.speed_mps || 0) >= 0.8 ? "지역사회 보행 가능" : "제한적 보행"}
                </Badge>
              </div>

              {clinical.recommendations && clinical.recommendations.length > 0 && (
                <div className="rounded-lg border bg-blue-50 p-4">
                  <p className="text-sm font-medium text-blue-700 mb-2">권장사항</p>
                  <ul className="list-disc list-inside space-y-1">
                    {clinical.recommendations.map((rec, idx) => (
                      <li key={idx} className="text-sm text-blue-600">{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* ═══ 6. 참고사항 ═══ */}
        {judgment && (
          <Card className="mb-6 border-gray-300">
            <CardContent className="p-4">
              <div className="flex gap-2">
                <Info className="h-4 w-4 text-gray-400 mt-0.5 shrink-0" />
                <div className="text-xs text-gray-500 leading-relaxed whitespace-pre-line">
                  {judgment.reference_note}
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* 액션 버튼 */}
        <div className="space-y-3">
          <Button className="w-full" size="lg">
            <Save className="mr-2 h-5 w-5" />
            결과 저장
          </Button>

          <div className="grid grid-cols-2 gap-3">
            <Button variant="outline">
              <FileText className="mr-2 h-4 w-4" />
              보고서 복사
            </Button>
            <Button variant="outline">
              <Share2 className="mr-2 h-4 w-4" />
              공유
            </Button>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <Button variant="secondary" onClick={handleNewTest}>
              <RefreshCw className="mr-2 h-4 w-4" />
              새 검사
            </Button>
            <Button variant="secondary" onClick={handleGoHome}>
              <Home className="mr-2 h-4 w-4" />
              홈으로
            </Button>
          </div>
        </div>

        {/* 검사 일시 */}
        <p className="mt-6 text-center text-sm text-gray-500">
          검사일:{" "}
          {new Date().toLocaleDateString("ko-KR", {
            year: "numeric",
            month: "long",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          })}
        </p>
      </main>

      {/* ═══ 영상 클립 모달 ═══ */}
      {clipModal.open && videoId && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
          onClick={() => setClipModal((prev) => ({ ...prev, open: false }))}
        >
          <div
            className="relative w-full max-w-2xl rounded-2xl bg-white overflow-hidden shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* 모달 헤더 */}
            <div className="flex items-center justify-between px-4 py-3 bg-gray-100 border-b">
              <div className="flex items-center gap-2">
                <Play className="h-4 w-4 text-blue-500" />
                <span className="font-medium text-sm">{clipModal.label}</span>
                <span className="text-xs text-gray-400">
                  ({clipModal.startS.toFixed(1)}s ~ {clipModal.endS.toFixed(1)}s)
                </span>
              </div>
              <button
                onClick={() => setClipModal((prev) => ({ ...prev, open: false }))}
                className="p-1 rounded-full hover:bg-gray-200 transition-colors"
              >
                <X className="h-5 w-5 text-gray-500" />
              </button>
            </div>
            {/* 영상 + Canvas overlay */}
            <div className="relative bg-black">
              <video
                ref={videoRef}
                src={`${BACKEND_URL}/api/video/${videoId}`}
                className="w-full aspect-video"
                controls
                playsInline
              />
              <canvas
                ref={clipCanvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
              />
            </div>
            {/* 구간 재생 + 슬로우모션 버튼 */}
            <div className="px-4 py-3 space-y-2">
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    if (videoRef.current) {
                      videoRef.current.playbackRate = 1.0;
                      videoRef.current.currentTime = clipModal.startS;
                      videoRef.current.play().catch(() => {});
                    }
                  }}
                  className="flex-1 flex items-center justify-center gap-2 py-2 rounded-lg bg-blue-500 text-white font-medium text-sm hover:bg-blue-600 transition-colors"
                >
                  <Play className="h-4 w-4" />
                  구간 재생
                </button>
                <button
                  onClick={() => {
                    if (videoRef.current) {
                      videoRef.current.playbackRate = 0.25;
                      videoRef.current.currentTime = clipModal.startS;
                      videoRef.current.play().catch(() => {});
                    }
                  }}
                  className="flex-1 flex items-center justify-center gap-2 py-2 rounded-lg bg-purple-500 text-white font-medium text-sm hover:bg-purple-600 transition-colors"
                >
                  <Gauge className="h-4 w-4" />
                  슬로우 모션 (0.25x)
                </button>
              </div>
              <div className="flex gap-1 justify-center">
                {[0.1, 0.25, 0.5, 1.0].map((rate) => (
                  <button
                    key={rate}
                    onClick={() => {
                      if (videoRef.current) {
                        videoRef.current.playbackRate = rate;
                      }
                    }}
                    className="px-3 py-1 rounded text-xs font-medium bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
                  >
                    {rate}x
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
