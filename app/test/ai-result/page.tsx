"use client";

import { useEffect, useState, useRef, useCallback, useMemo } from "react";
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
    keys: ["left_step_time_s", "right_step_time_s", "left_stride_time_s", "right_stride_time_s"],
  },
  {
    label: "비율 변수",
    keys: ["stance_ratio_pct", "swing_ratio_pct"],
  },
];

// ═══ Clip overlay drawing utilities ═══
function drawRoundedRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function drawGaugeBar(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number,
  value: number, min: number, max: number,
  normalMin: number, normalMax: number,
  color: string
) {
  const clamp01 = (v: number) => Math.min(1, Math.max(0, (v - min) / (max - min)));

  // Track background
  drawRoundedRect(ctx, x, y, w, h, h / 2);
  ctx.fillStyle = 'rgba(255,255,255,0.12)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Normal range highlight
  const normStartPx = x + clamp01(normalMin) * w;
  const normEndPx = x + clamp01(normalMax) * w;
  ctx.fillStyle = 'rgba(34,197,94,0.25)';
  ctx.fillRect(normStartPx, y, normEndPx - normStartPx, h);

  // Normal range labels
  ctx.font = `${Math.max(9, h * 0.7)}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.fillStyle = 'rgba(34,197,94,0.7)';
  ctx.fillText(`${normalMin}`, normStartPx, y - 3);
  ctx.fillText(`${normalMax}`, normEndPx, y - 3);

  // Value marker
  const markerX = x + clamp01(value) * w;
  ctx.beginPath();
  ctx.arc(markerX, y + h / 2, h * 0.75, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = 'white';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawJudgmentBanner(
  ctx: CanvasRenderingContext2D,
  cw: number,
  j: JudgmentVariable | null,
  title: string
): number {
  const bH = 40;
  ctx.fillStyle = 'rgba(0,0,0,0.8)';
  ctx.fillRect(0, 0, cw, bH);

  const fontSize = Math.max(13, cw * 0.028);
  ctx.font = `bold ${fontSize}px sans-serif`;
  ctx.textAlign = 'left';
  ctx.fillStyle = 'white';
  ctx.fillText(title, 12, bH / 2 + fontSize * 0.35);

  if (j) {
    const statusColor = j.color === 'green' ? '#22c55e'
      : j.color === 'orange' ? '#f97316' : '#9ca3af';
    const statusText = j.status === '정상' ? 'NORMAL'
      : j.status === 'N/A' ? 'N/A'
      : `${j.direction || ''} ${j.deviation != null ? j.deviation.toFixed(0) + '%' : j.status}`;
    ctx.textAlign = 'right';
    ctx.fillStyle = statusColor;
    ctx.font = `bold ${fontSize}px sans-serif`;
    ctx.fillText(statusText, cw - 12, bH / 2 + fontSize * 0.35);
  }
  return bH;
}

function drawArrowLine(
  ctx: CanvasRenderingContext2D,
  fromX: number, fromY: number, toX: number, toY: number,
  color: string, lineWidth: number = 3
) {
  const angle = Math.atan2(toY - fromY, toX - fromX);
  const headLen = Math.max(10, lineWidth * 4);

  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX - headLen * 0.6 * Math.cos(angle), toY - headLen * 0.6 * Math.sin(angle));
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - headLen * Math.cos(angle - Math.PI / 7), toY - headLen * Math.sin(angle - Math.PI / 7));
  ctx.lineTo(toX - headLen * Math.cos(angle + Math.PI / 7), toY - headLen * Math.sin(angle + Math.PI / 7));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

export default function AIResultPage() {
  const router = useRouter();
  const dispatch = useAppDispatch();
  const { config } = useAppSelector((state) => state.testSession);
  const { result, metrics, videoId, fileName, reset, status } = useAIAnalysis();
  const [waitCount, setWaitCount] = useState(0);
  const [clipModal, setClipModal] = useState<{
    open: boolean;
    label: string;
    startS: number;
    endS: number;
    variableName: string;
    foot: 'left' | 'right' | null;
    judgment: JudgmentVariable | null;
    targetStepNum?: number;  // 개별 분석에서 정확한 스텝 지정용
  }>({ open: false, label: "", startS: 0, endS: 0, variableName: "", foot: null, judgment: null });
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
  const evidenceClips: Record<string, { start_s: number; end_s: number; label: string; target_step_num?: number }> =
    (result as any)?.results?.evidence_clips || {};

  // Pre-compute variable judgment map for clip overlays
  const varMap = useMemo(() => {
    const map = new Map<string, JudgmentVariable>();
    const j = (metrics as any)?.judgment as GaitJudgment | undefined;
    if (j?.variables) {
      for (const v of j.variables) {
        map.set(v.variable_name, v);
      }
    }
    return map;
  }, [metrics]);

  const openClip = useCallback((variableName: string) => {
    const clip = evidenceClips[variableName];
    if (!clip) return;
    const foot = variableName.startsWith('left_') ? 'left' as const
      : variableName.startsWith('right_') ? 'right' as const : null;
    // L/R 변수는 base 변수명의 judgment를 사용 (e.g., left_step_time_s → step_time_s)
    const baseVarName = variableName.replace(/^(left|right)_/, '');
    setClipModal({
      open: true, label: clip.label,
      startS: clip.start_s, endS: clip.end_s,
      variableName, foot,
      judgment: varMap.get(variableName) || varMap.get(baseVarName) || null,
      targetStepNum: clip.target_step_num,
    });
  }, [evidenceClips, varMap]);

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

  // ═══ Clip Modal: Variable-specific canvas overlay ═══
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
    const cw = canvas.width;
    const ch = canvas.height;
    const scaleX = cw / (video.videoWidth || 1);
    const scaleY = ch / (video.videoHeight || 1);

    const entry = getTimelineEntry(currentTime);
    if (!entry) return;

    const varName = clipModal.variableName;
    const jv = clipModal.judgment;
    const foot = clipModal.foot;

    // Determine variable category
    const isVelocity = varName === 'gait_velocity_ms';
    const isCadence = varName === 'cadence_spm';
    const isDistance = varName.includes('step_length') || varName.includes('stride_length');
    const isTime = varName.includes('time_s');
    const isRatio = varName.includes('swing') || varName.includes('stance');

    // Common: draw ankle dots
    const drawAnkles = () => {
      ctx.beginPath();
      ctx.arc(entry.lx * scaleX, entry.ly * scaleY, 7, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(59, 130, 246, 0.9)';
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(entry.rx * scaleX, entry.ry * scaleY, 7, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();

      const ls = Math.max(10, cw * 0.018);
      ctx.font = `bold ${ls}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillStyle = '#93c5fd';
      ctx.fillText('L', entry.lx * scaleX, entry.ly * scaleY - 12);
      ctx.fillStyle = '#fca5a5';
      ctx.fillText('R', entry.rx * scaleX, entry.ry * scaleY - 12);
    };

    // ═══ VELOCITY ═══
    if (isVelocity) {
      drawJudgmentBanner(ctx, cw, jv, '보행 속도');
      drawAnkles();

      const spd = entry.spd;
      let perryLabel = '';
      let perryColor = '#22c55e';
      if (spd < 0.4) { perryLabel = 'Household Amb.'; perryColor = '#ef4444'; }
      else if (spd < 0.6) { perryLabel = 'Limited Community'; perryColor = '#f97316'; }
      else if (spd < 0.8) { perryLabel = 'Community Amb.'; perryColor = '#eab308'; }
      else if (spd < 1.0) { perryLabel = 'Full Community'; perryColor = '#22c55e'; }
      else if (spd < 1.4) { perryLabel = 'Normal Speed'; perryColor = '#22c55e'; }
      else { perryLabel = 'Fast Walker'; perryColor = '#3b82f6'; }

      // Big speed number
      const bigFs = Math.max(48, cw * 0.12);
      ctx.font = `bold ${bigFs}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fillText(spd.toFixed(2), cw / 2 + 2, ch * 0.42 + 2);
      ctx.fillStyle = 'white';
      ctx.fillText(spd.toFixed(2), cw / 2, ch * 0.42);

      // Unit
      ctx.font = `${Math.max(16, cw * 0.035)}px sans-serif`;
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.fillText('m/s', cw / 2, ch * 0.42 + bigFs * 0.4);

      // Perry badge
      const badgeFs = Math.max(14, cw * 0.03);
      ctx.font = `bold ${badgeFs}px sans-serif`;
      const badgeW = ctx.measureText(perryLabel).width + 24;
      const badgeH = badgeFs + 14;
      const badgeX = (cw - badgeW) / 2;
      const badgeY = ch * 0.48;
      drawRoundedRect(ctx, badgeX, badgeY, badgeW, badgeH, badgeH / 2);
      ctx.fillStyle = perryColor;
      ctx.fill();
      ctx.textAlign = 'center';
      ctx.fillStyle = 'white';
      ctx.fillText(perryLabel, cw / 2, badgeY + badgeH / 2 + badgeFs * 0.35);

      // Gauge bar
      const gY = ch * 0.62;
      const gW = cw * 0.75;
      const gX = (cw - gW) / 2;
      drawGaugeBar(ctx, gX, gY, gW, 14, spd, 0, 2.0, 1.0, 1.4, perryColor);

      ctx.font = `${Math.max(10, cw * 0.02)}px sans-serif`;
      ctx.textAlign = 'left';
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.fillText('0', gX, gY + 28);
      ctx.textAlign = 'right';
      ctx.fillText('2.0 m/s', gX + gW, gY + 28);

      // Elapsed time
      const elapsed = currentTime - timelineMeta.start_t;
      ctx.font = `bold ${Math.max(12, cw * 0.025)}px sans-serif`;
      ctx.textAlign = 'right';
      ctx.fillStyle = '#fbbf24';
      ctx.fillText(`${elapsed > 0 ? elapsed.toFixed(1) : '0.0'}s`, cw - 12, ch - 12);
    }

    // ═══ CADENCE ═══
    else if (isCadence) {
      drawJudgmentBanner(ctx, cw, jv, '케이던스');
      drawAnkles();

      const cad = entry.cad;

      // Big cadence number
      const bigFs = Math.max(48, cw * 0.12);
      ctx.font = `bold ${bigFs}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fillText(cad.toFixed(0), cw / 2 + 2, ch * 0.42 + 2);
      ctx.fillStyle = '#a78bfa';
      ctx.fillText(cad.toFixed(0), cw / 2, ch * 0.42);

      ctx.font = `${Math.max(16, cw * 0.035)}px sans-serif`;
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.fillText('steps/min', cw / 2, ch * 0.42 + bigFs * 0.4);

      // Gauge
      const gY = ch * 0.55;
      const gW = cw * 0.75;
      const gX = (cw - gW) / 2;
      drawGaugeBar(ctx, gX, gY, gW, 14, cad, 40, 160, 100, 120, '#a78bfa');

      ctx.font = `${Math.max(10, cw * 0.02)}px sans-serif`;
      ctx.textAlign = 'left';
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.fillText('40', gX, gY + 28);
      ctx.textAlign = 'right';
      ctx.fillText('160 spm', gX + gW, gY + 28);

      // Step rhythm dots (bottom)
      const dotY = ch * 0.75;
      const dotW = cw * 0.85;
      const dotX = (cw - dotW) / 2;
      const clipDur = clipModal.endS - clipModal.startS;

      ctx.fillStyle = 'rgba(255,255,255,0.1)';
      ctx.fillRect(dotX, dotY - 1.5, dotW, 3);

      const clipSteps = stepEventsAll.filter(
        s => s.time >= clipModal.startS - 0.1 && s.time <= clipModal.endS + 0.1
      );
      clipSteps.forEach(step => {
        const t = (step.time - clipModal.startS) / clipDur;
        const sx = dotX + t * dotW;
        ctx.beginPath();
        ctx.arc(sx, dotY, 6, 0, Math.PI * 2);
        ctx.fillStyle = step.leading_foot === 'left' ? '#3b82f6' : '#ef4444';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.5)';
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // Current time indicator
      const tNow = (currentTime - clipModal.startS) / clipDur;
      const nowX = dotX + Math.min(1, Math.max(0, tNow)) * dotW;
      ctx.beginPath();
      ctx.moveTo(nowX, dotY - 10);
      ctx.lineTo(nowX + 5, dotY - 16);
      ctx.lineTo(nowX - 5, dotY - 16);
      ctx.closePath();
      ctx.fillStyle = 'white';
      ctx.fill();

      // Step count
      ctx.font = `bold ${Math.max(12, cw * 0.025)}px sans-serif`;
      ctx.textAlign = 'left';
      ctx.fillStyle = '#34d399';
      ctx.fillText(`Steps: ${entry.cs}`, 12, ch - 12);
    }

    // ═══ DISTANCE (step_length, stride_length) — Phase 기반 HS 증명 ═══
    else if (isDistance) {
      const isStride = varName.includes('stride');
      const lineColor = foot === 'left' ? '#3b82f6' : foot === 'right' ? '#ef4444' : '#a78bfa';

      drawAnkles();

      // Find target step: targetStepNum이 있으면 정확한 스텝 사용, 없으면 시간 범위 검색
      let targetStep: typeof stepEventsAll[0] | undefined;
      if (clipModal.targetStepNum != null) {
        targetStep = stepEventsAll.find(s => s.step_num === clipModal.targetStepNum);
      }
      if (!targetStep) {
        const clipSteps = stepEventsAll.filter(
          s => s.time >= clipModal.startS - 0.5 && s.time <= clipModal.endS + 0.5
        );
        const footStepsInClip = foot ? clipSteps.filter(s => s.leading_foot === foot) : clipSteps;
        targetStep = isStride
          ? (footStepsInClip.length > 0 ? footStepsInClip[footStepsInClip.length - 1] : clipSteps[clipSteps.length - 1])
          : (footStepsInClip.length > 0 ? footStepsInClip[0] : clipSteps[0]);
      }

      if (targetStep) {
        const stepIdx = stepEventsAll.indexOf(targetStep);
        // Step: 1칸 전 (event[i-1]→event[i]), Stride: 2칸 전 (event[i-2]→event[i], 같은 발)
        const anchorOffset = isStride ? 2 : 1;
        const prevStep = stepIdx >= anchorOffset ? stepEventsAll[stepIdx - anchorOffset] : null;

        if (prevStep) {
          const anchorFoot = prevStep.leading_foot;
          // HS1: 앵커 발 좌표 (prevStep 시점에서 고정)
          const hs1X = (anchorFoot === 'left' ? prevStep.left_x : prevStep.right_x) * scaleX;
          const hs1Y = (prevStep.peak_y || entry.ly) * scaleY;
          // HS2: 리딩 발 좌표 (targetStep 시점에서 고정)
          const leadFoot = targetStep.leading_foot;
          const hs2X = (leadFoot === 'left' ? targetStep.left_x : targetStep.right_x) * scaleX;
          const hs2Y = (targetStep.peak_y || entry.ly) * scaleY;
          // 현재 리딩 발 위치 (실시간)
          const liveX = (leadFoot === 'left' ? entry.lx : entry.rx) * scaleX;
          const liveY = (leadFoot === 'left' ? entry.ly : entry.ry) * scaleY;

          const anchorColor = anchorFoot === 'left' ? '#3b82f6' : '#ef4444';
          const leadColor = leadFoot === 'left' ? '#3b82f6' : '#ef4444';
          const anchorLabel = anchorFoot === 'left' ? 'L' : 'R';
          const leadLabel = leadFoot === 'left' ? 'L' : 'R';

          // 실제 측정값 (per-step 데이터 우선, 없으면 judgment 평균)
          const stepCm = targetStep.step_length_cm;
          const strideCm = targetStep.stride_length_cm;
          const actualCm = isStride ? strideCm : stepCm;

          // ── Phase 판별 ──
          const phase = currentTime < prevStep.time ? 'pre'
            : currentTime < targetStep.time ? 'measuring'
            : 'done';

          const hsMarkerR = 12;
          const hsFs = Math.max(13, cw * 0.028);

          // ── Phase 1: HS 전 — 대기 ──
          if (phase === 'pre') {
            // 상단 안내
            const infoFs = Math.max(14, cw * 0.03);
            ctx.font = `bold ${infoFs}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.fillStyle = 'rgba(255,255,255,0.6)';
            ctx.fillText(`${anchorLabel} Heel Strike 대기 중...`, cw / 2, 30);
          }

          // ── Phase 2: 첫 HS 발생 → 앵커 고정, 리딩 발 이동 중 ──
          if (phase === 'measuring' || phase === 'done') {
            // HS1 마커 (앵커 발 — 고정 좌표)
            ctx.beginPath();
            ctx.arc(hs1X, hs1Y, hsMarkerR, 0, Math.PI * 2);
            ctx.fillStyle = anchorColor;
            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 3;
            ctx.stroke();

            // HS1 라벨
            const hs1LabelW = 52;
            const hs1LabelH = hsFs + 8;
            drawRoundedRect(ctx, hs1X - hs1LabelW / 2, hs1Y - hsMarkerR - hs1LabelH - 4, hs1LabelW, hs1LabelH, 4);
            ctx.fillStyle = anchorColor;
            ctx.fill();
            ctx.font = `bold ${hsFs}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.fillStyle = 'white';
            ctx.fillText(`${anchorLabel} HS`, hs1X, hs1Y - hsMarkerR - 8);
          }

          if (phase === 'measuring') {
            // 점선: 앵커 → 현재 발 위치 (이동 중)
            ctx.beginPath();
            ctx.moveTo(hs1X, hs1Y);
            ctx.lineTo(liveX, liveY);
            ctx.strokeStyle = leadColor;
            ctx.lineWidth = 3;
            ctx.setLineDash([8, 6]);
            ctx.stroke();
            ctx.setLineDash([]);

            // 실시간 거리 (측정 중)
            const ppm = timelineMeta.ppm || 100;
            const liveDist = (Math.abs(liveX / scaleX - hs1X / scaleX) / ppm) * 100;
            const midX = (hs1X + liveX) / 2;
            const midY = Math.min(hs1Y, liveY) - 24;

            const liveFs = Math.max(16, cw * 0.035);
            ctx.font = `bold ${liveFs}px sans-serif`;
            const liveText = `${liveDist.toFixed(1)} cm`;
            const tw = ctx.measureText(liveText).width;

            drawRoundedRect(ctx, midX - tw / 2 - 8, midY - liveFs - 2, tw + 16, liveFs + 10, 6);
            ctx.fillStyle = 'rgba(0,0,0,0.6)';
            ctx.fill();
            ctx.textAlign = 'center';
            ctx.fillStyle = 'rgba(255,255,255,0.7)';
            ctx.fillText(liveText, midX, midY);

            // 상단 안내: "측정 중..."
            ctx.font = `bold ${Math.max(12, cw * 0.025)}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.fillStyle = 'rgba(255,255,255,0.5)';
            ctx.fillText(`${leadLabel} Heel Strike 대기 중...`, cw / 2, 30);
          }

          // ── Phase 3: 두 번째 HS 완료 → 측정 확정 ──
          if (phase === 'done') {
            // HS2 마커 (리딩 발 — 고정 좌표)
            ctx.beginPath();
            ctx.arc(hs2X, hs2Y, hsMarkerR, 0, Math.PI * 2);
            ctx.fillStyle = leadColor;
            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 3;
            ctx.stroke();

            // HS2 라벨
            const hs2LabelW = 52;
            const hs2LabelH = hsFs + 8;
            drawRoundedRect(ctx, hs2X - hs2LabelW / 2, hs2Y - hsMarkerR - hs2LabelH - 4, hs2LabelW, hs2LabelH, 4);
            ctx.fillStyle = leadColor;
            ctx.fill();
            ctx.font = `bold ${hsFs}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.fillStyle = 'white';
            ctx.fillText(`${leadLabel} HS`, hs2X, hs2Y - hsMarkerR - 8);

            // 확정 측정선: 실선 화살표 (HS1 → HS2)
            // Glow
            ctx.beginPath();
            ctx.moveTo(hs1X, hs1Y);
            ctx.lineTo(hs2X, hs2Y);
            ctx.strokeStyle = 'rgba(255,255,255,0.1)';
            ctx.lineWidth = 28;
            ctx.stroke();

            drawArrowLine(ctx, hs1X, hs1Y, hs2X, hs2Y, lineColor, 4);

            // ── 확정 거리 라벨 (큰 글씨) ──
            const midX = (hs1X + hs2X) / 2;
            const midY = Math.min(hs1Y, hs2Y) - 30;
            const finalCm = actualCm != null ? actualCm : 0;

            const bigFs = Math.max(26, cw * 0.07);
            ctx.font = `bold ${bigFs}px sans-serif`;
            const finalText = `${finalCm.toFixed(1)} cm`;
            const tw = ctx.measureText(finalText).width;

            // 배경 박스
            const boxW = tw + 32;
            const boxH = bigFs + 20;
            drawRoundedRect(ctx, midX - boxW / 2, midY - bigFs - 6, boxW, boxH, 8);
            ctx.fillStyle = 'rgba(0,0,0,0.85)';
            ctx.fill();
            ctx.strokeStyle = lineColor;
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.textAlign = 'center';
            ctx.fillStyle = lineColor;
            ctx.fillText(finalText, midX, midY);

            // ── 하단 판정 바 ──
            const rangeMatch = jv?.normal_range?.match(/([\d.]+)\s*[~\u2013-]\s*([\d.]+)/);
            const normMin = rangeMatch ? parseFloat(rangeMatch[1]) : 55;
            const normMax = rangeMatch ? parseFloat(rangeMatch[2]) : 70;
            const statusColor = jv?.color === 'green' ? '#22c55e' : jv?.color === 'orange' ? '#f97316' : '#9ca3af';

            const barH = 40;
            const barY = ch - barH;
            ctx.fillStyle = 'rgba(0,0,0,0.8)';
            ctx.fillRect(0, barY, cw, barH);

            const botFs = Math.max(13, cw * 0.026);
            ctx.font = `bold ${botFs}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.fillStyle = statusColor;

            const isNormal = finalCm >= normMin && finalCm <= normMax;
            const verdictText = isNormal
              ? `${finalCm.toFixed(0)}cm  |  정상 범위 (${normMin}~${normMax}cm)`
              : `${finalCm.toFixed(0)}cm  |  정상 ${normMin}~${normMax}cm  |  ${jv?.direction || ''} ${jv?.deviation != null ? jv.deviation.toFixed(0) + '%' : ''}`;
            ctx.fillText(verdictText, cw / 2, barY + barH / 2 + botFs * 0.35);
          }
        }
      }
    }

    // ═══ TIME (step_time, stride_time) — HS phase 기반 (distance와 동일 구조) ═══
    else if (isTime) {
      const isStride = varName.includes('stride');
      const lineColor = foot === 'left' ? '#3b82f6' : foot === 'right' ? '#ef4444' : '#fbbf24';

      drawAnkles();

      // Find target step: targetStepNum이 있으면 정확한 스텝 사용
      let targetStep: typeof stepEventsAll[0] | undefined;
      if (clipModal.targetStepNum != null) {
        targetStep = stepEventsAll.find(s => s.step_num === clipModal.targetStepNum);
      }
      if (!targetStep) {
        const clipSteps = stepEventsAll.filter(
          s => s.time >= clipModal.startS - 0.5 && s.time <= clipModal.endS + 0.5
        );
        const footStepsInClip = foot ? clipSteps.filter(s => s.leading_foot === foot) : clipSteps;
        targetStep = isStride
          ? (footStepsInClip.length > 0 ? footStepsInClip[footStepsInClip.length - 1] : clipSteps[clipSteps.length - 1])
          : (footStepsInClip.length > 0 ? footStepsInClip[0] : clipSteps[0]);
      }

      if (targetStep) {
        const stepIdx = stepEventsAll.indexOf(targetStep);
        const anchorOffset = isStride ? 2 : 1;
        const prevStep = stepIdx >= anchorOffset ? stepEventsAll[stepIdx - anchorOffset] : null;

        if (prevStep) {
          const anchorFoot = prevStep.leading_foot;
          const hs1X = (anchorFoot === 'left' ? prevStep.left_x : prevStep.right_x) * scaleX;
          const hs1Y = (prevStep.peak_y || entry.ly) * scaleY;
          const leadFoot = targetStep.leading_foot;
          const hs2X = (leadFoot === 'left' ? targetStep.left_x : targetStep.right_x) * scaleX;
          const hs2Y = (targetStep.peak_y || entry.ly) * scaleY;

          const anchorColor = anchorFoot === 'left' ? '#3b82f6' : '#ef4444';
          const leadColor = leadFoot === 'left' ? '#3b82f6' : '#ef4444';
          const anchorLabel = anchorFoot === 'left' ? 'L' : 'R';
          const leadLabel = leadFoot === 'left' ? 'L' : 'R';

          // 실제 측정값: stride이면 stride_time_s, step이면 step_time_s
          const actualTime = isStride ? targetStep.stride_time_s : targetStep.step_time_s;
          const hs1Time = prevStep.time;
          const hs2Time = targetStep.time;

          const phase = currentTime < hs1Time ? 'pre'
            : currentTime < hs2Time ? 'measuring'
            : 'done';

          const hsMarkerR = 12;
          const hsFs = Math.max(13, cw * 0.028);

          // ── Phase 1: HS 전 ──
          if (phase === 'pre') {
            ctx.font = `bold ${Math.max(14, cw * 0.03)}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.fillStyle = 'rgba(255,255,255,0.6)';
            ctx.fillText(`${anchorLabel} Heel Strike 대기 중...`, cw / 2, 30);
          }

          // ── Phase 2+3 공통: HS1 마커 ──
          if (phase === 'measuring' || phase === 'done') {
            ctx.beginPath();
            ctx.arc(hs1X, hs1Y, hsMarkerR, 0, Math.PI * 2);
            ctx.fillStyle = anchorColor;
            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 3;
            ctx.stroke();

            const hs1LabelW = 52;
            const hs1LabelH = hsFs + 8;
            drawRoundedRect(ctx, hs1X - hs1LabelW / 2, hs1Y - hsMarkerR - hs1LabelH - 4, hs1LabelW, hs1LabelH, 4);
            ctx.fillStyle = anchorColor;
            ctx.fill();
            ctx.font = `bold ${hsFs}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.fillStyle = 'white';
            ctx.fillText(`${anchorLabel} HS`, hs1X, hs1Y - hsMarkerR - 8);
          }

          // ── Phase 2: 측정 중 — 스톱워치 카운트업 ──
          if (phase === 'measuring') {
            const elapsed = currentTime - hs1Time;

            // 큰 스톱워치
            const bigFs = Math.max(44, cw * 0.11);
            ctx.font = `bold ${bigFs}px monospace`;
            ctx.textAlign = 'center';
            ctx.fillStyle = 'rgba(0,0,0,0.5)';
            ctx.fillText(elapsed.toFixed(3), cw / 2 + 2, ch * 0.38 + 2);
            ctx.fillStyle = '#fbbf24';
            ctx.fillText(elapsed.toFixed(3), cw / 2, ch * 0.38);

            ctx.font = `${Math.max(14, cw * 0.03)}px sans-serif`;
            ctx.fillStyle = 'rgba(255,255,255,0.5)';
            ctx.fillText(`${leadLabel} Heel Strike 대기 중...`, cw / 2, ch * 0.38 + bigFs * 0.45);
          }

          // ── Phase 3: 측정 완료 ──
          if (phase === 'done') {
            // HS2 마커
            ctx.beginPath();
            ctx.arc(hs2X, hs2Y, hsMarkerR, 0, Math.PI * 2);
            ctx.fillStyle = leadColor;
            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 3;
            ctx.stroke();

            const hs2LabelW = 52;
            const hs2LabelH = hsFs + 8;
            drawRoundedRect(ctx, hs2X - hs2LabelW / 2, hs2Y - hsMarkerR - hs2LabelH - 4, hs2LabelW, hs2LabelH, 4);
            ctx.fillStyle = leadColor;
            ctx.fill();
            ctx.font = `bold ${hsFs}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.fillStyle = 'white';
            ctx.fillText(`${leadLabel} HS`, hs2X, hs2Y - hsMarkerR - 8);

            // 연결선 (시간이라 점선)
            ctx.beginPath();
            ctx.moveTo(hs1X, hs1Y);
            ctx.lineTo(hs2X, hs2Y);
            ctx.strokeStyle = 'rgba(251,191,36,0.4)';
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            ctx.stroke();
            ctx.setLineDash([]);

            // 확정 시간 (큰 글씨)
            const finalTime = actualTime != null ? actualTime : (hs2Time - hs1Time);
            const bigFs = Math.max(44, cw * 0.11);
            ctx.font = `bold ${bigFs}px monospace`;
            ctx.textAlign = 'center';
            ctx.fillStyle = 'rgba(0,0,0,0.5)';
            ctx.fillText(finalTime.toFixed(3), cw / 2 + 2, ch * 0.38 + 2);
            ctx.fillStyle = '#fbbf24';
            ctx.fillText(finalTime.toFixed(3), cw / 2, ch * 0.38);

            ctx.font = `${Math.max(16, cw * 0.035)}px sans-serif`;
            ctx.fillStyle = 'rgba(255,255,255,0.7)';
            ctx.fillText('seconds', cw / 2, ch * 0.38 + bigFs * 0.4);

            // 하단 판정 바
            const rangeMatch = jv?.normal_range?.match(/([\d.]+)\s*[~\u2013-]\s*([\d.]+)/);
            const normMin = rangeMatch ? parseFloat(rangeMatch[1]) : 0.4;
            const normMax = rangeMatch ? parseFloat(rangeMatch[2]) : 0.6;
            const statusColor = jv?.color === 'green' ? '#22c55e' : jv?.color === 'orange' ? '#f97316' : '#9ca3af';

            const barH = 40;
            const barY = ch - barH;
            ctx.fillStyle = 'rgba(0,0,0,0.8)';
            ctx.fillRect(0, barY, cw, barH);

            const botFs = Math.max(13, cw * 0.026);
            ctx.font = `bold ${botFs}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.fillStyle = statusColor;

            const isNormal = finalTime >= normMin && finalTime <= normMax;
            const typeLabel = isStride ? 'Stride Time' : 'Step Time';
            const verdictText = isNormal
              ? `${typeLabel}: ${finalTime.toFixed(3)}s  |  정상 (${normMin}~${normMax}s)`
              : `${typeLabel}: ${finalTime.toFixed(3)}s  |  정상 ${normMin}~${normMax}s  |  ${jv?.direction || ''} ${jv?.deviation != null ? jv.deviation.toFixed(0) + '%' : ''}`;
            ctx.fillText(verdictText, cw / 2, barY + barH / 2 + botFs * 0.35);
          }
        }
      }
    }

    // ═══ RATIO (swing/stance) ═══
    else if (isRatio) {
      const isSwing = varName.includes('swing');
      drawJudgmentBanner(ctx, cw, jv, isSwing ? 'Swing 비율' : 'Stance 비율');
      drawAnkles();

      // Foot state labels based on ankle velocity
      const prevEntry = getTimelineEntry(currentTime - 0.05);
      if (prevEntry) {
        const lvx = Math.abs(entry.lx - prevEntry.lx);
        const rvx = Math.abs(entry.rx - prevEntry.rx);
        const threshold = Math.max(lvx, rvx) * 0.4;

        const leftState = lvx > threshold ? 'SWING' : 'STANCE';
        const rightState = rvx > threshold ? 'SWING' : 'STANCE';

        const labelFs = Math.max(11, cw * 0.022);
        ctx.font = `bold ${labelFs}px sans-serif`;
        ctx.textAlign = 'center';

        // Left foot state badge
        const llx = entry.lx * scaleX;
        const lly = entry.ly * scaleY + 22;
        drawRoundedRect(ctx, llx - 30, lly - labelFs + 2, 60, labelFs + 6, 4);
        ctx.fillStyle = leftState === 'SWING' ? 'rgba(59,130,246,0.85)' : 'rgba(59,130,246,0.35)';
        ctx.fill();
        ctx.fillStyle = 'white';
        ctx.fillText(leftState, llx, lly);

        // Right foot state badge
        const rlx = entry.rx * scaleX;
        const rly = entry.ry * scaleY + 22;
        drawRoundedRect(ctx, rlx - 30, rly - labelFs + 2, 60, labelFs + 6, 4);
        ctx.fillStyle = rightState === 'SWING' ? 'rgba(239,68,68,0.85)' : 'rgba(239,68,68,0.35)';
        ctx.fill();
        ctx.fillStyle = 'white';
        ctx.fillText(rightState, rlx, rly);
      }

      // Ratio bars
      const barAreaY = ch * 0.35;
      const barW = cw * 0.7;
      const barX = (cw - barW) / 2;
      const barH = 22;

      // Left foot bar
      const lSwing = metricsData?.left_swing_pct || 40;
      const lStance = metricsData?.left_stance_pct || 60;

      ctx.font = `bold ${Math.max(12, cw * 0.025)}px sans-serif`;
      ctx.textAlign = 'right';
      ctx.fillStyle = '#93c5fd';
      ctx.fillText('L', barX - 8, barAreaY + barH / 2 + 5);

      drawRoundedRect(ctx, barX, barAreaY, barW, barH, 4);
      ctx.fillStyle = 'rgba(255,255,255,0.1)';
      ctx.fill();

      const lSwingW = (lSwing / 100) * barW;
      ctx.fillStyle = 'rgba(59,130,246,0.7)';
      ctx.fillRect(barX, barAreaY, lSwingW, barH);
      ctx.fillStyle = 'rgba(59,130,246,0.3)';
      ctx.fillRect(barX + lSwingW, barAreaY, barW - lSwingW, barH);

      ctx.font = `bold ${Math.max(10, cw * 0.02)}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillStyle = 'white';
      if (lSwingW > 40) ctx.fillText(`Sw ${lSwing.toFixed(0)}%`, barX + lSwingW / 2, barAreaY + barH / 2 + 4);
      if (barW - lSwingW > 40) ctx.fillText(`St ${lStance.toFixed(0)}%`, barX + lSwingW + (barW - lSwingW) / 2, barAreaY + barH / 2 + 4);

      // Right foot bar
      const rBarY = barAreaY + barH + 12;
      const rSwing = metricsData?.right_swing_pct || 40;
      const rStance = metricsData?.right_stance_pct || 60;

      ctx.font = `bold ${Math.max(12, cw * 0.025)}px sans-serif`;
      ctx.textAlign = 'right';
      ctx.fillStyle = '#fca5a5';
      ctx.fillText('R', barX - 8, rBarY + barH / 2 + 5);

      drawRoundedRect(ctx, barX, rBarY, barW, barH, 4);
      ctx.fillStyle = 'rgba(255,255,255,0.1)';
      ctx.fill();

      const rSwingW = (rSwing / 100) * barW;
      ctx.fillStyle = 'rgba(239,68,68,0.7)';
      ctx.fillRect(barX, rBarY, rSwingW, barH);
      ctx.fillStyle = 'rgba(239,68,68,0.3)';
      ctx.fillRect(barX + rSwingW, rBarY, barW - rSwingW, barH);

      ctx.font = `bold ${Math.max(10, cw * 0.02)}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillStyle = 'white';
      if (rSwingW > 40) ctx.fillText(`Sw ${rSwing.toFixed(0)}%`, barX + rSwingW / 2, rBarY + barH / 2 + 4);
      if (barW - rSwingW > 40) ctx.fillText(`St ${rStance.toFixed(0)}%`, barX + rSwingW + (barW - rSwingW) / 2, rBarY + barH / 2 + 4);

      // Normal 40% reference line
      const refX = barX + (40 / 100) * barW;
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.moveTo(refX, barAreaY - 8);
      ctx.lineTo(refX, rBarY + barH + 8);
      ctx.strokeStyle = 'rgba(34,197,94,0.7)';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.font = `${Math.max(9, cw * 0.018)}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillStyle = 'rgba(34,197,94,0.8)';
      ctx.fillText('40%', refX, barAreaY - 12);

      // Big ratio value (bottom center)
      if (jv && jv.measured_value != null) {
        const bigFs = Math.max(32, cw * 0.08);
        ctx.font = `bold ${bigFs}px sans-serif`;
        ctx.textAlign = 'center';
        const valColor = jv.color === 'green' ? '#22c55e' : jv.color === 'orange' ? '#f97316' : 'white';
        ctx.fillStyle = 'rgba(0,0,0,0.4)';
        ctx.fillText(`${jv.measured_value.toFixed(1)}%`, cw / 2 + 1, ch * 0.72 + 1);
        ctx.fillStyle = valColor;
        ctx.fillText(`${jv.measured_value.toFixed(1)}%`, cw / 2, ch * 0.72);

        ctx.font = `${Math.max(12, cw * 0.025)}px sans-serif`;
        ctx.fillStyle = 'rgba(255,255,255,0.6)';
        ctx.fillText(jv.display_name, cw / 2, ch * 0.72 + bigFs * 0.35);
      }
    }

    // ═══ FALLBACK ═══
    else {
      drawAnkles();

      const hudSize = Math.max(12, cw * 0.022);
      ctx.font = `bold ${hudSize}px sans-serif`;
      ctx.textAlign = 'right';
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(cw - 150, 8, 142, hudSize * 2.5 + 8);
      ctx.fillStyle = '#22d3ee';
      ctx.fillText(`Speed: ${entry.spd.toFixed(2)} m/s`, cw - 16, 8 + hudSize + 2);
      ctx.fillStyle = '#fbbf24';
      const elapsedT = currentTime - timelineMeta.start_t;
      ctx.fillText(`Time: ${elapsedT > 0 ? elapsedT.toFixed(2) : '0.00'}s`, cw - 16, 8 + hudSize * 2.2 + 2);
    }

    if (!video.paused) {
      clipAnimRef.current = requestAnimationFrame(renderClipOverlay);
    }
  }, [clipModal, getTimelineEntry, timelineMeta, stepEventsAll, metricsData]);

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
    if (value === null) return { label: "-", color: "text-gray-400", bg: "bg-gray-50", border: "border-gray-200" };
    if (value < 10) return { label: "정상", color: "text-green-700", bg: "bg-green-50", border: "border-green-200" };
    return { label: "비대칭", color: "text-red-700", bg: "bg-red-50", border: "border-red-200" };
  };

  const judgment = metrics.judgment as GaitJudgment | undefined;
  const velocity = judgment?.velocity;
  const clinical = metrics.clinical;

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
                {fileName && <span className="ml-2 text-green-500">| {fileName}</span>}
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
            <p className="text-xs text-gray-500 mt-1">전체 보행 구간의 L/R 평균값 기반</p>
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

        {/* ═══ 개별 스텝 분석 (Step) ═══ */}
        {stepEventsAll.length > 1 && timelineMeta && (() => {
          const validSteps = stepEventsAll.filter(s => s.step_length_cm != null);
          const meanStep = validSteps.length > 0
            ? validSteps.reduce((a, b) => a + b.step_length_cm!, 0) / validSteps.length : 0;
          const leftSteps = validSteps.filter(s => s.leading_foot === 'left');
          const rightSteps = validSteps.filter(s => s.leading_foot === 'right');
          const leftAvg = leftSteps.length > 0 ? leftSteps.reduce((a, b) => a + b.step_length_cm!, 0) / leftSteps.length : 0;
          const rightAvg = rightSteps.length > 0 ? rightSteps.reduce((a, b) => a + b.step_length_cm!, 0) / rightSteps.length : 0;
          const leftMin = leftSteps.length > 0 ? Math.min(...leftSteps.map(s => s.step_length_cm!)) : 0;
          const leftMax = leftSteps.length > 0 ? Math.max(...leftSteps.map(s => s.step_length_cm!)) : 0;
          const rightMin = rightSteps.length > 0 ? Math.min(...rightSteps.map(s => s.step_length_cm!)) : 0;
          const rightMax = rightSteps.length > 0 ? Math.max(...rightSteps.map(s => s.step_length_cm!)) : 0;

          return (
          <Card className="mb-6">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-lg">
                <Footprints className="h-5 w-5" />
                개별 Step 분석
              </CardTitle>
              <p className="text-xs text-gray-500 mt-1">각 스텝의 보폭(반대발→이 발)을 개별 확인</p>
            </CardHeader>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-gray-50">
                      <th className="text-center py-2 px-2 font-medium text-gray-500 w-10">#</th>
                      <th className="text-center py-2 px-1 font-medium text-gray-500 w-10">발</th>
                      <th className="text-center py-2 px-1 font-medium text-gray-500">보폭 (cm)</th>
                      <th className="text-center py-2 px-1 font-medium text-gray-500">시간 (s)</th>
                      <th className="text-center py-2 px-1 font-medium text-gray-500 w-10"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {stepEventsAll.filter(s => s.step_length_cm != null).map((step, idx) => {
                      const isLeft = step.leading_foot === 'left';
                      const footColor = isLeft ? 'text-blue-600' : 'text-red-600';
                      const bgColor = isLeft ? 'bg-blue-50/50' : 'bg-red-50/50';
                      const isOutlier = meanStep > 0
                        && Math.abs(step.step_length_cm! - meanStep) / meanStep > 0.2;

                      return (
                        <tr key={idx} className={`border-b last:border-b-0 ${bgColor}`}>
                          <td className="text-center py-2 px-2 text-xs text-gray-400 font-mono">
                            {step.step_num}
                          </td>
                          <td className={`text-center py-2 px-1 font-bold text-xs ${footColor}`}>
                            {isLeft ? 'L' : 'R'}
                          </td>
                          <td className={`text-center py-2 px-1 font-mono font-bold ${isOutlier ? 'text-orange-600' : ''}`}>
                            <span className="flex items-center justify-center gap-1">
                              {step.step_length_cm!.toFixed(1)}
                              {isOutlier && <AlertTriangle className="h-3 w-3 text-orange-500" />}
                            </span>
                          </td>
                          <td className="text-center py-2 px-1 font-mono text-gray-600">
                            {step.step_time_s != null ? step.step_time_s.toFixed(3) : '-'}
                          </td>
                          <td className="text-center py-2 px-1">
                            <button
                              onClick={() => {
                                const prevStep = stepEventsAll[stepEventsAll.indexOf(step) - 1];
                                const clipStart = prevStep ? prevStep.time - 0.1 : step.time - 0.3;
                                const clipEnd = step.time + 0.15;
                                const vn = isLeft ? 'left_step_length_cm' : 'right_step_length_cm';
                                setClipModal({
                                  open: true,
                                  label: `Step #${step.step_num} (${isLeft ? 'L' : 'R'})`,
                                  startS: clipStart,
                                  endS: clipEnd,
                                  variableName: vn,
                                  foot: step.leading_foot,
                                  judgment: varMap.get(vn) || null,
                                  targetStepNum: step.step_num,
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
            </CardContent>
          </Card>
          );
        })()}

        {/* ═══ 개별 Stride 분석 ═══ */}
        {stepEventsAll.length > 2 && timelineMeta && (() => {
          const validStrides = stepEventsAll.filter(s => s.stride_length_cm != null);
          if (validStrides.length === 0) return null;
          const meanStride = validStrides.reduce((a, b) => a + b.stride_length_cm!, 0) / validStrides.length;
          const leftStrides = validStrides.filter(s => s.leading_foot === 'left');
          const rightStrides = validStrides.filter(s => s.leading_foot === 'right');
          const leftAvg = leftStrides.length > 0 ? leftStrides.reduce((a, b) => a + b.stride_length_cm!, 0) / leftStrides.length : 0;
          const rightAvg = rightStrides.length > 0 ? rightStrides.reduce((a, b) => a + b.stride_length_cm!, 0) / rightStrides.length : 0;
          const leftMin = leftStrides.length > 0 ? Math.min(...leftStrides.map(s => s.stride_length_cm!)) : 0;
          const leftMax = leftStrides.length > 0 ? Math.max(...leftStrides.map(s => s.stride_length_cm!)) : 0;
          const rightMin = rightStrides.length > 0 ? Math.min(...rightStrides.map(s => s.stride_length_cm!)) : 0;
          const rightMax = rightStrides.length > 0 ? Math.max(...rightStrides.map(s => s.stride_length_cm!)) : 0;

          return (
          <Card className="mb-6">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-lg">
                <Activity className="h-5 w-5" />
                개별 Stride 분석
              </CardTitle>
              <p className="text-xs text-gray-500 mt-1">각 활보장(같은 발 1보행주기)을 개별 확인</p>
            </CardHeader>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-gray-50">
                      <th className="text-center py-2 px-2 font-medium text-gray-500 w-10">#</th>
                      <th className="text-center py-2 px-1 font-medium text-gray-500 w-10">발</th>
                      <th className="text-center py-2 px-1 font-medium text-gray-500">활보장 (cm)</th>
                      <th className="text-center py-2 px-1 font-medium text-gray-500">시간 (s)</th>
                      <th className="text-center py-2 px-1 font-medium text-gray-500 w-10"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {validStrides.map((step, idx) => {
                      const isLeft = step.leading_foot === 'left';
                      const footColor = isLeft ? 'text-blue-600' : 'text-red-600';
                      const bgColor = isLeft ? 'bg-blue-50/50' : 'bg-red-50/50';
                      const isOutlier = meanStride > 0
                        && Math.abs(step.stride_length_cm! - meanStride) / meanStride > 0.2;

                      return (
                        <tr key={idx} className={`border-b last:border-b-0 ${bgColor}`}>
                          <td className="text-center py-2 px-2 text-xs text-gray-400 font-mono">
                            {step.step_num}
                          </td>
                          <td className={`text-center py-2 px-1 font-bold text-xs ${footColor}`}>
                            {isLeft ? 'L' : 'R'}
                          </td>
                          <td className={`text-center py-2 px-1 font-mono font-bold ${isOutlier ? 'text-orange-600' : ''}`}>
                            <span className="flex items-center justify-center gap-1">
                              {step.stride_length_cm!.toFixed(1)}
                              {isOutlier && <AlertTriangle className="h-3 w-3 text-orange-500" />}
                            </span>
                          </td>
                          <td className="text-center py-2 px-1 font-mono text-gray-600">
                            {step.stride_time_s != null ? step.stride_time_s.toFixed(3) : '-'}
                          </td>
                          <td className="text-center py-2 px-1">
                            <button
                              onClick={() => {
                                const allIdx = stepEventsAll.indexOf(step);
                                const anchorStep = allIdx >= 2 ? stepEventsAll[allIdx - 2] : step;
                                const clipStart = anchorStep.time - 0.1;
                                const clipEnd = step.time + 0.15;
                                const vn = isLeft ? 'left_stride_length_cm' : 'right_stride_length_cm';
                                setClipModal({
                                  open: true,
                                  label: `Stride #${step.step_num} (${isLeft ? 'L' : 'R'})`,
                                  startS: clipStart,
                                  endS: clipEnd,
                                  variableName: vn,
                                  foot: step.leading_foot,
                                  judgment: varMap.get(vn) || null,
                                  targetStepNum: step.step_num,
                                });
                              }}
                              className={`p-1 rounded transition-colors ${isLeft ? 'bg-blue-100 hover:bg-blue-200' : 'bg-red-100 hover:bg-red-200'}`}
                              title={`Stride #${step.step_num} 재생`}
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
              <div className="px-4 py-3 bg-gray-50 border-t grid grid-cols-2 gap-3 text-xs">
                <div className="rounded-lg bg-blue-50 px-3 py-2 border border-blue-100">
                  <p className="font-bold text-blue-700 mb-1">L 평균: {leftAvg.toFixed(1)}cm</p>
                  <p className="text-blue-500">범위: {leftMin.toFixed(0)}~{leftMax.toFixed(0)}cm ({leftStrides.length}회)</p>
                </div>
                <div className="rounded-lg bg-red-50 px-3 py-2 border border-red-100">
                  <p className="font-bold text-red-700 mb-1">R 평균: {rightAvg.toFixed(1)}cm</p>
                  <p className="text-red-500">범위: {rightMin.toFixed(0)}~{rightMax.toFixed(0)}cm ({rightStrides.length}회)</p>
                </div>
              </div>
            </CardContent>
          </Card>
          );
        })()}

        {/* ═══ 4. 좌우 대칭성 - 시간 ═══ */}
        <Card className="mb-6">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Clock className="h-5 w-5" />
              시간 대칭성
            </CardTitle>
            <p className="text-xs text-gray-500 mt-1">전체 보행 구간의 L/R 평균값 기반</p>
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
                    <div className="flex justify-center gap-1">
                      {evidenceClips['left_step_time_s'] && (
                        <button onClick={() => openClip('left_step_time_s')}
                          className="p-1 rounded bg-blue-50 hover:bg-blue-100 transition-colors" title="L Step Time 재생">
                          <Play className="h-3.5 w-3.5 text-blue-500" />
                        </button>
                      )}
                      {evidenceClips['right_step_time_s'] && (
                        <button onClick={() => openClip('right_step_time_s')}
                          className="p-1 rounded bg-red-50 hover:bg-red-100 transition-colors" title="R Step Time 재생">
                          <Play className="h-3.5 w-3.5 text-red-500" />
                        </button>
                      )}
                    </div>
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
                    <div className="flex justify-center gap-1">
                      {evidenceClips['left_stride_time_s'] && (
                        <button onClick={() => openClip('left_stride_time_s')}
                          className="p-1 rounded bg-blue-50 hover:bg-blue-100 transition-colors" title="L Stride Time 재생">
                          <Play className="h-3.5 w-3.5 text-blue-500" />
                        </button>
                      )}
                      {evidenceClips['right_stride_time_s'] && (
                        <button onClick={() => openClip('right_stride_time_s')}
                          className="p-1 rounded bg-red-50 hover:bg-red-100 transition-colors" title="R Stride Time 재생">
                          <Play className="h-3.5 w-3.5 text-red-500" />
                        </button>
                      )}
                    </div>
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
            <p className="text-xs text-gray-500 mt-1">전체 보행 구간의 L/R 평균값 기반 · 정상: Swing 40% / Stance 60%</p>
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
                  <td className={`text-center font-mono font-bold ${getSymmetryStatus(metrics.swing_stance_si).color}`}>
                    {metrics.swing_stance_si?.toFixed(0) || "-"}
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

            {/* 종합 대칭성 지수 */}
            {metrics.overall_symmetry_index !== null && (() => {
              const siValue = metrics.overall_symmetry_index!;
              const siStatus = getSymmetryStatus(siValue);
              return (
                <div className={`mt-4 rounded-lg border p-4 ${siStatus.bg} ${siStatus.border}`}>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-medium text-gray-700">종합 대칭성 지수 (Overall SI)</span>
                    <div className="flex items-center gap-2">
                      <span className={`font-mono text-2xl font-bold ${siStatus.color}`}>
                        {siValue.toFixed(1)}%
                      </span>
                      <span className={`text-sm font-bold px-2 py-0.5 rounded ${siStatus.color} ${siStatus.bg}`}>
                        {siStatus.label}
                      </span>
                    </div>
                  </div>
                  {/* 시각적 바 */}
                  <div className="relative h-3 rounded-full bg-gray-200 mb-2">
                    <div
                      className={`h-full rounded-full transition-all ${siValue < 10 ? "bg-green-500" : "bg-red-500"}`}
                      style={{ width: `${Math.min(siValue / 30 * 100, 100)}%` }}
                    />
                    {/* 10% 기준선 */}
                    <div
                      className="absolute top-0 h-full w-0.5 bg-gray-600"
                      style={{ left: `${10 / 30 * 100}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-gray-500 mb-3">
                    <span>0% (완전 대칭)</span>
                    <span className="font-medium text-gray-600" style={{ marginLeft: `${10 / 30 * 100 - 8}%` }}>10% 기준</span>
                    <span>30%+</span>
                  </div>
                  {/* 해석 */}
                  <div className="text-xs text-gray-500 leading-relaxed space-y-1">
                    <p>
                      {siValue < 10
                        ? "좌우 보행이 대칭적입니다. 정상 범위 내 대칭성을 보입니다."
                        : `좌우 보행에 비대칭이 관찰됩니다 (SI ${siValue.toFixed(1)}%). 임상적 평가가 권장됩니다.`}
                    </p>
                    <p className="text-gray-400">
                      기준: SI &lt; 10% 정상 (Patterson et al. 2010; Herzog et al. 1989)
                    </p>
                  </div>
                </div>
              );
            })()}
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
