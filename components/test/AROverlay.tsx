"use client";

import { useAppSelector } from "@/store/hooks";
import { formatTime } from "@/lib/calculations";

interface AROverlayProps {
  showDistance?: boolean;
  showTimer?: boolean;
  showGuideLines?: boolean;
}

export function AROverlay({
  showDistance = true,
  showTimer = true,
  showGuideLines = true,
}: AROverlayProps) {
  const { timerValue, isTimerRunning, config, currentTrialIndex, currentMode } =
    useAppSelector((state) => state.testSession);

  return (
    <div className="pointer-events-none absolute inset-0">
      {/* Top Info Bar */}
      <div className="absolute left-0 right-0 top-0 flex items-center justify-between bg-gradient-to-b from-black/70 to-transparent p-4 text-white">
        <div>
          <p className="text-sm opacity-80">환자</p>
          <p className="font-medium">{config?.patientName || "-"}</p>
        </div>
        <div className="text-right">
          <p className="text-sm opacity-80">
            시행 {currentTrialIndex + 1} / {(config?.trialsPerMode || 3) * (config?.mode === "both" ? 2 : 1)}
          </p>
          <p className="font-medium">
            {currentMode === "comfortable" ? "편안한 속도" : "빠른 속도"}
          </p>
        </div>
      </div>

      {/* Center Timer */}
      {showTimer && (
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-center text-white">
          <div
            className={`text-6xl font-bold tabular-nums drop-shadow-lg ${
              isTimerRunning ? "text-green-400" : ""
            }`}
          >
            {formatTime(timerValue)}
          </div>
          {isTimerRunning && (
            <p className="mt-2 animate-pulse text-sm">측정 중...</p>
          )}
        </div>
      )}

      {/* Guide Lines */}
      {showGuideLines && (
        <>
          {/* Start Line */}
          <div className="absolute bottom-20 left-4 right-4">
            <div className="h-1 w-full bg-green-400/50" />
            <p className="mt-1 text-center text-xs text-green-400">시작선</p>
          </div>

          {/* Finish Line Indicator */}
          <div className="absolute right-4 top-1/2 -translate-y-1/2">
            <div className="h-32 w-1 bg-red-400/50" />
            <p className="mt-1 -rotate-90 transform text-xs text-red-400">10m</p>
          </div>
        </>
      )}

      {/* Distance Indicator */}
      {showDistance && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-full bg-black/50 px-4 py-2 text-white">
          <span className="text-2xl font-bold">{config?.distance || 10}</span>
          <span className="text-sm">m</span>
        </div>
      )}

      {/* Recording Indicator */}
      {isTimerRunning && (
        <div className="absolute right-4 top-20 flex items-center gap-2 rounded-full bg-red-500 px-3 py-1 text-sm text-white">
          <span className="h-2 w-2 animate-pulse rounded-full bg-white" />
          REC
        </div>
      )}
    </div>
  );
}
