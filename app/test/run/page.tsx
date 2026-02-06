"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAppSelector, useAppDispatch } from "@/store/hooks";
import {
  startTest,
  stopTest,
  addTrialResult,
  setCurrentMode,
  setStatus,
  TrialResult,
} from "@/store/slices/testSessionSlice";
import { Timer } from "@/components/test/Timer";
import { CameraView } from "@/components/test/CameraView";
import { AROverlay } from "@/components/test/AROverlay";
import { RestCountdown } from "@/components/test/RestCountdown";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { calculateSpeed, calculateCadence } from "@/lib/calculations";
import {
  Play,
  Square,
  RotateCcw,
  ArrowLeft,
  AlertCircle,
  CheckCircle,
} from "lucide-react";

export default function TestRunPage() {
  const router = useRouter();
  const dispatch = useAppDispatch();
  const {
    config,
    status,
    trials,
    currentTrialIndex,
    currentMode,
    isTimerRunning,
    timerValue,
  } = useAppSelector((state) => state.testSession);

  const [showRest, setShowRest] = useState(false);
  const [stepCount, setStepCount] = useState<number | null>(null);

  useEffect(() => {
    if (!config) {
      router.push("/test/setup");
    }
  }, [config, router]);

  if (!config) {
    return null;
  }

  const totalTrials =
    config.mode === "both" ? config.trialsPerMode * 2 : config.trialsPerMode;

  const currentModeTrials = trials.filter((t) => t.mode === currentMode);
  const completedModeTrials = currentModeTrials.length;

  const handleStart = () => {
    dispatch(startTest());
  };

  const handleStop = (time: number) => {
    dispatch(stopTest());

    const speed = calculateSpeed(time, config.distance);
    const cadence = stepCount ? calculateCadence(stepCount, time) : undefined;

    const trial: TrialResult = {
      trialNumber: completedModeTrials + 1,
      mode: currentMode,
      time,
      speed,
      stepCount: stepCount || undefined,
      cadence,
      strideLength: cadence ? (speed * 60) / (cadence / 2) : undefined,
      isValid: true,
    };

    dispatch(addTrialResult(trial));
    setStepCount(null);

    // Check if we need to rest or switch modes
    const newTrialCount = completedModeTrials + 1;

    if (newTrialCount >= config.trialsPerMode) {
      // Mode completed
      if (config.mode === "both" && currentMode === "comfortable") {
        // Switch to fast mode after rest
        setShowRest(true);
      } else {
        // All done
        router.push("/test/result");
      }
    } else if (newTrialCount < config.trialsPerMode) {
      // More trials in current mode - show rest
      setShowRest(true);
    }
  };

  const handleRestComplete = () => {
    setShowRest(false);

    // Check if we need to switch modes
    if (
      config.mode === "both" &&
      currentMode === "comfortable" &&
      currentModeTrials.length >= config.trialsPerMode
    ) {
      dispatch(setCurrentMode("fast"));
    }
  };

  const handleInvalidate = () => {
    if (isTimerRunning) {
      dispatch(stopTest());
    }
    // Reset without saving
  };

  // Show rest screen
  if (showRest) {
    return (
      <div className="min-h-screen bg-[hsl(var(--background))] p-4">
        <RestCountdown
          duration={config.restDuration}
          onComplete={handleRestComplete}
          trialNumber={currentTrialIndex}
          totalTrials={totalTrials}
        />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[hsl(var(--background))]">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-[hsl(var(--border))] bg-[hsl(var(--background))]/95 backdrop-blur">
        <div className="flex h-14 items-center justify-between px-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => router.push("/test/setup")}
            disabled={isTimerRunning}
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            설정
          </Button>
          <div className="text-center">
            <p className="text-sm font-medium">{config.patientName}</p>
            <Badge variant={currentMode === "comfortable" ? "default" : "secondary"}>
              {currentMode === "comfortable" ? "편안한 속도" : "빠른 속도"}
            </Badge>
          </div>
          <div className="w-16 text-right text-sm text-[hsl(var(--muted-foreground))]">
            {currentTrialIndex + 1}/{totalTrials}
          </div>
        </div>
      </header>

      {/* Camera View (if enabled) */}
      {config.useCamera && (
        <div className="relative">
          <CameraView isActive={true} />
          <AROverlay />
        </div>
      )}

      {/* Main Content */}
      <main className="p-4">
        {/* Timer Display */}
        <Card className="mb-4">
          <CardContent className="py-8">
            <Timer size="lg" onStop={handleStop} />
          </CardContent>
        </Card>

        {/* Trial Info */}
        <div className="mb-4 grid grid-cols-3 gap-2 text-center">
          <div className="rounded-lg bg-[hsl(var(--secondary))] p-3">
            <p className="text-2xl font-bold">{completedModeTrials + 1}</p>
            <p className="text-xs text-[hsl(var(--muted-foreground))]">
              현재 시행
            </p>
          </div>
          <div className="rounded-lg bg-[hsl(var(--secondary))] p-3">
            <p className="text-2xl font-bold">{config.distance}</p>
            <p className="text-xs text-[hsl(var(--muted-foreground))]">
              거리 (m)
            </p>
          </div>
          <div className="rounded-lg bg-[hsl(var(--secondary))] p-3">
            <p className="text-2xl font-bold">{config.trialsPerMode}</p>
            <p className="text-xs text-[hsl(var(--muted-foreground))]">
              총 시행
            </p>
          </div>
        </div>

        {/* Step Counter (if enabled) */}
        {config.useStepCounter && (
          <Card className="mb-4">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">걸음 수</span>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setStepCount((prev) => Math.max(0, (prev || 0) - 1))}
                    disabled={!isTimerRunning}
                  >
                    -
                  </Button>
                  <span className="w-12 text-center text-xl font-bold">
                    {stepCount || 0}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setStepCount((prev) => (prev || 0) + 1)}
                    disabled={!isTimerRunning}
                  >
                    +
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Control Buttons */}
        <div className="space-y-3">
          {!isTimerRunning ? (
            <Button className="w-full" size="lg" onClick={handleStart}>
              <Play className="mr-2 h-5 w-5" />
              시작
            </Button>
          ) : (
            <Button
              className="w-full"
              size="lg"
              variant="destructive"
              onClick={() => handleStop(timerValue)}
            >
              <Square className="mr-2 h-5 w-5" />
              정지
            </Button>
          )}

          <div className="grid grid-cols-2 gap-3">
            <Button
              variant="outline"
              onClick={handleInvalidate}
              disabled={!isTimerRunning && timerValue === 0}
            >
              <RotateCcw className="mr-2 h-4 w-4" />
              무효 처리
            </Button>
            <Button
              variant="outline"
              onClick={() => router.push("/test/result")}
              disabled={isTimerRunning || trials.length === 0}
            >
              <CheckCircle className="mr-2 h-4 w-4" />
              검사 종료
            </Button>
          </div>
        </div>

        {/* Previous Trials */}
        {currentModeTrials.length > 0 && (
          <Card className="mt-4">
            <CardContent className="p-4">
              <p className="mb-2 text-sm font-medium">이전 시행</p>
              <div className="space-y-1">
                {currentModeTrials.map((trial, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between rounded-md bg-[hsl(var(--secondary))]/50 px-3 py-2 text-sm"
                  >
                    <span>시행 {trial.trialNumber}</span>
                    <div className="flex items-center gap-2">
                      <span>{trial.time.toFixed(2)}s</span>
                      <span className="font-medium">
                        {trial.speed.toFixed(2)} m/s
                      </span>
                      {trial.isValid ? (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      ) : (
                        <AlertCircle className="h-4 w-4 text-red-500" />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Instructions */}
        <div className="mt-4 rounded-lg bg-[hsl(var(--muted))] p-4 text-sm text-[hsl(var(--muted-foreground))]">
          <p className="mb-2 font-medium">검사 지침</p>
          {currentMode === "comfortable" ? (
            <p>
              &ldquo;평소 걷는 속도로 편안하게 걸어주세요. 시작선에서 출발하여 끝선까지
              걸어가시면 됩니다.&rdquo;
            </p>
          ) : (
            <p>
              &ldquo;안전한 범위 내에서 가능한 빠르게 걸어주세요. 뛰지 않고 걷는 것이
              중요합니다.&rdquo;
            </p>
          )}
        </div>
      </main>
    </div>
  );
}
