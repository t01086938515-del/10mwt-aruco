"use client";

import { TrialResult } from "@/store/slices/testSessionSlice";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  getAverageFromTrials,
  getBestTrial,
  calculateCV,
  formatTime,
  formatSpeed,
} from "@/lib/calculations";
import { TrendingUp, Clock, Footprints, Activity } from "lucide-react";

interface ResultSummaryProps {
  trials: TrialResult[];
  mode: "comfortable" | "fast" | "both";
}

export function ResultSummary({ trials, mode }: ResultSummaryProps) {
  const comfortableTrials = trials.filter((t) => t.mode === "comfortable" && t.isValid);
  const fastTrials = trials.filter((t) => t.mode === "fast" && t.isValid);

  const comfortableAvgSpeed = getAverageFromTrials(comfortableTrials, "speed");
  const fastAvgSpeed = getAverageFromTrials(fastTrials, "speed");
  const comfortableAvgTime = getAverageFromTrials(comfortableTrials, "time");
  const fastAvgTime = getAverageFromTrials(fastTrials, "time");

  const comfortableBest = getBestTrial(comfortableTrials);
  const fastBest = getBestTrial(fastTrials);

  const comfortableCV = calculateCV(comfortableTrials, "speed");
  const fastCV = calculateCV(fastTrials, "speed");

  const renderModeResults = (
    modeLabel: string,
    modeTrials: TrialResult[],
    avgSpeed: number,
    avgTime: number,
    best: TrialResult | null,
    cv: number
  ) => {
    if (modeTrials.length === 0) return null;

    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center justify-between text-lg">
            <span>{modeLabel}</span>
            <Badge variant={avgSpeed >= 0.8 ? "success" : "warning"}>
              {avgSpeed >= 0.8 ? "지역사회 보행" : "제한적 보행"}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Main Stats */}
          <div className="mb-4 grid grid-cols-2 gap-4">
            <div className="rounded-lg bg-[hsl(var(--primary))]/10 p-4 text-center">
              <TrendingUp className="mx-auto mb-2 h-6 w-6 text-[hsl(var(--primary))]" />
              <p className="text-3xl font-bold text-[hsl(var(--primary))]">
                {formatSpeed(avgSpeed)}
              </p>
              <p className="text-sm text-[hsl(var(--muted-foreground))]">평균 속도 (m/s)</p>
            </div>
            <div className="rounded-lg bg-[hsl(var(--secondary))] p-4 text-center">
              <Clock className="mx-auto mb-2 h-6 w-6" />
              <p className="text-3xl font-bold">{formatTime(avgTime)}</p>
              <p className="text-sm text-[hsl(var(--muted-foreground))]">평균 시간</p>
            </div>
          </div>

          {/* Additional Stats */}
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="rounded-lg border border-[hsl(var(--border))] p-3">
              <p className="text-xl font-bold">{best ? formatSpeed(best.speed) : "-"}</p>
              <p className="text-xs text-[hsl(var(--muted-foreground))]">최고 속도</p>
            </div>
            <div className="rounded-lg border border-[hsl(var(--border))] p-3">
              <p className="text-xl font-bold">{cv.toFixed(1)}%</p>
              <p className="text-xs text-[hsl(var(--muted-foreground))]">변동계수</p>
            </div>
            <div className="rounded-lg border border-[hsl(var(--border))] p-3">
              <p className="text-xl font-bold">{modeTrials.length}회</p>
              <p className="text-xs text-[hsl(var(--muted-foreground))]">유효 시행</p>
            </div>
          </div>

          {/* Trial Details */}
          <div className="mt-4">
            <p className="mb-2 text-sm font-medium">시행별 결과</p>
            <div className="space-y-1">
              {modeTrials.map((trial, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between rounded-md bg-[hsl(var(--secondary))]/50 px-3 py-2 text-sm"
                >
                  <span>시행 {trial.trialNumber}</span>
                  <div className="flex items-center gap-4">
                    <span>{formatTime(trial.time)}</span>
                    <span className="font-medium">{formatSpeed(trial.speed)} m/s</span>
                    {trial.cadence && (
                      <span className="text-[hsl(var(--muted-foreground))]">
                        {trial.cadence.toFixed(0)} steps/min
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="space-y-4">
      {(mode === "comfortable" || mode === "both") &&
        renderModeResults(
          "편안한 속도",
          comfortableTrials,
          comfortableAvgSpeed,
          comfortableAvgTime,
          comfortableBest,
          comfortableCV
        )}

      {(mode === "fast" || mode === "both") &&
        renderModeResults(
          "빠른 속도",
          fastTrials,
          fastAvgSpeed,
          fastAvgTime,
          fastBest,
          fastCV
        )}
    </div>
  );
}
