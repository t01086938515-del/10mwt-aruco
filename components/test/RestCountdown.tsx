"use client";

import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Coffee, SkipForward } from "lucide-react";

interface RestCountdownProps {
  duration: number;
  onComplete: () => void;
  trialNumber: number;
  totalTrials: number;
}

export function RestCountdown({
  duration,
  onComplete,
  trialNumber,
  totalTrials,
}: RestCountdownProps) {
  const [timeLeft, setTimeLeft] = useState(duration);
  const [isPaused, setIsPaused] = useState(false);

  useEffect(() => {
    if (isPaused) return;

    if (timeLeft <= 0) {
      onComplete();
      return;
    }

    const timer = setInterval(() => {
      setTimeLeft((prev) => prev - 1);
    }, 1000);

    return () => clearInterval(timer);
  }, [timeLeft, isPaused, onComplete]);

  const progress = ((duration - timeLeft) / duration) * 100;

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center">
      <Card className="w-full max-w-md">
        <CardContent className="p-8 text-center">
          <div className="mb-6 flex items-center justify-center">
            <div className="rounded-full bg-[hsl(var(--primary))]/10 p-4">
              <Coffee className="h-8 w-8 text-[hsl(var(--primary))]" />
            </div>
          </div>

          <h2 className="mb-2 text-xl font-semibold">휴식 시간</h2>
          <p className="mb-6 text-[hsl(var(--muted-foreground))]">
            {trialNumber}/{totalTrials} 시행 완료
          </p>

          {/* Circular Progress */}
          <div className="relative mx-auto mb-6 h-40 w-40">
            <svg className="h-full w-full -rotate-90 transform">
              <circle
                cx="80"
                cy="80"
                r="70"
                stroke="hsl(var(--secondary))"
                strokeWidth="8"
                fill="none"
              />
              <circle
                cx="80"
                cy="80"
                r="70"
                stroke="hsl(var(--primary))"
                strokeWidth="8"
                fill="none"
                strokeLinecap="round"
                strokeDasharray={440}
                strokeDashoffset={440 - (440 * progress) / 100}
                className="transition-all duration-1000"
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-4xl font-bold">{formatTime(timeLeft)}</span>
            </div>
          </div>

          <p className="mb-6 text-sm text-[hsl(var(--muted-foreground))]">
            다음 시행까지 충분히 휴식하세요
          </p>

          <div className="flex gap-2">
            <Button
              variant="outline"
              className="flex-1"
              onClick={() => setIsPaused(!isPaused)}
            >
              {isPaused ? "계속" : "일시정지"}
            </Button>
            <Button className="flex-1" onClick={onComplete}>
              <SkipForward className="mr-2 h-4 w-4" />
              건너뛰기
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
