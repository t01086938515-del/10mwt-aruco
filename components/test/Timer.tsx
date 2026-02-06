"use client";

import { useEffect, useRef, useState } from "react";
import { useAppSelector, useAppDispatch } from "@/store/hooks";
import { updateTimer, stopTest } from "@/store/slices/testSessionSlice";
import { formatTime } from "@/lib/calculations";

interface TimerProps {
  onStop?: (time: number) => void;
  size?: "sm" | "md" | "lg";
}

export function Timer({ onStop, size = "lg" }: TimerProps) {
  const dispatch = useAppDispatch();
  const { isTimerRunning, timerValue } = useAppSelector((state) => state.testSession);
  const startTimeRef = useRef<number | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  useEffect(() => {
    if (isTimerRunning) {
      startTimeRef.current = performance.now() - timerValue * 1000;

      const updateTime = () => {
        if (startTimeRef.current !== null) {
          const elapsed = (performance.now() - startTimeRef.current) / 1000;
          dispatch(updateTimer(elapsed));
          animationFrameRef.current = requestAnimationFrame(updateTime);
        }
      };

      animationFrameRef.current = requestAnimationFrame(updateTime);
    } else {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isTimerRunning, dispatch]);

  const handleStop = () => {
    dispatch(stopTest());
    onStop?.(timerValue);
  };

  const sizeClasses = {
    sm: "text-2xl",
    md: "text-4xl",
    lg: "text-7xl",
  };

  return (
    <div className="text-center">
      <div
        className={`font-mono font-bold tabular-nums ${sizeClasses[size]} ${
          isTimerRunning ? "text-[hsl(var(--primary))]" : ""
        }`}
      >
        {formatTime(timerValue)}
      </div>
      {isTimerRunning && (
        <p className="mt-2 animate-pulse text-sm text-[hsl(var(--muted-foreground))]">
          측정 중...
        </p>
      )}
    </div>
  );
}
