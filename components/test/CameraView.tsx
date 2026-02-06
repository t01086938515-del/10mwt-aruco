"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Camera, CameraOff, AlertCircle } from "lucide-react";

interface CameraViewProps {
  isActive: boolean;
  onError?: (error: string) => void;
}

export function CameraView({ isActive, onError }: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const startCamera = async () => {
      if (!isActive) {
        stopCamera();
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "environment",
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          streamRef.current = stream;
          setHasPermission(true);
          setError(null);
        }
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "카메라 접근에 실패했습니다.";
        setError(errorMessage);
        setHasPermission(false);
        onError?.(errorMessage);
      }
    };

    startCamera();

    return () => {
      stopCamera();
    };
  }, [isActive, onError]);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  if (!isActive) {
    return (
      <div className="flex h-64 items-center justify-center rounded-lg bg-[hsl(var(--secondary))]">
        <div className="text-center text-[hsl(var(--muted-foreground))]">
          <CameraOff className="mx-auto mb-2 h-8 w-8" />
          <p>카메라가 비활성화되어 있습니다</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-64 items-center justify-center rounded-lg bg-[hsl(var(--destructive))]/10">
        <div className="text-center text-[hsl(var(--destructive))]">
          <AlertCircle className="mx-auto mb-2 h-8 w-8" />
          <p className="mb-2">{error}</p>
          <Button variant="outline" size="sm" onClick={() => setError(null)}>
            다시 시도
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="relative overflow-hidden rounded-lg bg-black">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="h-64 w-full object-cover"
      />
      {hasPermission && (
        <div className="absolute bottom-2 left-2 flex items-center gap-1 rounded-md bg-black/50 px-2 py-1 text-xs text-white">
          <Camera className="h-3 w-3" />
          <span>녹화 중</span>
        </div>
      )}
    </div>
  );
}
