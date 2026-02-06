"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  Maximize2,
} from "lucide-react";

interface VideoInfo {
  url: string;
  label: string;
  speed: number;
}

interface VideoCompareProps {
  video1: VideoInfo;
  video2: VideoInfo;
}

export default function VideoCompare({ video1, video2 }: VideoCompareProps) {
  const video1Ref = useRef<HTMLVideoElement>(null);
  const video2Ref = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // 동시 재생/정지
  const togglePlayPause = () => {
    if (isPlaying) {
      video1Ref.current?.pause();
      video2Ref.current?.pause();
    } else {
      video1Ref.current?.play();
      video2Ref.current?.play();
    }
    setIsPlaying(!isPlaying);
  };

  // 처음부터 다시 재생
  const restartVideos = () => {
    if (video1Ref.current) {
      video1Ref.current.currentTime = 0;
    }
    if (video2Ref.current) {
      video2Ref.current.currentTime = 0;
    }
    setCurrentTime(0);
    if (!isPlaying) {
      togglePlayPause();
    }
  };

  // 음소거 토글
  const toggleMute = () => {
    if (video1Ref.current) video1Ref.current.muted = !isMuted;
    if (video2Ref.current) video2Ref.current.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  // 시간 업데이트
  const handleTimeUpdate = () => {
    if (video1Ref.current) {
      setCurrentTime(video1Ref.current.currentTime);
    }
  };

  // 동영상 로드 시 duration 설정
  const handleLoadedMetadata = () => {
    if (video1Ref.current) {
      setDuration(video1Ref.current.duration);
    }
  };

  // 시간 포맷
  const formatTime = (time: number) => {
    const mins = Math.floor(time / 60);
    const secs = Math.floor(time % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  // 동영상 종료 시
  const handleEnded = () => {
    setIsPlaying(false);
  };

  // 시크바 조작
  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    if (video1Ref.current) video1Ref.current.currentTime = time;
    if (video2Ref.current) video2Ref.current.currentTime = time;
    setCurrentTime(time);
  };

  return (
    <div className="space-y-4">
      {/* 마리오 2P 스타일 분할 화면 */}
      <div className="relative rounded-xl overflow-hidden bg-black">
        {/* 상단 동영상 (기준 기록) */}
        <div className="relative border-b-4 border-yellow-500">
          <div className="absolute top-2 left-2 z-10 flex items-center gap-2">
            <Badge className="bg-blue-500 text-white font-bold">P1</Badge>
            <span className="text-white text-sm bg-black/50 px-2 py-1 rounded">
              {video1.label}
            </span>
          </div>
          <div className="absolute top-2 right-2 z-10">
            <Badge variant="outline" className="bg-black/50 text-white border-white">
              {video1.speed.toFixed(2)} m/s
            </Badge>
          </div>

          {video1.url ? (
            <video
              ref={video1Ref}
              src={video1.url}
              className="w-full h-[200px] md:h-[250px] object-cover"
              muted={isMuted}
              playsInline
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              onEnded={handleEnded}
            />
          ) : (
            <div className="w-full h-[200px] md:h-[250px] bg-gray-800 flex items-center justify-center">
              <div className="text-center text-gray-400">
                <div className="text-4xl mb-2">P1</div>
                <div>동영상 없음</div>
                <div className="text-sm mt-1">{video1.label}</div>
              </div>
            </div>
          )}
        </div>

        {/* 중앙 구분선 (마리오 스타일) */}
        <div className="absolute left-0 right-0 top-1/2 -translate-y-1/2 z-20 flex items-center justify-center pointer-events-none">
          <div className="bg-yellow-500 text-black font-bold px-4 py-1 rounded-full text-sm shadow-lg">
            VS
          </div>
        </div>

        {/* 하단 동영상 (비교 기록) */}
        <div className="relative border-t-4 border-yellow-500">
          <div className="absolute top-2 left-2 z-10 flex items-center gap-2">
            <Badge className="bg-green-500 text-white font-bold">P2</Badge>
            <span className="text-white text-sm bg-black/50 px-2 py-1 rounded">
              {video2.label}
            </span>
          </div>
          <div className="absolute top-2 right-2 z-10">
            <Badge variant="outline" className="bg-black/50 text-white border-white">
              {video2.speed.toFixed(2)} m/s
            </Badge>
          </div>

          {video2.url ? (
            <video
              ref={video2Ref}
              src={video2.url}
              className="w-full h-[200px] md:h-[250px] object-cover"
              muted={isMuted}
              playsInline
              onEnded={handleEnded}
            />
          ) : (
            <div className="w-full h-[200px] md:h-[250px] bg-gray-800 flex items-center justify-center">
              <div className="text-center text-gray-400">
                <div className="text-4xl mb-2">P2</div>
                <div>동영상 없음</div>
                <div className="text-sm mt-1">{video2.label}</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 재생 컨트롤 */}
      <div className="bg-[hsl(var(--accent))] rounded-lg p-4">
        {/* 시크바 */}
        <div className="flex items-center gap-3 mb-3">
          <span className="text-sm text-[hsl(var(--muted-foreground))] w-12">
            {formatTime(currentTime)}
          </span>
          <input
            type="range"
            min="0"
            max={duration || 100}
            value={currentTime}
            onChange={handleSeek}
            className="flex-1 h-2 bg-[hsl(var(--border))] rounded-lg appearance-none cursor-pointer accent-[hsl(var(--primary))]"
          />
          <span className="text-sm text-[hsl(var(--muted-foreground))] w-12 text-right">
            {formatTime(duration)}
          </span>
        </div>

        {/* 버튼들 */}
        <div className="flex items-center justify-center gap-3">
          <Button
            variant="outline"
            size="icon"
            onClick={restartVideos}
            title="처음부터"
          >
            <RotateCcw className="h-4 w-4" />
          </Button>

          <Button
            size="lg"
            onClick={togglePlayPause}
            className="w-20"
          >
            {isPlaying ? (
              <Pause className="h-5 w-5" />
            ) : (
              <Play className="h-5 w-5" />
            )}
          </Button>

          <Button
            variant="outline"
            size="icon"
            onClick={toggleMute}
            title={isMuted ? "음소거 해제" : "음소거"}
          >
            {isMuted ? (
              <VolumeX className="h-4 w-4" />
            ) : (
              <Volume2 className="h-4 w-4" />
            )}
          </Button>
        </div>

        {/* 동기화 상태 */}
        <div className="mt-3 text-center text-sm text-[hsl(var(--muted-foreground))]">
          두 동영상이 동시에 재생됩니다
        </div>
      </div>
    </div>
  );
}
