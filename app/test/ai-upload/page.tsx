"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAppSelector } from "@/store/hooks";
import { useAIAnalysis } from "@/lib/useAIAnalysis";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ArrowLeft,
  Upload,
  FileVideo,
  Play,
  AlertCircle,
  CheckCircle,
  Loader2,
  X,
} from "lucide-react";

export default function AIUploadPage() {
  const router = useRouter();
  const { config } = useAppSelector((state) => state.testSession);
  const { currentPatient } = useAppSelector((state) => state.patient);
  const { startAnalysis, status, error } = useAIAnalysis();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 설정이 없으면 설정 페이지로 리다이렉트
  useEffect(() => {
    if (!config) {
      router.push("/test/setup");
    }
  }, [config, router]);

  if (!config) {
    return null;
  }

  const handleFileSelect = useCallback((file: File) => {
    if (!file.type.startsWith("video/")) {
      alert("비디오 파일만 업로드할 수 있습니다.");
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setVideoPreview(url);
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFileSelect(file);
  };

  const handleClearFile = () => {
    setSelectedFile(null);
    if (videoPreview) {
      URL.revokeObjectURL(videoPreview);
      setVideoPreview(null);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile) return;

    const heightCm = config.patientHeight || currentPatient?.height;
    if (!heightCm) {
      alert("환자 키 정보가 없습니다. 설정으로 돌아가 키를 입력해주세요.");
      return;
    }

    await startAnalysis(selectedFile, {
      walkDistance: config.distance,
      patientHeightM: heightCm / 100,
    });

    // 분석 시작하면 분석 페이지로 이동
    router.push("/test/ai-analyze");
  };

  const isUploading = status === "uploading";
  const isConnecting = status === "connecting";
  const isProcessing = isUploading || isConnecting;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-gray-200 bg-gray-50/95 backdrop-blur">
        <div className="flex h-16 items-center justify-between px-4">
          <Button
            variant="ghost"
            onClick={() => router.push("/test/setup")}
            disabled={isProcessing}
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            설정으로
          </Button>
          <span className="font-semibold">AI 분석 - 영상 업로드</span>
          <div className="w-20" />
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-2xl p-4 md:p-6">
        {/* 환자 정보 */}
        <Card className="mb-6">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[hsl(var(--primary))]/10 text-[hsl(var(--primary))] font-bold">
                {config.patientName.charAt(0)}
              </div>
              <div>
                <p className="font-medium">{config.patientName}</p>
                <p className="text-sm text-gray-500">
                  측정 거리: {config.distance}m | 키: {config.patientHeight || currentPatient?.height || "미입력"}cm
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 영상 업로드 영역 */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <FileVideo className="h-5 w-5" />
              보행 영상 업로드
            </CardTitle>
          </CardHeader>
          <CardContent>
            {!selectedFile ? (
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
                  isDragOver
                    ? "border-[hsl(var(--primary))] bg-[hsl(var(--primary))]/5"
                    : "border-gray-300 hover:border-[hsl(var(--primary))]/50"
                }`}
              >
                <Upload className="mx-auto mb-4 h-12 w-12 text-gray-400" />
                <p className="mb-2 text-lg font-medium">
                  영상 파일을 드래그하거나 클릭하여 선택
                </p>
                <p className="text-sm text-gray-500">
                  MP4, MOV, AVI 파일 지원
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  onChange={handleInputChange}
                  className="hidden"
                />
              </div>
            ) : (
              <div className="space-y-4">
                {/* 비디오 미리보기 */}
                <div className="relative overflow-hidden rounded-lg bg-black">
                  <video
                    src={videoPreview || undefined}
                    controls
                    className="max-h-64 w-full"
                  />
                  <button
                    onClick={handleClearFile}
                    disabled={isProcessing}
                    className="absolute right-2 top-2 rounded-full bg-black/50 p-1 text-white hover:bg-black/70 disabled:opacity-50"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </div>

                {/* 파일 정보 */}
                <div className="flex items-center justify-between rounded-lg bg-gray-100 p-3">
                  <div className="flex items-center gap-2">
                    <FileVideo className="h-5 w-5 text-gray-500" />
                    <span className="text-sm font-medium">{selectedFile.name}</span>
                  </div>
                  <span className="text-sm text-gray-500">
                    {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
                  </span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* 상태 표시 */}
        {error && (
          <Card className="mb-6 border-red-200 bg-red-50">
            <CardContent className="flex items-center gap-3 p-4">
              <AlertCircle className="h-5 w-5 text-red-500" />
              <p className="text-red-700">{error}</p>
            </CardContent>
          </Card>
        )}

        {isProcessing && (
          <Card className="mb-6 border-blue-200 bg-blue-50">
            <CardContent className="flex items-center gap-3 p-4">
              <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
              <p className="text-blue-700">
                {isUploading ? "영상 업로드 중..." : "서버 연결 중..."}
              </p>
            </CardContent>
          </Card>
        )}

        {/* 촬영 가이드 */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="text-lg">촬영 가이드</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3">
              {[
                "카메라를 보행 경로의 측면에 고정하세요",
                "환자의 전신이 화면에 보이도록 촬영하세요",
                "보행 시작 전후로 2-3초 여유를 두세요",
                "환자가 프레임 안으로 걸어 들어오고 나가는 모습을 촬영하세요",
              ].map((tip, index) => (
                <li key={index} className="flex items-start gap-2">
                  <CheckCircle className="mt-0.5 h-4 w-4 shrink-0 text-green-500" />
                  <span className="text-sm">{tip}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        {/* 분석 시작 버튼 */}
        <Button
          onClick={handleStartAnalysis}
          disabled={!selectedFile || isProcessing}
          className="w-full"
          size="lg"
        >
          {isProcessing ? (
            <>
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
              처리 중...
            </>
          ) : (
            <>
              <Play className="mr-2 h-5 w-5" />
              AI 분석 시작
            </>
          )}
        </Button>
      </main>
    </div>
  );
}
