"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAppSelector, useAppDispatch } from "@/store/hooks";
import { setConfig, TestConfig, TestMode, CameraAngle } from "@/store/slices/testSessionSlice";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { User, Settings, ArrowRight, Cpu, Info, Video } from "lucide-react";

export function TestConfigForm() {
  const router = useRouter();
  const dispatch = useAppDispatch();
  const { config } = useAppSelector((state) => state.testSession);
  const { patientList, currentPatient } = useAppSelector((state) => state.patient);

  const [formData, setFormData] = useState({
    patientId: config?.patientId || currentPatient?.id || "",
    mode: (config?.mode || "both") as TestMode,
    trialsPerMode: config?.trialsPerMode || 3,
    restDuration: config?.restDuration || 60,
    distance: config?.distance || 10,
    cameraAngle: (config?.cameraAngle || "lateral") as CameraAngle,
    markerSize: config?.markerSize || 0.15,
  });

  const selectedPatient = patientList.find((p) => p.id === formData.patientId);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedPatient) return;

    const testConfig: TestConfig = {
      patientId: formData.patientId,
      patientName: selectedPatient.name,
      mode: formData.mode,
      trialsPerMode: formData.trialsPerMode,
      restDuration: formData.restDuration,
      useCamera: true,
      useStepCounter: false,
      distance: formData.distance,
      measurementMethod: "ai",
      cameraAngle: formData.cameraAngle,
      markerSize: formData.markerSize,
    };

    dispatch(setConfig(testConfig));
    router.push("/test/ai-upload");
  };

  const totalTrials =
    formData.mode === "both"
      ? formData.trialsPerMode * 2
      : formData.trialsPerMode;

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      {/* Patient Selection */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-lg">
            <User className="h-5 w-5" />
            환자 선택
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-4">
          <Select
            value={formData.patientId}
            onChange={(e) => setFormData({ ...formData, patientId: e.target.value })}
            options={patientList.map((p) => ({
              value: p.id,
              label: `${p.name} - ${p.diagnosis}`,
            }))}
            placeholder="환자를 선택하세요"
          />
          {selectedPatient && (
            <div className="mt-5 rounded-xl bg-[hsl(var(--secondary))] p-5">
              <div className="flex items-center gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-[hsl(var(--primary))]/10 text-lg font-bold text-[hsl(var(--primary))]">
                  {selectedPatient.name.charAt(0)}
                </div>
                <div className="space-y-1">
                  <p className="font-semibold text-base">{selectedPatient.name}</p>
                  <p className="text-sm text-[hsl(var(--muted-foreground))]">
                    {selectedPatient.diagnosis}
                  </p>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* AI 분석 설정 */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Cpu className="h-5 w-5" />
            AI 자동 분석
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-4 space-y-8">
          {/* 촬영 각도 선택 */}
          <div className="space-y-4">
            <label className="text-sm font-medium">촬영 각도</label>
            <div className="grid grid-cols-2 gap-5">
              <button
                type="button"
                onClick={() => setFormData({ ...formData, cameraAngle: "lateral" })}
                className={`relative flex flex-col items-center gap-4 rounded-xl border-2 p-6 transition-all ${
                  formData.cameraAngle === "lateral"
                    ? "border-[hsl(var(--primary))] bg-[hsl(var(--primary))]/5"
                    : "border-[hsl(var(--border))] hover:border-[hsl(var(--primary))]/50"
                }`}
              >
                <Video className={`h-10 w-10 ${
                  formData.cameraAngle === "lateral"
                    ? "text-[hsl(var(--primary))]"
                    : "text-[hsl(var(--muted-foreground))]"
                }`} />
                <span className="font-medium text-base">측면 촬영</span>
                <span className="text-sm text-[hsl(var(--muted-foreground))] text-center">
                  옆에서 보행 촬영
                </span>
                {formData.cameraAngle === "lateral" && (
                  <div className="absolute -right-1 -top-1 h-4 w-4 rounded-full bg-[hsl(var(--primary))]">
                    <svg className="h-4 w-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                )}
              </button>

              <button
                type="button"
                onClick={() => setFormData({ ...formData, cameraAngle: "posterior" })}
                className={`relative flex flex-col items-center gap-4 rounded-xl border-2 p-6 transition-all ${
                  formData.cameraAngle === "posterior"
                    ? "border-[hsl(var(--primary))] bg-[hsl(var(--primary))]/5"
                    : "border-[hsl(var(--border))] hover:border-[hsl(var(--primary))]/50"
                }`}
              >
                <Video className={`h-10 w-10 rotate-90 ${
                  formData.cameraAngle === "posterior"
                    ? "text-[hsl(var(--primary))]"
                    : "text-[hsl(var(--muted-foreground))]"
                }`} />
                <span className="font-medium text-base">후면 촬영</span>
                <span className="text-sm text-[hsl(var(--muted-foreground))] text-center">
                  뒤에서 보행 촬영
                </span>
                {formData.cameraAngle === "posterior" && (
                  <div className="absolute -right-1 -top-1 h-4 w-4 rounded-full bg-[hsl(var(--primary))]">
                    <svg className="h-4 w-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                )}
              </button>
            </div>
          </div>

          {/* 촬영 각도별 안내 */}
          <div className="rounded-lg bg-blue-50 p-5 dark:bg-blue-950/30">
            <div className="flex gap-3">
              <Info className="h-5 w-5 shrink-0 text-blue-600 dark:text-blue-400" />
              <div className="space-y-2 text-sm">
                <p className="font-medium text-blue-900 dark:text-blue-100">
                  {formData.cameraAngle === "lateral" ? "측면 촬영 안내" : "후면 촬영 안내"}
                </p>
                <ul className="list-inside list-disc space-y-1 text-blue-800 dark:text-blue-200">
                  {formData.cameraAngle === "lateral" ? (
                    <>
                      <li>측정 구간 양끝에 ArUco 마커 설치</li>
                      <li>보행 경로 옆에서 촬영</li>
                      <li>전신이 보이는 카메라 앵글</li>
                    </>
                  ) : (
                    <>
                      <li>측정 구간 시작점에 ArUco 마커 설치</li>
                      <li>보행 경로 뒤에서 촬영</li>
                      <li>환자가 카메라에서 멀어지는 방향으로 보행</li>
                    </>
                  )}
                </ul>
                <p className="text-blue-700 dark:text-blue-300">
                  → 속도, 케이던스, 보폭, 양측 대칭성 등 자동 분석
                </p>
              </div>
            </div>
          </div>

          {/* ArUco 마커 크기 */}
          <div className="space-y-4">
            <label className="text-sm font-medium">ArUco 마커 크기</label>
            <div className="flex items-center gap-4">
              <Input
                type="number"
                value={formData.markerSize}
                onChange={(e) =>
                  setFormData({ ...formData, markerSize: Number(e.target.value) })
                }
                min={0.05}
                max={0.5}
                step={0.01}
                className="w-28 h-11"
              />
              <span className="text-sm text-[hsl(var(--muted-foreground))]">m (기본: 15cm)</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Test Settings */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Settings className="h-5 w-5" />
            검사 설정
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-4 space-y-10">
          {/* Test Mode */}
          <div className="space-y-4">
            <label className="text-sm font-medium">검사 모드</label>
            <div className="grid grid-cols-3 gap-4">
              {[
                { value: "comfortable", label: "편안한 속도" },
                { value: "fast", label: "빠른 속도" },
                { value: "both", label: "모두" },
              ].map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() =>
                    setFormData({ ...formData, mode: option.value as TestMode })
                  }
                  className={`rounded-lg border p-5 text-sm font-medium transition-colors ${
                    formData.mode === option.value
                      ? "border-[hsl(var(--primary))] bg-[hsl(var(--primary))]/10 text-[hsl(var(--primary))]"
                      : "border-[hsl(var(--border))] hover:bg-[hsl(var(--accent))]"
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          {/* Trials per Mode */}
          <div className="space-y-4">
            <Slider
              label="시행 횟수 (모드당)"
              value={formData.trialsPerMode}
              onChange={(e) =>
                setFormData({ ...formData, trialsPerMode: Number(e.target.value) })
              }
              min={1}
              max={5}
              unit="회"
            />
          </div>

          {/* Rest Duration */}
          <div className="space-y-4">
            <Slider
              label="시행 간 휴식 시간"
              value={formData.restDuration}
              onChange={(e) =>
                setFormData({ ...formData, restDuration: Number(e.target.value) })
              }
              min={30}
              max={120}
              step={10}
              unit="초"
            />
          </div>

          {/* Distance */}
          <div className="space-y-4">
            <label className="text-sm font-medium">측정 거리</label>
            <div className="flex items-center gap-4">
              <Input
                type="number"
                value={formData.distance}
                onChange={(e) =>
                  setFormData({ ...formData, distance: Number(e.target.value) })
                }
                min={6}
                max={20}
                className="w-28 h-11"
              />
              <span className="text-sm text-[hsl(var(--muted-foreground))]">m</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Summary */}
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <p className="text-sm text-[hsl(var(--muted-foreground))]">총 시행 횟수</p>
              <p className="text-3xl font-bold">{totalTrials}회</p>
            </div>
            <div className="flex gap-3">
              {formData.mode !== "fast" && <Badge className="px-4 py-2">편안한 속도</Badge>}
              {formData.mode !== "comfortable" && <Badge variant="secondary" className="px-4 py-2">빠른 속도</Badge>}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Submit Button */}
      <Button
        type="submit"
        className="w-full h-14 text-base"
        size="lg"
        disabled={!selectedPatient}
      >
        영상 업로드
        <ArrowRight className="ml-2 h-5 w-5" />
      </Button>
    </form>
  );
}
