"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAppSelector, useAppDispatch } from "@/store/hooks";
import { resetSession } from "@/store/slices/testSessionSlice";
import { updatePatient } from "@/store/slices/patientSlice";
import { ResultSummary } from "@/components/test/ResultSummary";
import { ClinicalInterpretation } from "@/components/test/ClinicalInterpretation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getAverageFromTrials } from "@/lib/calculations";
import {
  ArrowLeft,
  Save,
  Share2,
  Home,
  RefreshCw,
  FileText,
  CheckCircle,
} from "lucide-react";

export default function TestResultPage() {
  const router = useRouter();
  const dispatch = useAppDispatch();
  const { config, trials } = useAppSelector((state) => state.testSession);
  const { patientList } = useAppSelector((state) => state.patient);
  const [isSaving, setIsSaving] = useState(false);
  const [isSaved, setIsSaved] = useState(false);

  useEffect(() => {
    if (!config || trials.length === 0) {
      router.push("/test/setup");
    }
  }, [config, trials, router]);

  if (!config || trials.length === 0) {
    return null;
  }

  const patient = patientList.find((p) => p.id === config.patientId);
  const patientAge = patient
    ? new Date().getFullYear() - new Date(patient.birth).getFullYear()
    : 65;

  const comfortableTrials = trials.filter(
    (t) => t.mode === "comfortable" && t.isValid
  );
  const comfortableAvgSpeed = getAverageFromTrials(comfortableTrials, "speed");

  const handleSave = async () => {
    setIsSaving(true);

    try {
      // Simulate saving to database
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Update patient's last test date and count
      if (patient) {
        dispatch(
          updatePatient({
            ...patient,
            lastTestDate: new Date().toISOString().split("T")[0],
            testCount: (patient.testCount || 0) + 1,
            updatedAt: new Date().toISOString(),
          })
        );
      }

      setIsSaved(true);
    } catch (error) {
      console.error("Failed to save:", error);
    } finally {
      setIsSaving(false);
    }
  };

  const handleNewTest = () => {
    dispatch(resetSession());
    router.push("/test/setup");
  };

  const handleGoHome = () => {
    dispatch(resetSession());
    router.push("/");
  };

  const handleExport = () => {
    // Create a simple text report
    const report = `
10m 보행 검사 결과 보고서
========================
환자명: ${config.patientName}
검사일: ${new Date().toLocaleDateString("ko-KR")}
검사 모드: ${config.mode === "both" ? "편안한/빠른 속도" : config.mode === "comfortable" ? "편안한 속도" : "빠른 속도"}

검사 결과:
${trials
  .map(
    (t) =>
      `- 시행 ${t.trialNumber} (${t.mode === "comfortable" ? "편안한" : "빠른"}): ${t.time.toFixed(2)}초, ${t.speed.toFixed(2)} m/s`
  )
  .join("\n")}

평균 속도: ${comfortableAvgSpeed.toFixed(2)} m/s
    `.trim();

    // Copy to clipboard
    navigator.clipboard.writeText(report).then(() => {
      alert("보고서가 클립보드에 복사되었습니다.");
    });
  };

  return (
    <div className="min-h-screen bg-[hsl(var(--background))]">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-[hsl(var(--border))] bg-[hsl(var(--background))]/95 backdrop-blur">
        <div className="flex h-14 items-center justify-between px-4">
          <Button variant="ghost" size="sm" onClick={handleGoHome}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            홈
          </Button>
          <span className="font-semibold">검사 결과</span>
          <div className="w-16" />
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-2xl p-4 md:p-6">
        {/* Success Banner */}
        <Card className="mb-6 border-green-500 bg-green-500/10">
          <CardContent className="flex items-center gap-4 p-4">
            <div className="rounded-full bg-green-500 p-2">
              <CheckCircle className="h-6 w-6 text-white" />
            </div>
            <div>
              <p className="font-semibold text-green-700 dark:text-green-400">
                검사가 완료되었습니다
              </p>
              <p className="text-sm text-green-600 dark:text-green-500">
                총 {trials.length}회 시행, {trials.filter((t) => t.isValid).length}회 유효
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Patient Info */}
        <Card className="mb-6">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">환자 정보</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-[hsl(var(--primary))]/10 text-xl font-bold text-[hsl(var(--primary))]">
                {config.patientName.charAt(0)}
              </div>
              <div>
                <p className="font-medium">{config.patientName}</p>
                <p className="text-sm text-[hsl(var(--muted-foreground))]">
                  {patient?.diagnosis || "-"}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Result Summary */}
        <div className="mb-6">
          <ResultSummary trials={trials} mode={config.mode} />
        </div>

        {/* Clinical Interpretation */}
        <div className="mb-6">
          <ClinicalInterpretation
            speed={comfortableAvgSpeed}
            age={patientAge}
            diagnosis={patient?.diagnosis}
          />
        </div>

        {/* Action Buttons */}
        <div className="space-y-3">
          <Button
            className="w-full"
            size="lg"
            onClick={handleSave}
            disabled={isSaving || isSaved}
          >
            {isSaved ? (
              <>
                <CheckCircle className="mr-2 h-5 w-5" />
                저장 완료
              </>
            ) : isSaving ? (
              <>
                <span className="mr-2 h-5 w-5 animate-spin rounded-full border-2 border-current border-t-transparent" />
                저장 중...
              </>
            ) : (
              <>
                <Save className="mr-2 h-5 w-5" />
                결과 저장
              </>
            )}
          </Button>

          <div className="grid grid-cols-2 gap-3">
            <Button variant="outline" onClick={handleExport}>
              <FileText className="mr-2 h-4 w-4" />
              보고서 복사
            </Button>
            <Button variant="outline" onClick={() => {}}>
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

        {/* Test Date */}
        <p className="mt-6 text-center text-sm text-[hsl(var(--muted-foreground))]">
          검사일: {new Date().toLocaleDateString("ko-KR", {
            year: "numeric",
            month: "long",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          })}
        </p>
      </main>
    </div>
  );
}
