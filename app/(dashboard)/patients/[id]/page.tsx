"use client";

import { useEffect, use } from "react";
import { useRouter } from "next/navigation";
import { useAppSelector, useAppDispatch } from "@/store/hooks";
import { setCurrentPatient, setPatientList } from "@/store/slices/patientSlice";
import { setConfig } from "@/store/slices/testSessionSlice";
import { PatientProfile } from "@/components/patient/PatientProfile";
import { TrendChart } from "@/components/patient/TrendChart";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { mockPatients, mockTestHistory } from "@/lib/mockData";
import { ArrowLeft, PlayCircle, Edit, FileText } from "lucide-react";

interface PageProps {
  params: Promise<{ id: string }>;
}

export default function PatientDetailPage({ params }: PageProps) {
  const { id } = use(params);
  const router = useRouter();
  const dispatch = useAppDispatch();
  const { currentPatient, patientList } = useAppSelector((state) => state.patient);

  useEffect(() => {
    // Load mock data if not loaded
    if (patientList.length === 0) {
      dispatch(setPatientList(mockPatients));
    }
  }, [dispatch, patientList.length]);

  useEffect(() => {
    // Find patient by ID
    const patient = patientList.find((p) => p.id === id);
    if (patient) {
      dispatch(setCurrentPatient(patient));
    }
  }, [id, patientList, dispatch]);

  const handleStartTest = () => {
    if (currentPatient) {
      dispatch(
        setConfig({
          patientId: currentPatient.id,
          patientName: currentPatient.name,
          mode: "both",
          trialsPerMode: 3,
          restDuration: 60,
          useCamera: false,
          useStepCounter: false,
          distance: 10,
        })
      );
      router.push("/test/setup");
    }
  };

  if (!currentPatient) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-[hsl(var(--primary))] border-t-transparent mx-auto mb-4" />
          <p className="text-[hsl(var(--muted-foreground))]">환자 정보를 불러오는 중...</p>
        </div>
      </div>
    );
  }

  // Get patient's test history
  const patientHistory = mockTestHistory
    .filter((t) => t.patientId === currentPatient.id)
    .map((t) => ({
      date: t.date,
      comfortableSpeed: t.comfortableSpeed,
      fastSpeed: t.fastSpeed,
    }))
    .reverse();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <Button variant="ghost" onClick={() => router.push("/")}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          돌아가기
        </Button>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => {}}>
            <Edit className="mr-2 h-4 w-4" />
            수정
          </Button>
          <Button onClick={handleStartTest}>
            <PlayCircle className="mr-2 h-4 w-4" />
            검사 시작
          </Button>
        </div>
      </div>

      {/* Patient Profile */}
      <PatientProfile patient={currentPatient} />

      {/* Trend Chart */}
      <TrendChart data={patientHistory} />

      {/* Test History */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">검사 이력</CardTitle>
        </CardHeader>
        <CardContent>
          {mockTestHistory.filter((t) => t.patientId === currentPatient.id).length === 0 ? (
            <div className="py-8 text-center text-[hsl(var(--muted-foreground))]">
              검사 기록이 없습니다.
            </div>
          ) : (
            <div className="space-y-2">
              {mockTestHistory
                .filter((t) => t.patientId === currentPatient.id)
                .map((test) => (
                  <div
                    key={test.id}
                    className="flex items-center justify-between rounded-lg border border-[hsl(var(--border))] p-3"
                  >
                    <div className="flex items-center gap-3">
                      <FileText className="h-5 w-5 text-[hsl(var(--muted-foreground))]" />
                      <div>
                        <p className="font-medium">{test.date}</p>
                        <div className="flex gap-2 text-sm text-[hsl(var(--muted-foreground))]">
                          <span>편안한 속도: {test.comfortableSpeed.toFixed(2)} m/s</span>
                          <span>|</span>
                          <span>빠른 속도: {test.fastSpeed.toFixed(2)} m/s</span>
                        </div>
                      </div>
                    </div>
                    <Badge
                      variant={test.comfortableSpeed >= 0.8 ? "success" : "warning"}
                    >
                      {test.comfortableSpeed >= 0.8 ? "지역사회 보행" : "제한적 보행"}
                    </Badge>
                  </div>
                ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
