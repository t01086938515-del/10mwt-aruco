"use client";

import { useState, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { mockPatients, mockTestHistory, TestRecord } from "@/lib/mockData";
import {
  ArrowLeft,
  GitCompare,
  TrendingUp,
  TrendingDown,
  Minus,
  Play,
  Pause,
  RotateCcw,
  Video,
  MapPin,
  Clock,
  Footprints,
} from "lucide-react";
import VideoCompare from "@/components/compare/VideoCompare";
import RealLifeCalculator from "@/components/compare/RealLifeCalculator";
import CompareChart from "@/components/compare/CompareChart";

interface RecordWithPatient extends TestRecord {
  patientName: string;
  diagnosis: string;
}

export default function ComparePage() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // URL에서 선택된 기록 ID 가져오기
  const selectedIds = searchParams.get("ids")?.split(",") || [];

  const [record1Id, setRecord1Id] = useState<string>(selectedIds[0] || "");
  const [record2Id, setRecord2Id] = useState<string>(selectedIds[1] || "");
  const [showVideoCompare, setShowVideoCompare] = useState(false);

  // 환자 정보 포함된 기록 목록
  const recordsWithPatient: RecordWithPatient[] = useMemo(() => {
    return mockTestHistory.map((test) => {
      const patient = mockPatients.find((p) => p.id === test.patientId);
      return {
        ...test,
        patientName: patient?.name || "Unknown",
        diagnosis: patient?.diagnosis || "",
      };
    });
  }, []);

  const record1 = recordsWithPatient.find((r) => r.id === record1Id);
  const record2 = recordsWithPatient.find((r) => r.id === record2Id);

  // 속도 차이 계산
  const getSpeedDiff = (speed1: number, speed2: number) => {
    const diff = speed2 - speed1;
    const percent = ((diff / speed1) * 100).toFixed(1);
    return { diff, percent };
  };

  // 개선도 표시 아이콘
  const ImprovementIcon = ({ diff }: { diff: number }) => {
    if (diff > 0.05) return <TrendingUp className="h-4 w-4 text-green-500" />;
    if (diff < -0.05) return <TrendingDown className="h-4 w-4 text-red-500" />;
    return <Minus className="h-4 w-4 text-gray-500" />;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <GitCompare className="h-6 w-6" />
            기록 비교
          </h1>
          <p className="text-[hsl(var(--muted-foreground))]">
            두 검사 기록을 선택하여 비교하세요
          </p>
        </div>
        <Button variant="outline" onClick={() => router.push("/records")}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          돌아가기
        </Button>
      </div>

      {/* Record Selection */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Record 1 Selection */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Badge className="bg-blue-500">1</Badge>
              기준 기록 선택
            </CardTitle>
          </CardHeader>
          <CardContent>
            <select
              value={record1Id}
              onChange={(e) => setRecord1Id(e.target.value)}
              className="w-full p-3 rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--background))] text-[hsl(var(--foreground))]"
            >
              <option value="">기록을 선택하세요</option>
              {recordsWithPatient.map((record) => (
                <option key={record.id} value={record.id} disabled={record.id === record2Id}>
                  {record.date} - {record.patientName} ({record.comfortableSpeed.toFixed(2)} m/s)
                </option>
              ))}
            </select>
            {record1 && (
              <div className="mt-3 p-3 bg-blue-500/10 rounded-lg">
                <div className="font-medium">{record1.patientName}</div>
                <div className="text-sm text-[hsl(var(--muted-foreground))]">
                  {record1.date} | {record1.diagnosis}
                </div>
                <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                  <div>편안한 속도: <span className="font-bold">{record1.comfortableSpeed.toFixed(2)} m/s</span></div>
                  <div>빠른 속도: <span className="font-bold">{record1.fastSpeed.toFixed(2)} m/s</span></div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Record 2 Selection */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Badge className="bg-green-500">2</Badge>
              비교 기록 선택
            </CardTitle>
          </CardHeader>
          <CardContent>
            <select
              value={record2Id}
              onChange={(e) => setRecord2Id(e.target.value)}
              className="w-full p-3 rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--background))] text-[hsl(var(--foreground))]"
            >
              <option value="">기록을 선택하세요</option>
              {recordsWithPatient.map((record) => (
                <option key={record.id} value={record.id} disabled={record.id === record1Id}>
                  {record.date} - {record.patientName} ({record.comfortableSpeed.toFixed(2)} m/s)
                </option>
              ))}
            </select>
            {record2 && (
              <div className="mt-3 p-3 bg-green-500/10 rounded-lg">
                <div className="font-medium">{record2.patientName}</div>
                <div className="text-sm text-[hsl(var(--muted-foreground))]">
                  {record2.date} | {record2.diagnosis}
                </div>
                <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                  <div>편안한 속도: <span className="font-bold">{record2.comfortableSpeed.toFixed(2)} m/s</span></div>
                  <div>빠른 속도: <span className="font-bold">{record2.fastSpeed.toFixed(2)} m/s</span></div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Comparison Results */}
      {record1 && record2 && (
        <>
          {/* Speed Comparison */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">속도 비교 분석</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Comfortable Speed Comparison */}
                <div className="p-4 rounded-lg bg-[hsl(var(--accent))]">
                  <div className="text-sm text-[hsl(var(--muted-foreground))] mb-2">편안한 보행 속도</div>
                  <div className="flex items-center justify-between">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-500">{record1.comfortableSpeed.toFixed(2)}</div>
                      <div className="text-xs text-[hsl(var(--muted-foreground))]">{record1.date}</div>
                    </div>
                    <div className="flex flex-col items-center px-4">
                      <ImprovementIcon diff={getSpeedDiff(record1.comfortableSpeed, record2.comfortableSpeed).diff} />
                      <div className={`text-lg font-bold ${
                        getSpeedDiff(record1.comfortableSpeed, record2.comfortableSpeed).diff > 0
                          ? 'text-green-500'
                          : getSpeedDiff(record1.comfortableSpeed, record2.comfortableSpeed).diff < 0
                            ? 'text-red-500'
                            : 'text-gray-500'
                      }`}>
                        {getSpeedDiff(record1.comfortableSpeed, record2.comfortableSpeed).diff > 0 ? '+' : ''}
                        {getSpeedDiff(record1.comfortableSpeed, record2.comfortableSpeed).percent}%
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-500">{record2.comfortableSpeed.toFixed(2)}</div>
                      <div className="text-xs text-[hsl(var(--muted-foreground))]">{record2.date}</div>
                    </div>
                  </div>
                </div>

                {/* Fast Speed Comparison */}
                <div className="p-4 rounded-lg bg-[hsl(var(--accent))]">
                  <div className="text-sm text-[hsl(var(--muted-foreground))] mb-2">빠른 보행 속도</div>
                  <div className="flex items-center justify-between">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-500">{record1.fastSpeed.toFixed(2)}</div>
                      <div className="text-xs text-[hsl(var(--muted-foreground))]">{record1.date}</div>
                    </div>
                    <div className="flex flex-col items-center px-4">
                      <ImprovementIcon diff={getSpeedDiff(record1.fastSpeed, record2.fastSpeed).diff} />
                      <div className={`text-lg font-bold ${
                        getSpeedDiff(record1.fastSpeed, record2.fastSpeed).diff > 0
                          ? 'text-green-500'
                          : getSpeedDiff(record1.fastSpeed, record2.fastSpeed).diff < 0
                            ? 'text-red-500'
                            : 'text-gray-500'
                      }`}>
                        {getSpeedDiff(record1.fastSpeed, record2.fastSpeed).diff > 0 ? '+' : ''}
                        {getSpeedDiff(record1.fastSpeed, record2.fastSpeed).percent}%
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-500">{record2.fastSpeed.toFixed(2)}</div>
                      <div className="text-xs text-[hsl(var(--muted-foreground))]">{record2.date}</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Cadence Comparison */}
              <div className="mt-4 grid grid-cols-2 gap-4">
                <div className="p-3 rounded-lg border border-[hsl(var(--border))]">
                  <div className="text-sm text-[hsl(var(--muted-foreground))]">편안한 케이던스</div>
                  <div className="flex items-center gap-2">
                    <span className="text-blue-500">{record1.comfortableCadence}</span>
                    <span className="text-[hsl(var(--muted-foreground))]">→</span>
                    <span className="text-green-500">{record2.comfortableCadence}</span>
                    <span className="text-xs text-[hsl(var(--muted-foreground))]">steps/min</span>
                  </div>
                </div>
                <div className="p-3 rounded-lg border border-[hsl(var(--border))]">
                  <div className="text-sm text-[hsl(var(--muted-foreground))]">빠른 케이던스</div>
                  <div className="flex items-center gap-2">
                    <span className="text-blue-500">{record1.fastCadence}</span>
                    <span className="text-[hsl(var(--muted-foreground))]">→</span>
                    <span className="text-green-500">{record2.fastCadence}</span>
                    <span className="text-xs text-[hsl(var(--muted-foreground))]">steps/min</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Chart Comparison */}
          <CompareChart record1={record1} record2={record2} />

          {/* Video Compare Toggle */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Video className="h-5 w-5" />
                  동영상 비교
                </CardTitle>
                <Button
                  variant={showVideoCompare ? "default" : "outline"}
                  onClick={() => setShowVideoCompare(!showVideoCompare)}
                >
                  {showVideoCompare ? "동영상 숨기기" : "동영상 비교 보기"}
                </Button>
              </div>
            </CardHeader>
            {showVideoCompare && (
              <CardContent>
                <VideoCompare
                  video1={{
                    url: record1.videoUrl || "",
                    label: `${record1.patientName} - ${record1.date}`,
                    speed: record1.comfortableSpeed,
                  }}
                  video2={{
                    url: record2.videoUrl || "",
                    label: `${record2.patientName} - ${record2.date}`,
                    speed: record2.comfortableSpeed,
                  }}
                />
              </CardContent>
            )}
          </Card>

          {/* Real Life Calculator */}
          <RealLifeCalculator
            comfortableSpeed={record2.comfortableSpeed}
            fastSpeed={record2.fastSpeed}
            patientName={record2.patientName}
          />
        </>
      )}
    </div>
  );
}
