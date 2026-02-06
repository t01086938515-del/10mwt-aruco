"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { mockPatients, mockTestHistory } from "@/lib/mockData";
import {
  ArrowLeft,
  Search,
  Calendar,
  FileText,
  ChevronRight,
  Filter,
  GitCompare,
  Check,
  X,
} from "lucide-react";

export default function RecordsPage() {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState("");
  const [dateFilter, setDateFilter] = useState("");
  const [compareMode, setCompareMode] = useState(false);
  const [selectedRecords, setSelectedRecords] = useState<string[]>([]);

  // 비교 모드 토글
  const toggleCompareMode = () => {
    setCompareMode(!compareMode);
    setSelectedRecords([]);
  };

  // 기록 선택/해제
  const toggleRecordSelection = (recordId: string) => {
    if (selectedRecords.includes(recordId)) {
      setSelectedRecords(selectedRecords.filter((id) => id !== recordId));
    } else if (selectedRecords.length < 2) {
      setSelectedRecords([...selectedRecords, recordId]);
    }
  };

  // 비교 페이지로 이동
  const goToCompare = () => {
    if (selectedRecords.length === 2) {
      router.push(`/records/compare?ids=${selectedRecords.join(",")}`);
    }
  };

  // Combine test history with patient info
  const records = mockTestHistory.map((test) => {
    const patient = mockPatients.find((p) => p.id === test.patientId);
    return {
      ...test,
      patientName: patient?.name || "Unknown",
      diagnosis: patient?.diagnosis || "",
    };
  });

  const filteredRecords = records.filter((record) => {
    const matchesSearch =
      record.patientName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      record.diagnosis.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesDate = !dateFilter || record.date.includes(dateFilter);
    return matchesSearch && matchesDate;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">검사 기록</h1>
          <p className="text-[hsl(var(--muted-foreground))]">
            모든 10MWT 검사 이력을 확인하세요
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant={compareMode ? "default" : "outline"}
            onClick={toggleCompareMode}
          >
            <GitCompare className="mr-2 h-4 w-4" />
            {compareMode ? "비교 취소" : "기록 비교"}
          </Button>
          <Button variant="outline" onClick={() => router.push("/")}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            돌아가기
          </Button>
        </div>
      </div>

      {/* Compare Mode Banner */}
      {compareMode && (
        <div className="flex items-center justify-between p-4 bg-[hsl(var(--primary))]/10 border border-[hsl(var(--primary))]/30 rounded-lg">
          <div className="flex items-center gap-3">
            <GitCompare className="h-5 w-5 text-[hsl(var(--primary))]" />
            <div>
              <div className="font-medium">비교할 기록 2개를 선택하세요</div>
              <div className="text-sm text-[hsl(var(--muted-foreground))]">
                선택됨: {selectedRecords.length}/2
              </div>
            </div>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSelectedRecords([])}
              disabled={selectedRecords.length === 0}
            >
              <X className="mr-1 h-4 w-4" />
              선택 초기화
            </Button>
            <Button
              size="sm"
              onClick={goToCompare}
              disabled={selectedRecords.length !== 2}
            >
              <Check className="mr-1 h-4 w-4" />
              비교하기
            </Button>
          </div>
        </div>
      )}

      {/* Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-col gap-3 sm:flex-row">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[hsl(var(--muted-foreground))]" />
              <Input
                placeholder="환자 이름 또는 진단명 검색..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
            <div className="relative w-full sm:w-48">
              <Calendar className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[hsl(var(--muted-foreground))]" />
              <Input
                type="date"
                value={dateFilter}
                onChange={(e) => setDateFilter(e.target.value)}
                className="pl-9"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Records List */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">검사 이력</CardTitle>
            <Badge variant="outline">{filteredRecords.length}건</Badge>
          </div>
        </CardHeader>
        <CardContent>
          {filteredRecords.length === 0 ? (
            <div className="py-8 text-center text-[hsl(var(--muted-foreground))]">
              검색 결과가 없습니다.
            </div>
          ) : (
            <div className="space-y-2">
              {filteredRecords.map((record, index) => {
                const isSelected = selectedRecords.includes(record.id);
                return (
                  <div
                    key={record.id}
                    className={`flex cursor-pointer items-center justify-between rounded-lg border p-4 transition-all duration-200 hover:bg-[hsl(var(--accent))] hover:translate-x-1 animate-fade-in ${
                      isSelected
                        ? "border-[hsl(var(--primary))] bg-[hsl(var(--primary))]/5"
                        : "border-[hsl(var(--border))]"
                    }`}
                    style={{ animationDelay: `${index * 0.05}s` }}
                    onClick={() => {
                      if (compareMode) {
                        toggleRecordSelection(record.id);
                      } else {
                        router.push(`/patients/${record.patientId}`);
                      }
                    }}
                  >
                    <div className="flex items-center gap-4">
                      {/* 비교 모드 체크박스 */}
                      {compareMode && (
                        <div
                          className={`w-6 h-6 rounded-full border-2 flex items-center justify-center transition-colors ${
                            isSelected
                              ? "bg-[hsl(var(--primary))] border-[hsl(var(--primary))]"
                              : "border-[hsl(var(--border))]"
                          }`}
                        >
                          {isSelected && <Check className="h-4 w-4 text-white" />}
                        </div>
                      )}
                      <div className="rounded-lg bg-[hsl(var(--primary))]/10 p-2">
                        <FileText className="h-5 w-5 text-[hsl(var(--primary))]" />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{record.patientName}</span>
                          <Badge variant="outline" className="text-xs">
                            {record.diagnosis}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-4 text-sm text-[hsl(var(--muted-foreground))]">
                          <span>{record.date}</span>
                          <span>|</span>
                          <span>
                            편안한: {record.comfortableSpeed.toFixed(2)} m/s
                          </span>
                          <span>|</span>
                          <span>빠른: {record.fastSpeed.toFixed(2)} m/s</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <Badge
                        variant={
                          record.comfortableSpeed >= 0.8 ? "success" : "warning"
                        }
                      >
                        {record.comfortableSpeed >= 0.8
                          ? "지역사회 보행"
                          : "제한적 보행"}
                      </Badge>
                      {!compareMode && (
                        <ChevronRight className="h-5 w-5 text-[hsl(var(--muted-foreground))]" />
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-3">
        <Card>
          <CardContent className="p-4 text-center">
            <p className="text-3xl font-bold text-[hsl(var(--primary))]">
              {records.length}
            </p>
            <p className="text-sm text-[hsl(var(--muted-foreground))]">
              총 검사 수
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <p className="text-3xl font-bold text-green-500">
              {records.filter((r) => r.comfortableSpeed >= 0.8).length}
            </p>
            <p className="text-sm text-[hsl(var(--muted-foreground))]">
              지역사회 보행
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <p className="text-3xl font-bold text-orange-500">
              {(
                records.reduce((sum, r) => sum + r.comfortableSpeed, 0) /
                records.length
              ).toFixed(2)}
            </p>
            <p className="text-sm text-[hsl(var(--muted-foreground))]">
              평균 속도 (m/s)
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
