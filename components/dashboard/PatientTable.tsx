"use client";

import { useRouter } from "next/navigation";
import { Patient } from "@/store/slices/patientSlice";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Search, ChevronRight, Calendar, Activity } from "lucide-react";

interface PatientTableProps {
  patients: Patient[];
  searchQuery: string;
  onSearchChange: (query: string) => void;
}

export function PatientTable({ patients, searchQuery, onSearchChange }: PatientTableProps) {
  const router = useRouter();

  const filteredPatients = patients.filter(
    (patient) =>
      patient.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      patient.diagnosis.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const formatDate = (dateString?: string) => {
    if (!dateString) return "-";
    const date = new Date(dateString);
    return date.toLocaleDateString("ko-KR", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  const calculateAge = (birth: string) => {
    const birthDate = new Date(birth);
    const today = new Date();
    let age = today.getFullYear() - birthDate.getFullYear();
    const monthDiff = today.getMonth() - birthDate.getMonth();
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
      age--;
    }
    return age;
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <CardTitle className="text-lg">환자 목록</CardTitle>
          <div className="relative w-full sm:w-64">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[hsl(var(--muted-foreground))]" />
            <Input
              placeholder="이름 또는 진단명 검색..."
              value={searchQuery}
              onChange={(e) => onSearchChange(e.target.value)}
              className="pl-9"
            />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {filteredPatients.length === 0 ? (
          <div className="py-8 text-center text-[hsl(var(--muted-foreground))]">
            {searchQuery ? "검색 결과가 없습니다." : "등록된 환자가 없습니다."}
          </div>
        ) : (
          <div className="space-y-2">
            {filteredPatients.map((patient, index) => (
              <div
                key={patient.id}
                className="flex cursor-pointer items-center justify-between rounded-lg border border-[hsl(var(--border))] p-3 transition-all duration-200 hover:bg-[hsl(var(--accent))] hover:translate-x-1 hover:border-[hsl(var(--primary))]/30 animate-fade-in group"
                style={{ animationDelay: `${index * 0.05}s` }}
                onClick={() => router.push(`/patients/${patient.id}`)}
              >
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[hsl(var(--primary))]/10 text-[hsl(var(--primary))]">
                    {patient.name.charAt(0)}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{patient.name}</span>
                      <span className="text-sm text-[hsl(var(--muted-foreground))]">
                        ({patient.gender === "male" ? "남" : "여"}, {calculateAge(patient.birth)}세)
                      </span>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-[hsl(var(--muted-foreground))]">
                      <Badge variant="outline" className="text-xs">
                        {patient.diagnosis}
                      </Badge>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <div className="hidden text-right sm:block">
                    <div className="flex items-center gap-1 text-sm">
                      <Calendar className="h-3 w-3" />
                      <span>{formatDate(patient.lastTestDate)}</span>
                    </div>
                    <div className="flex items-center gap-1 text-xs text-[hsl(var(--muted-foreground))]">
                      <Activity className="h-3 w-3" />
                      <span>검사 {patient.testCount || 0}회</span>
                    </div>
                  </div>
                  <ChevronRight className="h-5 w-5 text-[hsl(var(--muted-foreground))] transition-transform duration-200 group-hover:translate-x-1 group-hover:text-[hsl(var(--primary))]" />
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
