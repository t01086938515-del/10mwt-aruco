"use client";

import { Patient } from "@/store/slices/patientSlice";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { User, Calendar, Ruler, Weight, Activity } from "lucide-react";

interface PatientProfileProps {
  patient: Patient;
}

export function PatientProfile({ patient }: PatientProfileProps) {
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

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("ko-KR", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  const infoItems = [
    { label: "생년월일", value: formatDate(patient.birth) },
    { label: "성별", value: patient.gender === "male" ? "남성" : "여성" },
    { label: "신장", value: `${patient.height} cm` },
    { label: "체중", value: `${patient.weight} kg` },
    { label: "하지 길이", value: patient.legLength ? `${patient.legLength} cm` : "-" },
    {
      label: "마비측",
      value:
        patient.affectedSide === "left"
          ? "좌측"
          : patient.affectedSide === "right"
          ? "우측"
          : patient.affectedSide === "both"
          ? "양측"
          : "없음",
    },
    { label: "보조기구", value: patient.assistiveDevice || "없음" },
    { label: "AFO 착용", value: patient.afoUse ? "예" : "아니오" },
  ];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-4">
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-[hsl(var(--primary))]/10 text-2xl font-bold text-[hsl(var(--primary))]">
            {patient.name.charAt(0)}
          </div>
          <div>
            <CardTitle className="flex items-center gap-2">
              {patient.name}
              <span className="text-base font-normal text-[hsl(var(--muted-foreground))]">
                ({calculateAge(patient.birth)}세)
              </span>
            </CardTitle>
            <Badge variant="outline" className="mt-1">
              {patient.diagnosis}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          {infoItems.map((item) => (
            <div key={item.label} className="space-y-1">
              <p className="text-xs text-[hsl(var(--muted-foreground))]">{item.label}</p>
              <p className="font-medium">{item.value}</p>
            </div>
          ))}
        </div>

        <div className="mt-4 flex items-center gap-4 border-t border-[hsl(var(--border))] pt-4">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-[hsl(var(--muted-foreground))]" />
            <span className="text-sm">
              총 검사: <strong>{patient.testCount || 0}회</strong>
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Calendar className="h-4 w-4 text-[hsl(var(--muted-foreground))]" />
            <span className="text-sm">
              최근 검사:{" "}
              <strong>
                {patient.lastTestDate
                  ? new Date(patient.lastTestDate).toLocaleDateString("ko-KR")
                  : "-"}
              </strong>
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
