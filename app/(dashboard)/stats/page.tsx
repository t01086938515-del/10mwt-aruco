"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { useAppSelector } from "@/store/hooks";
import { ArrowLeft, TrendingUp, Users, Activity, Calendar } from "lucide-react";
import { useRouter } from "next/navigation";

export default function StatsPage() {
  const router = useRouter();
  const { patientList } = useAppSelector((state) => state.patient);

  // Mock statistics data
  const weeklyTests = [
    { day: "월", tests: 5 },
    { day: "화", tests: 8 },
    { day: "수", tests: 6 },
    { day: "목", tests: 7 },
    { day: "금", tests: 9 },
    { day: "토", tests: 3 },
    { day: "일", tests: 2 },
  ];

  const diagnosisDistribution = [
    { name: "뇌졸중", value: 35, color: "#3b82f6" },
    { name: "파킨슨병", value: 20, color: "#22c55e" },
    { name: "척수손상", value: 15, color: "#f59e0b" },
    { name: "고관절 수술", value: 18, color: "#8b5cf6" },
    { name: "기타", value: 12, color: "#6b7280" },
  ];

  const outcomeStats = [
    { category: "지역사회 보행", count: 45, percentage: 45 },
    { category: "제한적 보행", count: 35, percentage: 35 },
    { category: "가정 내 보행", count: 20, percentage: 20 },
  ];

  const stats = [
    {
      label: "이번 달 검사",
      value: 127,
      change: "+12%",
      icon: Activity,
      color: "text-blue-500",
    },
    {
      label: "활성 환자",
      value: patientList.length,
      change: "+5",
      icon: Users,
      color: "text-green-500",
    },
    {
      label: "평균 속도 향상",
      value: "0.15 m/s",
      change: "+8%",
      icon: TrendingUp,
      color: "text-purple-500",
    },
    {
      label: "이번 주 검사",
      value: 40,
      change: "+3",
      icon: Calendar,
      color: "text-orange-500",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">통계</h1>
          <p className="text-[hsl(var(--muted-foreground))]">
            검사 현황 및 분석 데이터
          </p>
        </div>
        <Button variant="outline" onClick={() => router.push("/")}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          돌아가기
        </Button>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {stats.map((stat) => (
          <Card key={stat.label}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <stat.icon className={`h-5 w-5 ${stat.color}`} />
                <span className="text-xs text-green-500">{stat.change}</span>
              </div>
              <p className="mt-2 text-2xl font-bold">{stat.value}</p>
              <p className="text-xs text-[hsl(var(--muted-foreground))]">
                {stat.label}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Weekly Tests Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">주간 검사 현황</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={weeklyTests}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="day" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar
                  dataKey="tests"
                  fill="hsl(var(--primary))"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Diagnosis Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">진단별 분포</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={diagnosisDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={2}
                    dataKey="value"
                    label={({ name, percent }) =>
                      `${name} ${((percent ?? 0) * 100).toFixed(0)}%`
                    }
                    labelLine={false}
                  >
                    {diagnosisDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Outcome Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">보행 수준 분포</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {outcomeStats.map((stat) => (
                <div key={stat.category}>
                  <div className="mb-1 flex items-center justify-between text-sm">
                    <span>{stat.category}</span>
                    <span className="font-medium">{stat.count}명</span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-[hsl(var(--secondary))]">
                    <div
                      className={`h-full rounded-full ${
                        stat.category === "지역사회 보행"
                          ? "bg-green-500"
                          : stat.category === "제한적 보행"
                          ? "bg-yellow-500"
                          : "bg-red-500"
                      }`}
                      style={{ width: `${stat.percentage}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
