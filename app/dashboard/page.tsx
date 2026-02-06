"use client";

import { useEffect, useState } from "react";
import { useAppSelector, useAppDispatch } from "@/store/hooks";
import { setPatientList, setSearchQuery } from "@/store/slices/patientSlice";
import { QuickActions } from "@/components/dashboard/QuickActions";
import { PatientTable } from "@/components/dashboard/PatientTable";
import { AddPatientModal } from "@/components/dashboard/AddPatientModal";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { mockPatients } from "@/lib/mockData";
import { Users, Activity, TrendingUp, Clock } from "lucide-react";

export default function DashboardPage() {
  const dispatch = useAppDispatch();
  const { user } = useAppSelector((state) => state.auth);
  const { patientList, searchQuery } = useAppSelector((state) => state.patient);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);

  useEffect(() => {
    // Load mock data on initial render
    if (patientList.length === 0) {
      dispatch(setPatientList(mockPatients));
    }
  }, [dispatch, patientList.length]);

  const stats = [
    {
      label: "전체 환자",
      value: patientList.length,
      icon: Users,
      color: "text-blue-500",
      bgColor: "bg-blue-500/10",
    },
    {
      label: "오늘 검사",
      value: 3,
      icon: Activity,
      color: "text-green-500",
      bgColor: "bg-green-500/10",
    },
    {
      label: "이번 주",
      value: 12,
      icon: TrendingUp,
      color: "text-purple-500",
      bgColor: "bg-purple-500/10",
    },
    {
      label: "평균 검사 시간",
      value: "8분",
      icon: Clock,
      color: "text-orange-500",
      bgColor: "bg-orange-500/10",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Greeting */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          안녕하세요, {user?.name || "치료사"}님
        </h2>
        <p className="text-gray-500 dark:text-gray-400">
          오늘도 좋은 하루 되세요.
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {stats.map((stat, index) => (
          <Card
            key={stat.label}
            hover
            className="animate-fade-in-up"
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            <CardContent className="flex items-center gap-3 p-3 sm:p-4">
              <div className={`rounded-lg ${stat.bgColor} p-2 flex items-center justify-center flex-shrink-0`}>
                <stat.icon className={`h-5 w-5 ${stat.color} flex-shrink-0`} />
              </div>
              <div className="min-w-0">
                <p className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white truncate">{stat.value}</p>
                <p className="text-xs text-gray-500 dark:text-gray-400 truncate">{stat.label}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Quick Actions */}
      <div>
        <h3 className="mb-3 text-lg font-semibold text-gray-900 dark:text-white">빠른 실행</h3>
        <QuickActions onAddPatient={() => setIsAddModalOpen(true)} />
      </div>

      {/* Patient List */}
      <PatientTable
        patients={patientList}
        searchQuery={searchQuery}
        onSearchChange={(query) => dispatch(setSearchQuery(query))}
      />

      {/* Add Patient Modal */}
      <AddPatientModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
      />
    </div>
  );
}
