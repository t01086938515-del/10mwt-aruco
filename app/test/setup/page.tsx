"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAppSelector, useAppDispatch } from "@/store/hooks";
import { setPatientList } from "@/store/slices/patientSlice";
import { TestConfigForm } from "@/components/test/TestConfigForm";
import { Button } from "@/components/ui/button";
import { mockPatients } from "@/lib/mockData";
import { ArrowLeft, Activity } from "lucide-react";

export default function TestSetupPage() {
  const router = useRouter();
  const dispatch = useAppDispatch();
  const { isAuthenticated } = useAppSelector((state) => state.auth);
  const { patientList } = useAppSelector((state) => state.patient);

  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/login");
    }
  }, [isAuthenticated, router]);

  useEffect(() => {
    if (patientList.length === 0) {
      dispatch(setPatientList(mockPatients));
    }
  }, [dispatch, patientList.length]);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-gray-200 bg-gray-50/95 backdrop-blur">
        <div className="flex h-16 items-center justify-between px-4">
          <Button variant="ghost" onClick={() => router.push("/")}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            돌아가기
          </Button>
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-[#0066cc]" />
            <span className="font-semibold">10MWT 설정</span>
          </div>
          <div className="w-20" /> {/* Spacer for centering */}
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-2xl p-4 md:p-8">
        <div className="mb-8">
          <h1 className="text-2xl font-bold">검사 설정</h1>
          <p className="mt-2 text-gray-500">
            10m 보행 검사를 시작하기 전에 설정을 확인하세요.
          </p>
        </div>

        <TestConfigForm />
      </main>
    </div>
  );
}
