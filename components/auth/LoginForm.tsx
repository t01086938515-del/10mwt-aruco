"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAppDispatch } from "@/store/hooks";
import { setUser, setError } from "@/store/slices/authSlice";
import { Button } from "@/components/ui/button";
import { LogIn, User, Lock, AlertCircle } from "lucide-react";

export function LoginForm() {
  const [employeeId, setEmployeeId] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const dispatch = useAppDispatch();
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setErrorMessage("");

    try {
      if (employeeId && password) {
        await new Promise((resolve) => setTimeout(resolve, 1000));

        // Mock user for demo
        dispatch(
          setUser({
            uid: "user-" + Date.now(),
            email: employeeId + "@walktest.pro",
            name: employeeId,
            hospital: "서울재활병원",
            employeeId: employeeId,
          })
        );

        router.push("/dashboard");
      } else {
        throw new Error("사번과 비밀번호를 입력해주세요.");
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "로그인에 실패했습니다.";
      setErrorMessage(message);
      dispatch(setError(message));
    } finally {
      setIsLoading(false);
    }
  };

  const handleDemoLogin = async () => {
    setIsLoading(true);
    setErrorMessage("");

    try {
      await new Promise((resolve) => setTimeout(resolve, 500));

      dispatch(
        setUser({
          uid: "demo-user-1",
          email: "demo@walktest.pro",
          name: "김치료사",
          hospital: "서울재활병원",
          employeeId: "PT001",
        })
      );

      router.push("/dashboard");
    } catch {
      setErrorMessage("데모 로그인에 실패했습니다.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full max-w-md rounded-2xl bg-white p-8 shadow-sm border border-gray-200">
      <h2 className="text-2xl font-bold text-center mb-2">로그인</h2>
      <p className="text-sm text-gray-500 text-center mb-6">
        사번과 비밀번호를 입력하세요
      </p>

      <form onSubmit={handleSubmit} className="space-y-4">
        {errorMessage && (
          <div className="flex items-center gap-2 rounded-lg bg-red-50 p-3 text-sm text-red-600">
            <AlertCircle className="h-4 w-4" />
            {errorMessage}
          </div>
        )}

        {/* Employee ID */}
        <div>
          <label htmlFor="employeeId" className="block text-sm font-medium text-gray-700 mb-1">
            사번
          </label>
          <div className="relative">
            <User className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
            <input
              id="employeeId"
              type="text"
              placeholder="PT001"
              value={employeeId}
              onChange={(e) => setEmployeeId(e.target.value)}
              className="w-full pl-10 pr-4 py-3 rounded-lg border border-gray-300 focus:border-[#0066cc] focus:ring-2 focus:ring-[#0066cc]/20 outline-none transition-all"
              disabled={isLoading}
            />
          </div>
        </div>

        {/* Password */}
        <div>
          <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">
            비밀번호
          </label>
          <div className="relative">
            <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
            <input
              id="password"
              type="password"
              placeholder="비밀번호 입력"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full pl-10 pr-4 py-3 rounded-lg border border-gray-300 focus:border-[#0066cc] focus:ring-2 focus:ring-[#0066cc]/20 outline-none transition-all"
              disabled={isLoading}
            />
          </div>
        </div>

        {/* Login Button */}
        <button
          type="submit"
          disabled={isLoading}
          className="w-full py-3 bg-[#0066cc] text-white font-semibold rounded-lg hover:bg-[#0055aa] transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
              로그인 중...
            </>
          ) : (
            <>
              <LogIn className="h-4 w-4" />
              로그인
            </>
          )}
        </button>

        {/* Divider */}
        <div className="relative my-2">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-gray-200" />
          </div>
          <div className="relative flex justify-center text-xs">
            <span className="bg-white px-2 text-gray-400">또는</span>
          </div>
        </div>

        {/* Demo Login */}
        <button
          type="button"
          onClick={handleDemoLogin}
          disabled={isLoading}
          className="w-full py-3 border border-gray-300 text-gray-700 font-medium rounded-lg hover:bg-gray-50 transition-colors disabled:opacity-50"
        >
          데모 계정으로 시작하기
        </button>

        {/* Signup Link */}
        <p className="mt-4 text-center text-sm text-gray-500">
          계정이 없으신가요?{" "}
          <button
            type="button"
            onClick={() => router.push("/signup")}
            className="text-[#0066cc] font-medium hover:underline"
          >
            회원가입
          </button>
        </p>
      </form>
    </div>
  );
}
