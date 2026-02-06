"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAppDispatch } from "@/store/hooks";
import { setUser } from "@/store/slices/authSlice";
import { ArrowLeft, Eye, EyeOff } from "lucide-react";

export default function SignupPage() {
  const router = useRouter();
  const dispatch = useAppDispatch();
  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    employeeId: "",
    password: "",
    passwordConfirm: "",
    hospital: "",
  });
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setError("");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    // Validation
    if (!formData.name || !formData.employeeId || !formData.password) {
      setError("필수 항목을 모두 입력해주세요.");
      return;
    }

    if (formData.password !== formData.passwordConfirm) {
      setError("비밀번호가 일치하지 않습니다.");
      return;
    }

    if (formData.password.length < 4) {
      setError("비밀번호는 4자 이상이어야 합니다.");
      return;
    }

    setIsLoading(true);

    // Simulate signup
    setTimeout(() => {
      dispatch(
        setUser({
          uid: "user-" + Date.now(),
          email: formData.employeeId + "@walktest.pro",
          name: formData.name,
          hospital: formData.hospital,
          employeeId: formData.employeeId,
        })
      );
      router.push("/dashboard");
    }, 1000);
  };

  // 데모 계정으로 빠른 가입
  const handleDemoSignup = () => {
    setIsLoading(true);
    setTimeout(() => {
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
    }, 500);
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gray-50 p-4">
      {/* Back Button */}
      <button
        onClick={() => router.push("/")}
        className="absolute top-6 left-6 flex items-center gap-2 text-sm text-gray-500 hover:text-gray-900 transition-colors"
      >
        <ArrowLeft size={16} />
        홈으로
      </button>

      {/* Logo */}
      <button
        onClick={() => router.push("/")}
        className="mb-6 flex items-center gap-3"
      >
        <img
          src="/logo.png"
          alt="WalkFlow"
          className="h-14 w-auto transition-all duration-200 hover:scale-110"
        />
      </button>

      {/* Signup Form */}
      <div className="w-full max-w-md rounded-2xl bg-white p-8 shadow-sm border border-gray-200">
        <h2 className="text-2xl font-bold text-center mb-6">회원가입</h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              이름 <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              name="name"
              value={formData.name}
              onChange={handleChange}
              placeholder="홍길동"
              className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:border-[#0066cc] focus:ring-2 focus:ring-[#0066cc]/20 outline-none transition-all"
            />
          </div>

          {/* Employee ID */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              사번 <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              name="employeeId"
              value={formData.employeeId}
              onChange={handleChange}
              placeholder="PT001"
              className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:border-[#0066cc] focus:ring-2 focus:ring-[#0066cc]/20 outline-none transition-all"
            />
          </div>

          {/* Password */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              비밀번호 <span className="text-red-500">*</span>
            </label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                name="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="4자 이상"
                className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:border-[#0066cc] focus:ring-2 focus:ring-[#0066cc]/20 outline-none transition-all pr-12"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
              >
                {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </div>
          </div>

          {/* Password Confirm */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              비밀번호 확인 <span className="text-red-500">*</span>
            </label>
            <input
              type={showPassword ? "text" : "password"}
              name="passwordConfirm"
              value={formData.passwordConfirm}
              onChange={handleChange}
              placeholder="비밀번호 재입력"
              className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:border-[#0066cc] focus:ring-2 focus:ring-[#0066cc]/20 outline-none transition-all"
            />
          </div>

          {/* Hospital (Optional) */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              소속 병원 <span className="text-gray-400 text-xs">(선택)</span>
            </label>
            <input
              type="text"
              name="hospital"
              value={formData.hospital}
              onChange={handleChange}
              placeholder="OO병원 재활의학과"
              className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:border-[#0066cc] focus:ring-2 focus:ring-[#0066cc]/20 outline-none transition-all"
            />
          </div>

          {/* Error Message */}
          {error && (
            <p className="text-red-500 text-sm text-center">{error}</p>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 bg-[#0066cc] text-white font-semibold rounded-lg hover:bg-[#0055aa] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? "가입 중..." : "회원가입"}
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

          {/* Demo Signup */}
          <button
            type="button"
            onClick={handleDemoSignup}
            disabled={isLoading}
            className="w-full py-3 border border-gray-300 text-gray-700 font-medium rounded-lg hover:bg-gray-50 transition-colors disabled:opacity-50"
          >
            데모 계정으로 시작하기
          </button>
        </form>

        {/* Login Link */}
        <p className="mt-6 text-center text-sm text-gray-500">
          이미 계정이 있으신가요?{" "}
          <button
            onClick={() => router.push("/login")}
            className="text-[#0066cc] font-medium hover:underline"
          >
            로그인
          </button>
        </p>
      </div>
    </div>
  );
}
