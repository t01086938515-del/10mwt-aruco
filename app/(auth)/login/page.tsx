"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAppSelector } from "@/store/hooks";
import { LoginForm } from "@/components/auth/LoginForm";
import { ArrowLeft } from "lucide-react";

export default function LoginPage() {
  const { isAuthenticated, loading } = useAppSelector((state) => state.auth);
  const router = useRouter();

  useEffect(() => {
    if (isAuthenticated && !loading) {
      router.push("/dashboard");
    }
  }, [isAuthenticated, loading, router]);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-[#0066cc] border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gray-50 p-4">
      {/* Back to Home */}
      <Link
        href="/"
        className="absolute top-6 left-6 flex items-center gap-2 text-sm text-gray-500 hover:text-gray-900 transition-colors no-underline"
      >
        <ArrowLeft size={16} />
        홈으로
      </Link>

      {/* Logo - Clickable (호버: 아이콘 커지고 그림자) */}
      <Link href="/" className="group mb-8 flex items-center gap-3 no-underline">
        <img
          src="/logo.png"
          alt="WalkFlow"
          className="h-14 w-auto transition-all duration-200 group-hover:scale-110"
        />
      </Link>

      <LoginForm />

      <p className="mt-8 text-center text-xs text-gray-400">
        전문 보행 분석 도구
      </p>
    </div>
  );
}
