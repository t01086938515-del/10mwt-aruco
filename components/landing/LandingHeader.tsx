"use client";

import { useAppSelector, useAppDispatch } from "@/store/hooks";
import { logout } from "@/store/slices/authSlice";
import { User, LogOut } from "lucide-react";
import { useRouter } from "next/navigation";

export function LandingHeader() {
  const router = useRouter();
  const dispatch = useAppDispatch();
  const { isAuthenticated, user } = useAppSelector((state) => state.auth);

  const handleLogout = () => {
    dispatch(logout());
    router.push("/");
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-white border-b border-gray-200">
      <div className="max-w-[1100px] mx-auto px-6 h-14 flex items-center justify-between">
        {/* Logo - 마크 + 로고명 */}
        <a href="/" className="flex items-center gap-1.5 hover:opacity-80 transition-opacity">
          <img src="/logo-mark.png" alt="WalkFlow" className="h-9 w-auto" />
          <span className="text-lg font-semibold text-gray-900 tracking-tight">WalkFlow</span>
        </a>

        {/* Navigation */}
        <nav className="hidden md:flex items-center gap-6">
          <a href="#features" className="text-sm text-gray-500 hover:text-gray-900">기능</a>
          <a href="#benefits" className="text-sm text-gray-500 hover:text-gray-900">장점</a>
          <a href="#how-it-works" className="text-sm text-gray-500 hover:text-gray-900">사용방법</a>
        </nav>

        {/* Auth Buttons - 일반 a 태그 사용 */}
        <div className="flex items-center gap-2">
          {isAuthenticated ? (
            <>
              {/* 로그인된 상태 */}
              <a
                href="/dashboard"
                className="flex items-center gap-2 text-sm text-gray-700 hover:text-gray-900 px-3 py-2"
              >
                <User size={16} />
                <span className="hidden sm:inline">{user?.name || "마이페이지"}</span>
              </a>
              <button
                onClick={handleLogout}
                className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-900 px-3 py-2"
              >
                <LogOut size={16} />
                <span className="hidden sm:inline">로그아웃</span>
              </button>
            </>
          ) : (
            <>
              {/* 로그인 안된 상태 - a 태그 사용 */}
              <a
                href="/login"
                className="text-sm text-gray-600 hover:text-gray-900 px-3 py-2"
              >
                로그인
              </a>
              <a
                href="/signup"
                className="px-4 py-2 bg-[#0066cc] text-white text-sm font-medium rounded-full hover:bg-[#0055aa] transition-colors"
              >
                회원가입
              </a>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
