"use client";

import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAppSelector, useAppDispatch } from "@/store/hooks";
import { logout } from "@/store/slices/authSlice";
import { Button } from "@/components/ui/button";
import { LogOut, Settings, User, Moon, Sun } from "lucide-react";
import { useState, useEffect } from "react";

export function Header() {
  const { user } = useAppSelector((state) => state.auth);
  const dispatch = useAppDispatch();
  const router = useRouter();
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const isDarkMode = document.documentElement.classList.contains("dark");
    setIsDark(isDarkMode);
  }, []);

  const toggleDarkMode = () => {
    document.documentElement.classList.toggle("dark");
    setIsDark(!isDark);
  };

  const handleLogout = () => {
    dispatch(logout());
    router.push("/login");
  };

  return (
    <header className="sticky top-0 z-40 border-b border-gray-200 dark:border-gray-700 bg-white/95 dark:bg-gray-900/95 backdrop-blur">
      <div className="flex h-14 items-center justify-between px-4 md:px-6">
        {/* Logo - 마크만 사용 */}
        <Link href="/" className="group flex items-center gap-2.5 no-underline">
          <img
            src="/logo-mark.png"
            alt="WalkFlow"
            className="h-9 w-auto transition-all duration-200 group-hover:scale-110"
          />
        </Link>

        {/* User Menu */}
        <div className="flex items-center gap-1.5">
          <Button variant="ghost" size="icon" onClick={toggleDarkMode} className="h-9 w-9">
            {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>

          <Button variant="ghost" size="icon" onClick={() => router.push("/settings")} className="h-9 w-9">
            <Settings className="h-4 w-4" />
          </Button>

          <div className="ml-1 hidden items-center gap-2 rounded-lg bg-gray-100 dark:bg-gray-800 px-3 py-1.5 sm:flex">
            <User className="h-4 w-4 text-gray-500 dark:text-gray-400" />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-200">{user?.name || "사용자"}</span>
          </div>

          <Button variant="ghost" size="icon" onClick={handleLogout} className="h-9 w-9">
            <LogOut className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </header>
  );
}
