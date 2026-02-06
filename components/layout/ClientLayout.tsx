"use client";

import { usePathname } from "next/navigation";
import Sidebar from "@/components/dashboard/Sidebar";

interface ClientLayoutProps {
  children: React.ReactNode;
}

// 사이드바를 숨길 경로 (로그인, 회원가입)
const hideSidebarPaths = ["/login", "/signup"];

export default function ClientLayout({ children }: ClientLayoutProps) {
  const pathname = usePathname();
  const shouldHideSidebar = hideSidebarPaths.includes(pathname);

  return (
    <>
      {!shouldHideSidebar && <Sidebar />}
      {children}
    </>
  );
}
