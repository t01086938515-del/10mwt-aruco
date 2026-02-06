"use client";

import { useState, useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import {
  Home,
  FileText,
  BarChart3,
  Settings,
  ChevronRight,
  ChevronDown,
  Menu,
  Footprints,
  ClipboardList,
} from "lucide-react";

interface NavItem {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  href?: string;
  description?: string;
  children?: NavItem[];
}

const navItems: NavItem[] = [
  {
    icon: Home,
    label: "홈",
    href: "/",
    description: "메인 화면",
  },
  {
    icon: ClipboardList,
    label: "검사",
    description: "보행 및 기능 검사",
    children: [
      {
        icon: Footprints,
        label: "10m Walk Test",
        href: "/dashboard",
        description: "10미터 보행 검사",
      },
      // 추후 검사 추가 예정
    ],
  },
  {
    icon: FileText,
    label: "검사 기록",
    href: "/records",
    description: "모든 검사 이력",
  },
  {
    icon: BarChart3,
    label: "통계",
    href: "/stats",
    description: "데이터 분석",
  },
  {
    icon: Settings,
    label: "설정",
    href: "/settings",
    description: "앱 설정",
  },
];

export default function Sidebar() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [expandedMenus, setExpandedMenus] = useState<string[]>(["검사"]);
  const [isMounted, setIsMounted] = useState(false);
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    setIsMounted(true);
  }, []);

  const toggleSubmenu = (label: string) => {
    setExpandedMenus((prev) =>
      prev.includes(label)
        ? prev.filter((item) => item !== label)
        : [...prev, label]
    );
  };

  const isActive = (href?: string) => {
    if (!href) return false;
    if (href === "/") return pathname === "/";
    return pathname === href || pathname.startsWith(href + "/");
  };

  const isParentActive = (item: NavItem) => {
    if (item.children) {
      return item.children.some((child) => isActive(child.href));
    }
    return false;
  };

  if (!isMounted) return null;

  const renderNavItem = (item: NavItem, depth = 0) => {
    const Icon = item.icon;
    const hasChildren = item.children && item.children.length > 0;
    const isMenuExpanded = expandedMenus.includes(item.label);
    const active = isActive(item.href) || isParentActive(item);

    return (
      <div key={item.label}>
        <button
          onClick={() => {
            if (hasChildren) {
              toggleSubmenu(item.label);
            } else if (item.href) {
              router.push(item.href);
              setIsExpanded(false);
            }
          }}
          className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-all duration-200 group ${
            depth > 0 ? "ml-4 pl-4" : ""
          } ${
            active && !hasChildren
              ? "bg-[#0066cc] text-white shadow-md"
              : hasChildren && isParentActive(item)
              ? "bg-blue-50 dark:bg-blue-900/20 text-[#0066cc] dark:text-blue-400"
              : "hover:bg-gray-50 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300"
          }`}
        >
          <Icon
            className={`h-5 w-5 flex-shrink-0 ${
              active && !hasChildren
                ? "text-white"
                : hasChildren && isParentActive(item)
                ? "text-[#0066cc] dark:text-blue-400"
                : "text-gray-400 group-hover:text-[#0066cc]"
            }`}
          />
          <div className="flex-1 text-left min-w-0">
            <div className="font-medium text-sm">{item.label}</div>
            {item.description && (
              <div
                className={`text-xs mt-0.5 truncate ${
                  active && !hasChildren ? "text-white/70" : "text-gray-400"
                }`}
              >
                {item.description}
              </div>
            )}
          </div>
          {hasChildren ? (
            <ChevronDown
              className={`h-4 w-4 flex-shrink-0 transition-transform ${
                isMenuExpanded ? "rotate-180" : ""
              } ${
                isParentActive(item)
                  ? "text-[#0066cc]"
                  : "text-gray-300 dark:text-gray-600"
              }`}
            />
          ) : (
            <ChevronRight
              className={`h-4 w-4 flex-shrink-0 transition-all ${
                active
                  ? "text-white opacity-100"
                  : "text-gray-300 dark:text-gray-600 opacity-0 group-hover:opacity-100 group-hover:translate-x-1"
              }`}
            />
          )}
        </button>

        {/* 서브메뉴 */}
        {hasChildren && isMenuExpanded && (
          <div className="mt-1 space-y-1">
            {item.children!.map((child) => renderNavItem(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <>
      {/* 호버 트리거 영역 */}
      <div
        className="fixed left-0 top-0 w-3 h-full z-[100] bg-transparent"
        onMouseEnter={() => setIsExpanded(true)}
      />

      {/* 메뉴 힌트 인디케이터 */}
      <div
        className={`fixed left-0 top-1/2 -translate-y-1/2 z-[90] transition-all duration-300 ${
          isExpanded ? "opacity-0 pointer-events-none" : "opacity-100"
        }`}
        onMouseEnter={() => setIsExpanded(true)}
      >
        <div className="flex items-center">
          <div className="w-1.5 h-24 bg-gradient-to-b from-transparent via-[#0066cc] to-transparent rounded-r-full" />
          <div className="ml-1 bg-[#0066cc] text-white p-1.5 rounded-r-lg shadow-lg">
            <Menu className="h-4 w-4" />
          </div>
        </div>
      </div>

      {/* 사이드바 패널 */}
      <div
        className={`fixed left-0 top-0 h-full bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 shadow-2xl z-[100] transition-transform duration-300 ease-out ${
          isExpanded ? "translate-x-0" : "-translate-x-full"
        }`}
        style={{ width: "280px" }}
        onMouseLeave={() => setIsExpanded(false)}
      >
        <div className="h-full flex flex-col">
          {/* 헤더 - 마크만 사용 */}
          <div className="p-5 border-b border-gray-100 dark:border-gray-800">
            <Link
              href="/"
              className="group flex items-center justify-center no-underline"
              onClick={() => setIsExpanded(false)}
            >
              <img
                src="/logo-mark.png"
                alt="WalkFlow"
                className="h-16 w-auto transition-all duration-200 group-hover:scale-105"
              />
            </Link>
          </div>

          {/* 네비게이션 */}
          <nav className="flex-1 p-2 space-y-1 overflow-y-auto">
            {navItems.map((item) => renderNavItem(item))}
          </nav>

          {/* 푸터 */}
          <div className="p-3 border-t border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50">
            <div className="text-xs text-gray-400 text-center">
              <p className="font-medium">WalkFlow v1.0.0</p>
            </div>
          </div>
        </div>
      </div>

      {/* 배경 오버레이 */}
      <div
        className={`fixed inset-0 bg-black/40 z-[95] transition-opacity duration-300 ${
          isExpanded ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
        onClick={() => setIsExpanded(false)}
      />
    </>
  );
}
