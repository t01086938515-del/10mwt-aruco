"use client";

import { useRouter } from "next/navigation";
import { Card, CardContent } from "@/components/ui/card";
import { UserPlus, PlayCircle, BarChart3, FileText } from "lucide-react";

interface QuickActionProps {
  onAddPatient: () => void;
}

export function QuickActions({ onAddPatient }: QuickActionProps) {
  const router = useRouter();

  const actions = [
    {
      icon: UserPlus,
      label: "환자 등록",
      description: "새 환자 추가",
      color: "bg-blue-500",
      onClick: onAddPatient,
    },
    {
      icon: PlayCircle,
      label: "검사 시작",
      description: "10MWT 실시",
      color: "bg-green-500",
      onClick: () => router.push("/test/setup"),
    },
    {
      icon: BarChart3,
      label: "통계",
      description: "분석 보기",
      color: "bg-purple-500",
      onClick: () => router.push("/stats"),
    },
    {
      icon: FileText,
      label: "기록",
      description: "검사 이력",
      color: "bg-orange-500",
      onClick: () => router.push("/records"),
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      {actions.map((action, index) => (
        <Card
          key={action.label}
          hover
          className="cursor-pointer group animate-fade-in-up"
          style={{ animationDelay: `${0.3 + index * 0.1}s` }}
          onClick={action.onClick}
        >
          <CardContent className="flex flex-col items-center justify-center p-3 sm:p-4">
            <div className={`mb-2 rounded-full ${action.color} p-2.5 sm:p-3 flex items-center justify-center transition-all duration-300 group-hover:scale-110 group-hover:shadow-lg`}>
              <action.icon className="h-5 w-5 sm:h-6 sm:w-6 text-white flex-shrink-0" />
            </div>
            <span className="font-medium text-sm text-gray-900 dark:text-white text-center">{action.label}</span>
            <span className="text-xs text-gray-500 dark:text-gray-400 text-center">
              {action.description}
            </span>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
