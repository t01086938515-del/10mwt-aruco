"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAppSelector } from "@/store/hooks";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import {
  ArrowLeft,
  User,
  Bell,
  Palette,
  Database,
  Shield,
  HelpCircle,
  Moon,
  Sun,
  Save,
} from "lucide-react";

export default function SettingsPage() {
  const router = useRouter();
  const { user } = useAppSelector((state) => state.auth);
  const [isDark, setIsDark] = useState(
    typeof document !== "undefined" && document.documentElement.classList.contains("dark")
  );

  const [settings, setSettings] = useState({
    defaultTrials: 3,
    defaultRestDuration: 60,
    defaultDistance: 10,
    autoSave: true,
    notifications: true,
    soundEffects: true,
    language: "ko",
  });

  const toggleDarkMode = () => {
    document.documentElement.classList.toggle("dark");
    setIsDark(!isDark);
  };

  const handleSave = () => {
    // Save settings to local storage or backend
    localStorage.setItem("walktest-settings", JSON.stringify(settings));
    alert("설정이 저장되었습니다.");
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">설정</h1>
          <p className="text-[hsl(var(--muted-foreground))]">
            앱 환경설정을 관리합니다
          </p>
        </div>
        <Button variant="outline" onClick={() => router.push("/")}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          돌아가기
        </Button>
      </div>

      {/* Profile Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <User className="h-5 w-5" />
            프로필
          </CardTitle>
          <CardDescription>계정 정보를 관리합니다</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">이름</label>
              <Input value={user?.name || ""} readOnly />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">이메일</label>
              <Input value={user?.email || ""} readOnly />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">소속 병원</label>
              <Input value={user?.hospital || ""} readOnly />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">사원번호</label>
              <Input value={user?.employeeId || ""} readOnly />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Test Default Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Database className="h-5 w-5" />
            검사 기본값
          </CardTitle>
          <CardDescription>검사 시 기본으로 적용되는 설정입니다</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <Slider
            label="기본 시행 횟수"
            value={settings.defaultTrials}
            onChange={(e) =>
              setSettings({ ...settings, defaultTrials: Number(e.target.value) })
            }
            min={1}
            max={5}
            unit="회"
          />
          <Slider
            label="기본 휴식 시간"
            value={settings.defaultRestDuration}
            onChange={(e) =>
              setSettings({
                ...settings,
                defaultRestDuration: Number(e.target.value),
              })
            }
            min={30}
            max={120}
            step={10}
            unit="초"
          />
          <div className="space-y-2">
            <label className="text-sm font-medium">기본 측정 거리</label>
            <Select
              value={settings.defaultDistance.toString()}
              onChange={(e) =>
                setSettings({
                  ...settings,
                  defaultDistance: Number(e.target.value),
                })
              }
              options={[
                { value: "6", label: "6m" },
                { value: "10", label: "10m (권장)" },
                { value: "14", label: "14m" },
              ]}
            />
          </div>
        </CardContent>
      </Card>

      {/* Appearance */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Palette className="h-5 w-5" />
            화면 설정
          </CardTitle>
          <CardDescription>앱의 외관을 설정합니다</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">다크 모드</p>
              <p className="text-sm text-[hsl(var(--muted-foreground))]">
                어두운 테마를 사용합니다
              </p>
            </div>
            <Button variant="outline" size="icon" onClick={toggleDarkMode}>
              {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            </Button>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">언어</label>
            <Select
              value={settings.language}
              onChange={(e) =>
                setSettings({ ...settings, language: e.target.value })
              }
              options={[
                { value: "ko", label: "한국어" },
                { value: "en", label: "English" },
              ]}
            />
          </div>
        </CardContent>
      </Card>

      {/* Notifications */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Bell className="h-5 w-5" />
            알림
          </CardTitle>
          <CardDescription>알림 및 소리 설정</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">알림 허용</p>
              <p className="text-sm text-[hsl(var(--muted-foreground))]">
                검사 완료 및 휴식 알림
              </p>
            </div>
            <input
              type="checkbox"
              checked={settings.notifications}
              onChange={(e) =>
                setSettings({ ...settings, notifications: e.target.checked })
              }
              className="h-5 w-5"
            />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">소리 효과</p>
              <p className="text-sm text-[hsl(var(--muted-foreground))]">
                타이머 및 알림 소리
              </p>
            </div>
            <input
              type="checkbox"
              checked={settings.soundEffects}
              onChange={(e) =>
                setSettings({ ...settings, soundEffects: e.target.checked })
              }
              className="h-5 w-5"
            />
          </div>
        </CardContent>
      </Card>

      {/* Data & Privacy */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Shield className="h-5 w-5" />
            데이터 및 개인정보
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">자동 저장</p>
              <p className="text-sm text-[hsl(var(--muted-foreground))]">
                검사 결과 자동 저장
              </p>
            </div>
            <input
              type="checkbox"
              checked={settings.autoSave}
              onChange={(e) =>
                setSettings({ ...settings, autoSave: e.target.checked })
              }
              className="h-5 w-5"
            />
          </div>
          <Button variant="outline" className="w-full">
            데이터 내보내기
          </Button>
          <Button variant="destructive" className="w-full">
            모든 데이터 삭제
          </Button>
        </CardContent>
      </Card>

      {/* Help */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <HelpCircle className="h-5 w-5" />
            도움말
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <Button variant="outline" className="w-full justify-start">
            사용 가이드
          </Button>
          <Button variant="outline" className="w-full justify-start">
            자주 묻는 질문
          </Button>
          <Button variant="outline" className="w-full justify-start">
            문의하기
          </Button>
          <div className="pt-4 text-center text-sm text-[hsl(var(--muted-foreground))]">
            WalkTest Pro v1.0.0
          </div>
        </CardContent>
      </Card>

      {/* Save Button */}
      <Button className="w-full" size="lg" onClick={handleSave}>
        <Save className="mr-2 h-5 w-5" />
        설정 저장
      </Button>
    </div>
  );
}
