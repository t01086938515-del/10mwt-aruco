"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  MapPin,
  Clock,
  Footprints,
  TrafficCone,
  ShoppingBag,
  Home,
  Building2,
  TreePine,
  Bus,
  Calculator,
} from "lucide-react";

interface RealLifeCalculatorProps {
  comfortableSpeed: number; // m/s
  fastSpeed: number; // m/s
  patientName: string;
}

// 미리 정의된 일상 거리들
const presetDistances = [
  { name: "횡단보도 (일반)", distance: 12, icon: TrafficCone, description: "신호 시간 약 20-30초" },
  { name: "횡단보도 (대로)", distance: 25, icon: TrafficCone, description: "신호 시간 약 40-60초" },
  { name: "집 → 엘리베이터", distance: 30, icon: Home, description: "아파트 복도" },
  { name: "버스정류장", distance: 100, icon: Bus, description: "가까운 정류장" },
  { name: "편의점", distance: 150, icon: ShoppingBag, description: "도보 접근" },
  { name: "동네 슈퍼마켓", distance: 300, icon: ShoppingBag, description: "일상 장보기" },
  { name: "공원 한 바퀴", distance: 400, icon: TreePine, description: "가벼운 산책" },
  { name: "대형마트", distance: 500, icon: Building2, description: "주간 장보기" },
];

export default function RealLifeCalculator({
  comfortableSpeed,
  fastSpeed,
  patientName,
}: RealLifeCalculatorProps) {
  const [customDistance, setCustomDistance] = useState<string>("");

  // 시간 계산 (초 단위)
  const calculateTime = (distance: number, speed: number): number => {
    if (speed <= 0) return 0;
    return distance / speed;
  };

  // 시간 포맷 (분:초)
  const formatTime = (seconds: number): string => {
    if (seconds === 0) return "-";
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    if (mins === 0) return `${secs}초`;
    return `${mins}분 ${secs}초`;
  };

  // 신호등 통과 가능 여부 판단
  const canCrossCrosswalk = (distance: number, signalTime: number): boolean => {
    const timeNeeded = calculateTime(distance, comfortableSpeed);
    return timeNeeded <= signalTime;
  };

  // 일반인 대비 비교 (일반 보행 속도 약 1.2-1.4 m/s)
  const normalWalkingSpeed = 1.3;
  const comparedToNormal = ((comfortableSpeed / normalWalkingSpeed) * 100).toFixed(0);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <Calculator className="h-5 w-5" />
          실생활 환산 계산기
        </CardTitle>
        <p className="text-sm text-[hsl(var(--muted-foreground))]">
          {patientName}님의 보행 속도로 일상 거리를 얼마나 걸어야 하는지 확인하세요
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* 속도 요약 */}
        <div className="grid grid-cols-3 gap-4 p-4 bg-[hsl(var(--accent))] rounded-lg">
          <div className="text-center">
            <div className="text-2xl font-bold text-[hsl(var(--primary))]">
              {comfortableSpeed.toFixed(2)}
            </div>
            <div className="text-xs text-[hsl(var(--muted-foreground))]">편안한 속도 (m/s)</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-500">
              {fastSpeed.toFixed(2)}
            </div>
            <div className="text-xs text-[hsl(var(--muted-foreground))]">빠른 속도 (m/s)</div>
          </div>
          <div className="text-center">
            <div className={`text-2xl font-bold ${
              parseInt(comparedToNormal) >= 80 ? 'text-green-500' :
              parseInt(comparedToNormal) >= 60 ? 'text-yellow-500' : 'text-red-500'
            }`}>
              {comparedToNormal}%
            </div>
            <div className="text-xs text-[hsl(var(--muted-foreground))]">일반인 대비</div>
          </div>
        </div>

        {/* 횡단보도 안전 분석 */}
        <div className="p-4 border border-[hsl(var(--border))] rounded-lg">
          <div className="flex items-center gap-2 mb-3">
            <TrafficCone className="h-5 w-5 text-yellow-500" />
            <span className="font-medium">횡단보도 안전 분석</span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className={`p-3 rounded-lg ${
              canCrossCrosswalk(12, 25) ? 'bg-green-500/10 border border-green-500' : 'bg-red-500/10 border border-red-500'
            }`}>
              <div className="text-sm font-medium">일반 횡단보도 (12m)</div>
              <div className="text-xs text-[hsl(var(--muted-foreground))]">신호 25초 기준</div>
              <div className="mt-1">
                <Badge variant={canCrossCrosswalk(12, 25) ? "success" : "destructive"}>
                  {canCrossCrosswalk(12, 25) ? "통과 가능" : "시간 부족"}
                </Badge>
                <span className="ml-2 text-sm">{formatTime(calculateTime(12, comfortableSpeed))}</span>
              </div>
            </div>
            <div className={`p-3 rounded-lg ${
              canCrossCrosswalk(25, 45) ? 'bg-green-500/10 border border-green-500' : 'bg-red-500/10 border border-red-500'
            }`}>
              <div className="text-sm font-medium">대로 횡단보도 (25m)</div>
              <div className="text-xs text-[hsl(var(--muted-foreground))]">신호 45초 기준</div>
              <div className="mt-1">
                <Badge variant={canCrossCrosswalk(25, 45) ? "success" : "destructive"}>
                  {canCrossCrosswalk(25, 45) ? "통과 가능" : "시간 부족"}
                </Badge>
                <span className="ml-2 text-sm">{formatTime(calculateTime(25, comfortableSpeed))}</span>
              </div>
            </div>
          </div>
        </div>

        {/* 일상 거리 계산 */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <MapPin className="h-5 w-5 text-[hsl(var(--primary))]" />
            <span className="font-medium">일상 거리 소요 시간</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {presetDistances.map((preset) => {
              const Icon = preset.icon;
              const comfortTime = calculateTime(preset.distance, comfortableSpeed);
              const fastTime = calculateTime(preset.distance, fastSpeed);

              return (
                <div
                  key={preset.name}
                  className="p-3 border border-[hsl(var(--border))] rounded-lg hover:bg-[hsl(var(--accent))] transition-colors"
                >
                  <div className="flex items-start gap-3">
                    <div className="p-2 rounded-lg bg-[hsl(var(--primary))]/10">
                      <Icon className="h-4 w-4 text-[hsl(var(--primary))]" />
                    </div>
                    <div className="flex-1">
                      <div className="font-medium text-sm">{preset.name}</div>
                      <div className="text-xs text-[hsl(var(--muted-foreground))]">
                        {preset.distance}m · {preset.description}
                      </div>
                      <div className="mt-2 flex items-center gap-4 text-sm">
                        <div>
                          <span className="text-[hsl(var(--muted-foreground))]">편안: </span>
                          <span className="font-medium">{formatTime(comfortTime)}</span>
                        </div>
                        <div>
                          <span className="text-[hsl(var(--muted-foreground))]">빠른: </span>
                          <span className="font-medium text-orange-500">{formatTime(fastTime)}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* 사용자 지정 거리 */}
        <div className="p-4 bg-[hsl(var(--accent))] rounded-lg">
          <div className="flex items-center gap-2 mb-3">
            <Footprints className="h-5 w-5" />
            <span className="font-medium">직접 거리 입력</span>
          </div>
          <div className="flex gap-3">
            <div className="relative flex-1">
              <Input
                type="number"
                placeholder="거리 입력"
                value={customDistance}
                onChange={(e) => setCustomDistance(e.target.value)}
                className="pr-8"
              />
              <span className="absolute right-3 top-1/2 -translate-y-1/2 text-sm text-[hsl(var(--muted-foreground))]">
                m
              </span>
            </div>
            {customDistance && parseFloat(customDistance) > 0 && (
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-1">
                  <Clock className="h-4 w-4 text-[hsl(var(--muted-foreground))]" />
                  <span>편안: </span>
                  <span className="font-bold">
                    {formatTime(calculateTime(parseFloat(customDistance), comfortableSpeed))}
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <span>빠른: </span>
                  <span className="font-bold text-orange-500">
                    {formatTime(calculateTime(parseFloat(customDistance), fastSpeed))}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 보행 능력 평가 */}
        <div className="p-4 border-2 border-[hsl(var(--primary))]/30 rounded-lg bg-[hsl(var(--primary))]/5">
          <div className="font-medium mb-2">보행 능력 평가</div>
          <div className="space-y-2 text-sm">
            {comfortableSpeed >= 0.8 ? (
              <div className="flex items-center gap-2 text-green-500">
                <Badge variant="success">지역사회 보행 가능</Badge>
                <span>독립적인 외출 및 일상 활동 가능</span>
              </div>
            ) : comfortableSpeed >= 0.6 ? (
              <div className="flex items-center gap-2 text-yellow-500">
                <Badge variant="warning">제한적 지역사회 보행</Badge>
                <span>짧은 거리 외출 가능, 긴 거리는 보조 필요</span>
              </div>
            ) : comfortableSpeed >= 0.4 ? (
              <div className="flex items-center gap-2 text-orange-500">
                <Badge className="bg-orange-500">가정 내 보행</Badge>
                <span>실내 이동 가능, 외출 시 보조 필요</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-red-500">
                <Badge variant="destructive">보행 보조 필요</Badge>
                <span>이동 시 보조기구 또는 도움 필요</span>
              </div>
            )}

            <div className="mt-3 pt-3 border-t border-[hsl(var(--border))]">
              <div className="text-[hsl(var(--muted-foreground))]">
                <strong>참고:</strong> 횡단보도를 안전하게 건너려면 최소 0.6 m/s 이상의 보행 속도가 필요합니다.
                현재 속도는 일반인(1.3 m/s) 대비 <strong>{comparedToNormal}%</strong> 수준입니다.
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
