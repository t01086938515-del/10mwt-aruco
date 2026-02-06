"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { generateClinicalInterpretation } from "@/lib/calculations";
import { AlertTriangle, Users, TrendingUp, Lightbulb, ShieldCheck, ShieldAlert } from "lucide-react";

interface ClinicalInterpretationProps {
  speed: number;
  age: number;
  diagnosis?: string;
}

export function ClinicalInterpretation({ speed, age, diagnosis }: ClinicalInterpretationProps) {
  const interpretation = generateClinicalInterpretation(speed, age, diagnosis);

  const fallRiskIcons = {
    high: ShieldAlert,
    moderate: AlertTriangle,
    low: ShieldCheck,
  };

  const FallRiskIcon = fallRiskIcons[interpretation.fallRisk.level];

  const fallRiskColors = {
    high: "text-red-500 bg-red-500/10",
    moderate: "text-yellow-500 bg-yellow-500/10",
    low: "text-green-500 bg-green-500/10",
  };

  const communityColors = {
    household: "text-red-500 bg-red-500/10",
    limited: "text-yellow-500 bg-yellow-500/10",
    full: "text-green-500 bg-green-500/10",
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">임상 해석</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Risk Assessment */}
        <div className="grid gap-4 sm:grid-cols-2">
          {/* Fall Risk */}
          <div
            className={`rounded-lg p-4 ${
              fallRiskColors[interpretation.fallRisk.level]
            }`}
          >
            <div className="flex items-center gap-3">
              <FallRiskIcon className="h-8 w-8" />
              <div>
                <p className="font-medium">{interpretation.fallRisk.description}</p>
                <p className="text-sm opacity-80">낙상 위험도</p>
              </div>
            </div>
          </div>

          {/* Community Ambulation */}
          <div
            className={`rounded-lg p-4 ${
              communityColors[interpretation.communityAmbulation.level]
            }`}
          >
            <div className="flex items-center gap-3">
              <Users className="h-8 w-8" />
              <div>
                <p className="font-medium">
                  {interpretation.communityAmbulation.description}
                </p>
                <p className="text-sm opacity-80">지역사회 보행 수준</p>
              </div>
            </div>
          </div>
        </div>

        {/* Normative Comparison */}
        {interpretation.percentile !== null && (
          <div className="rounded-lg border border-[hsl(var(--border))] p-4">
            <div className="flex items-center gap-3">
              <TrendingUp className="h-6 w-6 text-[hsl(var(--primary))]" />
              <div>
                <p className="font-medium">{interpretation.comparison}</p>
                <p className="text-sm text-[hsl(var(--muted-foreground))]">
                  동일 연령대 백분위: {interpretation.percentile}%
                </p>
              </div>
            </div>

            {/* Percentile Bar */}
            <div className="mt-3">
              <div className="relative h-2 w-full rounded-full bg-[hsl(var(--secondary))]">
                <div
                  className="absolute h-full rounded-full bg-[hsl(var(--primary))]"
                  style={{ width: `${interpretation.percentile}%` }}
                />
                <div
                  className="absolute -top-1 h-4 w-1 rounded-full bg-[hsl(var(--primary))]"
                  style={{ left: `${interpretation.percentile}%` }}
                />
              </div>
              <div className="mt-1 flex justify-between text-xs text-[hsl(var(--muted-foreground))]">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>
          </div>
        )}

        {/* Recommendations */}
        <div>
          <div className="mb-3 flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-[hsl(var(--primary))]" />
            <h4 className="font-medium">권장 사항</h4>
          </div>
          <ul className="space-y-2">
            {interpretation.recommendations.map((rec, index) => (
              <li
                key={index}
                className="flex items-start gap-2 rounded-md bg-[hsl(var(--secondary))]/50 p-3 text-sm"
              >
                <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-[hsl(var(--primary))]" />
                {rec}
              </li>
            ))}
          </ul>
        </div>

        {/* Clinical Thresholds Reference */}
        <div className="rounded-lg bg-[hsl(var(--muted))] p-4">
          <p className="mb-2 text-sm font-medium">임상 기준 참고</p>
          <div className="grid grid-cols-2 gap-2 text-xs text-[hsl(var(--muted-foreground))]">
            <div>
              <span className="inline-block h-2 w-2 rounded-full bg-red-500" /> {"<"} 0.6 m/s: 높은
              낙상 위험
            </div>
            <div>
              <span className="inline-block h-2 w-2 rounded-full bg-yellow-500" /> 0.6-0.8 m/s:
              중등도 위험
            </div>
            <div>
              <span className="inline-block h-2 w-2 rounded-full bg-green-500" /> {">"} 0.8 m/s:
              지역사회 보행
            </div>
            <div>
              <span className="inline-block h-2 w-2 rounded-full bg-blue-500" /> {">"} 1.2 m/s:
              완전 독립 보행
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
