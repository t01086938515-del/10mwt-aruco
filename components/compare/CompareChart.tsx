"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ScatterChart,
  Scatter,
  ZAxis,
  Cell,
  ReferenceLine,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart3, Radar as RadarIcon, ScatterChart as ScatterIcon } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";

interface RecordData {
  date: string;
  patientName: string;
  comfortableSpeed: number;
  fastSpeed: number;
  comfortableCadence: number;
  fastCadence: number;
}

interface CompareChartProps {
  record1: RecordData;
  record2: RecordData;
}

export default function CompareChart({ record1, record2 }: CompareChartProps) {
  const [chartType, setChartType] = useState<"bar" | "radar" | "scatter">("bar");

  // 바 차트 데이터
  const barData = [
    {
      name: "편안한 속도",
      기준: record1.comfortableSpeed,
      비교: record2.comfortableSpeed,
      unit: "m/s",
    },
    {
      name: "빠른 속도",
      기준: record1.fastSpeed,
      비교: record2.fastSpeed,
      unit: "m/s",
    },
    {
      name: "편안한 케이던스",
      기준: record1.comfortableCadence,
      비교: record2.comfortableCadence,
      unit: "steps/min",
    },
    {
      name: "빠른 케이던스",
      기준: record1.fastCadence,
      비교: record2.fastCadence,
      unit: "steps/min",
    },
  ];

  // 레이더 차트 데이터 (정규화)
  const normalizeValue = (value: number, max: number) => (value / max) * 100;

  const radarData = [
    {
      subject: "편안한 속도",
      기준: normalizeValue(record1.comfortableSpeed, 1.5),
      비교: normalizeValue(record2.comfortableSpeed, 1.5),
      fullMark: 100,
    },
    {
      subject: "빠른 속도",
      기준: normalizeValue(record1.fastSpeed, 2.0),
      비교: normalizeValue(record2.fastSpeed, 2.0),
      fullMark: 100,
    },
    {
      subject: "편안한 케이던스",
      기준: normalizeValue(record1.comfortableCadence, 140),
      비교: normalizeValue(record2.comfortableCadence, 140),
      fullMark: 100,
    },
    {
      subject: "빠른 케이던스",
      기준: normalizeValue(record1.fastCadence, 160),
      비교: normalizeValue(record2.fastCadence, 160),
      fullMark: 100,
    },
  ];

  // XY 산점도 데이터 (속도 vs 케이던스)
  const scatterData1 = [
    { x: record1.comfortableCadence, y: record1.comfortableSpeed, name: "기준-편안한", type: "기준" },
    { x: record1.fastCadence, y: record1.fastSpeed, name: "기준-빠른", type: "기준" },
  ];

  const scatterData2 = [
    { x: record2.comfortableCadence, y: record2.comfortableSpeed, name: "비교-편안한", type: "비교" },
    { x: record2.fastCadence, y: record2.fastSpeed, name: "비교-빠른", type: "비교" },
  ];

  // 개선도 데이터
  const improvementData = [
    {
      name: "편안한 속도",
      개선율: ((record2.comfortableSpeed - record1.comfortableSpeed) / record1.comfortableSpeed * 100),
    },
    {
      name: "빠른 속도",
      개선율: ((record2.fastSpeed - record1.fastSpeed) / record1.fastSpeed * 100),
    },
    {
      name: "편안한 케이던스",
      개선율: ((record2.comfortableCadence - record1.comfortableCadence) / record1.comfortableCadence * 100),
    },
    {
      name: "빠른 케이던스",
      개선율: ((record2.fastCadence - record1.fastCadence) / record1.fastCadence * 100),
    },
  ];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded-lg p-3 shadow-lg">
          <p className="font-medium mb-2">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(2) : entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const ScatterTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded-lg p-3 shadow-lg">
          <p className="font-medium mb-1">{data.name}</p>
          <p className="text-sm">케이던스: {data.x} steps/min</p>
          <p className="text-sm">속도: {data.y.toFixed(2)} m/s</p>
        </div>
      );
    }
    return null;
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between flex-wrap gap-2">
          <CardTitle className="text-lg flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            평가 지표 그래프
          </CardTitle>
          <div className="flex gap-2">
            <Button
              variant={chartType === "bar" ? "default" : "outline"}
              size="sm"
              onClick={() => setChartType("bar")}
            >
              <BarChart3 className="h-4 w-4 mr-1" />
              막대
            </Button>
            <Button
              variant={chartType === "radar" ? "default" : "outline"}
              size="sm"
              onClick={() => setChartType("radar")}
            >
              <RadarIcon className="h-4 w-4 mr-1" />
              레이더
            </Button>
            <Button
              variant={chartType === "scatter" ? "default" : "outline"}
              size="sm"
              onClick={() => setChartType("scatter")}
            >
              <ScatterIcon className="h-4 w-4 mr-1" />
              XY
            </Button>
          </div>
        </div>
        <div className="flex gap-4 text-sm mt-2">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-blue-500" />
            <span>기준: {record1.date}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-green-500" />
            <span>비교: {record2.date}</span>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* 메인 비교 차트 */}
          <div className="h-[300px]">
            {chartType === "bar" && (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
                  />
                  <YAxis
                    tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Bar dataKey="기준" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="비교" fill="#22c55e" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}

            {chartType === "radar" && (
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <PolarGrid stroke="hsl(var(--border))" />
                  <PolarAngleAxis
                    dataKey="subject"
                    tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                  />
                  <PolarRadiusAxis
                    angle={30}
                    domain={[0, 100]}
                    tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
                  />
                  <Radar
                    name="기준"
                    dataKey="기준"
                    stroke="#3b82f6"
                    fill="#3b82f6"
                    fillOpacity={0.3}
                  />
                  <Radar
                    name="비교"
                    dataKey="비교"
                    stroke="#22c55e"
                    fill="#22c55e"
                    fillOpacity={0.3}
                  />
                  <Legend />
                  <Tooltip content={<CustomTooltip />} />
                </RadarChart>
              </ResponsiveContainer>
            )}

            {chartType === "scatter" && (
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis
                    type="number"
                    dataKey="x"
                    name="케이던스"
                    unit=" steps/min"
                    domain={['dataMin - 10', 'dataMax + 10']}
                    tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
                    label={{ value: '케이던스 (steps/min)', position: 'bottom', offset: 0, fill: "hsl(var(--muted-foreground))" }}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    name="속도"
                    unit=" m/s"
                    domain={['dataMin - 0.1', 'dataMax + 0.1']}
                    tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
                    label={{ value: '속도 (m/s)', angle: -90, position: 'insideLeft', fill: "hsl(var(--muted-foreground))" }}
                  />
                  <ZAxis range={[100, 100]} />
                  {/* 지역사회 보행 기준선 */}
                  <ReferenceLine y={0.8} stroke="#f59e0b" strokeDasharray="5 5" label={{ value: '지역사회 보행 (0.8)', fill: '#f59e0b', fontSize: 10 }} />
                  <Tooltip content={<ScatterTooltip />} />
                  <Legend />
                  <Scatter name="기준" data={scatterData1} fill="#3b82f6">
                    {scatterData1.map((entry, index) => (
                      <Cell key={`cell-1-${index}`} fill="#3b82f6" />
                    ))}
                  </Scatter>
                  <Scatter name="비교" data={scatterData2} fill="#22c55e">
                    {scatterData2.map((entry, index) => (
                      <Cell key={`cell-2-${index}`} fill="#22c55e" />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* XY 그래프 설명 */}
          {chartType === "scatter" && (
            <div className="p-3 bg-[hsl(var(--accent))] rounded-lg text-sm">
              <p className="font-medium mb-1">XY 그래프 해석</p>
              <p className="text-[hsl(var(--muted-foreground))]">
                X축: 케이던스(분당 걸음 수), Y축: 보행 속도.
                오른쪽 위로 이동할수록 보행 능력이 향상됨을 의미합니다.
                노란색 점선은 지역사회 보행 기준(0.8 m/s)입니다.
              </p>
            </div>
          )}

          {/* 개선율 차트 */}
          <div>
            <h4 className="text-sm font-medium mb-3 text-[hsl(var(--muted-foreground))]">
              개선율 (%)
            </h4>
            <div className="h-[150px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={improvementData}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis
                    type="number"
                    tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
                    domain={['dataMin - 5', 'dataMax + 5']}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
                  />
                  <Tooltip
                    formatter={(value: number) => [`${value.toFixed(1)}%`, '개선율']}
                  />
                  <ReferenceLine x={0} stroke="hsl(var(--border))" />
                  <Bar dataKey="개선율" radius={[0, 4, 4, 0]}>
                    {improvementData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.개선율 >= 0 ? "#22c55e" : "#ef4444"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* 수치 요약 테이블 */}
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[hsl(var(--border))]">
                  <th className="text-left py-2 px-3">지표</th>
                  <th className="text-center py-2 px-3 text-blue-500">기준 ({record1.date})</th>
                  <th className="text-center py-2 px-3 text-green-500">비교 ({record2.date})</th>
                  <th className="text-center py-2 px-3">변화</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-[hsl(var(--border))]">
                  <td className="py-2 px-3">편안한 속도 (m/s)</td>
                  <td className="text-center py-2 px-3">{record1.comfortableSpeed.toFixed(2)}</td>
                  <td className="text-center py-2 px-3">{record2.comfortableSpeed.toFixed(2)}</td>
                  <td className={`text-center py-2 px-3 font-medium ${
                    record2.comfortableSpeed > record1.comfortableSpeed ? 'text-green-500' :
                    record2.comfortableSpeed < record1.comfortableSpeed ? 'text-red-500' : ''
                  }`}>
                    {record2.comfortableSpeed > record1.comfortableSpeed ? '+' : ''}
                    {((record2.comfortableSpeed - record1.comfortableSpeed) / record1.comfortableSpeed * 100).toFixed(1)}%
                  </td>
                </tr>
                <tr className="border-b border-[hsl(var(--border))]">
                  <td className="py-2 px-3">빠른 속도 (m/s)</td>
                  <td className="text-center py-2 px-3">{record1.fastSpeed.toFixed(2)}</td>
                  <td className="text-center py-2 px-3">{record2.fastSpeed.toFixed(2)}</td>
                  <td className={`text-center py-2 px-3 font-medium ${
                    record2.fastSpeed > record1.fastSpeed ? 'text-green-500' :
                    record2.fastSpeed < record1.fastSpeed ? 'text-red-500' : ''
                  }`}>
                    {record2.fastSpeed > record1.fastSpeed ? '+' : ''}
                    {((record2.fastSpeed - record1.fastSpeed) / record1.fastSpeed * 100).toFixed(1)}%
                  </td>
                </tr>
                <tr className="border-b border-[hsl(var(--border))]">
                  <td className="py-2 px-3">편안한 케이던스 (steps/min)</td>
                  <td className="text-center py-2 px-3">{record1.comfortableCadence}</td>
                  <td className="text-center py-2 px-3">{record2.comfortableCadence}</td>
                  <td className={`text-center py-2 px-3 font-medium ${
                    record2.comfortableCadence > record1.comfortableCadence ? 'text-green-500' :
                    record2.comfortableCadence < record1.comfortableCadence ? 'text-red-500' : ''
                  }`}>
                    {record2.comfortableCadence > record1.comfortableCadence ? '+' : ''}
                    {((record2.comfortableCadence - record1.comfortableCadence) / record1.comfortableCadence * 100).toFixed(1)}%
                  </td>
                </tr>
                <tr>
                  <td className="py-2 px-3">빠른 케이던스 (steps/min)</td>
                  <td className="text-center py-2 px-3">{record1.fastCadence}</td>
                  <td className="text-center py-2 px-3">{record2.fastCadence}</td>
                  <td className={`text-center py-2 px-3 font-medium ${
                    record2.fastCadence > record1.fastCadence ? 'text-green-500' :
                    record2.fastCadence < record1.fastCadence ? 'text-red-500' : ''
                  }`}>
                    {record2.fastCadence > record1.fastCadence ? '+' : ''}
                    {((record2.fastCadence - record1.fastCadence) / record1.fastCadence * 100).toFixed(1)}%
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
