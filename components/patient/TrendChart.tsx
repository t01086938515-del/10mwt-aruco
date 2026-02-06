"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface TrendData {
  date: string;
  comfortableSpeed: number;
  fastSpeed: number;
}

interface TrendChartProps {
  data: TrendData[];
  title?: string;
}

export function TrendChart({ data, title = "보행 속도 변화" }: TrendChartProps) {
  if (data.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex h-64 items-center justify-center text-[hsl(var(--muted-foreground))]">
            검사 데이터가 없습니다.
          </div>
        </CardContent>
      </Card>
    );
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return `${date.getMonth() + 1}/${date.getDate()}`;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis
                dataKey="date"
                tickFormatter={formatDate}
                tick={{ fontSize: 12 }}
              />
              <YAxis
                domain={[0, "auto"]}
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `${value.toFixed(1)}`}
                label={{
                  value: "m/s",
                  angle: -90,
                  position: "insideLeft",
                  fontSize: 12,
                }}
              />
              <Tooltip
                formatter={(value) => [`${Number(value).toFixed(2)} m/s`]}
                labelFormatter={(label) => `날짜: ${label}`}
              />
              <Legend />
              {/* Community ambulation threshold */}
              <ReferenceLine
                y={0.8}
                stroke="#f59e0b"
                strokeDasharray="5 5"
                label={{
                  value: "지역사회 보행 기준",
                  position: "insideTopRight",
                  fontSize: 10,
                  fill: "#f59e0b",
                }}
              />
              <Line
                type="monotone"
                dataKey="comfortableSpeed"
                name="편안한 속도"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={{ r: 4 }}
                activeDot={{ r: 6 }}
              />
              <Line
                type="monotone"
                dataKey="fastSpeed"
                name="빠른 속도"
                stroke="#22c55e"
                strokeWidth={2}
                dot={{ r: 4 }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
