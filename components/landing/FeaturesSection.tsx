"use client";

import { Timer, Camera, LineChart, FileText, Smartphone, Zap } from "lucide-react";

const features = [
  {
    icon: Timer,
    title: "스마트 타이머",
    description: "정밀한 시간 측정과 자동 기록으로 검사 효율을 높입니다.",
    color: "text-blue-500",
    bgColor: "bg-blue-50",
  },
  {
    icon: Camera,
    title: "AR 거리 측정",
    description: "카메라를 통해 10m 구간을 AR로 표시하고 정확한 거리를 보장합니다.",
    color: "text-green-500",
    bgColor: "bg-green-50",
  },
  {
    icon: LineChart,
    title: "자동 속도 계산",
    description: "보행 속도를 자동 계산하고 정상 범위와 비교합니다.",
    color: "text-purple-500",
    bgColor: "bg-purple-50",
  },
  {
    icon: FileText,
    title: "임상 해석 리포트",
    description: "낙상 위험도, 기능적 보행 수준 등 임상 해석을 자동 생성합니다.",
    color: "text-amber-500",
    bgColor: "bg-amber-50",
  },
  {
    icon: Smartphone,
    title: "모바일 최적화",
    description: "태블릿, 스마트폰에서 완벽하게 동작합니다.",
    color: "text-pink-500",
    bgColor: "bg-pink-50",
  },
  {
    icon: Zap,
    title: "빠른 기록 저장",
    description: "결과가 자동 저장되고 변화 추이를 한눈에 파악합니다.",
    color: "text-red-500",
    bgColor: "bg-red-50",
  },
];

export function FeaturesSection() {
  return (
    <section id="features" className="py-20 bg-gray-50">
      <div className="max-w-[1000px] mx-auto px-6">
        {/* Section Header */}
        <div className="text-center mb-12">
          <span className="text-sm font-medium text-[#0066cc] mb-2 block">
            FEATURES
          </span>
          <h2 className="text-2xl md:text-3xl font-bold mb-3">
            검사에 필요한 모든 기능
          </h2>
          <p className="text-gray-500 max-w-[400px] mx-auto text-sm">
            10미터 보행 검사를 위한 전문 도구들을 한 곳에서
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="group p-5 bg-white rounded-xl border border-gray-100 hover:border-gray-200 hover:shadow-md transition-all duration-200"
            >
              <div className={`w-10 h-10 ${feature.bgColor} rounded-lg flex items-center justify-center mb-3 group-hover:scale-105 transition-transform`}>
                <feature.icon className={`w-5 h-5 ${feature.color}`} />
              </div>
              <h3 className="font-semibold mb-1.5">{feature.title}</h3>
              <p className="text-sm text-gray-500 leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
