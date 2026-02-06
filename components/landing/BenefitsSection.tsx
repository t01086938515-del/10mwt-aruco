"use client";

import { Check, Clock, Shield, TrendingUp } from "lucide-react";

const benefits = [
  {
    icon: Clock,
    title: "검사 시간 50% 단축",
    description: "수동 측정 대비 검사 준비와 기록 시간을 대폭 줄여줍니다.",
    stats: "평균 8분 → 4분",
  },
  {
    icon: Shield,
    title: "표준화된 검사 프로토콜",
    description: "일관된 검사 방법으로 측정 오차를 최소화합니다.",
    stats: "오차율 0.8% 이하",
  },
  {
    icon: TrendingUp,
    title: "객관적인 치료 효과 확인",
    description: "보행 속도 변화를 그래프로 시각화하여 효과를 보여줍니다.",
    stats: "추이 분석 자동화",
  },
];

const checkList = [
  "별도 장비 없이 스마트폰만으로 검사",
  "환자 데이터 안전하게 로컬 저장",
  "PDF 리포트 내보내기 지원",
  "다중 시행 검사 및 평균 계산",
  "실시간 음성 안내 기능",
  "오프라인에서도 사용 가능",
];

export function BenefitsSection() {
  return (
    <section id="benefits" className="py-20">
      <div className="max-w-[1000px] mx-auto px-6">
        {/* Section Header */}
        <div className="text-center mb-12">
          <span className="text-sm font-medium text-[#0066cc] mb-2 block">
            WHY WALKTEST PRO
          </span>
          <h2 className="text-2xl md:text-3xl font-bold mb-3">
            왜 WalkTest Pro인가요?
          </h2>
          <p className="text-gray-500 max-w-[400px] mx-auto text-sm">
            전문적인 보행 검사를 누구나 쉽고 정확하게
          </p>
        </div>

        {/* Benefits Cards */}
        <div className="grid md:grid-cols-3 gap-5 mb-12">
          {benefits.map((benefit) => (
            <div
              key={benefit.title}
              className="p-6 bg-white rounded-xl border border-gray-100"
            >
              <div className="w-11 h-11 bg-blue-50 rounded-xl flex items-center justify-center mb-4">
                <benefit.icon className="w-5 h-5 text-[#0066cc]" />
              </div>
              <h3 className="text-lg font-semibold mb-2">{benefit.title}</h3>
              <p className="text-sm text-gray-500 mb-3 leading-relaxed">
                {benefit.description}
              </p>
              <span className="inline-block px-2.5 py-1 bg-blue-50 rounded-full text-xs font-medium text-[#0066cc]">
                {benefit.stats}
              </span>
            </div>
          ))}
        </div>

        {/* Checklist */}
        <div className="bg-gray-50 rounded-2xl p-6 md:p-8">
          <div className="grid sm:grid-cols-2 gap-3">
            {checkList.map((item) => (
              <div key={item} className="flex items-center gap-2.5">
                <div className="w-5 h-5 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <Check className="w-3 h-3 text-green-600" />
                </div>
                <span className="text-sm text-gray-700">{item}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
