"use client";

const steps = [
  {
    number: "1",
    title: "환자 선택",
    description: "환자 목록에서 선택하거나 새 환자를 등록합니다.",
  },
  {
    number: "2",
    title: "검사 설정",
    description: "보조도구, 시행 횟수, 휴식 시간을 설정합니다.",
  },
  {
    number: "3",
    title: "AR 거리 설정",
    description: "카메라로 바닥을 비추면 10m 구간이 AR로 표시됩니다.",
  },
  {
    number: "4",
    title: "검사 시작",
    description: "타이머가 작동하고 끝선 통과 시 자동 정지됩니다.",
  },
  {
    number: "5",
    title: "결과 확인",
    description: "보행 속도와 임상 해석이 즉시 표시되고 자동 저장됩니다.",
  },
];

export function HowItWorksSection() {
  return (
    <section id="how-it-works" className="py-20 bg-gray-50">
      <div className="max-w-[800px] mx-auto px-6">
        {/* Section Header */}
        <div className="text-center mb-12">
          <span className="text-sm font-medium text-[#0066cc] mb-2 block">
            HOW IT WORKS
          </span>
          <h2 className="text-2xl md:text-3xl font-bold mb-3">
            5단계로 간단하게
          </h2>
          <p className="text-gray-500 text-sm">
            복잡한 설정 없이 바로 검사를 시작하세요
          </p>
        </div>

        {/* Steps - Simple Vertical List */}
        <div className="space-y-4">
          {steps.map((step, index) => (
            <div
              key={step.number}
              className="flex items-start gap-4 p-4 bg-white rounded-xl border border-gray-100"
            >
              <div className="w-8 h-8 bg-[#0066cc] rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-sm font-bold text-white">{step.number}</span>
              </div>
              <div className="pt-0.5">
                <h3 className="font-semibold mb-1">{step.title}</h3>
                <p className="text-sm text-gray-500">{step.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
