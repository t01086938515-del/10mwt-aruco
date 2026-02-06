"use client";

import Link from "next/link";
import { ArrowRight, Play, ChevronDown } from "lucide-react";

export function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20">
      {/* Gradient Orbs Background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-[500px] h-[500px] rounded-full bg-[hsl(202,100%,50%)] opacity-15 blur-[120px]" />
        <div className="absolute top-1/3 right-1/4 w-[400px] h-[400px] rounded-full bg-[hsl(120,100%,34%)] opacity-15 blur-[120px]" />
        <div className="absolute bottom-1/4 left-1/3 w-[300px] h-[300px] rounded-full bg-[hsl(260,100%,70%)] opacity-10 blur-[100px]" />
      </div>

      <div className="relative z-10 max-w-[900px] mx-auto px-6 pb-20 text-center">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-gray-100 rounded-full mb-5">
          <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          <span className="text-sm text-gray-600">
            간편하고 정확한 보행 분석
          </span>
        </div>

        {/* Headline */}
        <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight mb-4 leading-tight">
          정확하고 빠른
          <br />
          <span className="text-[#0066cc]">10미터 보행 검사</span>
        </h1>

        {/* Subheadline */}
        <p className="text-base sm:text-lg md:text-xl text-gray-500 max-w-[540px] mx-auto mb-8 leading-relaxed">
          AR 기술과 스마트 타이머로 보행 속도를 정밀하게 측정하고,
          자동 분석으로 임상 해석까지 한 번에 제공합니다.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-3 mb-12">
          <Link
            href="/dashboard"
            className="group flex items-center gap-2 px-7 py-3.5 bg-[#0066cc] text-white font-semibold rounded-full hover:bg-[#0055aa] hover:shadow-lg hover:shadow-blue-500/25 transition-all"
          >
            검사 시작하기
            <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
          </Link>
          <button className="flex items-center gap-2 px-7 py-3.5 bg-white border border-gray-200 text-gray-700 font-medium rounded-full hover:bg-gray-50 hover:border-gray-300 transition-all">
            <Play size={16} className="text-[#0066cc]" />
            데모 영상 보기
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-6 max-w-[420px] mx-auto">
          <div>
            <p className="text-2xl sm:text-3xl font-bold text-gray-900">500+</p>
            <p className="text-xs sm:text-sm text-gray-500">치료사 사용</p>
          </div>
          <div>
            <p className="text-2xl sm:text-3xl font-bold text-gray-900">10,000+</p>
            <p className="text-xs sm:text-sm text-gray-500">검사 완료</p>
          </div>
          <div>
            <p className="text-2xl sm:text-3xl font-bold text-gray-900">99.2%</p>
            <p className="text-xs sm:text-sm text-gray-500">정확도</p>
          </div>
        </div>
      </div>

      {/* Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1 text-gray-400">
        <span className="text-[10px] tracking-widest uppercase">Scroll</span>
        <ChevronDown size={20} className="animate-bounce" />
      </div>
    </section>
  );
}
