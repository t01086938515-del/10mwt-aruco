"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";

export function CTASection() {
  return (
    <section className="py-16">
      <div className="max-w-[900px] mx-auto px-6">
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-[#0066cc] to-[#004499] p-10 md:p-12 text-center">
          {/* Background Decoration */}
          <div className="absolute top-0 right-0 w-[300px] h-[300px] bg-white/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none" />
          <div className="absolute bottom-0 left-0 w-[200px] h-[200px] bg-white/10 rounded-full blur-3xl translate-y-1/2 -translate-x-1/2 pointer-events-none" />

          <div className="relative z-10">
            <h2 className="text-2xl md:text-3xl font-bold text-white mb-3">
              지금 바로 시작하세요
            </h2>
            <p className="text-white/80 max-w-[400px] mx-auto mb-6 text-sm leading-relaxed">
              별도의 설치나 회원가입 없이 바로 사용할 수 있습니다.
              전문적인 10미터 보행 검사를 경험해보세요.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
              <Link
                href="/dashboard"
                className="group flex items-center gap-2 px-6 py-3 bg-white text-[#0066cc] font-semibold rounded-full hover:shadow-lg transition-shadow"
              >
                검사 시작하기
                <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link
                href="/login"
                className="px-6 py-3 text-white font-medium rounded-full border border-white/30 hover:bg-white/10 transition-colors text-sm"
              >
                로그인
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
