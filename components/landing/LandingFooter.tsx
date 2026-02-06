"use client";

import Link from "next/link";

const footerLinks = {
  product: {
    title: "제품",
    links: [
      { label: "기능", href: "#features" },
      { label: "장점", href: "#benefits" },
      { label: "사용방법", href: "#how-it-works" },
    ],
  },
  resources: {
    title: "리소스",
    links: [
      { label: "사용 가이드", href: "#" },
      { label: "FAQ", href: "#" },
    ],
  },
  company: {
    title: "회사",
    links: [
      { label: "소개", href: "#" },
      { label: "문의하기", href: "#" },
    ],
  },
};

export function LandingFooter() {
  return (
    <footer className="bg-gray-900 text-white py-12">
      <div className="max-w-[1000px] mx-auto px-6">
        {/* Main Footer Content */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-10">
          {/* Logo & Description */}
          <div className="col-span-2 md:col-span-1">
            <Link href="/" className="group flex items-center gap-2 mb-3 no-underline">
              <img
                src="/logo-mark.png"
                alt="WalkFlow"
                className="h-10 w-auto transition-all duration-200 group-hover:scale-110"
              />
            </Link>
            <p className="text-xs text-gray-400 leading-relaxed">
              전문 보행 검사 도구
            </p>
          </div>

          {/* Links */}
          {Object.entries(footerLinks).map(([key, section]) => (
            <div key={key}>
              <h4 className="font-medium text-sm mb-3">{section.title}</h4>
              <ul className="space-y-2">
                {section.links.map((link) => (
                  <li key={link.label}>
                    <a
                      href={link.href}
                      className="text-xs text-gray-400 hover:text-white transition-colors"
                    >
                      {link.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Divider */}
        <div className="border-t border-gray-800 pt-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
            <p className="text-xs text-gray-500">
              © 2025 WalkFlow. All rights reserved.
            </p>
            <div className="flex items-center gap-4">
              <a href="#" className="text-xs text-gray-500 hover:text-white transition-colors">
                이용약관
              </a>
              <a href="#" className="text-xs text-gray-500 hover:text-white transition-colors">
                개인정보처리방침
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
