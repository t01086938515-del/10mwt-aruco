"use client";

import * as React from "react";

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "secondary" | "destructive" | "outline" | "success" | "warning";
}

function Badge({ className = "", variant = "default", ...props }: BadgeProps) {
  const variants = {
    default:
      "bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] shadow hover:bg-[hsl(var(--primary))]/80",
    secondary:
      "bg-[hsl(var(--secondary))] text-[hsl(var(--secondary-foreground))] hover:bg-[hsl(var(--secondary))]/80",
    destructive:
      "bg-[hsl(var(--destructive))] text-[hsl(var(--destructive-foreground))] shadow hover:bg-[hsl(var(--destructive))]/80",
    outline: "text-[hsl(var(--foreground))] border border-[hsl(var(--border))]",
    success:
      "bg-[hsl(var(--success))] text-[hsl(var(--success-foreground))] shadow hover:bg-[hsl(var(--success))]/80",
    warning:
      "bg-[hsl(var(--warning))] text-[hsl(var(--warning-foreground))] shadow hover:bg-[hsl(var(--warning))]/80",
  };

  return (
    <div
      className={`inline-flex items-center rounded-md px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-[hsl(var(--ring))] focus:ring-offset-2 ${variants[variant]} ${className}`}
      {...props}
    />
  );
}

export { Badge };
