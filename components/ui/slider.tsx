"use client";

import * as React from "react";

export interface SliderProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "type"> {
  label?: string;
  showValue?: boolean;
  unit?: string;
}

const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  ({ className = "", label, showValue = true, unit = "", min = 0, max = 100, value, ...props }, ref) => {
    const percentage = ((Number(value) - Number(min)) / (Number(max) - Number(min))) * 100;

    return (
      <div className={`w-full ${className}`}>
        {(label || showValue) && (
          <div className="mb-4 flex items-center justify-between">
            {label && <span className="text-sm font-medium">{label}</span>}
            {showValue && (
              <span className="text-base font-semibold text-[hsl(var(--primary))]">
                {value}
                {unit}
              </span>
            )}
          </div>
        )}
        <div className="relative py-1">
          <input
            ref={ref}
            type="range"
            min={min}
            max={max}
            value={value}
            className="h-3 w-full cursor-pointer appearance-none rounded-lg bg-[hsl(var(--secondary))] accent-[hsl(var(--primary))]"
            style={{
              background: `linear-gradient(to right, hsl(var(--primary)) 0%, hsl(var(--primary)) ${percentage}%, hsl(var(--secondary)) ${percentage}%, hsl(var(--secondary)) 100%)`,
            }}
            {...props}
          />
        </div>
      </div>
    );
  }
);
Slider.displayName = "Slider";

export { Slider };
