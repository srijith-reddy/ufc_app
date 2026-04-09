import { cn } from "@/lib/utils";
import { formatPercent } from "@/lib/api";

export function ProbabilityBar({
  leftLabel,
  rightLabel,
  leftValue,
  rightValue,
  className,
}: {
  leftLabel: string;
  rightLabel: string;
  leftValue: number;
  rightValue: number;
  className?: string;
}) {
  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex items-center justify-between text-xs uppercase tracking-[0.18em] text-white/45">
        <span>{leftLabel}</span>
        <span>{rightLabel}</span>
      </div>
      <div className="relative h-3 overflow-hidden rounded-full bg-white/8">
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-[#f6c56d] via-[#ff9159] to-[#ff6b35]"
          style={{ width: `${Math.max(0, Math.min(100, leftValue * 100))}%` }}
        />
        <div
          className="absolute inset-y-0 right-0 rounded-full bg-gradient-to-r from-[#63d4ff] to-[#2a9dff]"
          style={{ width: `${Math.max(0, Math.min(100, rightValue * 100))}%` }}
        />
      </div>
      <div className="flex items-center justify-between text-sm text-white/88">
        <span>{formatPercent(leftValue)}</span>
        <span>{formatPercent(rightValue)}</span>
      </div>
    </div>
  );
}
