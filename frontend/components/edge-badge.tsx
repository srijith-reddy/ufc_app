import { cn } from "@/lib/utils";
import type { ValueState } from "@/types/api";

const toneClasses: Record<string, string> = {
  positive: "border-cyan-400/30 bg-cyan-400/10 text-cyan-200",
  neutral: "border-white/10 bg-white/5 text-white/75",
  negative: "border-rose-400/30 bg-rose-400/10 text-rose-200",
  muted: "border-white/10 bg-white/5 text-white/55",
};

export function EdgeBadge({
  valueState,
  className,
}: {
  valueState: ValueState;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-medium uppercase tracking-[0.18em]",
        toneClasses[valueState.tone] ?? toneClasses.muted,
        className,
      )}
    >
      {valueState.label}
    </span>
  );
}
