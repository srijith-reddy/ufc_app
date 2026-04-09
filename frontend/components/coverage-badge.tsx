import { cn } from "@/lib/utils";
import type { CoverageMeta } from "@/types/api";

const toneClasses: Record<string, string> = {
  success: "border-emerald-400/30 bg-emerald-400/10 text-emerald-200",
  warning: "border-amber-300/30 bg-amber-300/10 text-amber-100",
  muted: "border-white/10 bg-white/5 text-white/72",
};

export function CoverageBadge({
  coverage,
  className,
}: {
  coverage: CoverageMeta;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em]",
        toneClasses[coverage.tone] ?? toneClasses.muted,
        className,
      )}
    >
      {coverage.label}
    </span>
  );
}
