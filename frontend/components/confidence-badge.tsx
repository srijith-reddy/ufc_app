import { cn } from "@/lib/utils";
import type { ConfidenceMeta } from "@/types/api";

const toneClasses: Record<string, string> = {
  strong: "border-white/15 bg-white/8 text-white",
  medium: "border-cyan-400/25 bg-cyan-400/10 text-cyan-100",
  cautious: "border-amber-300/25 bg-amber-300/10 text-amber-100",
};

export function ConfidenceBadge({
  confidence,
  className,
}: {
  confidence: ConfidenceMeta;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-medium uppercase tracking-[0.18em]",
        toneClasses[confidence.tone],
        className,
      )}
    >
      {confidence.label}
    </span>
  );
}
