import { AlertCircle } from "lucide-react";

import type { UnsupportedFight } from "@/types/api";

export function UnsupportedFightRow({ fight }: { fight: UnsupportedFight }) {
  return (
    <div className="rounded-[24px] border border-white/10 bg-white/[0.03] p-5">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-2">
          <p className="text-lg font-semibold tracking-[-0.03em] text-white">
            {fight.fighter_a} vs {fight.fighter_b}
          </p>
          <p className="text-sm font-medium uppercase tracking-[0.18em] text-amber-100/78">
            {fight.reason_label}
          </p>
          <p className="max-w-2xl text-sm leading-7 text-white/58">{fight.reason}</p>
        </div>
        <div className="inline-flex items-center gap-2 rounded-full border border-amber-300/20 bg-amber-300/10 px-3 py-2 text-xs uppercase tracking-[0.18em] text-amber-100/80">
          <AlertCircle className="h-4 w-4" />
          Unavailable
        </div>
      </div>
    </div>
  );
}
