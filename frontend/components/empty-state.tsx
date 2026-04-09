import type { ReactNode } from "react";

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="rounded-[28px] border border-white/10 bg-white/[0.03] p-8 shadow-halo">
      <div className="max-w-xl space-y-3">
        <h3 className="text-xl font-semibold tracking-[-0.03em] text-white">{title}</h3>
        <p className="text-sm leading-7 text-white/60">{description}</p>
        {action ? <div className="pt-3">{action}</div> : null}
      </div>
    </div>
  );
}
