import { SkeletonFightCard } from "@/components/skeleton-fight-card";

export default function GlobalLoading() {
  return (
    <div className="px-6 py-20 lg:px-10">
      <div className="mx-auto max-w-7xl space-y-6">
        <div className="h-4 w-28 animate-pulse rounded-full bg-white/10" />
        <div className="h-16 w-2/3 animate-pulse rounded-[24px] bg-white/10" />
        <div className="grid gap-6 lg:grid-cols-2">
          <SkeletonFightCard />
          <SkeletonFightCard />
        </div>
      </div>
    </div>
  );
}
