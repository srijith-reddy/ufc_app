export function SkeletonFightCard() {
  return (
    <div className="animate-pulse rounded-[28px] border border-white/10 bg-white/[0.03] p-6">
      <div className="space-y-4">
        <div className="h-3 w-24 rounded-full bg-white/10" />
        <div className="h-8 w-3/4 rounded-full bg-white/10" />
        <div className="h-3 w-full rounded-full bg-white/10" />
        <div className="h-3 w-5/6 rounded-full bg-white/10" />
        <div className="h-20 rounded-[20px] bg-white/6" />
      </div>
    </div>
  );
}
