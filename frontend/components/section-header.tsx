export function SectionHeader({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="max-w-2xl space-y-4">
      <p className="text-xs font-semibold uppercase tracking-[0.35em] text-cyan-200/80">
        {eyebrow}
      </p>
      <h2 className="text-3xl font-semibold tracking-[-0.04em] text-white sm:text-4xl">
        {title}
      </h2>
      <p className="text-base leading-7 text-white/64 sm:text-lg">{description}</p>
    </div>
  );
}
