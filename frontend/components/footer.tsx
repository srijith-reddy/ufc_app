import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-t border-white/8 bg-black/40">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 px-6 py-10 text-sm text-white/48 lg:flex-row lg:items-end lg:justify-between lg:px-10">
        <div className="max-w-xl space-y-2">
          <p className="text-sm font-medium uppercase tracking-[0.28em] text-white/40">
            Octagon Intel
          </p>
          <p className="leading-7">
            Calibrated prefight intelligence for UFC cards. No post-fight leakage. No fake certainty.
          </p>
        </div>
        <div className="space-y-2 text-left lg:text-right">
          <p>Model edge compares calibrated probabilities against market price when odds are available.</p>
          <p>
            Use responsibly. This product is analytical support, not a guarantee of outcomes.
          </p>
          <p>
            <Link href="/events" className="text-cyan-200 transition hover:text-white">
              Browse supported events
            </Link>
          </p>
        </div>
      </div>
    </footer>
  );
}
