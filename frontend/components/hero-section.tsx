import Link from "next/link";
import { ArrowRight, ShieldCheck, Sparkles } from "lucide-react";

import { formatTimestamp } from "@/lib/api";
import type { EventSummary } from "@/types/api";

import { CoverageBadge } from "./coverage-badge";
import { Reveal } from "./reveal";

export function HeroSection({ featuredEvent }: { featuredEvent: EventSummary | null }) {
  return (
    <section className="relative overflow-hidden px-6 py-20 lg:px-10 lg:py-28">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(255,107,53,0.2),transparent_32%),radial-gradient(circle_at_80%_10%,rgba(99,212,255,0.16),transparent_30%),linear-gradient(180deg,rgba(255,255,255,0.04),transparent_30%)]" />
      <div className="absolute inset-x-0 top-0 h-full bg-grid bg-[size:72px_72px] opacity-[0.07]" />

      <div className="relative mx-auto grid max-w-7xl gap-12 lg:grid-cols-[1.15fr_0.85fr] lg:items-end">
        <Reveal>
          <div className="space-y-8">
            <div className="inline-flex items-center gap-2 rounded-full border border-white/12 bg-white/[0.04] px-4 py-2 text-xs font-semibold uppercase tracking-[0.28em] text-white/72">
              <ShieldCheck className="h-4 w-4 text-cyan-200" />
              Prefight-only
              <span className="text-white/25">/</span>
              Calibrated
              <span className="text-white/25">/</span>
              Deterministic
            </div>

            <div className="max-w-4xl space-y-6">
              <h1 className="text-5xl font-semibold leading-[0.94] tracking-[-0.06em] text-white sm:text-6xl lg:text-7xl">
                Calibrated UFC betting intelligence for supported fight cards.
              </h1>
              <p className="max-w-2xl text-lg leading-8 text-white/64 sm:text-xl">
                Event-level prediction coverage grounded in real available data. No post-fight leakage.
                No fake explanations. No false certainty where support does not exist.
              </p>
            </div>

            <div className="flex flex-col gap-4 sm:flex-row">
              <Link
                href={featuredEvent ? `/events/${featuredEvent.event_id}` : "/events"}
                className="inline-flex items-center justify-center gap-2 rounded-full bg-white px-6 py-3 text-sm font-semibold text-[#05090e] transition hover:translate-y-[-1px] hover:bg-cyan-100"
              >
                View featured event
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="/events"
                className="inline-flex items-center justify-center rounded-full border border-white/12 bg-white/[0.03] px-6 py-3 text-sm font-semibold text-white/78 transition hover:border-white/24 hover:bg-white/[0.06] hover:text-white"
              >
                Browse event coverage
              </Link>
            </div>
          </div>
        </Reveal>

        <Reveal delay={0.08}>
          <div className="rounded-[32px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.08),rgba(255,255,255,0.02))] p-6 shadow-halo backdrop-blur">
            <div className="flex items-start justify-between gap-4">
              <div className="space-y-3">
                <p className="text-xs font-semibold uppercase tracking-[0.32em] text-white/45">
                  Featured Event
                </p>
                <div>
                  <h2 className="text-3xl font-semibold tracking-[-0.04em] text-white">
                    {featuredEvent?.title ?? "Coverage syncing"}
                  </h2>
                  <p className="mt-2 text-sm leading-7 text-white/58">
                    {featuredEvent?.subtitle ??
                      "Supported fight cards will appear here once local event data is available."}
                  </p>
                </div>
              </div>
              {featuredEvent ? <CoverageBadge coverage={featuredEvent.coverage} /> : null}
            </div>

            <div className="mt-8 grid gap-4 rounded-[24px] border border-white/8 bg-black/25 p-5 sm:grid-cols-2">
              <div>
                <p className="text-xs uppercase tracking-[0.22em] text-white/38">Coverage</p>
                <p className="mt-2 text-2xl font-semibold tracking-[-0.04em] text-white">
                  {featuredEvent
                    ? `${featuredEvent.supported_count}/${featuredEvent.total_count}`
                    : "N/A"}
                </p>
                <p className="mt-1 text-sm text-white/52">supported bouts on the current card</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.22em] text-white/38">Market refresh</p>
                <p className="mt-2 text-2xl font-semibold tracking-[-0.04em] text-white">
                  {formatTimestamp(featuredEvent?.odds_last_updated ?? null)}
                </p>
                <p className="mt-1 text-sm text-white/52">latest captured odds snapshot</p>
              </div>
            </div>

            <div className="mt-6 rounded-[24px] border border-white/8 bg-gradient-to-br from-[#ff6b35]/10 to-[#63d4ff]/10 p-5">
              <div className="flex items-start gap-3">
                <Sparkles className="mt-0.5 h-5 w-5 text-gold" />
                <div className="space-y-2">
                  <p className="text-sm font-medium text-white">
                    Why this feels different from raw odds
                  </p>
                  <p className="text-sm leading-7 text-white/62">
                    Market pricing is shown alongside calibrated model probability so you can spot agreement,
                    disagreement, and true no-bet spots without pretending every matchup has clean support.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
