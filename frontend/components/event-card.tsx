import Link from "next/link";
import { ArrowUpRight } from "lucide-react";

import { formatEventMeta, formatTimestamp } from "@/lib/api";
import type { EventSummary } from "@/types/api";

import { CoverageBadge } from "./coverage-badge";

export function EventCard({ event }: { event: EventSummary }) {
  const meta = formatEventMeta(event.date, event.venue, event.location);

  return (
    <Link
      href={`/events/${event.event_id}`}
      className="group rounded-[30px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.02))] p-6 shadow-halo transition duration-300 hover:border-white/18 hover:bg-[linear-gradient(180deg,rgba(255,255,255,0.08),rgba(255,255,255,0.03))] hover:translate-y-[-2px]"
    >
      <div className="flex items-start justify-between gap-4">
        <CoverageBadge coverage={event.coverage} />
        <span className="rounded-full border border-white/10 p-2 text-white/48 transition group-hover:border-cyan-300/25 group-hover:text-cyan-100">
          <ArrowUpRight className="h-4 w-4" />
        </span>
      </div>

      <div className="mt-8 space-y-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.26em] text-white/38">
            Event intelligence
          </p>
          <h3 className="mt-3 text-3xl font-semibold tracking-[-0.04em] text-white">
            {event.title}
          </h3>
          <p className="mt-2 text-sm leading-7 text-white/60">{event.subtitle}</p>
        </div>

        <div className="grid gap-3 rounded-[22px] border border-white/8 bg-black/25 p-4 sm:grid-cols-3">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-white/35">Support</p>
            <p className="mt-2 text-xl font-semibold text-white">
              {event.supported_count}/{event.total_count}
            </p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-white/35">Unavailable</p>
            <p className="mt-2 text-xl font-semibold text-white">{event.unsupported_count}</p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-white/35">Coverage</p>
            <p className="mt-2 text-xl font-semibold text-white">
              {(event.coverage_ratio * 100).toFixed(0)}%
            </p>
          </div>
        </div>
      </div>

      <div className="mt-6 flex flex-col gap-2 text-sm text-white/52">
        <span>{event.featured_matchup ?? "Matchup details appear on the event page."}</span>
        <span>{meta || "Event metadata unavailable in current local card payload."}</span>
        <span>
          {event.odds_last_updated
            ? `Odds refresh: ${formatTimestamp(event.odds_last_updated)}`
            : event.timeline === "future"
              ? "Odds not synced yet for this event."
              : "Historical odds snapshot unavailable."}
        </span>
      </div>
    </Link>
  );
}
