import Link from "next/link";
import { ChevronLeft, Clock3 } from "lucide-react";

import { CoverageBadge } from "@/components/coverage-badge";
import { EmptyState } from "@/components/empty-state";
import { EventWorkspace } from "@/components/event-workspace";
import { Reveal } from "@/components/reveal";
import { formatEventMeta, formatTimestamp, getEventDetail } from "@/lib/api";
import type { EventDetailResponse } from "@/types/api";

export const dynamic = "force-dynamic";

export default async function EventDetailPage({
  params,
}: {
  params: Promise<{ eventNumber: string }>;
}) {
  const { eventNumber: eventId } = await params;

  let payload: EventDetailResponse | null = null;
  let errorMessage: string | null = null;

  try {
    payload = await getEventDetail(eventId);
  } catch (error) {
    errorMessage = error instanceof Error ? error.message : "Unable to load event detail.";
  }

  if (!payload) {
    return (
      <section className="px-6 py-20 lg:px-10">
        <div className="mx-auto max-w-7xl">
          <EmptyState
            title={`Unable to load ${eventId}`}
            description={errorMessage ?? "This event could not be resolved from the backend API."}
            action={
              <Link
                href="/events"
                className="inline-flex rounded-full border border-white/12 bg-white/[0.04] px-5 py-3 text-sm font-medium text-white transition hover:bg-white/[0.08]"
              >
                Back to events
              </Link>
            }
          />
        </div>
      </section>
    );
  }

  const meta = formatEventMeta(payload.hero.date, payload.hero.venue, payload.hero.location);

  return (
    <section className="px-6 py-14 lg:px-10 lg:py-20">
      <div className="mx-auto max-w-7xl space-y-14">
        <Reveal>
          <Link
            href="/events"
            className="inline-flex items-center gap-2 text-sm text-white/52 transition hover:text-white"
          >
            <ChevronLeft className="h-4 w-4" />
            Back to event index
          </Link>
        </Reveal>

        <Reveal delay={0.04}>
          <div className="overflow-hidden rounded-[36px] border border-white/10 bg-[radial-gradient(circle_at_top_left,rgba(255,107,53,0.18),transparent_30%),radial-gradient(circle_at_80%_20%,rgba(99,212,255,0.14),transparent_32%),linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.03))] p-8 shadow-halo sm:p-10">
            <div className="flex flex-col gap-8 lg:flex-row lg:items-end lg:justify-between">
              <div className="max-w-3xl space-y-5">
                <CoverageBadge coverage={payload.hero.coverage} />
                <div>
                  <h1 className="text-4xl font-semibold tracking-[-0.06em] text-white sm:text-6xl">
                    {payload.hero.title}
                  </h1>
                  <p className="mt-4 max-w-2xl text-base leading-8 text-white/64 sm:text-lg">
                    {payload.hero.subtitle}
                  </p>
                </div>
                <div className="flex flex-wrap gap-3 text-sm text-white/56">
                  <span>{payload.hero.summary}</span>
                  <span>{meta || "Venue and date metadata unavailable in current local event payload."}</span>
                </div>
              </div>

              <div className="w-full max-w-sm rounded-[28px] border border-white/10 bg-black/25 p-5">
                <div className="flex items-start gap-3">
                  <Clock3 className="mt-1 h-5 w-5 text-cyan-200" />
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.24em] text-white/38">
                      Odds freshness
                    </p>
                    <p className="mt-2 text-lg font-medium text-white">
                      {payload.hero.odds_last_updated
                        ? formatTimestamp(payload.hero.odds_last_updated)
                        : "Pending"}
                    </p>
                    <p className="mt-2 text-sm leading-7 text-white/54">
                      {payload.hero.odds_status}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Reveal>

        {payload.supported_fights.length || payload.unsupported_fights.length ? (
          <Reveal delay={0.04}>
            <EventWorkspace
              eventId={payload.event.event_id}
              eventTitle={payload.hero.title}
              supportedFights={payload.supported_fights}
              unsupportedFights={payload.unsupported_fights}
            />
          </Reveal>
        ) : (
          <EmptyState
            title={payload.event.has_card ? "No supported fights on this card" : "No local fight card is stored for this event"}
            description={
              payload.event.has_card
                ? "This event is present locally, but the current prefight artifact set could not produce valid fight-level predictions for any matchup."
                : "This archived event has local metadata or odds history, but not a saved fight card artifact yet."
            }
          />
        )}
      </div>
    </section>
  );
}
