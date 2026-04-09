import Link from "next/link";
import { BarChart3, BellRing, BrainCircuit, Radar, ScanSearch, Shield, Sparkles } from "lucide-react";

import { EmptyState } from "@/components/empty-state";
import { EventCard } from "@/components/event-card";
import { HeroSection } from "@/components/hero-section";
import { Reveal } from "@/components/reveal";
import { SectionHeader } from "@/components/section-header";
import { getEvents } from "@/lib/api";
import type { EventsResponse } from "@/types/api";

export const dynamic = "force-dynamic";

const methodology = [
  {
    icon: Shield,
    title: "Prefight integrity",
    copy: "Only prefight-available fighter information is used. No post-fight leakage, no retroactive features, no hidden future context.",
  },
  {
    icon: Radar,
    title: "Coverage discipline",
    copy: "Each bout is checked for fighter resolution, snapshot support, and feature construction. Unsupported fights are shown explicitly instead of being forced.",
  },
  {
    icon: BarChart3,
    title: "Calibrated probabilities",
    copy: "Raw model output is calibrated before it reaches the product so event pages communicate measured probability rather than fake certainty.",
  },
];

const differentiation = [
  "Model edge compares calibrated probability against live market price where odds exist.",
  "Unavailable predictions are surfaced clearly with reasons such as missing snapshot support or feature build failure.",
  "Every page is designed around event-level decision making rather than dumping notebook artifacts into a UI.",
];

const advancedTools = [
  {
    icon: ScanSearch,
    title: "Odds Screen",
    copy: "Bookmaker-aware event pricing with best-line context, market breadth, and clean fallback behavior when only aggregate odds are available.",
  },
  {
    icon: BellRing,
    title: "Email Notifications",
    copy: "Stage price and edge alerts directly from supported events so the alert workflow feels native to the product instead of bolted on later.",
  },
  {
    icon: BrainCircuit,
    title: "Research Assistant",
    copy: "Grounded matchup research generated only from current prefight features, model outputs, and live market context.",
  },
];

export default async function HomePage() {
  let eventsPayload: EventsResponse | null = null;
  let errorMessage: string | null = null;

  try {
    eventsPayload = await getEvents();
  } catch (error) {
    errorMessage = error instanceof Error ? error.message : "Unable to load event coverage.";
  }

  return (
    <>
      <HeroSection featuredEvent={eventsPayload?.featured_event ?? null} />

      <section className="px-6 py-20 lg:px-10">
        <div className="mx-auto max-w-7xl space-y-10">
          <Reveal>
            <SectionHeader
              eyebrow="How It Works"
              title="Built like a product, not a notebook wrapper."
              description="The experience is centered on event-level decision quality: which bouts are supported, where the model leans, how that compares to the market, and when the platform should explicitly say pass."
            />
          </Reveal>

          <div className="grid gap-6 lg:grid-cols-3">
            {methodology.map((item, index) => (
              <Reveal key={item.title} delay={index * 0.06}>
                <div className="h-full rounded-[28px] border border-white/10 bg-white/[0.03] p-6 shadow-halo">
                  <item.icon className="h-6 w-6 text-cyan-200" />
                  <h3 className="mt-8 text-2xl font-semibold tracking-[-0.04em] text-white">
                    {item.title}
                  </h3>
                  <p className="mt-4 text-sm leading-7 text-white/58">{item.copy}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section className="px-6 py-12 lg:px-10">
        <div className="mx-auto grid max-w-7xl gap-10 lg:grid-cols-[0.9fr_1.1fr] lg:items-start">
          <Reveal>
            <SectionHeader
              eyebrow="Why This Differs"
              title="More useful than raw odds, more honest than fake AI confidence."
              description="The platform does not treat every card as fully supported. It earns trust by separating supportable fights from unavailable ones and by keeping the model-market comparison grounded in real prefight inputs."
            />
          </Reveal>

          <Reveal delay={0.08}>
            <div className="rounded-[32px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.03))] p-6 shadow-halo">
              <div className="space-y-5">
                {differentiation.map((item) => (
                  <div key={item} className="flex gap-4 rounded-[24px] border border-white/8 bg-black/25 p-5">
                    <Sparkles className="mt-1 h-5 w-5 shrink-0 text-gold" />
                    <p className="text-sm leading-7 text-white/64">{item}</p>
                  </div>
                ))}
              </div>
            </div>
          </Reveal>
        </div>
      </section>

      <section className="px-6 py-16 lg:px-10">
        <div className="mx-auto max-w-7xl space-y-10">
          <Reveal>
            <SectionHeader
              eyebrow="Advanced Tools"
              title="More than a prediction feed."
              description="The platform now includes a premium odds screen, staged email alert workflow, and a grounded research assistant layered directly on top of the event intelligence engine."
            />
          </Reveal>

          <div className="grid gap-6 lg:grid-cols-3">
            {advancedTools.map((item, index) => (
              <Reveal key={item.title} delay={index * 0.06}>
                <div className="h-full rounded-[28px] border border-white/10 bg-white/[0.03] p-6 shadow-halo">
                  <item.icon className="h-6 w-6 text-gold" />
                  <h3 className="mt-8 text-2xl font-semibold tracking-[-0.04em] text-white">
                    {item.title}
                  </h3>
                  <p className="mt-4 text-sm leading-7 text-white/58">{item.copy}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section className="px-6 py-20 lg:px-10">
        <div className="mx-auto max-w-7xl space-y-10">
          <Reveal>
            <SectionHeader
              eyebrow="Featured Coverage"
              title="Current event intelligence."
              description="Use the supported event index as the front door into the platform. Each card tells you exactly how much of the slate is truly predictable from the current artifact set."
            />
          </Reveal>

          {eventsPayload?.featured_event ? (
            <Reveal delay={0.08}>
              <EventCard event={eventsPayload.featured_event} />
            </Reveal>
          ) : (
            <EmptyState
              title="Event coverage is not available yet"
              description={
                errorMessage ??
                "No supported event summaries were returned by the backend. Once cards and artifacts are available, featured event intelligence will appear here."
              }
              action={
                <Link
                  href="/events"
                  className="inline-flex rounded-full border border-white/12 bg-white/[0.04] px-5 py-3 text-sm font-medium text-white transition hover:bg-white/[0.08]"
                >
                  Open events index
                </Link>
              }
            />
          )}
        </div>
      </section>
    </>
  );
}
