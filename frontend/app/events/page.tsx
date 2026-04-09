import { EmptyState } from "@/components/empty-state";
import { EventCard } from "@/components/event-card";
import { Reveal } from "@/components/reveal";
import { SectionHeader } from "@/components/section-header";
import { getEvents } from "@/lib/api";
import type { EventSummary } from "@/types/api";

export const dynamic = "force-dynamic";

export default async function EventsPage() {
  let futureEvents: EventSummary[] = [];
  let errorMessage: string | null = null;

  try {
    const payload = await getEvents();
    futureEvents = Array.isArray(payload.future_events)
      ? payload.future_events
      : (payload.events ?? []).filter((event) => event.timeline === "future");
  } catch (error) {
    errorMessage = error instanceof Error ? error.message : "Unable to load events.";
  }

  return (
    <section className="px-6 py-20 lg:px-10">
      <div className="mx-auto max-w-7xl space-y-12">
        <Reveal>
          <SectionHeader
            eyebrow="Event Index"
            title="Upcoming UFC events."
            description="Every card the platform is currently tracking for prefight fight intelligence."
          />
        </Reveal>

        {futureEvents.length ? (
          <div className="grid gap-6 xl:grid-cols-2">
            {futureEvents.map((event, index) => (
              <Reveal key={event.event_id} delay={index * 0.04}>
                <EventCard event={event} />
              </Reveal>
            ))}
          </div>
        ) : (
          <EmptyState
            title="No upcoming events are synced yet"
            description={
              errorMessage ??
              "Run the card scraper or let the daily GitHub workflow pull the next UFC and Fight Night cards into the repo."
            }
          />
        )}
      </div>
    </section>
  );
}
