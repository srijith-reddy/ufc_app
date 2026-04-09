import { EmptyState } from "@/components/empty-state";
import { EventCard } from "@/components/event-card";
import { Reveal } from "@/components/reveal";
import { SectionHeader } from "@/components/section-header";
import { getEvents } from "@/lib/api";
import type { EventSummary } from "@/types/api";

export const dynamic = "force-dynamic";

export default async function EventsPage() {
  let futureEvents: EventSummary[] = [];
  let pastEvents: EventSummary[] = [];
  let errorMessage: string | null = null;

  try {
    const payload = await getEvents();
    futureEvents = Array.isArray(payload.future_events)
      ? payload.future_events
      : (payload.events ?? []).filter((event) => event.timeline === "future");
    pastEvents = Array.isArray(payload.past_events)
      ? payload.past_events
      : (payload.events ?? []).filter((event) => event.timeline === "past");
  } catch (error) {
    errorMessage = error instanceof Error ? error.message : "Unable to load events.";
  }

  return (
    <section className="px-6 py-20 lg:px-10">
      <div className="mx-auto max-w-7xl space-y-12">
        <Reveal>
          <SectionHeader
            eyebrow="Event Index"
            title="Browse upcoming and archived UFC intelligence."
            description="Future events are separated from archived cards so the product reads like a real event lifecycle, not a flat dump of local files."
          />
        </Reveal>

        {futureEvents.length || pastEvents.length ? (
          <div className="space-y-14">
            <div className="space-y-6">
              <SectionHeader
                eyebrow="Future Events"
                title="Upcoming cards"
                description="These are the events the product is currently tracking for upcoming fight intelligence."
              />
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
                  title="No future events are synced yet"
                  description="Run the card scraper or let the daily GitHub workflow pull the next UFC and Fight Night cards into the repo."
                />
              )}
            </div>

            <div className="space-y-6">
              <SectionHeader
                eyebrow="Past Events"
                title="Archived local events"
                description="Historical events are kept separately so past cards and odds snapshots do not clutter the upcoming event workflow."
              />
              {pastEvents.length ? (
                <div className="grid gap-6 xl:grid-cols-2">
                  {pastEvents.map((event, index) => (
                    <Reveal key={event.event_id} delay={index * 0.04}>
                      <EventCard event={event} />
                    </Reveal>
                  ))}
                </div>
              ) : (
                <EmptyState
                  title="No archived events are stored locally"
                  description="Once older event cards or historical odds snapshots are saved, they will appear here."
                />
              )}
            </div>
          </div>
        ) : (
          <EmptyState
            title="No event cards are currently available"
            description={
              errorMessage ??
              "Populate local card data and refresh the API to expose event intelligence here."
            }
          />
        )}
      </div>
    </section>
  );
}
