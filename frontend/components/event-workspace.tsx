"use client";

import { useMemo, useState } from "react";
import { BellRing, BrainCircuit, ChartCandlestick, ClipboardList, ShieldAlert } from "lucide-react";

import type { SupportedFight, UnsupportedFight } from "@/types/api";

import { AlertsStudio } from "./alerts-studio";
import { BettingLab } from "./betting-lab";
import { EmptyState } from "./empty-state";
import { FightCardDeck } from "./fight-card-deck";
import { OddsScreen } from "./odds-screen";
import { ResearchAssistant } from "./research-assistant";
import { UnsupportedFightRow } from "./unsupported-fight-row";

type TabId = "overview" | "market" | "research" | "alerts" | "unavailable";

const tabMeta = {
  overview: {
    label: "Overview",
    icon: ClipboardList,
    title: "Fight card and best supported reads.",
    description:
      "Lead with the card itself. Open any fight for the full breakdown, then drop into bankroll simulation underneath.",
  },
  market: {
    label: "Odds Screen",
    icon: ChartCandlestick,
    title: "Bookmaker context and market structure.",
    description:
      "Best prices, market breadth, and sportsbook rows when the current local artifact includes them.",
  },
  research: {
    label: "Research",
    icon: BrainCircuit,
    title: "Grounded matchup research.",
    description:
      "Ask about supported fights using only the current prefight payload, market context, and calibrated outputs.",
  },
  alerts: {
    label: "Alerts",
    icon: BellRing,
    title: "Price and edge notifications.",
    description:
      "Stage alert rules for the current event without cluttering the core fight-card view.",
  },
  unavailable: {
    label: "Unavailable",
    icon: ShieldAlert,
    title: "Explicitly unsupported bouts.",
    description:
      "Keep trust high by calling out unsupported matchups and the exact reason each one is unavailable.",
  },
} satisfies Record<TabId, { label: string; icon: React.ComponentType<{ className?: string }>; title: string; description: string }>;

export function EventWorkspace({
  eventId,
  eventTitle,
  supportedFights,
  unsupportedFights,
}: {
  eventId: string;
  eventTitle: string;
  supportedFights: SupportedFight[];
  unsupportedFights: UnsupportedFight[];
}) {
  const [activeTab, setActiveTab] = useState<TabId>("overview");

  const tabs = useMemo(
    () =>
      ([
        ["overview", supportedFights.length],
        ["market", supportedFights.length],
        ["research", supportedFights.length],
        ["alerts", supportedFights.length],
        ["unavailable", unsupportedFights.length],
      ] as const).filter(([, count], index) => (index === 4 ? true : count > 0)),
    [supportedFights.length, unsupportedFights.length],
  );

  const currentTab = tabMeta[activeTab];

  return (
    <div className="space-y-8">
      <div className="sticky top-[88px] z-20 rounded-[26px] border border-white/10 bg-[#071018]/88 p-3 backdrop-blur-xl">
        <div className="flex flex-wrap gap-2">
          {tabs.map(([tabId, count]) => {
            const meta = tabMeta[tabId];
            const active = activeTab === tabId;
            const Icon = meta.icon;
            return (
              <button
                key={tabId}
                type="button"
                onClick={() => setActiveTab(tabId)}
                className={`inline-flex items-center gap-2 rounded-full border px-4 py-2.5 text-sm transition ${
                  active
                    ? "border-white bg-white text-[#071018]"
                    : "border-white/10 bg-white/[0.03] text-white/70 hover:border-white/18 hover:text-white"
                }`}
              >
                <Icon className="h-4 w-4" />
                {meta.label}
                <span
                  className={`rounded-full px-2 py-0.5 text-[11px] ${
                    active ? "bg-[#071018]/10 text-[#071018]" : "bg-white/[0.06] text-white/55"
                  }`}
                >
                  {count}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="space-y-3">
        <p className="text-xs font-semibold uppercase tracking-[0.3em] text-cyan-200/72">
          {currentTab.label}
        </p>
        <h2 className="text-3xl font-semibold tracking-[-0.05em] text-white sm:text-4xl">
          {currentTab.title}
        </h2>
        <p className="max-w-3xl text-base leading-7 text-white/62">{currentTab.description}</p>
      </div>

      {activeTab === "overview" ? (
        <div className="space-y-10">
          {supportedFights.length ? (
            <>
              <FightCardDeck fights={supportedFights} />
              <BettingLab fights={supportedFights} />
            </>
          ) : (
            <EmptyState
              title="No supported fights on this card"
              description="This event is present locally, but the current prefight artifact set could not produce valid fight-level predictions for any matchup."
            />
          )}
        </div>
      ) : null}

      {activeTab === "market" ? <OddsScreen fights={supportedFights} /> : null}

      {activeTab === "research" ? (
        <ResearchAssistant eventTitle={eventTitle} fights={supportedFights} />
      ) : null}

      {activeTab === "alerts" ? (
        <AlertsStudio eventId={eventId} fights={supportedFights} />
      ) : null}

      {activeTab === "unavailable" ? (
        unsupportedFights.length ? (
          <div className="grid gap-4">
            {unsupportedFights.map((fight) => (
              <UnsupportedFightRow key={fight.id} fight={fight} />
            ))}
          </div>
        ) : (
          <EmptyState
            title="No unavailable fights on this card"
            description="Every bout on this event card is currently supported by the prefight pipeline."
          />
        )
      ) : null}
    </div>
  );
}
