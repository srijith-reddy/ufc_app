"use client";

import { useMemo, useState } from "react";
import { BrainCircuit, Send, Sparkles } from "lucide-react";

import { formatPercent } from "@/lib/api";
import type { FeatureComparisonItem, SupportedFight } from "@/types/api";

type AssistantMessage = {
  id: string;
  role: "assistant" | "user";
  text: string;
  fight?: SupportedFight | null;
  sections?: Array<{ title: string; items: string[] }>;
};

function normalizeText(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9 ]+/g, " ").replace(/\s+/g, " ").trim();
}

function compareLine(item: FeatureComparisonItem, fight: SupportedFight) {
  const left = item.fighter_a_display ?? "N/A";
  const right = item.fighter_b_display ?? "N/A";
  if (item.advantage === "a") {
    return `${fight.fighter_a} leads ${item.label.toLowerCase()} (${left} vs ${right}).`;
  }
  if (item.advantage === "b") {
    return `${fight.fighter_b} leads ${item.label.toLowerCase()} (${right} vs ${left}).`;
  }
  return `${item.label} is effectively even (${left} vs ${right}).`;
}

function buildFightResponse(fight: SupportedFight): AssistantMessage {
  const striking = fight.feature_comparison.filter((item) =>
    ["slpm", "sapm", "str_acc", "str_def"].includes(item.key),
  );
  const grappling = fight.feature_comparison.filter((item) =>
    ["td_avg", "td_def", "sub_avg"].includes(item.key),
  );
  const profile = fight.feature_comparison.filter((item) =>
    ["reach", "height", "age", "layoff", "recent_form_3"].includes(item.key),
  );

  return {
    id: `assistant-${fight.id}`,
    role: "assistant",
    fight,
    text: `${fight.model_lean} ${fight.edge_pick !== null ? `The model gives ${fight.favored_fighter} a ${formatPercent(fight.edge_pick, 1)} edge over market pricing.` : "Current market pricing is too thin to score a clean edge."}`,
    sections: [
      {
        title: "Matchup read",
        items: [
          `${fight.favored_fighter} is the current model side at ${formatPercent(fight.pick_probability, 1)} win probability.`,
          `${fight.confidence.label} profile with ${fight.value_state.label.toLowerCase()} market state.`,
        ],
      },
      {
        title: "Striking",
        items: striking.slice(0, 3).map((item) => compareLine(item, fight)),
      },
      {
        title: "Grappling",
        items: grappling.slice(0, 3).map((item) => compareLine(item, fight)),
      },
      {
        title: "Profile",
        items: profile.slice(0, 3).map((item) => compareLine(item, fight)),
      },
    ].filter((section) => section.items.length),
  };
}

function buildOverviewResponse(fights: SupportedFight[]): AssistantMessage {
  const positive = [...fights]
    .filter((fight) => fight.edge_pick !== null && fight.edge_pick > 0)
    .sort((left, right) => (right.edge_pick ?? 0) - (left.edge_pick ?? 0))
    .slice(0, 3);

  return {
    id: "assistant-overview",
    role: "assistant",
    text:
      positive.length > 0
        ? `Top current value signals on this card are ${positive.map((fight) => fight.favored_fighter).join(", ")}. Ask for a matchup comparison to get the full prefight breakdown.`
        : "This card has supported fights, but the current odds snapshot does not show a strong positive edge cluster right now.",
    sections: [
      {
        title: "Top edges",
        items:
          positive.length > 0
            ? positive.map(
                (fight) =>
                  `${fight.favored_fighter}: ${formatPercent(fight.pick_probability, 1)} model probability with ${formatPercent(fight.edge_pick, 1)} edge.`,
              )
            : ["No supported fight currently clears a large positive edge threshold."],
      },
    ],
  };
}

function findFight(prompt: string, fights: SupportedFight[]) {
  const normalizedPrompt = normalizeText(prompt);

  let direct = fights.find((fight) => {
    const a = normalizeText(fight.fighter_a);
    const b = normalizeText(fight.fighter_b);
    return normalizedPrompt.includes(a) && normalizedPrompt.includes(b);
  });
  if (direct) {
    return direct;
  }

  direct = fights.find((fight) => {
    const lastA = normalizeText(fight.fighter_a.split(" ").slice(-1)[0] ?? "");
    const lastB = normalizeText(fight.fighter_b.split(" ").slice(-1)[0] ?? "");
    return normalizedPrompt.includes(lastA) && normalizedPrompt.includes(lastB);
  });
  if (direct) {
    return direct;
  }

  return fights.find((fight) => {
    const a = normalizeText(fight.fighter_a);
    const b = normalizeText(fight.fighter_b);
    return normalizedPrompt.includes(a) || normalizedPrompt.includes(b);
  });
}

export function ResearchAssistant({
  eventTitle,
  fights,
}: {
  eventTitle: string;
  fights: SupportedFight[];
}) {
  const suggestedPrompts = useMemo(() => {
    const first = fights[0];
    const second = fights[1];
    return [
      first ? `Compare ${first.fighter_a} and ${first.fighter_b}` : "Break down the featured fight",
      "Where is the biggest edge on this card?",
      second ? `Tape study ${second.fighter_a}` : "Who carries the cleanest striking edge?",
      "Summarize the top betting signals",
    ];
  }, [fights]);

  const [messages, setMessages] = useState<AssistantMessage[]>([buildOverviewResponse(fights)]);
  const [input, setInput] = useState("");

  function submitPrompt(prompt: string) {
    const trimmed = prompt.trim();
    if (!trimmed) {
      return;
    }

    const nextMessages: AssistantMessage[] = [
      ...messages,
      {
        id: `user-${messages.length}`,
        role: "user",
        text: trimmed,
      },
    ];

    const matchedFight = findFight(trimmed, fights);
    const assistantMessage =
      matchedFight !== undefined
        ? buildFightResponse(matchedFight)
        : buildOverviewResponse(fights);

    setMessages([...nextMessages, { ...assistantMessage, id: `${assistantMessage.id}-${messages.length}` }]);
    setInput("");
  }

  return (
    <div className="rounded-[32px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.03))] p-6 shadow-halo">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="max-w-3xl">
          <p className="text-xs font-semibold uppercase tracking-[0.26em] text-cyan-200/72">
            Research Assistant
          </p>
          <h3 className="mt-3 text-2xl font-semibold tracking-[-0.04em] text-white">
            Grounded matchup research for {eventTitle}.
          </h3>
          <p className="mt-3 text-sm leading-7 text-white/58">
            This assistant only summarizes supported fights from the current prefight snapshot, odds data,
            and calibrated model outputs. No invented narratives, no hidden post-fight context.
          </p>
        </div>

        <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 text-xs uppercase tracking-[0.16em] text-white/55">
          <BrainCircuit className="h-4 w-4 text-cyan-200" />
          Grounded mode
        </div>
      </div>

      <div className="mt-8 space-y-5">
        {messages.map((message) => (
          <div
            key={message.id}
            className={message.role === "user" ? "flex justify-end" : "flex justify-start"}
          >
            <div
              className={
                message.role === "user"
                  ? "max-w-3xl rounded-[24px] bg-[#ff6b35] px-5 py-4 text-white shadow-glow"
                  : "max-w-4xl rounded-[28px] border border-white/10 bg-white/[0.03] px-5 py-5 text-white/84"
              }
            >
              <p className="text-sm leading-7">{message.text}</p>

              {message.sections?.length ? (
                <div className="mt-5 grid gap-4 lg:grid-cols-2">
                  {message.sections.map((section) => (
                    <div key={`${message.id}-${section.title}`} className="rounded-[22px] border border-white/8 bg-black/20 p-4">
                      <div className="flex items-center gap-2">
                        <Sparkles className="h-4 w-4 text-cyan-200" />
                        <p className="text-sm font-medium text-white">{section.title}</p>
                      </div>
                      <div className="mt-3 space-y-2 text-sm leading-7 text-white/64">
                        {section.items.map((item) => (
                          <p key={item}>{item}</p>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        {suggestedPrompts.map((prompt) => (
          <button
            key={prompt}
            type="button"
            onClick={() => submitPrompt(prompt)}
            className="rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-sm text-white/72 transition hover:border-white/16 hover:bg-white/[0.06] hover:text-white"
          >
            {prompt}
          </button>
        ))}
      </div>

      <div className="mt-6 rounded-[28px] border border-white/10 bg-black/20 p-3">
        <div className="flex flex-col gap-3">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask about a fighter, matchup, edge, or tape-study angle..."
            className="min-h-[120px] w-full resize-none rounded-[22px] border border-white/8 bg-transparent px-4 py-3 text-base text-white outline-none placeholder:text-white/34"
          />
          <div className="flex items-center justify-between gap-3">
            <p className="text-xs leading-6 text-white/38">
              Answers are generated only from the current supported fight payload.
            </p>
            <button
              type="button"
              onClick={() => submitPrompt(input)}
              className="inline-flex items-center gap-2 rounded-full bg-white px-5 py-3 text-sm font-medium text-[#071018] transition hover:bg-cyan-100"
            >
              Send
              <Send className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
