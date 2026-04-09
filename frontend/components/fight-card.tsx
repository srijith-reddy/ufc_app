"use client";

import { motion } from "framer-motion";
import { ChevronRight } from "lucide-react";

import { formatPercent } from "@/lib/api";
import type { SupportedFight } from "@/types/api";

import { ConfidenceBadge } from "./confidence-badge";
import { EdgeBadge } from "./edge-badge";
import { ProbabilityBar } from "./probability-bar";

export function FightCard({
  fight,
  onSelect,
}: {
  fight: SupportedFight;
  onSelect: (fight: SupportedFight) => void;
}) {
  return (
    <motion.button
      whileHover={{ y: -3 }}
      transition={{ duration: 0.18 }}
      onClick={() => onSelect(fight)}
      className="w-full rounded-[30px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.07),rgba(255,255,255,0.03))] p-6 text-left shadow-halo transition hover:border-white/18"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <ConfidenceBadge confidence={fight.confidence} />
          <EdgeBadge valueState={fight.value_state} />
        </div>
        <div className="inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.22em] text-white/40">
          Open breakdown
          <ChevronRight className="h-4 w-4" />
        </div>
      </div>

      <div className="mt-8 grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
        <div className="grid gap-4 sm:grid-cols-3 xl:grid-cols-2">
          <div className="rounded-[22px] border border-white/8 bg-black/25 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-white/35">Pick side</p>
            <p className="mt-3 text-2xl font-semibold tracking-[-0.05em] text-white">
              {fight.favored_fighter}
            </p>
            <p className="mt-1 text-sm text-white/52">
              {formatPercent(fight.pick_probability, 1)} calibrated win probability
            </p>
          </div>
          <div className="rounded-[22px] border border-white/8 bg-black/25 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-white/35">Model edge</p>
            <p className="mt-3 text-2xl font-semibold tracking-[-0.05em] text-white">
              {formatPercent(fight.edge_pick, 1)}
            </p>
            <p className="mt-1 text-sm text-white/52">Versus market price on the selected side</p>
          </div>
          <div className="rounded-[22px] border border-white/8 bg-black/25 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-white/35">Market price</p>
            <p className="mt-3 text-2xl font-semibold tracking-[-0.05em] text-white">
              {fight.odds_available && fight.best_odds_decimal
                ? fight.best_odds_decimal.toFixed(2)
                : "N/A"}
            </p>
            <p className="mt-1 text-sm text-white/52">
              {fight.odds_available ? "Best decimal odds found for the pick side" : "Odds unavailable"}
            </p>
          </div>
          <div className="rounded-[22px] border border-white/8 bg-black/25 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-white/35">Kelly signal</p>
            <p className="mt-3 text-2xl font-semibold tracking-[-0.05em] text-white">
              {formatPercent(fight.kelly_fraction, 1)}
            </p>
            <p className="mt-1 text-sm text-white/52">Full Kelly fraction before your multiplier</p>
          </div>
          <div className="rounded-[22px] border border-white/8 bg-black/25 p-4 sm:col-span-3 xl:col-span-2">
            <p className="text-xs uppercase tracking-[0.2em] text-white/35">Grounded signals</p>
            <div className="mt-3 flex flex-wrap gap-2">
              {fight.insight_chips.length ? (
                fight.insight_chips.map((chip) => (
                  <span
                    key={`${fight.id}-${chip.label}-${chip.fighter}`}
                    className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-2 text-xs text-white/74"
                  >
                    {chip.label}: {chip.fighter}
                  </span>
                ))
              ) : (
                <span className="text-sm text-white/52">
                  Feature-based supporting signals were limited, but the prefight model still produced a valid probability.
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="rounded-[26px] border border-white/8 bg-black/20 p-5">
          <div className="flex items-start justify-between gap-5">
            <div className="space-y-2 text-right ml-auto">
              <div className="text-[11px] uppercase tracking-[0.22em] text-white/35">Supported bout</div>
              <div className="space-y-1">
                <h3 className="text-3xl font-semibold tracking-[-0.05em] text-white">
                  {fight.fighter_a}
                </h3>
                <p className="text-xs uppercase tracking-[0.26em] text-white/30">vs</p>
                <h3 className="text-3xl font-semibold tracking-[-0.05em] text-white">
                  {fight.fighter_b}
                </h3>
              </div>
            </div>
          </div>

          <div className="mt-6">
            <ProbabilityBar
              leftLabel={fight.fighter_a}
              rightLabel={fight.fighter_b}
              leftValue={fight.probability_a}
              rightValue={fight.probability_b}
            />
          </div>

          <p className="mt-5 max-w-2xl text-sm leading-7 text-white/62 xl:ml-auto xl:text-right">
            {fight.model_lean}
          </p>
        </div>
      </div>
    </motion.button>
  );
}
