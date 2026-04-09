"use client";

import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";

import { formatPercent } from "@/lib/api";
import type { FeatureComparisonItem, SupportedFight } from "@/types/api";

import { ConfidenceBadge } from "./confidence-badge";
import { EdgeBadge } from "./edge-badge";
import { ProbabilityBar } from "./probability-bar";

function FeatureRow({
  item,
  fighterA,
  fighterB,
}: {
  item: FeatureComparisonItem;
  fighterA: string;
  fighterB: string;
}) {
  return (
    <div className="grid gap-3 rounded-[20px] border border-white/8 bg-white/[0.03] p-4 sm:grid-cols-[1fr_auto_1fr] sm:items-center">
      <div className={item.advantage === "a" ? "text-white" : "text-white/55"}>
        <p className="text-xs uppercase tracking-[0.18em] text-white/35">{fighterA}</p>
        <p className="mt-2 text-lg font-medium">{item.fighter_a_display ?? "N/A"}</p>
      </div>
      <div className="text-center">
        <p className="text-xs uppercase tracking-[0.18em] text-white/35">{item.label}</p>
      </div>
      <div className={item.advantage === "b" ? "text-white sm:text-right" : "text-white/55 sm:text-right"}>
        <p className="text-xs uppercase tracking-[0.18em] text-white/35">{fighterB}</p>
        <p className="mt-2 text-lg font-medium">{item.fighter_b_display ?? "N/A"}</p>
      </div>
    </div>
  );
}

export function FightBreakdownModal({
  fight,
  onClose,
}: {
  fight: SupportedFight | null;
  onClose: () => void;
}) {
  return (
    <AnimatePresence>
      {fight ? (
        <motion.div
          className="fixed inset-0 z-50 flex items-end justify-center bg-black/70 p-4 backdrop-blur-md lg:items-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, y: 24, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 16, scale: 0.98 }}
            transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
            onClick={(event) => event.stopPropagation()}
            className="max-h-[92vh] w-full max-w-5xl overflow-y-auto rounded-[34px] border border-white/10 bg-[#071018] p-6 shadow-halo sm:p-8"
          >
            <div className="flex items-start justify-between gap-4">
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  <ConfidenceBadge confidence={fight.confidence} />
                  <EdgeBadge valueState={fight.value_state} />
                </div>
                <div>
                  <h3 className="text-3xl font-semibold tracking-[-0.05em] text-white sm:text-4xl">
                    {fight.fighter_a} vs {fight.fighter_b}
                  </h3>
                  <p className="mt-3 max-w-3xl text-sm leading-7 text-white/62">{fight.model_lean}</p>
                </div>
              </div>
              <button
                type="button"
                onClick={onClose}
                className="rounded-full border border-white/10 bg-white/[0.04] p-3 text-white/70 transition hover:border-white/18 hover:bg-white/[0.08] hover:text-white"
                aria-label="Close fight breakdown"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="mt-8 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
              <div className="rounded-[28px] border border-white/10 bg-white/[0.03] p-5">
                <ProbabilityBar
                  leftLabel={fight.fighter_a}
                  rightLabel={fight.fighter_b}
                  leftValue={fight.probability_a}
                  rightValue={fight.probability_b}
                />
              </div>

              <div className="grid gap-4 sm:grid-cols-3 lg:grid-cols-1">
                <div className="rounded-[24px] border border-white/10 bg-white/[0.03] p-5">
                  <p className="text-xs uppercase tracking-[0.2em] text-white/35">Pick probability</p>
                  <p className="mt-2 text-3xl font-semibold tracking-[-0.05em] text-white">
                    {formatPercent(fight.pick_probability)}
                  </p>
                </div>
                <div className="rounded-[24px] border border-white/10 bg-white/[0.03] p-5">
                  <p className="text-xs uppercase tracking-[0.2em] text-white/35">Market probability</p>
                  <p className="mt-2 text-3xl font-semibold tracking-[-0.05em] text-white">
                    {formatPercent(fight.market_probability_pick)}
                  </p>
                </div>
                <div className="rounded-[24px] border border-white/10 bg-white/[0.03] p-5">
                  <p className="text-xs uppercase tracking-[0.2em] text-white/35">Kelly fraction</p>
                  <p className="mt-2 text-3xl font-semibold tracking-[-0.05em] text-white">
                    {formatPercent(fight.kelly_fraction, 1)}
                  </p>
                </div>
              </div>
            </div>

            <div className="mt-8 space-y-4">
              <p className="text-xs font-semibold uppercase tracking-[0.3em] text-white/38">
                Grounded comparison
              </p>
              <div className="grid gap-3">
                {fight.feature_comparison.map((item) => (
                  <FeatureRow
                    key={item.key}
                    item={item}
                    fighterA={fight.fighter_a}
                    fighterB={fight.fighter_b}
                  />
                ))}
              </div>
            </div>

            <div className="mt-8 space-y-4">
              <p className="text-xs font-semibold uppercase tracking-[0.3em] text-white/38">
                Prefight signals
              </p>
              <div className="flex flex-wrap gap-3">
                {fight.insight_chips.length ? (
                  fight.insight_chips.map((chip) => (
                    <div
                      key={`${fight.id}-${chip.label}-${chip.fighter}`}
                      className="rounded-[20px] border border-white/10 bg-white/[0.04] px-4 py-3"
                    >
                      <p className="text-xs uppercase tracking-[0.2em] text-white/35">{chip.label}</p>
                      <p className="mt-2 text-sm font-medium text-white">{chip.fighter}</p>
                      <p className="mt-1 text-sm text-white/56">{chip.detail}</p>
                    </div>
                  ))
                ) : (
                  <div className="rounded-[20px] border border-white/10 bg-white/[0.04] px-4 py-3 text-sm text-white/58">
                    No extra signal chips were promoted for this matchup because the available prefight feature differences were not materially large enough.
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
