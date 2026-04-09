"use client";

import { useMemo, useState } from "react";
import { Coins, Flame, LineChart, ShieldAlert } from "lucide-react";

import { formatPercent } from "@/lib/api";
import {
  expectedValuePerDollar,
  qualifyBets,
  simulateBankroll,
  type BettingSettings,
  type QualifiedBet,
} from "@/lib/betting";
import type { SupportedFight } from "@/types/api";

function formatCurrency(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

function SettingRow({
  label,
  value,
  children,
}: {
  label: string;
  value: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-3 rounded-[24px] border border-white/10 bg-white/[0.03] p-4">
      <div className="flex items-center justify-between gap-4">
        <p className="text-sm text-white/64">{label}</p>
        <span className="text-sm font-medium text-white">{value}</span>
      </div>
      {children}
    </div>
  );
}

export function BettingLab({ fights }: { fights: SupportedFight[] }) {
  const [settings, setSettings] = useState<BettingSettings>({
    bankroll: 1000,
    kellyMultiplier: 0.5,
    minEdge: 0.02,
    minExpectedValue: 0.01,
    simulations: 5000,
  });

  const bets = useMemo(() => qualifyBets(fights, settings), [fights, settings]);
  const simulation = useMemo(() => simulateBankroll(bets, settings), [bets, settings]);
  const topBet = bets[0] ?? null;
  const strongestEdge = bets.reduce<QualifiedBet | null>(
    (best, current) => (best === null || current.edge > best.edge ? current : best),
    null,
  );
  const strongestKelly = bets.reduce<QualifiedBet | null>(
    (best, current) =>
      best === null || current.fullKellyFraction > best.fullKellyFraction ? current : best,
    null,
  );

  return (
    <div className="grid gap-8 xl:grid-cols-[340px_1fr]">
      <aside className="space-y-4 rounded-[32px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.03))] p-5 shadow-halo">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.26em] text-cyan-200/72">
            Betting Settings
          </p>
          <h3 className="mt-3 text-2xl font-semibold tracking-[-0.04em] text-white">
            Event bankroll lab
          </h3>
          <p className="mt-3 text-sm leading-7 text-white/58">
            Calibrated probabilities, live market prices, and conservative Kelly sizing for a cleaner
            event-level betting view.
          </p>
        </div>

        <SettingRow label="Starting bankroll" value={formatCurrency(settings.bankroll)}>
          <input
            type="range"
            min={250}
            max={10000}
            step={250}
            value={settings.bankroll}
            onChange={(event) =>
              setSettings((current) => ({ ...current, bankroll: Number(event.target.value) }))
            }
            className="w-full accent-[#ff6b35]"
          />
        </SettingRow>

        <SettingRow label="Kelly multiplier" value={`${settings.kellyMultiplier.toFixed(2)}x`}>
          <input
            type="range"
            min={0.1}
            max={1}
            step={0.05}
            value={settings.kellyMultiplier}
            onChange={(event) =>
              setSettings((current) => ({
                ...current,
                kellyMultiplier: Number(event.target.value),
              }))
            }
            className="w-full accent-[#ff6b35]"
          />
        </SettingRow>

        <SettingRow label="Minimum edge" value={formatPercent(settings.minEdge, 1)}>
          <input
            type="range"
            min={0}
            max={0.12}
            step={0.005}
            value={settings.minEdge}
            onChange={(event) =>
              setSettings((current) => ({ ...current, minEdge: Number(event.target.value) }))
            }
            className="w-full accent-[#63d4ff]"
          />
        </SettingRow>

        <SettingRow
          label="Minimum EV per dollar"
          value={`${settings.minExpectedValue.toFixed(3)}`}
        >
          <input
            type="range"
            min={0}
            max={0.2}
            step={0.005}
            value={settings.minExpectedValue}
            onChange={(event) =>
              setSettings((current) => ({
                ...current,
                minExpectedValue: Number(event.target.value),
              }))
            }
            className="w-full accent-[#63d4ff]"
          />
        </SettingRow>

        <SettingRow label="Monte Carlo runs" value={settings.simulations.toLocaleString()}>
          <input
            type="range"
            min={500}
            max={10000}
            step={500}
            value={settings.simulations}
            onChange={(event) =>
              setSettings((current) => ({
                ...current,
                simulations: Number(event.target.value),
              }))
            }
            className="w-full accent-[#f6c56d]"
          />
        </SettingRow>
      </aside>

      <div className="space-y-8">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <div className="rounded-[28px] border border-white/10 bg-white/[0.03] p-5">
            <div className="flex items-center gap-3 text-white/72">
              <Coins className="h-5 w-5 text-gold" />
              <p className="text-sm uppercase tracking-[0.18em]">Bets Simulated</p>
            </div>
            <p className="mt-4 text-4xl font-semibold tracking-[-0.05em] text-white">
              {bets.length}
            </p>
            <p className="mt-2 text-sm text-white/54">Qualified using edge, EV, and Kelly filters.</p>
          </div>

          <div className="rounded-[28px] border border-white/10 bg-white/[0.03] p-5">
            <div className="flex items-center gap-3 text-white/72">
              <LineChart className="h-5 w-5 text-cyan-200" />
              <p className="text-sm uppercase tracking-[0.18em]">Median Final Bankroll</p>
            </div>
            <p className="mt-4 text-4xl font-semibold tracking-[-0.05em] text-white">
              {simulation ? formatCurrency(simulation.median) : "N/A"}
            </p>
          </div>

          <div className="rounded-[28px] border border-white/10 bg-white/[0.03] p-5">
            <div className="flex items-center gap-3 text-white/72">
              <Flame className="h-5 w-5 text-[#ff6b35]" />
              <p className="text-sm uppercase tracking-[0.18em]">Chance To Grow</p>
            </div>
            <p className="mt-4 text-4xl font-semibold tracking-[-0.05em] text-white">
              {simulation ? formatPercent(simulation.chanceToGrow, 1) : "N/A"}
            </p>
          </div>

          <div className="rounded-[28px] border border-white/10 bg-white/[0.03] p-5">
            <div className="flex items-center gap-3 text-white/72">
              <ShieldAlert className="h-5 w-5 text-amber-200" />
              <p className="text-sm uppercase tracking-[0.18em]">Drawdown &gt; 50%</p>
            </div>
            <p className="mt-4 text-4xl font-semibold tracking-[-0.05em] text-white">
              {simulation ? formatPercent(simulation.drawdownOverHalf, 1) : "N/A"}
            </p>
          </div>
        </div>

        {bets.length ? (
          <div className="grid gap-4 xl:grid-cols-3">
            <div className="rounded-[28px] border border-emerald-300/14 bg-emerald-300/[0.08] p-5">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-100/70">
                Best EV Signal
              </p>
              <p className="mt-3 text-xl font-semibold tracking-[-0.04em] text-white">
                {topBet?.pick}
              </p>
              <p className="mt-1 text-sm text-white/58">{topBet?.fightLabel}</p>
              <p className="mt-4 text-sm leading-7 text-emerald-100/88">
                {topBet ? `Expected value of +${topBet.expectedValue.toFixed(3)} per dollar at ${topBet.decimalOdds.toFixed(2)} decimal odds.` : null}
              </p>
            </div>

            <div className="rounded-[28px] border border-cyan-300/14 bg-cyan-300/[0.08] p-5">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-100/70">
                Largest Market Gap
              </p>
              <p className="mt-3 text-xl font-semibold tracking-[-0.04em] text-white">
                {strongestEdge ? formatPercent(strongestEdge.edge, 1) : "N/A"}
              </p>
              <p className="mt-1 text-sm text-white/58">{strongestEdge?.fightLabel}</p>
              <p className="mt-4 text-sm leading-7 text-cyan-100/88">
                {strongestEdge ? `${strongestEdge.pick} carries the biggest model-market disagreement on the card under your current filters.` : null}
              </p>
            </div>

            <div className="rounded-[28px] border border-amber-200/14 bg-amber-200/[0.08] p-5">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-amber-100/70">
                Total Stake Exposure
              </p>
              <p className="mt-3 text-xl font-semibold tracking-[-0.04em] text-white">
                {simulation ? formatCurrency(simulation.totalStake) : "N/A"}
              </p>
              <p className="mt-1 text-sm text-white/58">{strongestKelly?.fightLabel ?? "No current Kelly signals"}</p>
              <p className="mt-4 text-sm leading-7 text-amber-100/88">
                {strongestKelly
                  ? `${strongestKelly.pick} is the largest Kelly sizing signal. Lower the multiplier if this exposure feels too aggressive.`
                  : "No wager clears the full set of current thresholds."}
              </p>
            </div>
          </div>
        ) : null}

        {simulation ? (
          <div className="rounded-[32px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.03))] p-6 shadow-halo">
            <h3 className="text-2xl font-semibold tracking-[-0.04em] text-white">
              Bankroll simulation
            </h3>
            <p className="mt-3 max-w-3xl text-sm leading-7 text-white/58">
              This simulates the card thousands of times using calibrated model probability, current best
              odds, and your selected Kelly multiplier. It answers: if I repeat this event strategy, what
              usually happens?
            </p>

            <div className="mt-8 grid gap-5 md:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-[24px] border border-white/8 bg-black/25 p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-white/38">5% worst case</p>
                <p className="mt-2 text-3xl font-semibold text-white">{formatCurrency(simulation.p5)}</p>
              </div>
              <div className="rounded-[24px] border border-white/8 bg-black/25 p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-white/38">Median</p>
                <p className="mt-2 text-3xl font-semibold text-white">{formatCurrency(simulation.median)}</p>
              </div>
              <div className="rounded-[24px] border border-white/8 bg-black/25 p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-white/38">95% best case</p>
                <p className="mt-2 text-3xl font-semibold text-white">{formatCurrency(simulation.p95)}</p>
              </div>
              <div className="rounded-[24px] border border-white/8 bg-black/25 p-4">
                <p className="text-xs uppercase tracking-[0.18em] text-white/38">Mean / Std Dev</p>
                <p className="mt-2 text-xl font-semibold text-white">
                  {formatCurrency(simulation.mean)} / {formatCurrency(simulation.stdDev)}
                </p>
              </div>
            </div>
          </div>
        ) : null}

        <div className="rounded-[32px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.03))] p-6 shadow-halo">
          <h3 className="text-2xl font-semibold tracking-[-0.04em] text-white">Qualified bets</h3>
          <p className="mt-3 max-w-3xl text-sm leading-7 text-white/58">
            Only bets that clear your edge, EV, and Kelly thresholds are included. This keeps the event page
            focused on usable decisions rather than every supported fight with a line attached.
          </p>

          {bets.length ? (
            <div className="mt-6 overflow-hidden rounded-[24px] border border-white/10">
              <div className="hidden grid-cols-[2.2fr_1.4fr_1fr_1fr_1fr_1fr] bg-white/[0.04] px-4 py-3 text-xs uppercase tracking-[0.16em] text-white/45 md:grid">
                <span>Fight</span>
                <span>Pick</span>
                <span>p_model</span>
                <span>best odds</span>
                <span>full Kelly</span>
                <span>stake</span>
              </div>
              {bets.map((bet) => (
                <div key={bet.id} className="border-t border-white/8">
                  <div className="grid grid-cols-[2.2fr_1.4fr_1fr_1fr_1fr_1fr] px-4 py-4 text-sm text-white/84 max-md:hidden">
                    <span>{bet.fightLabel}</span>
                    <span>{bet.pick}</span>
                    <span>{formatPercent(bet.pickProbability, 1)}</span>
                    <span>{bet.decimalOdds.toFixed(2)}</span>
                    <span>{formatPercent(bet.fullKellyFraction, 1)}</span>
                    <span>{formatCurrency(bet.stakeDollars)}</span>
                  </div>

                  <div className="space-y-3 px-4 py-4 md:hidden">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <p className="text-base font-medium text-white">{bet.pick}</p>
                        <p className="mt-1 text-sm text-white/56">{bet.fightLabel}</p>
                      </div>
                      <div className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-xs font-medium text-cyan-100">
                        {formatPercent(bet.edge, 1)} edge
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-3 text-sm text-white/72">
                      <div className="rounded-[18px] border border-white/8 bg-black/20 p-3">
                        <p className="text-[11px] uppercase tracking-[0.16em] text-white/38">Model</p>
                        <p className="mt-2 text-base font-medium text-white">
                          {formatPercent(bet.pickProbability, 1)}
                        </p>
                      </div>
                      <div className="rounded-[18px] border border-white/8 bg-black/20 p-3">
                        <p className="text-[11px] uppercase tracking-[0.16em] text-white/38">Best odds</p>
                        <p className="mt-2 text-base font-medium text-white">
                          {bet.decimalOdds.toFixed(2)}
                        </p>
                      </div>
                      <div className="rounded-[18px] border border-white/8 bg-black/20 p-3">
                        <p className="text-[11px] uppercase tracking-[0.16em] text-white/38">Full Kelly</p>
                        <p className="mt-2 text-base font-medium text-white">
                          {formatPercent(bet.fullKellyFraction, 1)}
                        </p>
                      </div>
                      <div className="rounded-[18px] border border-white/8 bg-black/20 p-3">
                        <p className="text-[11px] uppercase tracking-[0.16em] text-white/38">Stake</p>
                        <p className="mt-2 text-base font-medium text-white">
                          {formatCurrency(bet.stakeDollars)}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="mt-6 rounded-[24px] border border-white/10 bg-white/[0.03] p-5 text-sm leading-7 text-white/58">
              No bets qualified under the current settings. Lower the minimum edge or EV threshold if you want
              to study a wider slate.
            </div>
          )}

          <p className="mt-5 text-xs leading-6 text-white/40">
            Research use only. Simulation outputs reflect current local odds snapshots plus calibrated model
            probabilities, and they should be treated as scenario analysis rather than certainty.
          </p>
        </div>
      </div>
    </div>
  );
}
