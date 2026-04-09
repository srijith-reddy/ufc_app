"use client";

import { useMemo, useState } from "react";
import { BarChart3, Layers3, ScanSearch } from "lucide-react";

import type { SportsbookQuote, SupportedFight } from "@/types/api";

function americanToDecimal(odds: number | null) {
  if (odds === null || odds === 0) {
    return null;
  }
  if (odds > 0) {
    return 1 + odds / 100;
  }
  return 1 + 100 / Math.abs(odds);
}

function formatAmerican(odds: number | null) {
  if (odds === null) {
    return "N/A";
  }
  return odds > 0 ? `+${odds}` : `${odds}`;
}

function bestAmerican(oddsList: number[]) {
  if (!oddsList.length) {
    return null;
  }
  return [...oddsList].sort(
    (left, right) => (americanToDecimal(right) ?? 0) - (americanToDecimal(left) ?? 0),
  )[0];
}

function quoteCount(fight: SupportedFight) {
  return fight.sportsbook_quotes.length || Math.max(fight.odds_list_a.length, fight.odds_list_b.length);
}

function highlightQuote(quote: SportsbookQuote, bestA: number | null, bestB: number | null) {
  return quote.fighter_a_price === bestA || quote.fighter_b_price === bestB;
}

export function OddsScreen({ fights }: { fights: SupportedFight[] }) {
  const [showAllQuotes, setShowAllQuotes] = useState(false);

  const fightsWithOdds = useMemo(
    () => fights.filter((fight) => fight.odds_available || fight.odds_list_a.length || fight.odds_list_b.length),
    [fights],
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 rounded-[32px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.03))] p-6 shadow-halo lg:flex-row lg:items-center lg:justify-between">
        <div className="max-w-3xl">
          <p className="text-xs font-semibold uppercase tracking-[0.26em] text-cyan-200/72">
            Odds Screen
          </p>
          <h3 className="mt-3 text-2xl font-semibold tracking-[-0.04em] text-white">
            Market view for supported fights.
          </h3>
          <p className="mt-3 text-sm leading-7 text-white/58">
            Best line, market breadth, and sportsbook-level rows when the current odds artifact includes
            bookmaker quotes. Older snapshots still render aggregated market stats cleanly.
          </p>
        </div>

        <div className="inline-flex rounded-full border border-white/10 bg-white/[0.04] p-1.5">
          <button
            type="button"
            onClick={() => setShowAllQuotes(false)}
            className={`rounded-full px-4 py-2 text-sm transition ${
              !showAllQuotes ? "bg-white text-[#071018]" : "text-white/65 hover:text-white"
            }`}
          >
            Best books
          </button>
          <button
            type="button"
            onClick={() => setShowAllQuotes(true)}
            className={`rounded-full px-4 py-2 text-sm transition ${
              showAllQuotes ? "bg-white text-[#071018]" : "text-white/65 hover:text-white"
            }`}
          >
            All quotes
          </button>
        </div>
      </div>

      <div className="space-y-5">
        {fightsWithOdds.map((fight) => {
          const bestA = bestAmerican(fight.odds_list_a);
          const bestB = bestAmerican(fight.odds_list_b);
          const quotes = showAllQuotes
            ? fight.sportsbook_quotes
            : fight.sportsbook_quotes.filter((quote) => highlightQuote(quote, bestA, bestB));

          return (
            <div
              key={`odds-${fight.id}`}
              className="overflow-hidden rounded-[30px] border border-white/10 bg-white/[0.03]"
            >
              <div className="border-b border-white/8 px-5 py-5 sm:px-6">
                <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.22em] text-white/36">
                      {fight.value_state.label}
                    </p>
                    <h4 className="mt-3 text-2xl font-semibold tracking-[-0.04em] text-white">
                      {fight.fighter_a} vs {fight.fighter_b}
                    </h4>
                    <p className="mt-3 text-sm leading-7 text-white/58">{fight.model_lean}</p>
                  </div>

                  <div className="grid gap-3 sm:grid-cols-3">
                    <div className="rounded-[20px] border border-white/8 bg-black/20 px-4 py-3">
                      <p className="text-[11px] uppercase tracking-[0.16em] text-white/36">Best pick line</p>
                      <p className="mt-2 text-xl font-semibold text-white">
                        {fight.best_odds_decimal ? fight.best_odds_decimal.toFixed(2) : "N/A"}
                      </p>
                    </div>
                    <div className="rounded-[20px] border border-white/8 bg-black/20 px-4 py-3">
                      <p className="text-[11px] uppercase tracking-[0.16em] text-white/36">Edge</p>
                      <p className="mt-2 text-xl font-semibold text-white">
                        {fight.edge_pick !== null ? `${(fight.edge_pick * 100).toFixed(1)}%` : "N/A"}
                      </p>
                    </div>
                    <div className="rounded-[20px] border border-white/8 bg-black/20 px-4 py-3">
                      <p className="text-[11px] uppercase tracking-[0.16em] text-white/36">Books / samples</p>
                      <p className="mt-2 text-xl font-semibold text-white">{quoteCount(fight)}</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid gap-4 px-5 py-5 sm:grid-cols-2 sm:px-6 xl:grid-cols-4">
                <div className="rounded-[22px] border border-white/8 bg-black/20 p-4">
                  <div className="flex items-center gap-2 text-white/45">
                    <BarChart3 className="h-4 w-4" />
                    <p className="text-[11px] uppercase tracking-[0.16em]">Best A</p>
                  </div>
                  <p className="mt-3 text-lg font-semibold text-white">{formatAmerican(bestA)}</p>
                  <p className="mt-1 text-sm text-white/52">{fight.fighter_a}</p>
                </div>
                <div className="rounded-[22px] border border-white/8 bg-black/20 p-4">
                  <div className="flex items-center gap-2 text-white/45">
                    <BarChart3 className="h-4 w-4" />
                    <p className="text-[11px] uppercase tracking-[0.16em]">Best B</p>
                  </div>
                  <p className="mt-3 text-lg font-semibold text-white">{formatAmerican(bestB)}</p>
                  <p className="mt-1 text-sm text-white/52">{fight.fighter_b}</p>
                </div>
                <div className="rounded-[22px] border border-white/8 bg-black/20 p-4">
                  <div className="flex items-center gap-2 text-white/45">
                    <Layers3 className="h-4 w-4" />
                    <p className="text-[11px] uppercase tracking-[0.16em]">Market breadth</p>
                  </div>
                  <p className="mt-3 text-lg font-semibold text-white">
                    {Math.max(fight.odds_list_a.length, fight.odds_list_b.length)} samples
                  </p>
                  <p className="mt-1 text-sm text-white/52">Across the current local snapshot.</p>
                </div>
                <div className="rounded-[22px] border border-white/8 bg-black/20 p-4">
                  <div className="flex items-center gap-2 text-white/45">
                    <ScanSearch className="h-4 w-4" />
                    <p className="text-[11px] uppercase tracking-[0.16em]">Bookmaker rows</p>
                  </div>
                  <p className="mt-3 text-lg font-semibold text-white">
                    {fight.sportsbook_quotes.length || "Legacy"}
                  </p>
                  <p className="mt-1 text-sm text-white/52">
                    {fight.sportsbook_quotes.length
                      ? "Named sportsbook quotes available."
                      : "Refresh odds to capture book-level rows in new artifacts."}
                  </p>
                </div>
              </div>

              {quotes.length ? (
                <div className="border-t border-white/8">
                  <div className="hidden grid-cols-[1.2fr_1fr_1fr] bg-white/[0.04] px-6 py-3 text-xs uppercase tracking-[0.16em] text-white/45 md:grid">
                    <span>Sportsbook</span>
                    <span>{fight.fighter_a}</span>
                    <span>{fight.fighter_b}</span>
                  </div>
                  {quotes.map((quote) => (
                    <div
                      key={`${fight.id}-${quote.sportsbook}`}
                      className="border-t border-white/8 px-6 py-4 max-md:space-y-3 md:grid md:grid-cols-[1.2fr_1fr_1fr] md:items-center"
                    >
                      <div className="flex items-center gap-3">
                        <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-xs font-medium text-white/78">
                          {quote.sportsbook}
                        </span>
                        {!showAllQuotes && highlightQuote(quote, bestA, bestB) ? (
                          <span className="text-[11px] uppercase tracking-[0.14em] text-cyan-200/74">
                            Best current row
                          </span>
                        ) : null}
                      </div>
                      <div className="flex items-center justify-between rounded-[18px] border border-white/8 bg-black/15 px-4 py-3 md:border-0 md:bg-transparent md:px-0 md:py-0">
                        <span className="text-sm text-white/48 md:hidden">{fight.fighter_a}</span>
                        <span className="text-base font-medium text-white">{formatAmerican(quote.fighter_a_price)}</span>
                      </div>
                      <div className="flex items-center justify-between rounded-[18px] border border-white/8 bg-black/15 px-4 py-3 md:border-0 md:bg-transparent md:px-0 md:py-0">
                        <span className="text-sm text-white/48 md:hidden">{fight.fighter_b}</span>
                        <span className="text-base font-medium text-white">{formatAmerican(quote.fighter_b_price)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="border-t border-white/8 px-6 py-5 text-sm leading-7 text-white/56">
                  This odds artifact carries valid market prices for the model, but not named sportsbook rows.
                  The next provider-backed refresh will populate the full bookmaker screen.
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
