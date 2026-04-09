import type { SupportedFight } from "@/types/api";

export interface QualifiedBet {
  id: string;
  fightLabel: string;
  pick: string;
  pickProbability: number;
  marketProbability: number | null;
  edge: number;
  decimalOdds: number;
  fullKellyFraction: number;
  stakeDollars: number;
  expectedValue: number;
}

export interface BettingSettings {
  bankroll: number;
  kellyMultiplier: number;
  minEdge: number;
  minExpectedValue: number;
  simulations: number;
}

export interface SimulationSummary {
  median: number;
  p5: number;
  p95: number;
  chanceToGrow: number;
  mean: number;
  stdDev: number;
  drawdownOverHalf: number;
  totalStake: number;
}

export function expectedValuePerDollar(probability: number, decimalOdds: number) {
  if (!Number.isFinite(probability) || !Number.isFinite(decimalOdds) || decimalOdds <= 1) {
    return Number.NaN;
  }
  return probability * decimalOdds - 1;
}

export function qualifyBets(
  fights: SupportedFight[],
  settings: BettingSettings,
): QualifiedBet[] {
  return fights
    .filter((fight) => fight.odds_available && fight.best_odds_decimal && fight.edge_pick !== null)
    .map((fight) => {
      const decimalOdds = fight.best_odds_decimal as number;
      const expectedValue = expectedValuePerDollar(fight.pick_probability, decimalOdds);
      const stakeDollars =
        settings.bankroll * Math.max(0, fight.kelly_fraction ?? 0) * settings.kellyMultiplier;

      return {
        id: fight.id,
        fightLabel: `${fight.fighter_a} vs ${fight.fighter_b}`,
        pick: fight.favored_fighter,
        pickProbability: fight.pick_probability,
        marketProbability: fight.market_probability_pick,
        edge: fight.edge_pick ?? 0,
        decimalOdds,
        fullKellyFraction: fight.kelly_fraction ?? 0,
        stakeDollars,
        expectedValue,
      };
    })
    .filter(
      (bet) =>
        bet.edge >= settings.minEdge &&
        bet.expectedValue >= settings.minExpectedValue &&
        bet.fullKellyFraction > 0,
    )
    .sort((a, b) => b.expectedValue - a.expectedValue);
}

function percentile(sortedValues: number[], q: number) {
  if (!sortedValues.length) {
    return 0;
  }
  const index = (sortedValues.length - 1) * q;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) {
    return sortedValues[lower];
  }
  const weight = index - lower;
  return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
}

export function simulateBankroll(
  bets: QualifiedBet[],
  settings: BettingSettings,
): SimulationSummary | null {
  if (!bets.length || settings.simulations <= 0) {
    return null;
  }

  const outcomes: number[] = [];
  let chanceToGrow = 0;
  let drawdownOverHalf = 0;

  for (let i = 0; i < settings.simulations; i += 1) {
    let bankroll = settings.bankroll;
    let hitDrawdown = false;

    for (const bet of bets) {
      const stake = bankroll * bet.fullKellyFraction * settings.kellyMultiplier;
      if (stake <= 0) {
        continue;
      }

      if (Math.random() < bet.pickProbability) {
        bankroll += stake * (bet.decimalOdds - 1);
      } else {
        bankroll -= stake;
      }

      if (!hitDrawdown && bankroll <= settings.bankroll * 0.5) {
        hitDrawdown = true;
      }

      if (bankroll <= 0) {
        bankroll = 0;
        break;
      }
    }

    if (hitDrawdown) {
      drawdownOverHalf += 1;
    }
    if (bankroll > settings.bankroll) {
      chanceToGrow += 1;
    }
    outcomes.push(bankroll);
  }

  outcomes.sort((a, b) => a - b);
  const mean = outcomes.reduce((sum, value) => sum + value, 0) / outcomes.length;
  const variance =
    outcomes.reduce((sum, value) => sum + (value - mean) ** 2, 0) / outcomes.length;

  return {
    median: percentile(outcomes, 0.5),
    p5: percentile(outcomes, 0.05),
    p95: percentile(outcomes, 0.95),
    chanceToGrow: chanceToGrow / outcomes.length,
    mean,
    stdDev: Math.sqrt(variance),
    drawdownOverHalf: drawdownOverHalf / outcomes.length,
    totalStake: bets.reduce((sum, bet) => sum + bet.stakeDollars, 0),
  };
}
