export type CoverageTone = "success" | "warning" | "muted";
export type ValueTone = "positive" | "neutral" | "negative" | "muted";
export type ConfidenceTone = "strong" | "medium" | "cautious";

export interface CoverageMeta {
  label: string;
  tone: CoverageTone;
  description: string;
}

export interface EventSummary {
  event_id: string;
  event_number: number | null;
  title: string;
  subtitle: string;
  date: string | null;
  venue: string | null;
  location: string | null;
  timeline: "future" | "past";
  has_card: boolean;
  has_odds: boolean;
  coverage_status: string;
  coverage: CoverageMeta;
  supported_count: number;
  unsupported_count: number;
  total_count: number;
  coverage_ratio: number;
  featured_matchup: string | null;
  odds_last_updated: string | null;
}

export interface EventsResponse {
  events: EventSummary[];
  future_events: EventSummary[];
  past_events: EventSummary[];
  featured_event: EventSummary | null;
}

export interface ConfidenceMeta {
  label: string;
  tone: ConfidenceTone;
  score: number;
}

export interface ValueState {
  label: string;
  tone: ValueTone;
}

export interface InsightChip {
  label: string;
  fighter: string;
  tone: string;
  detail: string;
}

export interface FeatureComparisonItem {
  key: string;
  label: string;
  fighter_a_value: number | null;
  fighter_b_value: number | null;
  fighter_a_display: string | null;
  fighter_b_display: string | null;
  advantage: "a" | "b" | null;
}

export interface SportsbookQuote {
  sportsbook: string;
  updated_at: string | null;
  fighter_a_price: number | null;
  fighter_b_price: number | null;
  fighter_a_decimal: number | null;
  fighter_b_decimal: number | null;
}

export interface SupportedFight {
  id: string;
  fighter_a: string;
  fighter_b: string;
  probability_a: number;
  probability_b: number;
  favored_fighter: string;
  pick_side: "a" | "b";
  pick_probability: number;
  confidence: ConfidenceMeta;
  odds_available: boolean;
  best_odds_decimal: number | null;
  market_probability_a: number | null;
  market_probability_pick: number | null;
  edge_pick: number | null;
  value_state: ValueState;
  kelly_fraction: number | null;
  insight_chips: InsightChip[];
  model_lean: string;
  feature_comparison: FeatureComparisonItem[];
  odds_list_a: number[];
  odds_list_b: number[];
  sportsbook_quotes: SportsbookQuote[];
}

export interface UnsupportedFight {
  id: string;
  fighter_a: string;
  fighter_b: string;
  status: string;
  reason_label: string;
  reason: string;
  description: string;
}

export interface EventDetailResponse {
  event: EventSummary;
  hero: {
    title: string;
    subtitle: string;
    date: string | null;
    venue: string | null;
    location: string | null;
    coverage: CoverageMeta;
    summary: string;
    odds_last_updated: string | null;
    odds_status: string;
  };
  supported_fights: SupportedFight[];
  unsupported_fights: UnsupportedFight[];
}
