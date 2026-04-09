"use client";

import { useEffect, useMemo, useState } from "react";
import { BellRing, Mail, Trash2 } from "lucide-react";

import { formatPercent } from "@/lib/api";
import type { SupportedFight } from "@/types/api";

type AlertType = "price" | "edge";

type AlertRule = {
  id: string;
  email: string;
  fightId: string;
  fightLabel: string;
  pick: string;
  type: AlertType;
  threshold: number;
  sportsbook: string;
  createdAt: string;
};

const STORAGE_KEY = "octagon-intel-alerts-v1";

function formatCurrencyLikeOdds(value: number) {
  return value.toFixed(2);
}

function alertStatus(rule: AlertRule, fight: SupportedFight | undefined) {
  if (!fight) {
    return {
      label: "Missing Fight",
      tone: "muted",
      detail: "This matchup is no longer present in the current local event payload.",
    };
  }

  if (rule.type === "price") {
    const current = fight.best_odds_decimal;
    if (current !== null && current >= rule.threshold) {
      return {
        label: "Triggered",
        tone: "positive",
        detail: `Current best line ${current.toFixed(2)} has reached your target.`,
      };
    }
    return {
      label: "Monitoring",
      tone: "neutral",
      detail: `Current best line is ${current ? current.toFixed(2) : "N/A"} against your ${rule.threshold.toFixed(2)} target.`,
    };
  }

  const currentEdge = fight.edge_pick;
  if (currentEdge !== null && currentEdge >= rule.threshold) {
    return {
      label: "Triggered",
      tone: "positive",
      detail: `Current model edge ${formatPercent(currentEdge, 1)} has reached your trigger.`,
    };
  }
  return {
    label: "Monitoring",
    tone: "neutral",
    detail: `Current edge is ${formatPercent(currentEdge, 1)} against your ${formatPercent(rule.threshold, 1)} trigger.`,
  };
}

function toneClasses(tone: string) {
  if (tone === "positive") {
    return "border-emerald-300/20 bg-emerald-300/[0.10] text-emerald-100";
  }
  if (tone === "muted") {
    return "border-white/10 bg-white/[0.04] text-white/70";
  }
  return "border-cyan-300/20 bg-cyan-300/[0.10] text-cyan-100";
}

export function AlertsStudio({
  eventId,
  fights,
}: {
  eventId: string;
  fights: SupportedFight[];
}) {
  const [alerts, setAlerts] = useState<AlertRule[]>([]);
  const [email, setEmail] = useState("");
  const [fightId, setFightId] = useState(fights[0]?.id ?? "");
  const [type, setType] = useState<AlertType>("price");
  const [threshold, setThreshold] = useState("2.00");
  const [sportsbook, setSportsbook] = useState("Any sportsbook");

  const fightsById = useMemo(() => Object.fromEntries(fights.map((fight) => [fight.id, fight])), [fights]);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return;
      }
      const parsed = JSON.parse(raw) as AlertRule[];
      setAlerts(parsed.filter((rule) => rule.id.startsWith(`${eventId}:`)));
    } catch {
      setAlerts([]);
    }
  }, [eventId]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      const existing = raw ? (JSON.parse(raw) as AlertRule[]) : [];
      const scoped = existing.filter((rule) => !rule.id.startsWith(`${eventId}:`));
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify([...scoped, ...alerts]));
    } catch {
      return;
    }
  }, [alerts, eventId]);

  function createAlert() {
    const selectedFight = fightsById[fightId];
    const parsedThreshold = Number(threshold);
    if (!selectedFight || !email.trim() || Number.isNaN(parsedThreshold)) {
      return;
    }

    const nextRule: AlertRule = {
      id: `${eventId}:${selectedFight.id}:${Date.now()}`,
      email: email.trim(),
      fightId: selectedFight.id,
      fightLabel: `${selectedFight.fighter_a} vs ${selectedFight.fighter_b}`,
      pick: selectedFight.favored_fighter,
      type,
      threshold: parsedThreshold,
      sportsbook: sportsbook.trim() || "Any sportsbook",
      createdAt: new Date().toISOString(),
    };

    setAlerts((current) => [nextRule, ...current]);
    setThreshold(type === "price" ? "2.00" : "0.05");
  }

  function removeAlert(id: string) {
    setAlerts((current) => current.filter((rule) => rule.id !== id));
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
      <div className="rounded-[32px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.03))] p-6 shadow-halo">
        <p className="text-xs font-semibold uppercase tracking-[0.26em] text-cyan-200/72">
          Email Notifications
        </p>
        <h3 className="mt-3 text-2xl font-semibold tracking-[-0.04em] text-white">
          Set price and edge alerts.
        </h3>
        <p className="mt-3 text-sm leading-7 text-white/58">
          Stage alert rules for this event with your email target, preferred sportsbook label, and a trigger
          threshold. Rules persist in this browser so the product can model the alert workflow cleanly while
          delivery infrastructure is wired separately.
        </p>

        <div className="mt-6 space-y-4">
          <div className="space-y-2">
            <label className="text-xs uppercase tracking-[0.16em] text-white/40">Email</label>
            <input
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
              className="w-full rounded-[18px] border border-white/10 bg-black/20 px-4 py-3 text-white outline-none placeholder:text-white/32"
            />
          </div>

          <div className="space-y-2">
            <label className="text-xs uppercase tracking-[0.16em] text-white/40">Fight</label>
            <select
              value={fightId}
              onChange={(event) => setFightId(event.target.value)}
              className="w-full rounded-[18px] border border-white/10 bg-black/20 px-4 py-3 text-white outline-none"
            >
              {fights.map((fight) => (
                <option key={fight.id} value={fight.id}>
                  {fight.fighter_a} vs {fight.fighter_b}
                </option>
              ))}
            </select>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <label className="text-xs uppercase tracking-[0.16em] text-white/40">Trigger type</label>
              <select
                value={type}
                onChange={(event) => {
                  const nextType = event.target.value as AlertType;
                  setType(nextType);
                  setThreshold(nextType === "price" ? "2.00" : "0.05");
                }}
                className="w-full rounded-[18px] border border-white/10 bg-black/20 px-4 py-3 text-white outline-none"
              >
                <option value="price">Best odds at or above</option>
                <option value="edge">Model edge at or above</option>
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-xs uppercase tracking-[0.16em] text-white/40">
                {type === "price" ? "Target decimal odds" : "Target edge"}
              </label>
              <input
                value={threshold}
                onChange={(event) => setThreshold(event.target.value)}
                className="w-full rounded-[18px] border border-white/10 bg-black/20 px-4 py-3 text-white outline-none placeholder:text-white/32"
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-xs uppercase tracking-[0.16em] text-white/40">Sportsbook preference</label>
            <input
              value={sportsbook}
              onChange={(event) => setSportsbook(event.target.value)}
              className="w-full rounded-[18px] border border-white/10 bg-black/20 px-4 py-3 text-white outline-none placeholder:text-white/32"
            />
          </div>

          <button
            type="button"
            onClick={createAlert}
            className="inline-flex items-center gap-2 rounded-full bg-white px-5 py-3 text-sm font-medium text-[#071018] transition hover:bg-cyan-100"
          >
            <Mail className="h-4 w-4" />
            Save alert
          </button>
        </div>
      </div>

      <div className="rounded-[32px] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.03))] p-6 shadow-halo">
        <div className="flex items-center gap-3">
          <BellRing className="h-5 w-5 text-cyan-200" />
          <h3 className="text-2xl font-semibold tracking-[-0.04em] text-white">My notifications</h3>
        </div>
        <p className="mt-3 text-sm leading-7 text-white/58">
          Alert rules are evaluated against the current local odds snapshot so you can see which ones are
          already live and which ones are still waiting for price movement.
        </p>

        <div className="mt-6 space-y-4">
          {alerts.length ? (
            alerts.map((rule) => {
              const status = alertStatus(rule, fightsById[rule.fightId]);
              return (
                <div
                  key={rule.id}
                  className="rounded-[24px] border border-white/10 bg-black/20 p-5"
                >
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                    <div className="space-y-2">
                      <p className="text-sm text-white/50">{rule.fightLabel}</p>
                      <p className="text-xl font-semibold tracking-[-0.03em] text-white">{rule.pick}</p>
                      <p className="text-sm text-white/58">{rule.email}</p>
                    </div>

                    <div className="flex items-center gap-3">
                      <span className={`rounded-full border px-3 py-1 text-xs font-medium ${toneClasses(status.tone)}`}>
                        {status.label}
                      </span>
                      <button
                        type="button"
                        onClick={() => removeAlert(rule.id)}
                        className="rounded-full border border-white/10 bg-white/[0.03] p-2 text-white/58 transition hover:text-white"
                        aria-label="Delete alert"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </div>

                  <div className="mt-4 grid gap-3 sm:grid-cols-3">
                    <div className="rounded-[18px] border border-white/8 bg-white/[0.03] p-3">
                      <p className="text-[11px] uppercase tracking-[0.16em] text-white/38">Trigger</p>
                      <p className="mt-2 text-base font-medium text-white">
                        {rule.type === "price"
                          ? `${formatCurrencyLikeOdds(rule.threshold)} decimal`
                          : formatPercent(rule.threshold, 1)}
                      </p>
                    </div>
                    <div className="rounded-[18px] border border-white/8 bg-white/[0.03] p-3">
                      <p className="text-[11px] uppercase tracking-[0.16em] text-white/38">Sportsbook</p>
                      <p className="mt-2 text-base font-medium text-white">{rule.sportsbook}</p>
                    </div>
                    <div className="rounded-[18px] border border-white/8 bg-white/[0.03] p-3">
                      <p className="text-[11px] uppercase tracking-[0.16em] text-white/38">Status</p>
                      <p className="mt-2 text-sm leading-6 text-white/70">{status.detail}</p>
                    </div>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="rounded-[24px] border border-white/10 bg-white/[0.03] p-5 text-sm leading-7 text-white/58">
              No alert rules for this event yet. Add a price or edge trigger to start tracking opportunities.
            </div>
          )}
        </div>

        <p className="mt-5 text-xs leading-6 text-white/38">
          This build persists alert rules locally in the browser. Production delivery can be attached to a mail
          worker such as Resend or SMTP without changing the product surface.
        </p>
      </div>
    </div>
  );
}
