"use client";

import { useState } from "react";

import type { SupportedFight } from "@/types/api";

import { FightBreakdownModal } from "./fight-breakdown-modal";
import { FightCard } from "./fight-card";

export function FightCardDeck({ fights }: { fights: SupportedFight[] }) {
  const [selectedFight, setSelectedFight] = useState<SupportedFight | null>(null);

  return (
    <>
      <div className="grid gap-6">
        {fights.map((fight) => (
          <FightCard key={fight.id} fight={fight} onSelect={setSelectedFight} />
        ))}
      </div>
      <FightBreakdownModal fight={selectedFight} onClose={() => setSelectedFight(null)} />
    </>
  );
}
