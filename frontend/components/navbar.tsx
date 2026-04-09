"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

const navItems = [
  { href: "/", label: "Home" },
  { href: "/events", label: "Events" },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-40 border-b border-white/8 bg-[#05090e]/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4 lg:px-10">
        <Link href="/" className="group inline-flex items-center gap-3">
          <span className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-gradient-to-br from-[#ff6b35]/25 to-[#63d4ff]/20 text-sm font-semibold tracking-[0.2em] text-white shadow-glow">
            UFC
          </span>
          <div>
            <div className="text-sm font-semibold uppercase tracking-[0.28em] text-white/55">
              Octagon Intel
            </div>
            <div className="text-base font-medium tracking-[-0.03em] text-white transition group-hover:text-cyan-100">
              Prefight betting intelligence
            </div>
          </div>
        </Link>

        <nav className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] p-1.5">
          {navItems.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "rounded-full px-4 py-2 text-sm transition",
                  active
                    ? "bg-white text-[#06090d]"
                    : "text-white/65 hover:bg-white/8 hover:text-white",
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
