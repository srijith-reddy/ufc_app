"""
Fighter name normalization.

This is the single canonical implementation used by:
  - core/ (inference, eligibility)
  - app.py (Streamlit)
  - scrape_cards.py
  - scrape_odds.py
  - api/main.py
  - tests/

All four entry points previously duplicated this code. Any change to
normalization logic must happen here only.
"""
from __future__ import annotations
import re
import unicodedata


def normalize_name(name: str) -> str:
    """
    Return lowercase ASCII-only normalized fighter name.

    Steps:
      1. NFKD Unicode decomposition (strips accents, ligatures, etc.)
      2. ASCII encode/decode (drops any remaining non-ASCII bytes)
      3. Lowercase
      4. Strip non-alpha, non-space characters (apostrophes, hyphens, dots)
      5. Collapse multiple spaces
    """
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def name_aliases(name: str) -> set[str]:
    """
    Generate all reasonable name variants for robust odds/profile lookup.

    Handles:
      - first/last swap (common in non-English name orderings)
      - compound surnames like "Cortes Acosta" → "cortesacosta"
      - reversed full names
    """
    base = normalize_name(name)
    parts = base.split()

    aliases: set[str] = {base}

    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]
        middle = parts[1:-1]

        aliases.add(f"{last} {first}")
        aliases.add(f"{first} {last}")

        if middle:
            compound_spaced = " ".join(middle + [last])   # e.g. "cortes acosta"
            compound_nospace = "".join(middle + [last])   # e.g. "cortesacosta"

            aliases.update({
                f"{first} {compound_spaced}",
                f"{first} {compound_nospace}",
                compound_spaced,
                compound_nospace,
                f"{compound_spaced} {first}",
                f"{compound_nospace} {first}",
            })

        aliases.add(" ".join(reversed(parts)))

    return {a.strip() for a in aliases if a.strip()}
