# -*- coding: utf-8 -*-
"""Build candidate reference list (>=40) with DOI verification via CrossRef."""
from __future__ import annotations
import re, json, time, urllib.request, urllib.parse

BASE = r"D:\TT paper\0811Temporal Pathway"
TXT1 = BASE + r"\docs\refs_text\A_Heterogeneous_Graph.txt"
TXT2 = BASE + r"\docs\refs_text\Risk_stratification_and_pathway.txt"
OUTJSON = BASE + r"\work\refs_candidates.json"


def extract_entries(path):
    t = open(path, encoding="utf-8").read()
    idx = max([m.end() for m in re.finditer(r"\n(References|REFERENCES|Reference)\s*\n", t)] or [-1])
    refs = t[idx:]
    refs = re.sub(r"===== PAGE \d+ =====", " ", refs)
    refs = re.sub(r"\s+", " ", refs)
    # entries start with "N." or "\tN.\t"
    entries = re.split(r"(?:^|\s)(?:\d{1,2}\.|\t\d{1,2}\.\t)\s*", refs)
    out = []
    for e in entries:
        e = e.strip()
        if len(e) < 20:
            continue
        m = re.search(r"doi[:/]?\s*(10\.\d{4,9}/[^\s,;]+)", e, re.I)
        doi = m.group(1).rstrip(".,") if m else None
        out.append({"raw": e[:220], "doi": doi})
    return out


def crossref_by_title(title, mailto="admin@example.com"):
    q = urllib.parse.quote(title[:120])
    url = f"https://api.crossref.org/works?query.bibliographic={q}&rows=3&mailto={mailto}"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.loads(r.read().decode())
        items = data.get("message", {}).get("items", [])
        for it in items:
            ttl = it.get("title", [""])[0] if it.get("title") else ""
            year = ""
            for k in ["published-print", "published-online", "issued"]:
                if it.get(k, {}).get("date-parts"):
                    year = it[k]["date-parts"][0][0]; break
            score = it.get("score", 0)
            if ttl and score > 10:
                return {"doi": it["DOI"], "title": ttl, "year": year, "score": round(score, 1)}
    except Exception as ex:
        return {"error": str(ex)}
    return None


def main():
    pool = {}
    for path, tag in [(TXT1, "S"), (TXT2, "P")]:
        for ent in extract_entries(path):
            key = ent["raw"][:80]
            pool.setdefault(key, {"raw": ent["raw"], "doi": ent["doi"], "src": tag})
    print(f"raw entries: {len(pool)}", flush=True)
    # fill missing DOIs via CrossRef by title
    missing = [v for v in pool.values() if not v["doi"]]
    print(f"missing DOI: {len(missing)}", flush=True)
    for i, v in enumerate(missing):
        title = re.sub(r"^(.*?\.)\s*", "", v["raw"])[:120]
        res = crossref_by_title(title)
        if res and res.get("doi"):
            v["doi"] = res["doi"]; v["xref"] = res
        time.sleep(0.4)
        if (i + 1) % 10 == 0:
            print(f"  xref {i+1}/{len(missing)} done", flush=True)
    with open(OUTJSON, "w", encoding="utf-8") as f:
        json.dump(list(pool.values()), f, ensure_ascii=False, indent=1)
    withdoi = [v for v in pool.values() if v.get("doi")]
    print(f"total={len(pool)}, with DOI={len(withdoi)}", flush=True)


if __name__ == "__main__":
    main()
