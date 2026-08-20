# -*- coding: utf-8 -*-
"""Verify every candidate DOI via CrossRef works API; report invalid/unresolvable."""
from __future__ import annotations
import json, time, urllib.request, urllib.parse

BASE = r"D:\TT paper\0811Temporal Pathway"
POOL = BASE + r"\work\refs_candidates.json"
OUT = BASE + r"\work\refs_verified.json"


def check(doi, mailto="admin@example.com"):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={mailto}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ref-check/1.0 (mailto:%s)" % mailto})
        with urllib.request.urlopen(req, timeout=25) as r:
            d = json.loads(r.read().decode())
        m = d["message"]
        title = (m.get("title") or [""])[0]
        year = ""
        for k in ["published-print", "published-online", "issued"]:
            if m.get(k, {}).get("date-parts"):
                year = m[k]["date-parts"][0][0]; break
        auth = m.get("author", [])
        first = (auth[0].get("family", "") if auth else "")
        return {"ok": True, "title": title, "year": year, "first_author": first,
                "container": (m.get("container-title") or [""])[0]}
    except Exception as ex:
        return {"ok": False, "error": str(ex)[:120]}


def main():
    pool = json.load(open(POOL, encoding="utf-8"))
    seen = set(); uniq = []
    for v in pool:
        doi = (v.get("doi") or "").lower().strip()
        if not doi or doi in seen:
            continue
        seen.add(doi); uniq.append(v)
    print(f"unique DOIs to verify: {len(uniq)}", flush=True)
    results = []
    n_ok = 0
    for i, v in enumerate(uniq):
        r = check(v["doi"])
        r["doi"] = v["doi"]
        r["raw"] = v["raw"]
        r["src"] = v.get("src", "")
        if r["ok"]:
            n_ok += 1
        results.append(r)
        print(f"  [{i+1}/{len(uniq)}] {'OK ' if r['ok'] else 'BAD'} {v['doi'][:60]}", flush=True)
        time.sleep(0.35)
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"verified: {n_ok}/{len(uniq)}", flush=True)


if __name__ == "__main__":
    main()
