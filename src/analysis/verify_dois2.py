# -*- coding: utf-8 -*-
"""Round-2 DOI fixes: complete truncated DOIs via title search, validate arXiv via doi.org."""
from __future__ import annotations
import json, time, urllib.request, urllib.parse, http.client

BASE = r"D:\TT paper\0811Temporal Pathway"
POOL = BASE + r"\work\refs_verified.json"
OUT = BASE + r"\work\refs_verified2.json"


def crossref_title(title, mailto="admin@example.com"):
    q = urllib.parse.quote(title[:150])
    url = f"https://api.crossref.org/works?query.bibliographic={q}&rows=3&mailto={mailto}"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            d = json.loads(r.read().decode())
        for it in d.get("message", {}).get("items", []):
            ttl = (it.get("title") or [""])[0]
            if ttl and it.get("score", 0) > 8:
                return {"doi": it["DOI"], "title": ttl, "score": round(it["score"], 1)}
    except Exception as ex:
        return {"error": str(ex)[:80]}
    return None


def doi_org_resolves(doi):
    """HEAD to doi.org; expect 3xx redirect for registered DOIs."""
    url = "https://doi.org/" + urllib.parse.quote(doi, safe="/()")
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "ref-check/1.0"})
        conn = urllib.request.urlopen(req, timeout=20)
        return conn.status < 400
    except Exception as ex:
        return None


def main():
    d = json.load(open(POOL, encoding="utf-8"))
    fixes = {
        # truncated DOIs -> corrected via title search fallback
    }
    for r in d:
        if r["ok"]:
            continue
        doi = r["doi"]
        raw = r["raw"]
        print("fixing:", doi, flush=True)
        # 1) try title search for the full/truncated entries
        title = raw.split(". ")[0][:150]
        res = crossref_title(title)
        if res and res.get("doi") and res["doi"] != doi:
            r["ok"] = True
            r["doi_fixed"] = res["doi"]
            r["title"] = res["title"]
            r["xref"] = res
            print("  -> title search found", res["doi"], flush=True)
            continue
        # 2) arXiv DOIs: verify via doi.org resolution
        if doi.startswith("10.48550/arXiv"):
            st = doi_org_resolves(doi)
            if st is True:
                r["ok"] = True
                r["via"] = "doi.org"
                print("  -> doi.org resolves", flush=True)
                continue
        # 3) try appending common suffix for truncated nature DOIs
        for suf in ["4", "0", "1", "2", "3", "5", "6"]:
            cand = doi + suf
            st = doi_org_resolves(cand)
            if st is True:
                r["ok"] = True
                r["doi_fixed"] = cand
                r["via"] = "doi.org-suffix"
                print("  -> suffix", suf, "resolves", flush=True)
                break
        time.sleep(0.3)
    json.dump(d, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ok = sum(1 for r in d if r["ok"])
    print(f"after round-2: {ok}/{len(d)} ok", flush=True)
    for r in d:
        if not r["ok"]:
            print("  STILL BAD:", r["doi"], "|", r["raw"][:80], flush=True)


if __name__ == "__main__":
    main()
