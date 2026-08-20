# -*- coding: utf-8 -*-
"""Round-3: authoritative verification.
- arXiv DOIs (10.48550/arXiv.*) validated via doi.org resolution (DataCite).
- truncated Nature DOIs completed via suffix probing on doi.org.
- non-arXiv/non-truncated verified via CrossRef works API.
"""
from __future__ import annotations
import json, time, urllib.request, urllib.parse

BASE = r"D:\TT paper\0811Temporal Pathway"
POOL = BASE + r"\work\refs_verified.json"
OUT = BASE + r"\work\refs_verified3.json"


def doi_org_resolves(doi):
    url = "https://doi.org/" + urllib.parse.quote(doi, safe="/()")
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "ref-check/1.0"})
        conn = urllib.request.urlopen(req, timeout=20)
        return conn.status < 400
    except Exception as ex:
        return None


def crossref_check(doi):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto=admin@example.com"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ref-check/1.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            m = json.loads(r.read().decode())["message"]
        return {"ok": True,
                "title": (m.get("title") or [""])[0],
                "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
                "container": (m.get("container-title") or [""])[0]}
    except Exception:
        return None


def main():
    d = json.load(open(POOL, encoding="utf-8"))
    fixed_nature = {
        "10.1038/s41746-025-02138-": "10.1038/s41746-025-02138-4",
        "10.1038/s41598-": None,  # probe suffixes
    }
    for r in d:
        doi = r["doi"]
        if r.get("ok") and not r.get("via") in ("title",):
            # re-verify clean entries via CrossRef
            r["ok"] = False
        # arXiv -> doi.org
        if doi.startswith("10.48550/arXiv"):
            st = doi_org_resolves(doi)
            r["ok"] = bool(st)
            r["via"] = "doi.org-arxiv" if r["ok"] else "FAIL"
            r.pop("doi_fixed", None); r.pop("xref", None)
            print(doi, "->", "OK" if r["ok"] else "FAIL", flush=True)
            continue
        # truncated nature DOIs
        if doi in fixed_nature:
            base_doi = fixed_nature[doi]
            ok = False
            if base_doi:
                st = doi_org_resolves(base_doi)
                if st:
                    r["ok"] = True; r["doi_fixed"] = base_doi; r["via"] = "doi.org"
            else:
                for suf in ["0","1","2","3","4","5","6","7","8","9"]:
                    cand = doi + suf
                    if doi_org_resolves(cand):
                        r["ok"] = True; r["doi_fixed"] = cand; r["via"] = "doi.org-suffix"; break
            r.pop("xref", None)
            print(doi, "->", "OK " + r.get("doi_fixed","") if r["ok"] else "FAIL", flush=True)
            continue
        # everything else -> CrossRef
        res = crossref_check(doi)
        r["ok"] = bool(res and res["ok"])
        if res and res["ok"]:
            r["title"] = res["title"]; r["year"] = res["year"]; r["container"] = res["container"]
            r["via"] = "crossref"
        r.pop("doi_fixed", None); r.pop("xref", None)
        print(doi, "->", "OK" if r["ok"] else "FAIL", flush=True)
        time.sleep(0.3)
    json.dump(d, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ok = [r for r in d if r["ok"]]
    print(f"VERIFIED: {len(ok)}/{len(d)}", flush=True)
    for r in d:
        if not r["ok"]:
            print("  BAD:", r["doi"], "|", r["raw"][:90], flush=True)


if __name__ == "__main__":
    main()
