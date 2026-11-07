"""Build a post-cutoff Wikipedia corpus.

Takes English Wikipedia articles *created* in a recent window (default: the
last 30 days, the most `recentchanges` keeps), plus the articles they link to
(bridge entities for multi-hop questions) and the articles those link to
(third hop). Everything is fetched as plain text through the MediaWiki API,
serially and politely.

Outputs (in --out):
  corpus.jsonl   one doc per line: id, title, text, kind (new|linked|hop3), created
  graph.json     for each doc: titles it links to that are also mentioned in its text
  manifest.json  window, counts, seed, fetch time

Runs anywhere with internet. We run it on Kaggle because Wikimedia rate-limits
the shared egress IP of our dev sandbox.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
UA = "RAG-benchmark/0.1 (https://github.com/omnaiduu/RAG; research corpus builder)"
DROP_SECTIONS = {
    "references", "external links", "see also", "notes", "further reading",
    "bibliography", "sources", "citations", "footnotes", "notes and references",
    "works cited", "gallery",
}
SKIP_TITLE = re.compile(r"^(List of|Lists of|Index of|Outline of|Timeline of)|\(disambiguation\)")

session = requests.Session()
session.headers["User-Agent"] = UA


def api(params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
    for attempt in range(8):
        try:
            r = session.get(API, params=params, timeout=60)
            if r.status_code == 429 or "maxlag" in r.text[:200]:
                raise RuntimeError(f"throttled {r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"  retry in {wait}s: {exc}", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError("API kept failing")


def new_articles(start: str, end: str, min_bytes: int) -> list[dict]:
    out, cont = [], {}
    while True:
        data = api({
            "action": "query", "list": "recentchanges", "rctype": "new",
            "rcnamespace": "0", "rcshow": "!redirect|!bot", "rcprop": "title|timestamp|sizes",
            "rclimit": "500", "rcdir": "newer", "rcstart": start, "rcend": end, **cont,
        })
        for rc in data["query"]["recentchanges"]:
            if rc.get("newlen", 0) >= min_bytes and not SKIP_TITLE.search(rc["title"]):
                out.append({"title": rc["title"], "created": rc["timestamp"]})
        if "continue" not in data:
            return out
        cont = data["continue"]
        time.sleep(0.2)


def page_info(titles: list[str]) -> dict[str, dict]:
    """Current length, disambiguation flag and redirect target, 50 titles per call."""
    info = {}
    for i in range(0, len(titles), 50):
        data = api({
            "action": "query", "prop": "info|pageprops", "ppprop": "disambiguation",
            "titles": "|".join(titles[i:i + 50]), "redirects": "1",
        })
        for p in data["query"].get("pages", []):
            if p.get("missing"):
                continue
            info[p["title"]] = {
                "length": p.get("length", 0),
                "disambig": "disambiguation" in p.get("pageprops", {}),
            }
        time.sleep(0.2)
    return info


def clean(text: str, max_words: int) -> str:
    keep, skipping = [], False
    for line in text.splitlines():
        m = re.match(r"^(=+)\s*(.*?)\s*\1\s*$", line)
        if m:
            level, name = len(m.group(1)), m.group(2).strip().lower()
            if level == 2:
                skipping = name in DROP_SECTIONS
            if not skipping:
                keep.append(f"\n## {m.group(2).strip()}" if level == 2 else f"\n### {m.group(2).strip()}")
            continue
        if not skipping and line.strip():
            keep.append(line.strip())
    words = "\n".join(keep).split(" ")
    return " ".join(words[:max_words]).strip()


def fetch_page(title: str, max_words: int) -> dict | None:
    data = api({
        "action": "query", "prop": "extracts|links", "explaintext": "1",
        "exsectionformat": "wiki", "plnamespace": "0", "pllimit": "max",
        "titles": title, "redirects": "1",
    })
    pages = data["query"].get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    p = pages[0]
    text = clean(p.get("extract", ""), max_words)
    if len(text.split()) < 120:
        return None
    links = [l["title"] for l in p.get("links", [])]
    low = text.lower()
    mentioned = [t for t in links if len(t) > 3 and not SKIP_TITLE.search(t)
                 and re.sub(r"\s*\(.*?\)$", "", t).lower() in low]
    return {"title": p["title"], "text": text, "mentioned_links": mentioned}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus_out")
    ap.add_argument("--days", type=int, default=29)
    ap.add_argument("--n-new", type=int, default=700)
    ap.add_argument("--bridges-per-doc", type=int, default=3)
    ap.add_argument("--hop3-per-bridge", type=int, default=1)
    ap.add_argument("--min-bytes", type=int, default=7000)
    ap.add_argument("--max-words", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    start = (now - dt.timedelta(days=args.days)).isoformat().replace("+00:00", "Z")
    end = now.isoformat().replace("+00:00", "Z")

    print(f"listing new articles {start} .. {end}", flush=True)
    cands = new_articles(start, end, args.min_bytes)
    print(f"  {len(cands)} candidates >= {args.min_bytes} bytes", flush=True)
    info = page_info([c["title"] for c in cands])
    cands = [c for c in cands if c["title"] in info and not info[c["title"]]["disambig"]]
    rng.shuffle(cands)

    docs: dict[str, dict] = {}
    graph: dict[str, list[str]] = {}

    def add(title: str, kind: str, created: str | None = None) -> dict | None:
        if title in docs:
            return docs[title]
        page = fetch_page(title, args.max_words)
        time.sleep(0.1)
        if page is None or page["title"] in docs:
            return docs.get(page["title"]) if page else None
        doc = {"id": f"d{len(docs):05d}", "title": page["title"], "text": page["text"],
               "kind": kind, "created": created}
        docs[page["title"]] = doc
        graph[page["title"]] = page["mentioned_links"]
        return doc

    new_titles = []
    for c in cands:
        if len(new_titles) >= args.n_new:
            break
        d = add(c["title"], "new", c["created"])
        if d and graph[d["title"]]:
            new_titles.append(d["title"])
            if len(new_titles) % 50 == 0:
                print(f"  new {len(new_titles)}/{args.n_new}  docs={len(docs)}", flush=True)
    new_set = set(new_titles)

    print("fetching bridge (hop-2) and hop-3 documents", flush=True)
    for i, t in enumerate(new_titles):
        links = graph[t]
        # Prefer links that are themselves new articles (both hops unseen).
        ordered = [l for l in links if l in new_set] + [l for l in links if l not in new_set]
        picked = 0
        for l in ordered:
            if picked >= args.bridges_per_doc:
                break
            b = add(l, "linked")
            if not b:
                continue
            picked += 1
            hop3 = [x for x in graph[b["title"]] if x != t]
            rng.shuffle(hop3)
            for x in hop3[: args.hop3_per_bridge]:
                add(x, "hop3")
        if (i + 1) % 50 == 0:
            print(f"  bridges for {i + 1}/{len(new_titles)}  docs={len(docs)}", flush=True)

    with open(out / "corpus.jsonl", "w") as f:
        for d in docs.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    (out / "graph.json").write_text(json.dumps(graph, ensure_ascii=False))
    kinds = {}
    for d in docs.values():
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    manifest = {"window_start": start, "window_end": end, "fetched_at": end,
                "n_docs": len(docs), "kinds": kinds, "args": vars(args)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
