"""Grounding for the Field Guide chat (RAG-lite).

Order: (1) the prediction/species context sent by the client, (2) KB animals named in the
latest user message, (3) a live lookup for any other animal: Wikipedia search ->
Wikidata (taxon name P225, IUCN status P141) -> iNaturalist (is it an animal? observation
count, common name). Earlier user messages are scanned for KB animals when the latest one
names none (follow-up questions such as "how long do they live?").

All outbound requests use the shared ``httpx.AsyncClient`` (generic User-Agent) and a
short timeout; failures degrade to "no live sources" instead of failing the chat.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.services.knowledge import SpeciesKB

log = logging.getLogger("wildlens.chat")

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
INAT_API = "https://api.inaturalist.org/v1"

IUCN_LABELS = {
    "LC": "Least Concern", "NT": "Near Threatened", "VU": "Vulnerable", "EN": "Endangered",
    "CR": "Critically Endangered", "EW": "Extinct in the Wild", "EX": "Extinct",
    "DD": "Data Deficient", "NE": "Not Evaluated",
}
WIKIDATA_IUCN = {
    "Q211005": "LC", "Q719675": "NT", "Q278113": "VU", "Q11394": "EN", "Q219127": "CR",
    "Q239509": "EW", "Q237350": "EX", "Q3245245": "DD", "Q3350324": "NE",
    "Q96377276": "EN",  # newer "endangered" item used by some P141 statements
}
INAT_IUCN = {0: "NE", 5: "DD", 10: "LC", 20: "NT", 30: "VU", 40: "EN", 50: "CR", 60: "EW", 70: "EX"}
ANIMAL_ICONIC = {"Animalia", "Mammalia", "Aves", "Reptilia", "Amphibia", "Actinopterygii",
                 "Mollusca", "Arachnida", "Insecta"}
ICONIC_GROUP = {"Mammalia": "Mammal", "Aves": "Bird", "Reptilia": "Reptile", "Amphibia": "Amphibian",
                "Actinopterygii": "Fish", "Mollusca": "Mollusc", "Arachnida": "Arachnid", "Insecta": "Insect"}

STOPWORDS = set("""
a an the and or but if then so of to in on at by for with from about into over under as is are was were be
been being am do does did doing have has had having can could should would will shall may might must i me my
we our you your he she it its they them their this that these those there here what which who whom whose why
how when where tell explain describe give show know please thanks thank hi hello hey also more most some any
all much many very really just like than vs versus compare between eat eats eating live lives living life
lifespan habitat diet found find big large small size long fast speed weight heavy tall old age species animal
animals facts fact info information interesting fun kind kinds type types status endangered conservation it's
what's whats where's dangerous friendly pet pets baby babies called name names do don't doesn't anything something
""".split())

PROJECT_RE = re.compile(
    r"\b(wildlens|this (project|app|site|model)|your (model|training|dataset)|the model|resnet|accuracy|"
    r"dataset|classes|classifier|heatmap|grad-?cam|cam|onnx|trained|training)\b",
    re.I,
)


@dataclass
class Source:
    id: int
    title: str
    url: Optional[str]
    text: str
    kind: str  # wikipedia | inaturalist | wikidata | kb | project

    def public(self) -> Dict[str, Any]:
        return {"id": self.id, "title": self.title, "url": self.url, "kind": self.kind}


@dataclass
class Subject:
    name: str
    scientific_name: Optional[str] = None
    slug: Optional[str] = None  # set when the animal is one of the 80 KB classes
    summary: Optional[str] = None
    status_code: Optional[str] = None
    status_label: Optional[str] = None
    status_source: Optional[str] = None
    is_group: bool = False
    group_note: Optional[str] = None
    animal_group: Optional[str] = None
    taxonomy: Dict[str, Optional[str]] = field(default_factory=dict)
    observations: Optional[int] = None
    model_accuracy: Optional[float] = None
    source_ids: List[int] = field(default_factory=list)
    cite: Dict[str, List[int]] = field(default_factory=dict)  # fact -> source ids (summary/status/observations)

    def public(self) -> Dict[str, Any]:
        return {"name": self.name, "scientific_name": self.scientific_name, "slug": self.slug}


@dataclass
class Grounding:
    subjects: List[Subject] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    project: bool = False
    retrieval: str = "none"  # kb | live | none | unavailable
    query: Optional[str] = None

    def add_source(self, title: str, url: Optional[str], text: str, kind: str) -> int:
        sid = len(self.sources) + 1
        self.sources.append(Source(sid, title, url, text, kind))
        return sid


def extract_query(text: str, max_words: int = 6) -> str:
    """Keyword query for Wikipedia search: drop question words/stopwords."""
    words = re.findall(r"[A-Za-z][A-Za-z\-']*", text)
    kept = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
    return " ".join(kept[:max_words])


def _first_sentences(text: str, n: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return " ".join(parts[:n])


class _TTLCache:
    def __init__(self, ttl_s: float = 6 * 3600, max_items: int = 256):
        self.ttl, self.max = ttl_s, max_items
        self._d: Dict[str, Tuple[float, Any]] = {}

    def get(self, k: str) -> Any:
        v = self._d.get(k)
        if v and time.monotonic() - v[0] < self.ttl:
            return v[1]
        return None

    def set(self, k: str, v: Any) -> None:
        if len(self._d) >= self.max:
            self._d.pop(next(iter(self._d)))
        self._d[k] = (time.monotonic(), v)


class Retriever:
    def __init__(self, kb: SpeciesKB, http: httpx.AsyncClient, timeout_s: float = 10.0,
                 live_lookup: bool = True, model_accuracy: Optional[float] = None):
        self.kb = kb
        self.http = http
        self.timeout_s = timeout_s
        self.live_lookup = live_lookup
        self.model_accuracy = model_accuracy
        self._cache = _TTLCache()

    # ------------------------------------------------------------ public
    async def retrieve(self, messages: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None) -> Grounding:
        g = Grounding()
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        latest = user_msgs[-1] if user_msgs else ""

        entries: List[Dict[str, Any]] = []
        ctx = (context or {}).get("species_slug") or (context or {}).get("label")
        ctx_entry = self.kb.resolve(ctx) if ctx else None
        if ctx_entry:
            entries.append(ctx_entry)
        named = self.kb.match_text(latest)
        for e in named:
            if e not in entries:
                entries.append(e)

        live: Optional[Dict[str, Any]] = None
        if not named and self.live_lookup:
            q = extract_query(latest)
            if q:
                g.query = q
                try:
                    live = await asyncio.wait_for(self._live(q), timeout=self.timeout_s)
                except (asyncio.TimeoutError, httpx.HTTPError, ValueError, KeyError) as exc:
                    log.info("live lookup failed for %r: %s", q, type(exc).__name__)
                    g.retrieval = "unavailable"

        if not entries and live is None:
            for prev in reversed(user_msgs[:-1]):
                found = self.kb.match_text(prev)
                if found:
                    entries.extend(found)
                    break

        for e in entries[:2]:
            self._add_kb(g, e)
        if live is not None:
            self._add_live(g, live)
        if g.subjects:
            if live is not None:
                g.retrieval = "kb+live" if entries else "live"
            else:
                g.retrieval = "kb"
        if PROJECT_RE.search(latest):
            g.project = True
            g.add_source("WildLens project facts", None, self._project_text(), "project")
        return g

    # ------------------------------------------------------------ KB
    def _add_kb(self, g: Grounding, e: Dict[str, Any]) -> None:
        st = e.get("iucn_status") or {}
        tax = e.get("taxonomy") or {}
        ms = e.get("model_stats") or {}
        s = Subject(
            name=e.get("common_name") or e["label"], scientific_name=e.get("scientific_name"), slug=e["slug"],
            summary=e.get("summary"), status_code=st.get("code"), status_label=st.get("label"),
            status_source=st.get("source"), is_group=bool(e.get("is_group")), group_note=e.get("rank_note"),
            animal_group=e.get("animal_group"), taxonomy=tax, observations=e.get("inaturalist_observations"),
            model_accuracy=ms.get("test_accuracy"),
        )
        if e.get("summary"):
            sid = g.add_source(f"Wikipedia: {e.get('wikipedia_title') or s.name}",
                               e.get("wikipedia_url"), e["summary"], "wikipedia")
            s.source_ids.append(sid)
            s.cite["summary"] = [sid]
        facts = [f"Label in the WildLens model: {e['label']}",
                 f"Scientific name / taxon: {e.get('scientific_name')} (rank: {e.get('rank')})",
                 f"Note: {e.get('rank_note')}",
                 "Taxonomy: " + ", ".join(f"{k} {v}" for k, v in tax.items() if v),
                 f"Animal group: {e.get('animal_group')}",
                 f"IUCN Red List status: {st.get('label')}" + (f" ({st.get('code')}; source: {st.get('source')})" if st.get("code") else "")]
        if e.get("inaturalist_observations") is not None:
            facts.append(f"iNaturalist observations (taxon {(e.get('inaturalist_taxon') or {}).get('name')}): "
                         f"{e['inaturalist_observations']:,}")
        if ms.get("test_accuracy") is not None:
            facts.append(f"WildLens ResNet50 test accuracy on this class: {ms['test_accuracy'] * 100:.1f}% "
                         f"({ms.get('test_images')} test images)")
        inat_id = e.get("inaturalist_taxon_id")
        url = f"https://www.inaturalist.org/taxa/{inat_id}" if inat_id else (st.get("source_url") or None)
        sid = g.add_source(f"WildLens species data: {s.name} (Wikidata, iNaturalist, GBIF)", url,
                           "\n".join(facts), "kb")
        s.source_ids.append(sid)
        s.cite["status"] = s.cite["observations"] = s.cite["model"] = [sid]
        g.subjects.append(s)

    # ------------------------------------------------------------ live lookup
    async def _live(self, q: str) -> Optional[Dict[str, Any]]:
        cached = self._cache.get(q.lower())
        if cached is not None:
            return cached or None
        r = await self.http.get(WIKI_API, params={
            "action": "query", "generator": "search", "gsrsearch": q, "gsrlimit": 3, "gsrnamespace": 0,
            "prop": "pageprops|extracts|info", "ppprop": "wikibase_item", "exintro": 1, "explaintext": 1,
            "inprop": "url", "redirects": 1, "format": "json", "formatversion": 2,
        })
        r.raise_for_status()
        pages = sorted(((r.json().get("query") or {}).get("pages") or []), key=lambda p: p.get("index", 99))
        pages = [p for p in pages if (p.get("pageprops") or {}).get("wikibase_item")]
        if not pages:
            self._cache.set(q.lower(), {})
            return None
        ids = [p["pageprops"]["wikibase_item"] for p in pages]
        r = await self.http.get(WIKIDATA_API, params={"action": "wbgetentities", "ids": "|".join(ids),
                                                      "props": "claims", "format": "json"})
        r.raise_for_status()
        ents = r.json().get("entities") or {}
        for p in pages:
            qid = p["pageprops"]["wikibase_item"]
            claims = (ents.get(qid) or {}).get("claims") or {}
            taxon = _claim_values(claims, "P225")
            if not taxon:
                continue  # not a taxon article
            sci = taxon[0]
            inat = await self._inat(sci)
            if inat is not None and inat.get("iconic_taxon_name") not in ANIMAL_ICONIC:
                continue  # a plant, fungus, ... — out of scope
            p141 = [v.get("id") for v in _claim_values(claims, "P141", ranked=True) if isinstance(v, dict)]
            code = WIKIDATA_IUCN.get(p141[0]) if p141 else None
            status_source = "IUCN Red List via Wikidata (P141)" if code else None
            if not code and inat and inat.get("iucn"):
                code, status_source = inat["iucn"], "IUCN Red List via iNaturalist"
            result = {
                "title": p.get("title"), "url": p.get("fullurl"), "extract": p.get("extract") or "",
                "qid": qid, "scientific_name": sci, "status_code": code, "status_source": status_source,
                "inat": inat,
            }
            self._cache.set(q.lower(), result)
            return result
        self._cache.set(q.lower(), {})
        return None

    async def _inat(self, name: str) -> Optional[Dict[str, Any]]:
        try:
            r = await self.http.get(f"{INAT_API}/taxa", params={"q": name, "per_page": 5, "is_active": "true"})
            r.raise_for_status()
        except httpx.HTTPError:
            return None
        res = r.json().get("results") or []
        exact = [t for t in res if (t.get("name") or "").lower() == name.lower()] or res[:1]
        if not exact:
            return None
        t = exact[0]
        cs = t.get("conservation_status") or {}
        iucn = INAT_IUCN.get(cs.get("iucn")) if cs.get("place") is None and "iucn" in (cs.get("authority") or "").lower() else None
        return {"id": t.get("id"), "name": t.get("name"), "rank": t.get("rank"),
                "common_name": t.get("preferred_common_name"), "observations": t.get("observations_count"),
                "iconic_taxon_name": t.get("iconic_taxon_name"), "iucn": iucn}

    def _add_live(self, g: Grounding, d: Dict[str, Any]) -> None:
        inat = d.get("inat") or {}
        code = d.get("status_code")
        s = Subject(
            name=d["title"], scientific_name=d.get("scientific_name"), summary=d.get("extract"),
            status_code=code, status_label=IUCN_LABELS.get(code) if code else None,
            status_source=d.get("status_source"), animal_group=ICONIC_GROUP.get(inat.get("iconic_taxon_name") or ""),
            observations=inat.get("observations"),
        )
        if d.get("extract"):
            sid = g.add_source(f"Wikipedia: {d['title']}", d.get("url"), d["extract"], "wikipedia")
            s.source_ids.append(sid)
            s.cite["summary"] = [sid]
        facts = [f"Scientific name: {d.get('scientific_name')}"]
        if code:
            facts.append(f"IUCN Red List status: {IUCN_LABELS.get(code)} ({code}; source: {d.get('status_source')})")
        else:
            facts.append("IUCN Red List status: not found in Wikidata or iNaturalist")
        if d.get("qid"):
            sid = g.add_source(f"Wikidata: {d.get('scientific_name')}",
                               f"https://www.wikidata.org/wiki/{d['qid']}", "\n".join(facts), "wikidata")
            s.source_ids.append(sid)
            s.cite["status"] = [sid]
        if inat.get("id"):
            itxt = [f"iNaturalist taxon: {inat.get('name')} (rank {inat.get('rank')})"]
            if inat.get("common_name"):
                itxt.append(f"Common name: {inat['common_name']}")
            if inat.get("observations") is not None:
                itxt.append(f"Observations on iNaturalist: {inat['observations']:,}")
            if inat.get("iconic_taxon_name"):
                itxt.append(f"Group: {inat['iconic_taxon_name']}")
            if inat.get("iucn"):
                itxt.append(f"IUCN Red List status (global): {IUCN_LABELS.get(inat['iucn'])} ({inat['iucn']})")
            sid = g.add_source(f"iNaturalist: {inat.get('common_name') or inat.get('name')}",
                               f"https://www.inaturalist.org/taxa/{inat['id']}", "\n".join(itxt), "inaturalist")
            s.source_ids.append(sid)
            s.cite["observations"] = [sid]
            if code and "iNaturalist" in (d.get("status_source") or ""):
                s.cite["status"] = [sid]
        g.subjects.append(s)

    def _project_text(self) -> str:
        acc = self.model_accuracy
        acc_s = f"{acc * 100:.1f}%" if acc else "about 80%"
        return (
            "WildLens identifies animals in photos among 80 classes (e.g. Bear, Zebra, Moths and butterflies; "
            "several classes are groups rather than single species). The production model is a ResNet50 "
            f"fine-tuned on the project dataset, exported to ONNX; test top-1 accuracy {acc_s}, top-5 about 97%. "
            "Four models were compared (ResNet50, ConvNeXt-Tiny, EfficientNetV2-S, Swin-Tiny). "
            "Predictions come with a class activation map (CAM, equal to Grad-CAM for this architecture) showing "
            "which image regions drove the decision. The model was trained on tight crops, so full scenes or "
            "animals outside the 80 classes can be misclassified; low-confidence predictions (top-1 < 50%) are "
            "flagged. Species facts come from Wikipedia, Wikidata, iNaturalist and GBIF."
        )


def _claim_values(claims: Dict[str, Any], pid: str, ranked: bool = False) -> List[Any]:
    items = claims.get(pid) or []
    if ranked:
        pref = [c for c in items if c.get("rank") == "preferred"]
        items = pref or [c for c in items if c.get("rank") != "deprecated"]
    out = []
    for c in items:
        v = ((c.get("mainsnak") or {}).get("datavalue") or {}).get("value")
        if v is not None:
            out.append(v)
    return out


def first_sentences(text: str, n: int) -> str:
    return _first_sentences(text, n)
