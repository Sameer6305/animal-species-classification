"""GET /api/species and GET /api/species/{slug} (pre-built knowledge base)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.deps import get_kb
from app.schemas import ErrorResponse, SpeciesDetail, SpeciesListResponse, SpeciesSummary
from app.services.knowledge import SpeciesKB

router = APIRouter(tags=["species"])


@router.get(
    "/species",
    response_model=SpeciesListResponse,
    summary="List the 80 species/classes",
    description="Atlas listing in class order. `q` searches common/scientific names and taxonomy "
    "(case-insensitive substring); `group` filters by animal group (Mammal, Bird, Reptile, Amphibian, Fish, "
    "Insect, Arachnid, Crustacean, Mollusc, Other); `status` by IUCN code (LC, NT, VU, EN, CR, EW, EX, DD, NE) "
    "or `NA` for entries without a species-level status (groups, domestic forms, hybrid).",
)
def list_species(
    q: Optional[str] = Query(None, max_length=100, description="Search text."),
    group: Optional[str] = Query(None, max_length=32, description="Animal group, e.g. `Mammal`."),
    status: Optional[str] = Query(None, max_length=8, description="IUCN code or `NA`."),
    kb: SpeciesKB = Depends(get_kb),
) -> SpeciesListResponse:
    rows = [
        SpeciesSummary(
            slug=e["slug"],
            label=e["label"],
            common_name=e.get("common_name") or e["label"],
            scientific_name=e.get("scientific_name"),
            is_group=bool(e.get("is_group")),
            animal_group=e.get("animal_group"),
            thumb_url=(e.get("image") or {}).get("thumb_url"),
            iucn_code=(e.get("iucn_status") or {}).get("code"),
            iucn_label=(e.get("iucn_status") or {}).get("label"),
            model_accuracy=(e.get("model_stats") or {}).get("test_accuracy"),
        )
        for e in kb.search(q=q, group=group, status=status)
    ]
    return SpeciesListResponse(count=len(rows), total=len(kb), species=rows)


@router.get(
    "/species/{slug}",
    response_model=SpeciesDetail,
    responses={404: {"model": ErrorResponse}},
    summary="Full knowledge-base entry for one species/class",
)
def get_species(slug: str, kb: SpeciesKB = Depends(get_kb)) -> SpeciesDetail:
    e = kb.get(slug)
    if e is None:
        raise HTTPException(status_code=404, detail=f"Unknown species {slug!r}.")
    return SpeciesDetail(**e)
