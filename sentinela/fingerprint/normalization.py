"""Forma de comparação de termos — Documento 112.7D, seção 10.

A normalização ocorre em duas camadas:

1. **Normalização canônica** — `normalize_term()` da Scientific Taxonomy
   (112.7A), reutilizada SEM alteração. Produz o `normalized_term` registrado
   em `MatchEvidence` (ex.: "MERRA-2" → "merra-2").
2. **Forma de comparação** — derivada da forma canônica pela dobra de
   pontuação (pontuação tratada como separador), usada SOMENTE para localizar
   e comparar termos taxonômicos. Nunca é persistida.

A forma de comparação não traduz, não lematiza, não aplica stemming e não
substitui termos por inferência. Números relevantes são preservados
("merra-2" → "merra 2", nunca "merra").
"""

from __future__ import annotations

import re

from sentinela.taxonomy.models import normalize_term

_WS_RE = re.compile(r"\s+")


def comparison_form(term: str) -> str:
    """Forma de comparação: normalização canônica + dobra de pontuação.

    Tudo que não for letra ou número Unicode vira separador de token; espaços
    são colapsados. Ex.: "MERRA-2" → "merra 2", "SO₂" → "so2",
    "Espírito Santo" → "espirito santo".
    """
    canonical = normalize_term(term)
    folded = "".join(c if c.isalnum() else " " for c in canonical)
    return _WS_RE.sub(" ", folded).strip()


def comparison_tokens(term: str) -> tuple[str, ...]:
    """Tokens da forma de comparação — a unidade de fronteira de palavra."""
    form = comparison_form(term)
    return tuple(form.split(" ")) if form else ()


__all__ = ["comparison_form", "comparison_tokens", "normalize_term"]
