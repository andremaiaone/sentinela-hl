"""Extração de termos do Event — Documento 112.7D, seções 7, 18 e 21.

Lê SOMENTE os campos autorizados do contrato `Event` e produz os termos
candidatos à resolução taxonômica. Dois modos (seção 18):

- **discreto** — cada valor é um termo independente (keywords, category,
  scientific_area, country, entities);
- **texto** — o campo é um texto livre onde termos taxonômicos serão
  localizados por fronteira de palavra (title, summary, evidence).

Contratos flexíveis:

- `entities` aceita lista de strings ou lista de objetos com "name" (seção
  21). Formas desconhecidas não causam crash: são ignoradas de forma explícita
  com diagnóstico técnico (log).
- `evidence` contribui somente com texto claramente identificado: string pura
  ou objeto com "text"/"excerpt" string (seção 7.3). Estruturas arbitrárias
  são ignoradas — nunca convertidas recursivamente em texto.

Campos contextuais (epistemic_status, confidence_score, occurred_at) não
geram conceitos nem são copiados para a saída (seção 7.4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sentinela.core.models import Event
from sentinela.fingerprint.models import FingerprintField

log = logging.getLogger(__name__)

#: chaves de `evidence` cujo valor string é tratado como texto identificado.
_EVIDENCE_TEXT_KEYS = ("text", "excerpt")


@dataclass(frozen=True)
class FieldTerm:
    """Um termo candidato extraído de um campo autorizado."""
    field: FingerprintField
    input_term: str          # valor original, sem qualquer normalização
    is_text: bool            # True: busca textual; False: termo discreto


def _discrete(field: FingerprintField, value: Any) -> FieldTerm | None:
    """Um valor discreto vira termo se for string não vazia."""
    if isinstance(value, str) and value.strip():
        return FieldTerm(field=field, input_term=value, is_text=False)
    return None


def _entity_terms(raw: list[Any]) -> list[FieldTerm]:
    terms: list[FieldTerm] = []
    for item in raw:
        if isinstance(item, str):
            term = _discrete(FingerprintField.ENTITY, item)
        elif isinstance(item, dict):
            term = _discrete(FingerprintField.ENTITY, item.get("name"))
            if term is None:
                log.warning("entidade sem 'name' string ignorada: %r", item)
        else:
            log.warning("entidade em forma desconhecida ignorada: %r", item)
            term = None
        if term is not None:
            terms.append(term)
    return terms


def _evidence_terms(raw: list[Any]) -> list[FieldTerm]:
    terms: list[FieldTerm] = []
    for item in raw:
        text: Any = None
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            for key in _EVIDENCE_TEXT_KEYS:
                if isinstance(item.get(key), str):
                    text = item[key]
                    break
            else:
                log.warning("evidence sem texto identificado ignorada: %r", item)
        else:
            log.warning("evidence em forma desconhecida ignorada: %r", item)
        if isinstance(text, str) and text.strip():
            terms.append(
                FieldTerm(
                    field=FingerprintField.EVIDENCE,
                    input_term=text,
                    is_text=True,
                )
            )
    return terms


def extract_terms(event: Event) -> list[FieldTerm]:
    """Extrai os termos candidatos do evento, em ordem fixa de campos."""
    terms: list[FieldTerm] = []

    if event.title.strip():
        terms.append(
            FieldTerm(
                field=FingerprintField.TITLE,
                input_term=event.title,
                is_text=True,
            )
        )

    if event.summary and event.summary.strip():
        terms.append(
            FieldTerm(
                field=FingerprintField.SUMMARY,
                input_term=event.summary,
                is_text=True,
            )
        )

    for field, value in (
        (FingerprintField.CATEGORY, event.category),
        (FingerprintField.SCIENTIFIC_AREA, event.scientific_area),
        (FingerprintField.COUNTRY, event.country),
    ):
        term = _discrete(field, value)
        if term is not None:
            terms.append(term)

    for keyword in event.keywords:
        term = _discrete(FingerprintField.KEYWORD, keyword)
        if term is not None:
            terms.append(term)

    terms.extend(_entity_terms(list(event.entities)))
    terms.extend(_evidence_terms(list(event.evidence)))
    return terms


__all__ = ["FieldTerm", "extract_terms"]
