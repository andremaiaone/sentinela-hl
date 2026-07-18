"""Testes de normalização e forma de comparação — Documento 112.7D, seções 10 e 40.2."""
from __future__ import annotations

from sentinela.fingerprint.normalization import comparison_form, comparison_tokens
from sentinela.taxonomy.models import normalize_term


def test_caixa():
    assert comparison_form("CALIPSO") == "calipso"


def test_acentos():
    assert comparison_form("Espírito Santo") == "espirito santo"
    assert comparison_form("Erupção") == "erupcao"


def test_espacos_colapsados():
    assert comparison_form("  mineral   dust  ") == "mineral dust"


def test_pontuacao_vira_separador():
    assert comparison_form("MERRA-2") == "merra 2"
    assert comparison_form("no-fly zone") == "no fly zone"
    assert comparison_form("dust (mineral)") == "dust mineral"


def test_numeros_preservados():
    assert comparison_form("MERRA-2") == "merra 2"
    assert comparison_form("850 hPa") == "850 hpa"
    assert comparison_form("SO₂") == "so2"


def test_termos_compostos():
    assert comparison_form("Coronal Mass Ejection") == "coronal mass ejection"
    assert comparison_tokens("coronal mass ejection") == ("coronal", "mass", "ejection")


def test_camada_canonica_da_taxonomia_intacta():
    # a forma canônica (normalize_term) NÃO dobra pontuação (seção 10).
    assert normalize_term("MERRA-2") == "merra-2"
    assert normalize_term("Espírito Santo") == "espirito santo"


def test_texto_so_pontuacao_sem_tokens():
    assert comparison_form("— —") == ""
    assert comparison_tokens("---") == ()


def test_sem_traducao_lematizacao_ou_stemming():
    # "alisios" não vira "trade winds"; plural não vira singular.
    assert comparison_form("alisios") == "alisios"
    assert comparison_form("aerosols") == "aerosols"
