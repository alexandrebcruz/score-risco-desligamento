"""Carregamento da configuração central (config.yaml).

Resolve a raiz do projeto de forma robusta (procura o config.yaml subindo na
árvore de diretórios), de modo que tanto notebooks quanto testes consigam
carregar a mesma configuração sem depender do diretório de trabalho atual.
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import yaml


def find_project_root(start: Path | None = None) -> Path:
    """Sobe na árvore de diretórios até achar config.yaml."""
    p = (start or Path(__file__)).resolve()
    for parent in [p, *p.parents]:
        if (parent / "config.yaml").exists():
            return parent
    raise FileNotFoundError("config.yaml não encontrado a partir de %s" % p)


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Lê o config.yaml e injeta caminhos absolutos resolvidos em `cfg['abs']`."""
    root = find_project_root()
    with open(root / "config.yaml", "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["root"] = root
    # Resolve cada caminho relativo para absoluto e garante que exista.
    cfg["abs"] = {}
    for key, rel in cfg["paths"].items():
        abs_path = (root / rel).resolve()
        abs_path.mkdir(parents=True, exist_ok=True)
        cfg["abs"][key] = abs_path
    return cfg


def anos_validos(cfg: dict | None = None) -> list[int]:
    """Anos a usar no cálculo das taxas, respeitando exclusão de anos atípicos."""
    cfg = cfg or load_config()
    anos = list(cfg["anos"])
    if cfg.get("excluir_anos_atipicos"):
        anos = [a for a in anos if a not in set(cfg.get("anos_atipicos", []))]
    return anos


if __name__ == "__main__":
    c = load_config()
    print("Raiz do projeto:", c["root"])
    print("Anos válidos para taxas:", anos_validos(c))
    print("Modo sintético:", c["synthetic_mode"])
