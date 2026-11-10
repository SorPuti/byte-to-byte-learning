import os
import urllib.request

import config

PT_BR_SOURCES = {
    33056: "Histórias Sem Data",
    53101: "A Mão e a Luva",
    54829: "Memórias Póstumas de Brás Cubas",
    55682: "Quincas Borba",
    55752: "Dom Casmurro",
    55797: "Memorial de Aires",
    56737: "Esaú e Jacó",
    57001: "Papéis Avulsos",
    57895: "Os Trabalhadores do Mar",
    59620: "Queda que as Mulheres Têm para os Tolos",
    61653: "Poesias Completas",
    67162: "Helena",
    67780: "Yaya Garcia",
    67935: "Relíquias de Casa Velha",
}

RAW_DIR = os.path.join(config.DATA_DIR, "raw")


def strip_gutenberg(text: str) -> str:
    lines = text.splitlines()
    start = next((i + 1 for i, line in enumerate(lines) if "*** START OF" in line), 0)
    end = next((i for i, line in enumerate(lines) if "*** END OF" in line), len(lines))
    return "\n".join(lines[start:end]).strip() + "\n"


def _download(book_id: int) -> str:
    path = os.path.join(RAW_DIR, f"pg{book_id}.txt")
    if not os.path.exists(path):
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        os.makedirs(RAW_DIR, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as resp:
            with open(path, "wb") as f:
                f.write(resp.read())
    with open(path, "rb") as f:
        return f.read().decode("utf-8-sig", errors="replace")


def build_corpus(out_path: str = config.CORPUS_FILE) -> str:
    parts = []
    for book_id, title in PT_BR_SOURCES.items():
        body = strip_gutenberg(_download(book_id))
        parts.append(body)
        print(f"  {title}: {len(body):,} caracteres")
    corpus = "\n\n".join(parts)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"Corpus salvo em {out_path}: {len(corpus):,} caracteres")
    return out_path


def ensure_corpus(path: str = config.CORPUS_FILE) -> str:
    if not os.path.exists(path):
        build_corpus(path)
    return path


if __name__ == "__main__":
    build_corpus()
