# face_filter.py — Decide if photo B may be used: reject if person A appears in it.
# Usage:
#   python face_filter.py build_db --refs_dir refs/ [--db_path encodings.pkl]
#   python face_filter.py check  --photo foto.jpg [--person Name] [--tolerance 0.6] [--model hog]
#
# Exit codes: 0 = person NOT found (photo may be used), 1 = person FOUND (reject),
#             2 = usage/processing error.

import argparse
import pickle
import sys
from pathlib import Path

import face_recognition

DEFAULT_DB = Path("encodings.pkl")
DEFAULT_TOLERANCE = 0.6


def build_db(refs_dir: Path, db_path: Path, model: str = "hog") -> None:
    """Encode all reference photos: refs_dir/<person_name>/*.jpg"""
    if not refs_dir.is_dir():
        print(f"erro: diretório de referências não existe: {refs_dir}", file=sys.stderr)
        sys.exit(2)

    names, encodings = [], []
    for filepath in sorted(refs_dir.glob("*/*")):
        if filepath.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)
        locs = face_recognition.face_locations(image, model=model)
        if not locs:
            print(f"aviso: nenhuma face em {filepath}; ignorada", file=sys.stderr)
            continue
        for enc in face_recognition.face_encodings(image, locs):
            names.append(name)
            encodings.append(enc)
        print(f"{name}: {filepath.name} -> {len(locs)} face(s)")

    with db_path.open("wb") as f:
        pickle.dump({"names": names, "encodings": encodings}, f)
    print(f"banco salvo: {db_path} ({len(encodings)} encodings, {len(set(names))} pessoa(s))")


def check(photo: Path, db_path: Path, person: str | None, tolerance: float, model: str = "hog") -> None:
    if not photo.is_file():
        print(f"erro: foto não encontrada: {photo}", file=sys.stderr)
        sys.exit(2)
    if not db_path.is_file():
        print(f"erro: banco não encontrado ({db_path}). Rode primeiro: "
              f"python face_filter.py build_db --refs_dir refs/", file=sys.stderr)
        sys.exit(2)

    with db_path.open("rb") as f:
        db = pickle.load(f)
    names, known = db["names"], db["encodings"]

    image = face_recognition.load_image_file(photo)
    locs = face_recognition.face_locations(image, model=model)
    targets = face_recognition.face_encodings(image, locs)
    print(f"{photo}: {len(locs)} face(s) detectada(s)")

    # Unknown reference names only matter when --person is given.
    wanted = {i for i, n in enumerate(names) if person is None or n == person}
    if person is not None and not wanted:
        print(f"erro: pessoa '{person}' não existe no banco. Pessoas conhecidas: "
              f"{sorted(set(names))}", file=sys.stderr)
        sys.exit(2)

    hits = []
    for idx, target in zip(locs, targets):
        if not wanted:
            break
        distances = face_recognition.face_distance([known[i] for i in wanted], target)
        best = int(distances.argmin())
        if distances[best] < tolerance:
            name = names[list(wanted)[best]]
            hits.append((idx, name, float(distances[best])))
            print(f"  face em {idx}: MATCH {name} (dist={distances[best]:.4f} < {tolerance})")
        else:
            print(f"  face em {idx}: estranha (melhor dist={distances[best]:.4f})")

    if hits:
        print(f"DECISÃO: REJEITAR foto — pessoa A detectada ({hits[0][1]})")
        sys.exit(1)
    print("DECISÃO: APROVADA — pessoa A não está na foto")
    sys.exit(0)


def main() -> None:
    p = argparse.ArgumentParser(description="Filtra foto: rejeita se pessoa A estiver nela.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build_db", help="gera encodings de referência")
    b.add_argument("--refs_dir", type=Path, default=Path("refs"))
    b.add_argument("--db_path", type=Path, default=DEFAULT_DB)
    b.add_argument("--model", default="hog", choices=["hog", "cnn"])

    c = sub.add_parser("check", help="verifica uma foto")
    c.add_argument("--photo", type=Path, required=True)
    c.add_argument("--person", help="verificar somente esta pessoa; default: qualquer")
    c.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    c.add_argument("--db_path", type=Path, default=DEFAULT_DB)
    c.add_argument("--model", default="hog", choices=["hog", "cnn"])

    args = p.parse_args()
    if args.cmd == "build_db":
        build_db(args.refs_dir, args.db_path, args.model)
    else:
        check(args.photo, args.db_path, args.person, args.tolerance, args.model)


if __name__ == "__main__":
    main()