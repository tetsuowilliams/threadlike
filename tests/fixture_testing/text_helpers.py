import hashlib


def norm_text(s: str) -> str:
    # minimal normalization so small changes don't explode hashes
    return " ".join(s.lower().split())

def text_hash(s: str) -> str:
    return hashlib.sha256(norm_text(s).encode("utf-8")).hexdigest()