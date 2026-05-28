from __future__ import annotations

import hashlib


def generate_text_hash(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def generate_case_id(raw_text: str) -> str:
    text_hash = generate_text_hash(raw_text)
    return f"case_{text_hash[:12]}"


def generate_section_id(case_id: str, section_index: int) -> str:
    if section_index < 1:
        raise ValueError("section_index must be greater than or equal to 1")
    return f"{case_id}_sec_{section_index:03d}"
