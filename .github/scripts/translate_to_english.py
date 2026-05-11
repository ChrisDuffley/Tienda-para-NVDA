#!/usr/bin/env python3
"""
Translate untranslated Spanish msgid entries in an NVDA addon .po file to English.

Uses the GitHub Models API (GPT-4o-mini) authenticated via the GITHUB_TOKEN
environment variable — no external accounts or paid services required.

Usage: translate_to_english.py <path/to/nvda.po>

Environment variables:
  GITHUB_TOKEN  (required) — automatically provided in every GitHub Actions job.

Exit codes:
  0 - One or more strings were translated and saved
  1 - Error (bad args, file not found, missing token, fatal API failure)
  2 - Nothing to translate (all entries already have a msgstr)
"""
import json
import os
import sys
import time
from pathlib import Path

import polib
from openai import OpenAI

# GitHub Models OpenAI-compatible endpoint — authenticated with GITHUB_TOKEN.
GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"
MODEL = "gpt-4o-mini"

# System prompt that gives the model enough context to produce idiomatic
# NVDA addon UI strings rather than raw literal translations.
SYSTEM_PROMPT = (
    "You are a professional localiser for NVDA (NonVisual Desktop Access) screen reader add-ons. "
    "Your task is to translate Spanish UI strings into natural, idiomatic English.\n\n"
    "Rules you must follow:\n"
    "- Preserve accelerator-key prefixes exactly as-is (e.g. '&Cerrar' → '&Close', not 'To &close').\n"
    "- Preserve all Python-style format placeholders exactly (e.g. '{}', '{0}', '{name}').\n"
    "- Keep the same capitalisation style (title-case stays title-case, sentence-case stays sentence-case).\n"
    "- Do NOT add or remove trailing punctuation unless the source has it.\n"
    "- Do NOT translate proper nouns such as 'NVDA', 'NV Access', 'NVDA.ES', 'Tienda'.\n"
    "- Return ONLY the translated string — no explanation, no quotes, no extra text.\n"
    "- If the input is empty or whitespace, return exactly the same empty/whitespace string."
)

# How many strings to pack into a single API call.
# Smaller batches give faster feedback on failures; larger batches are more
# efficient.  30 is a good balance for UI strings.
BATCH_SIZE = 30

# Brief pause between batches to stay well within rate limits.
BATCH_DELAY_S = 0.5

# Pause before retrying an individual string after a batch failure.
RETRY_DELAY_S = 1.0


def _make_client(token: str) -> OpenAI:
    return OpenAI(base_url=GITHUB_MODELS_BASE_URL, api_key=token)


def _translate_batch(client: OpenAI, texts: list[str]) -> list[str]:
    """
    Send *texts* to the model as a single structured request and return the
    translations in the same order.  Each item is sent as a numbered entry so
    the model can return them in a JSON array without confusing multi-line
    strings.
    """
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    user_prompt = (
        f"Translate each of the following {len(texts)} Spanish UI string(s) to English. "
        f'Return a JSON object with a single key "translations" whose value is an array '
        f"of exactly {len(texts)} translated string(s) in the same order.\n\n"
        f"{numbered}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # Low temperature for consistent, reliable output.
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    # The model returns a JSON object; we asked for an array but the
    # json_object response_format wraps it.  Handle both shapes.
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        results = parsed
    elif isinstance(parsed, dict):
        # Common shapes: {"translations": [...]} or {"1": "...", "2": "..."}
        if "translations" in parsed:
            results = parsed["translations"]
        else:
            # Numbered-key dict — sort by key and extract values.
            results = [v for _, v in sorted(parsed.items(), key=lambda kv: int(kv[0]))]
    else:
        raise ValueError(f"Unexpected response shape: {type(parsed)}")

    if len(results) != len(texts):
        raise ValueError(
            f"Model returned {len(results)} translation(s) for {len(texts)} input(s)."
        )
    return [str(r) for r in results]


def _match_newline_envelope(msgid: str, translation: str) -> str:
    """
    msgfmt requires that if msgid begins/ends with '\\n', msgstr must too.
    The AI strips leading/trailing newlines from the text it receives, so we
    re-apply them here to avoid 'entries do not both begin with \\n' errors.
    """
    if msgid.startswith("\n") and not translation.startswith("\n"):
        translation = "\n" + translation
    if msgid.endswith("\n") and not translation.endswith("\n"):
        translation = translation + "\n"
    return translation


def translate_po_file(po_path: str, token: str) -> bool:
    """
    Read *po_path*, translate every untranslated entry from Spanish to English
    using GitHub Models, save the file, and return True if at least one entry
    was translated.
    """
    po = polib.pofile(po_path, encoding="utf-8")

    candidates = [
        entry
        for entry in po.untranslated_entries()
        if entry.msgid and not entry.obsolete and entry.msgid.strip()
    ]

    if not candidates:
        print("All entries are already translated — nothing to do.")
        return False

    print(
        f"Translating {len(candidates)} untranslated string(s) "
        f"from Spanish → English using GitHub Models ({MODEL}) …"
    )
    client = _make_client(token)
    translated_count = 0

    for batch_start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[batch_start : batch_start + BATCH_SIZE]
        texts = [e.msgid for e in batch]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(candidates) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches} ({len(texts)} strings) …")

        try:
            results = _translate_batch(client, texts)
            for entry, result in zip(batch, results):
                if result:
                    entry.msgstr = _match_newline_envelope(entry.msgid, result)
                    translated_count += 1

        except Exception as exc:
            # Batch failed — retry each string individually to minimise loss.
            print(f"  Batch {batch_num} failed ({exc!r}), retrying one-by-one …")
            for entry in batch:
                try:
                    time.sleep(RETRY_DELAY_S)
                    (result,) = _translate_batch(client, [entry.msgid])
                    if result:
                        entry.msgstr = _match_newline_envelope(entry.msgid, result)
                        translated_count += 1
                except Exception as inner:
                    print(f"    Skipping {entry.msgid[:60]!r} — {inner!r}")

        if batch_start + BATCH_SIZE < len(candidates):
            time.sleep(BATCH_DELAY_S)

    if translated_count:
        po.save(po_path)
        print(f"Saved {translated_count} new translation(s) to {po_path}")
        return True

    print("No successful translations were obtained from the model.")
    return False


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <path/to/nvda.po>")
        return 1

    po_path = sys.argv[1]
    if not Path(po_path).is_file():
        print(f"Error: file not found: {po_path}")
        return 1

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("Error: GITHUB_TOKEN environment variable is not set.")
        return 1

    try:
        changed = translate_po_file(po_path, token)
    except Exception as exc:
        print(f"Fatal error: {exc!r}")
        return 1

    return 0 if changed else 2


if __name__ == "__main__":
    sys.exit(main())
