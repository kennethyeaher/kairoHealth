"""
LLM based field extraction using zero shot JSON mode prompting.

A structured prompt asks Claude to read OCR text and emit a JSON object matching
FIELD_SCHEMA. Hallucinated keys (fields not in the schema) are dropped during
postprocessing. Output is written to results/predictions_llm.json with the same
per document keys as the rule based predictions, so the evaluator can compare
them directly.

Sampling is pinned to temperature 0 and every raw response is cached to
data/llm_cache, keyed by a hash of the model name and the exact prompt. The
cache is committed with the repository, so a fresh clone can reproduce every
downstream number without an API key and without spending anything. Changing
the model or the prompt changes the key, so stale answers are never reused.

Run:
    python -m src.extract_llm              # use the cache where it has an answer
    python -m src.extract_llm --no-cache   # ignore the cache and call the API
    python -m src.extract_llm --refresh    # call the API and overwrite the cache

Outputs:
    results/predictions_llm.json    14 fields per document, written incrementally
    data/llm_cache/*.json           one raw response per document
"""

import argparse
import hashlib
import json
import os
import random
import re
import time

import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv

from config import FIELD_SCHEMA, LLM_CACHE_DIR, OCR_RESULTS_PATH, RESULTS_DIR


# API settings

MODEL = "claude-sonnet-4-5"

# Pinned to 0 so a rerun against the same OCR text gives the same answer.
# Without this the pipeline cannot be reproduced, only approximated.
TEMPERATURE = 0
MAX_TOKENS = 1000

# Statuses worth trying again. Everything else, including a bad request or a
# bad key, is a real failure and is raised immediately rather than retried.
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
MAX_ATTEMPTS = 5
FIRST_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 16.0

# Client is built on first use, not at import, so the module still loads when no
# key is present. That is what lets a cached run work without credentials.
_client = None


def get_client():
    """Return the shared Anthropic client, building it on first use.

    Returns
    client
        An authenticated Anthropic client.

    Raises
        RuntimeError when ANTHROPIC_API_KEY is not set, with a message pointing
        at the cache as the way to run without one.
    """
    global _client

    if _client is None:
        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or run with the committed cache in data/llm_cache to "
                "reproduce results without calling the API."
            )
        _client = Anthropic(api_key=api_key)

    return _client


# Prompt

# Zero shot extraction prompt. Design choices:
#   - Schema is explicit with format hints, so Claude knows the expected
#     shape rather than inventing keys.
#   - I instruct null for missing fields so the JSON is always valid
#     against the schema (the evaluator handles None gracefully).
#   - "Return ONLY the JSON object" prevents prose wrapping.
#   - I deliberately do NOT ask Claude to correct OCR errors, the
#     experiment is about how the LLM handles raw OCR, not how well
#     it can repair the input.
#
# Editing this template changes the cache key for every document, which forces a
# full paid rerun. That is intended. A cached answer should never belong to a
# prompt that no longer exists.
PROMPT_TEMPLATE = """Extracting structured data from an OCR processed pediatric triage form. The OCR text may contain character errors, smudged or misread checkboxes, and out-of-order text from the form's multi-column layout.

Return a JSON object with exactly these fields, using null for any field you cannot find with reasonable confidence:

{{
  "date": "DD/MM/YYYY format",
  "time": "HH:MM 24hour format",
  "ampm": "AM or PM",
  "name": "patient name",
  "age": "age with unit (e.g. '24 months')",
  "sex": "M or F",
  "presenting_complaint": "free text describing chief complaint",
  "triage_color": "RED, YELLOW, or GREEN, whichever is marked with X",
  "rr": "respiratory rate as integer string",
  "hr": "heart rate as integer string",
  "sat": "oxygen saturation as integer string",
  "temperature": "temperature as decimal string (such as '38.5')",
  "weight": "weight in kg as decimal string",
  "muac": "MUAC in mm as integer string"
}}

Return ONLY the JSON object, no other text, no markdown fencing.

OCR TEXT:
{ocr_text}
"""


# response cache

def cache_key(prompt):
    """Build the cache file name for one prompt.

    The key covers the model name, the sampling temperature, and the full
    prompt text, so any change to the request produces a different key.

    Parameters
    prompt
        The exact prompt string that would be sent to the API.

    Returns
    key
        A hex digest used as the cache file stem.
    """
    payload = f"{MODEL}|{TEMPERATURE}|{prompt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_cache(prompt):
    """Return the cached raw response for a prompt, or None on a miss.

    Parameters
    prompt
        The prompt whose cached response is wanted.

    Returns
    raw_response
        The stored response text, or None when nothing is cached. A cache file
        that cannot be parsed is treated as a miss rather than an error, so one
        corrupt file never blocks a run.
    """
    path = LLM_CACHE_DIR / f"{cache_key(prompt)}.json"
    if not path.exists():
        return None

    try:
        with open(path) as handle:
            return json.load(handle)["response"]
    except (json.JSONDecodeError, KeyError):
        print(f"  cache file {path.name} is unreadable, calling the API instead")
        return None


def write_cache(prompt, raw_response):
    """Store one raw response so later runs do not need the API.

    Parameters
    prompt
        The prompt that produced the response.
    raw_response
        The exact text the model returned, before any parsing.

    Returns
    None
    """
    path = LLM_CACHE_DIR / f"{cache_key(prompt)}.json"
    record = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "prompt": prompt,
        "response": raw_response,
    }
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2)


# API call

def is_retryable(error):
    """Decide whether an API error is worth trying again.

    Connection and timeout failures are always transient. Status errors are
    retried only for the codes in RETRYABLE_STATUS_CODES, so a bad key or a
    malformed request fails fast instead of being retried five times.

    Parameters
    error
        The exception raised by the Anthropic client.

    Returns
    should_retry
        True when another attempt is worthwhile.
    """
    if isinstance(error, anthropic.APIConnectionError):
        return True
    if isinstance(error, anthropic.APIStatusError):
        return error.status_code in RETRYABLE_STATUS_CODES
    return False


def call_model(prompt):
    """Send one prompt to the API, retrying transient failures with backoff.

    Waits grow from FIRST_BACKOFF_SECONDS up to MAX_BACKOFF_SECONDS, with a
    small random offset so parallel runs do not retry in lockstep.

    Parameters
    prompt
        The prompt to send.

    Returns
    raw_response
        The text of the model's first content block, stripped.

    Raises
        The original API error when it is not retryable, or when the last
        attempt fails.
    """
    backoff = FIRST_BACKOFF_SECONDS

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = get_client().messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()

        except (anthropic.APIConnectionError, anthropic.APIStatusError) as error:
            if not is_retryable(error) or attempt == MAX_ATTEMPTS:
                raise

            wait = min(backoff, MAX_BACKOFF_SECONDS) + random.uniform(0, 0.3)
            print(f"  attempt {attempt} failed ({type(error).__name__}), "
                  f"retrying in {wait:.1f}s")
            time.sleep(wait)
            backoff *= 2


# extraction

def parse_response(raw_response):
    """Turn one raw model response into a dict matching FIELD_SCHEMA.

    Parameters
    raw_response
        The text the model returned.

    Returns
    fields
        Dict of exactly the 14 schema fields. Keys outside the schema are
        dropped. A response that is not valid JSON yields all None, which the
        evaluator scores as a total miss.
    """
    # Strip markdown code fences if Claude added them despite the "no fencing"
    # instruction, observed occasionally in practice.
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_response)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {field: None for field in FIELD_SCHEMA}

    if not isinstance(parsed, dict):
        return {field: None for field in FIELD_SCHEMA}

    return {field: parsed.get(field) for field in FIELD_SCHEMA}


def extract_llm(ocr_text, use_cache=True, refresh_cache=False):
    """Extract the schema fields from one OCR text.

    Parameters
    ocr_text
        Raw Tesseract output for one document.
    use_cache
        Whether a stored response may be used instead of calling the API.
    refresh_cache
        Whether to call the API and overwrite any stored response.

    Returns
    fields, was_cached
        The 14 extracted fields, and whether the answer came from the cache.
    """
    prompt = PROMPT_TEMPLATE.format(ocr_text=ocr_text)

    if use_cache and not refresh_cache:
        cached = read_cache(prompt)
        if cached is not None:
            return parse_response(cached), True

    raw_response = call_model(prompt)
    if use_cache or refresh_cache:
        write_cache(prompt, raw_response)

    return parse_response(raw_response), False


# entry point

def parse_arguments():
    """Read the command line options for this stage.

    Returns
    arguments
        Namespace with the no_cache and refresh flags.
    """
    parser = argparse.ArgumentParser(
        description="Extract triage form fields from OCR text using Claude.")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="ignore stored responses and do not write new ones")
    parser.add_argument(
        "--refresh", action="store_true",
        help="call the API for every document and overwrite the cache")
    return parser.parse_args()


def main():
    """Extract fields from every OCR result and save predictions."""
    arguments = parse_arguments()
    use_cache = not arguments.no_cache

    with open(OCR_RESULTS_PATH) as handle:
        ocr_results = json.load(handle)

    output_path = RESULTS_DIR / "predictions_llm.json"
    predictions = {}
    cache_hits = 0
    total = len(ocr_results)

    for index, (document_id, document) in enumerate(ocr_results.items(), start=1):
        fields, was_cached = extract_llm(
            document["text"], use_cache=use_cache, refresh_cache=arguments.refresh)
        predictions[document_id] = fields
        cache_hits += was_cached

        source = "cached" if was_cached else "api"
        print(f"[{index}/{total}] {document_id} ({source})")

        # Written every document so an interrupted run keeps its work.
        with open(output_path, "w") as handle:
            json.dump(predictions, handle, indent=2)

    print(f"\nLLM predictions written to {output_path}")
    print(f"Predicted {len(predictions)} documents "
          f"x {len(FIELD_SCHEMA)} fields each")
    print(f"{cache_hits} of {total} answers came from the cache")


if __name__ == "__main__":
    main()
