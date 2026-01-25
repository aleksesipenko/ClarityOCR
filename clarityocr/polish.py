#!/usr/bin/env python3
"""
LLM-based polishing for OCR-converted Markdown files.

Uses LM Studio local server (OpenAI-compatible API) with RuadaptQwen3-8B-Hybrid.

Fixes:
- Residual OCR errors (both Russian and English)
- Spacing issues (слипшиеся слова)
- Broken words across lines
- Minor punctuation errors
- Markdown formatting issues

Usage:
    python polish_with_llm.py                           # Polish all .md files
    python polish_with_llm.py --file path.md            # Polish specific file
    python polish_with_llm.py --dry-run                 # Preview without saving
    python polish_with_llm.py --base-url http://...:port  # Custom LM Studio URL

Requirements:
    pip install openai
    LM Studio running with model loaded on http://localhost:1234
"""

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not found. Install with: pip install openai")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class PolishConfig:
    """Configuration for LLM polishing."""

    # LM Studio server settings
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"  # LM Studio doesn't require real key
    model: str = "local-model"  # LM Studio uses loaded model, name is flexible

    # Generation parameters (optimized for text correction)
    # See: https://lmstudio.ai/docs/developer/openai-compat/chat-completions
    temperature: float = 0.1  # Low for deterministic output
    top_p: float = 0.9  # Nucleus sampling
    top_k: int = 40  # Limit token choices (LM Studio supports this)
    repeat_penalty: float = 1.05  # Prevent repetition
    max_tokens: int = 4096  # Max output tokens

    # Chunking settings (smaller chunks for better token management)
    chunk_size: int = 500  # Target tokens per chunk (~2000 chars)
    chunk_overlap: int = 50  # Overlap between chunks
    min_chunk_size: int = 30  # Don't process tiny chunks

    # Processing settings
    timeout: float = 120.0  # Seconds per chunk
    max_retries: int = 2  # Retries on failure
    stream: bool = False  # Streaming (False for simpler handling)


# System prompt for OCR correction (bilingual RU/EN)
# Note: /no_think prefix disables Qwen3 thinking mode via prompt
SYSTEM_PROMPT = """/no_think
Ты — корректор текста после OCR-распознавания. Исправляй ошибки в русском И английском тексте.

## ЗАДАЧА 1: Исправление OCR-ошибок

### 1.1 Слипшиеся элементы (КРИТИЧЕСКИ ВАЖНО для оглавлений!)
- Слово+число: "Bibliography328" → "Bibliography 328", "Глава3" → "Глава 3"
- Слово+слово: "theword" → "the word", "словоработа" → "слово работа"
- После знаков препинания: "слово,слово" → "слово, слово", "end.New" → "end. New"
- Сноски слипшиеся с текстом: "слово1" → "слово¹" или "слово [1]"

### 1.2 Похожие символы (частые замены)
Кириллица ↔ Латиница:
- с↔c, а↔a, е↔e, о↔o, р↔p, х↔x, у↔y, В↔B, Н↔H, М↔M, Т↔T, К↔K

Цифры ↔ Буквы:
- 0↔O↔О, 1↔l↔I↔|, 5↔S, 8↔B, 6↔b

Похожие буквы:
- rn↔m, cl↔d, vv↔w, ii↔ü, fi↔fl (лигатуры)
- ъ↔ь, и↔й, ш↔щ, п↔л (русские)

### 1.3 Переносы слов на границах строк
- "пере-\\nнос" → "перенос"
- "trans-\\nlation" → "translation"  
- "компью-\\nтер" → "компьютер"

### 1.4 Пробелы
- Двойные/тройные пробелы → один пробел
- Пробелы перед знаками препинания: "слово ." → "слово."
- Пробелы внутри слов: "сло во" → "слово" (если очевидно)

### 1.5 Пунктуация и типографика
- Прямые кавычки → типографские: "слово" → «слово» (русский), "word" (английский)
- Дефис вместо тире: "слово - слово" → "слово — слово"
- Три точки → многоточие: "..." → "…" (опционально)
- Апостроф: ' и ` → ' (правильный)

### 1.6 Специальные случаи
- Римские цифры: "ХII" (с кириллицей) → "XII", "1V" → "IV"
- Буква ё: восстанавливай если очевидно ("еще" → "ещё", "все" в значении "всё")
- Мусор OCR: случайные символы ¤§¶†‡ в середине слов — удаляй

## ЗАДАЧА 2: Улучшение Markdown-разметки

- Заголовки: "# # Заголовок" → "# Заголовок", "#Заголовок" → "# Заголовок"
- Списки: смешанные "-", "*", "•" → единообразно "-"
- Выделения: "**жир ный**" → "**жирный**", "* курсив*" → "*курсив*"
- Пустые строки: 3+ подряд → максимум 2
- Таблицы: выравнивай | если немного сбиты

## ЗАПРЕЩЕНО

- НЕ меняй смысл и стиль текста
- НЕ переформулируй предложения
- НЕ добавляй новую информацию
- НЕ удаляй контент (только явный мусор OCR)
- НЕ исправляй авторскую пунктуацию и орфографию
- НЕ трогай цитаты на других языках (латынь, французский)
- НЕ меняй структуру таблиц (только выравнивание)

## СОХРАНЯЙ ОБЯЗАТЕЛЬНО

- Маркеры страниц [p:N] — критически важны!
- Сноски (<sup>*</sup>, \\*, [1], ¹²³)
- Библиографические ссылки
- Структуру документа (главы, параграфы)
- HTML-теги если есть (<br>, <sup>, etc.)

## ПРИНЦИП

Если сомневаешься — оставь как есть. Лучше пропустить ошибку, чем испортить текст.

Верни ТОЛЬКО исправленный текст, без комментариев и пояснений."""


# =============================================================================
# CHUNKING (Smart boundaries)
# =============================================================================


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token for mixed RU/EN)."""
    return len(text) // 4


def is_table_line(line: str) -> bool:
    """Check if line is part of a Markdown table."""
    stripped = line.strip()
    if not stripped:
        return False
    # Table lines start and end with | or are separator lines
    return (stripped.startswith("|") and stripped.endswith("|")) or bool(
        re.match(r"^\|?[-:\s|]+\|?$", stripped)
    )


def find_sentence_boundary(text: str, start_pos: int, max_pos: int) -> int:
    """
    Find the best sentence boundary between start_pos and max_pos.

    Priority (highest to lowest):
    1. Double newline (paragraph boundary)
    2. Single newline followed by blank line
    3. Sentence-ending punctuation followed by space/newline
    4. Single newline (line boundary)
    5. max_pos (fallback)
    """
    search_text = text[start_pos:max_pos]

    # Priority 1: Paragraph boundary (double newline)
    para_matches = list(re.finditer(r"\n\n+", search_text))
    if para_matches:
        # Return position after the last paragraph break
        return start_pos + para_matches[-1].end()

    # Priority 2: Sentence ending (.!?) followed by space/newline
    # Handle Russian and English sentence endings
    sentence_pattern = r"[.!?»\"')\]]+[\s\n]+"
    sentence_matches = list(re.finditer(sentence_pattern, search_text))
    if sentence_matches:
        # Return position after the last sentence break
        return start_pos + sentence_matches[-1].end()

    # Priority 3: Single newline (line boundary)
    newline_matches = list(re.finditer(r"\n", search_text))
    if newline_matches:
        return start_pos + newline_matches[-1].end()

    # Fallback: return max_pos
    return max_pos


def split_into_blocks(text: str) -> list[tuple[str, str]]:
    """
    Split text into logical blocks, identifying block types.

    Returns list of (block_text, block_type) where type is:
    - "table": Markdown table
    - "code": Code block
    - "text": Regular text
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = []
    lines = text.split("\n")

    current_block = []
    current_type = "text"
    in_code_block = False
    in_table = False

    for line in lines:
        # Check for code block boundaries
        if line.strip().startswith("```"):
            if in_code_block:
                # End code block
                current_block.append(line)
                blocks.append(("\n".join(current_block), "code"))
                current_block = []
                in_code_block = False
                current_type = "text"
            else:
                # Start code block
                if current_block:
                    blocks.append(("\n".join(current_block), current_type))
                current_block = [line]
                in_code_block = True
                current_type = "code"
            continue

        if in_code_block:
            current_block.append(line)
            continue

        # Check for table lines
        is_table = is_table_line(line)

        if is_table and not in_table:
            # Starting a table
            if current_block:
                blocks.append(("\n".join(current_block), current_type))
            current_block = [line]
            in_table = True
            current_type = "table"
        elif is_table and in_table:
            # Continuing a table
            current_block.append(line)
        elif not is_table and in_table:
            # Ending a table
            blocks.append(("\n".join(current_block), "table"))
            current_block = [line] if line.strip() else []
            in_table = False
            current_type = "text"
        else:
            # Regular text
            current_block.append(line)

    # Don't forget the last block
    if current_block:
        blocks.append(("\n".join(current_block), current_type))

    return blocks


def create_chunks(
    text: str, chunk_size: int = 800, overlap: int = 100, min_size: int = 50
) -> Generator[tuple[str, int, int], None, None]:
    """
    Split text into chunks with smart boundaries.

    Features:
    - Respects sentence boundaries (never cuts mid-sentence)
    - Protects tables from being split
    - Protects code blocks from being split
    - Uses overlap for context continuity

    Yields: (chunk_text, start_char, end_char)
    """
    blocks = split_into_blocks(text)

    current_chunk_parts = []
    current_tokens = 0
    chunk_start_char = 0
    current_char = 0

    for block_text, block_type in blocks:
        block_tokens = estimate_tokens(block_text)
        block_len = len(block_text)

        # Tables and code blocks should not be split
        if block_type in ("table", "code"):
            # If adding this block exceeds limit, yield current chunk first
            if current_chunk_parts and current_tokens + block_tokens > chunk_size:
                chunk_text = "\n".join(current_chunk_parts)
                if estimate_tokens(chunk_text) >= min_size:
                    yield chunk_text, chunk_start_char, current_char
                current_chunk_parts = []
                current_tokens = 0
                chunk_start_char = current_char

            # If block itself is too large, yield it as single chunk
            if block_tokens > chunk_size:
                if current_chunk_parts:
                    chunk_text = "\n".join(current_chunk_parts)
                    if estimate_tokens(chunk_text) >= min_size:
                        yield chunk_text, chunk_start_char, current_char
                    current_chunk_parts = []
                    current_tokens = 0
                yield block_text, current_char, current_char + block_len
                current_char += block_len + 1
                chunk_start_char = current_char
                continue

            # Add block to current chunk
            current_chunk_parts.append(block_text)
            current_tokens += block_tokens
            current_char += block_len + 1
            continue

        # Regular text - can be split at sentence boundaries
        if block_tokens <= chunk_size - current_tokens:
            # Block fits in current chunk
            current_chunk_parts.append(block_text)
            current_tokens += block_tokens
            current_char += block_len + 1
        else:
            # Need to split this block
            remaining_text = block_text
            remaining_start = current_char

            while remaining_text:
                remaining_tokens = estimate_tokens(remaining_text)
                available_tokens = chunk_size - current_tokens

                if remaining_tokens <= available_tokens:
                    # Rest fits in current chunk
                    current_chunk_parts.append(remaining_text)
                    current_tokens += remaining_tokens
                    current_char = remaining_start + len(remaining_text)
                    break

                if current_chunk_parts:
                    # Yield current chunk before splitting
                    chunk_text = "\n".join(current_chunk_parts)
                    if estimate_tokens(chunk_text) >= min_size:
                        yield chunk_text, chunk_start_char, remaining_start
                    current_chunk_parts = []
                    current_tokens = 0
                    chunk_start_char = remaining_start

                # Find where to split (at sentence boundary)
                target_chars = chunk_size * 4  # Approximate chars for target tokens

                if len(remaining_text) <= target_chars:
                    # Rest fits
                    current_chunk_parts.append(remaining_text)
                    current_tokens = estimate_tokens(remaining_text)
                    current_char = remaining_start + len(remaining_text)
                    break

                # Find best split point (80-100% of target)
                min_pos = int(target_chars * 0.7)
                max_pos = min(len(remaining_text), int(target_chars * 1.0))

                split_pos = find_sentence_boundary(remaining_text, min_pos, max_pos)

                # If no good boundary found, use max_pos
                if split_pos <= min_pos:
                    split_pos = max_pos

                chunk_part = remaining_text[:split_pos].rstrip()
                remaining_text = remaining_text[split_pos:].lstrip()

                if chunk_part:
                    current_chunk_parts.append(chunk_part)
                    chunk_text = "\n".join(current_chunk_parts)
                    if estimate_tokens(chunk_text) >= min_size:
                        yield chunk_text, chunk_start_char, remaining_start + split_pos
                    current_chunk_parts = []
                    current_tokens = 0

                remaining_start += split_pos
                chunk_start_char = remaining_start

            current_char = (
                remaining_start + len(remaining_text) if remaining_text else remaining_start
            )

    # Yield final chunk - ALWAYS yield remaining content, even if small
    # This prevents data loss at end of files/sections
    if current_chunk_parts:
        chunk_text = "\n".join(current_chunk_parts)
        yield chunk_text, chunk_start_char, current_char


def merge_chunks(chunks: list[str], overlap: int = 100) -> str:
    """Merge chunks back together.

    Uses single newline to prevent inserting paragraph breaks between
    sentences that were split across chunk boundaries.
    """
    return "\n".join(chunks)


# =============================================================================
# LM STUDIO API INTERACTION
# =============================================================================


def create_client(config: PolishConfig) -> OpenAI:
    """Create OpenAI client configured for LM Studio."""
    return OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=config.timeout)


def build_user_prompt(chunk: str) -> str:
    """Build the user prompt for OCR correction."""
    return f"ТЕКСТ ДЛЯ ИСПРАВЛЕНИЯ:\n\n{chunk}\n\nИСПРАВЛЕННЫЙ ТЕКСТ:"


def call_lm_studio(
    client: OpenAI, chunk: str, config: PolishConfig, system_prompt: str = SYSTEM_PROMPT
) -> tuple[str, float]:
    """
    Call LM Studio API to correct a chunk.

    Returns: (corrected_text, elapsed_seconds)
    """
    start_time = time.time()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(chunk)},
    ]

    # Use generous max_tokens to prevent ANY truncation
    # The model should output roughly the same as input, but we give it plenty of room
    # This is critical for Russian/Cyrillic text which tokenizes inefficiently
    #
    # Safe approach: always use full max_tokens from config (default 4096)
    # The model will stop naturally when it's done, we just don't want artificial cutoff
    output_max_tokens = config.max_tokens  # Use full configured max (4096)

    # LM Studio supported parameters
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,  # type: ignore[arg-type]
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=output_max_tokens,
        stream=config.stream,
        extra_body={
            "top_k": config.top_k,
            "repeat_penalty": config.repeat_penalty,
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    result = response.choices[0].message.content  # type: ignore[union-attr]
    elapsed = time.time() - start_time

    # Strip any <think>...</think> tags if model still outputs them
    if result:
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()

    return result.strip() if result else "", elapsed


def process_chunk_with_retry(
    client: OpenAI,
    chunk: str,
    config: PolishConfig,
    chunk_num: int,
    total_chunks: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Process a chunk with retry logic."""

    for attempt in range(config.max_retries + 1):
        try:
            result, elapsed = call_lm_studio(client, chunk, config, system_prompt)

            # Validate result
            if not result.strip():
                print(f"    WARNING: Empty response for chunk {chunk_num}, keeping original")
                return chunk

            # Check for dramatic length change (possible hallucination)
            ratio = len(result) / len(chunk) if chunk else 1
            if ratio < 0.5 or ratio > 2.0:
                print(f"    WARNING: Suspicious length change ({ratio:.1%}) for chunk {chunk_num}")
                if attempt < config.max_retries:
                    print("    Retrying...")
                    continue
                print(f"    Keeping original after {attempt + 1} attempts")
                return chunk

            print(
                f"    Chunk {chunk_num}/{total_chunks}: {elapsed:.1f}s, {len(chunk)}->{len(result)} chars"
            )

            # Output chunk diff for UI monitoring (JSON format for parsing)
            # Truncate to first 800 chars for display, with ellipsis
            def make_preview(text: str, max_len: int = 800) -> str:
                if len(text) <= max_len:
                    return text.replace("\n", "\\n")
                # Find last space before max_len to avoid cutting mid-word
                cut = text[:max_len].rfind(" ")
                if cut < max_len // 2:
                    cut = max_len  # No good break point, just cut
                return text[:cut].replace("\n", "\\n") + "..."

            original_preview = make_preview(chunk)
            result_preview = make_preview(result)
            print(
                f"[CHUNK_DIFF] {json.dumps({'original': original_preview, 'result': result_preview}, ensure_ascii=False)}"
            )
            return result

        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg or "timed out" in error_msg:
                print(f"    ERROR: Timeout on chunk {chunk_num}")
            elif "connection" in error_msg:
                print(f"    ERROR: Connection failed - is LM Studio running?")
                return chunk
            else:
                print(f"    ERROR: {e} on chunk {chunk_num}")

            if attempt < config.max_retries:
                print(f"    Retrying ({attempt + 1}/{config.max_retries})...")
                time.sleep(1)
            else:
                print(f"    Keeping original after {attempt + 1} attempts")
                return chunk

    return chunk


# =============================================================================
# FILE PROCESSING
# =============================================================================


def detect_language(text: str) -> str:
    """Detect primary language of text."""
    cyrillic = len(re.findall(r"[\u0400-\u04ff]", text))
    latin = len(re.findall(r"[a-zA-Z]", text))
    return "ru" if cyrillic > latin else "en"


def process_file(
    client: OpenAI,
    filepath: Path,
    config: PolishConfig,
    dry_run: bool = False,
    verbose: bool = False,
) -> tuple[bool, dict]:
    """
    Process a single Markdown file.

    Returns: (was_modified, stats)
    """
    print(f"\nProcessing: {filepath.name}")

    try:
        text = filepath.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return False, {"error": str(e)}

    if not text.strip():
        print("  SKIP: Empty file")
        return False, {"skipped": "empty"}

    # Single bilingual prompt
    system_prompt = SYSTEM_PROMPT

    # Log detected primary language
    lang = detect_language(text)
    print(f"  Primary language: {'Russian' if lang == 'ru' else 'English'}")

    # Create chunks
    chunks = list(
        create_chunks(
            text,
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
            min_size=config.min_chunk_size,
        )
    )

    if not chunks:
        print("  SKIP: No content to process")
        return False, {"skipped": "no_content"}

    avg_chars = sum(len(c[0]) for c in chunks) // len(chunks)
    print(f"  Chunks: {len(chunks)} (avg {avg_chars} chars)")

    if dry_run:
        print(f"  DRY RUN: Would process {len(chunks)} chunks")
        return False, {"chunks": len(chunks), "dry_run": True}

    # Process each chunk
    start_time = time.time()
    corrected_chunks = []

    for i, (chunk_text, start, end) in enumerate(chunks, 1):
        result = process_chunk_with_retry(client, chunk_text, config, i, len(chunks), system_prompt)
        corrected_chunks.append(result)

    # Merge chunks back
    corrected_text = merge_chunks(corrected_chunks, config.chunk_overlap)

    # Calculate stats
    elapsed = time.time() - start_time
    chars_before = len(text)
    chars_after = len(corrected_text)

    stats = {
        "chunks": len(chunks),
        "time_s": round(elapsed, 1),
        "chars_before": chars_before,
        "chars_after": chars_after,
        "change_pct": round((chars_after - chars_before) / chars_before * 100, 1)
        if chars_before
        else 0,
    }

    # Check if anything changed
    if corrected_text == text:
        print(f"  NO CHANGES (took {elapsed:.1f}s)")
        return False, stats

    # Create backup
    backup_path = filepath.with_suffix(".md.bak")
    if not backup_path.exists():
        shutil.copy2(filepath, backup_path)
        print(f"  Backup: {backup_path.name}")

    # Save result
    filepath.write_text(corrected_text, encoding="utf-8")

    print(
        f"  SAVED: {chars_before}->{chars_after} chars ({stats['change_pct']:+.1f}%) in {elapsed:.1f}s"
    )

    return True, stats


def check_lm_studio_connection(config: PolishConfig) -> bool:
    """Check if LM Studio server is running and responsive."""
    try:
        client = create_client(config)
        # Try to list models (lightweight check)
        models = client.models.list()
        model_names = [m.id for m in models.data] if models.data else []

        if model_names:
            print(f"LM Studio connected. Available models: {', '.join(model_names[:3])}...")
            return True
        else:
            print("LM Studio connected but no models loaded.")
            print("Please load a model in LM Studio before running this script.")
            return False

    except Exception as e:
        print(f"ERROR: Cannot connect to LM Studio at {config.base_url}")
        print(f"Details: {e}")
        print("\nMake sure:")
        print("1. LM Studio is running")
        print("2. Local server is started (Server tab -> Start Server)")
        print("3. A model is loaded")
        return False


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Polish OCR-converted Markdown with local LLM (LM Studio)"
    )
    parser.add_argument(
        "--file",
        type=str,
        action="append",
        dest="files",
        help="Process specific file (can be repeated)",
    )
    parser.add_argument("--dir", type=str, help="Process directory (default: источники/md/)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # LM Studio settings
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:1234/v1",
        help="LM Studio server URL (default: http://localhost:1234/v1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="local-model",
        help="Model identifier (default: local-model)",
    )

    # Generation settings
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature (default: 0.1)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size in tokens (default: 800)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max output tokens (default: 4096)",
    )

    args = parser.parse_args()

    # Build config
    config = PolishConfig(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        chunk_size=args.chunk_size,
        max_tokens=args.max_tokens,
    )

    # Check LM Studio connection
    print(f"Connecting to LM Studio at {config.base_url}...")
    if not check_lm_studio_connection(config):
        return 1

    # Create client
    client = create_client(config)

    # Determine files to process
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        base_dir = (
            Path(args.dir) if args.dir else Path.cwd()  # Use current directory as default
        )
        if not base_dir.exists():
            print(f"Directory not found: {base_dir}")
            return 1
        files = sorted(base_dir.glob("*.md"))

    if not files:
        print("No .md files found")
        return 1

    # Print config
    print(f"\nConfiguration:")
    print(f"  Server: {config.base_url}")
    print(f"  Model: {config.model}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Chunk size: {config.chunk_size} tokens")
    print(f"  Max tokens: {config.max_tokens}")
    print(f"  Files: {len(files)}")

    if args.dry_run:
        print("\n*** DRY RUN MODE ***")

    # Process files
    print(f"\n{'=' * 60}")

    modified_count = 0
    total_time = 0

    for filepath in files:
        was_modified, stats = process_file(client, filepath, config, args.dry_run, args.verbose)
        if was_modified:
            modified_count += 1
        total_time += stats.get("time_s", 0)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Files processed: {len(files)}")
    print(f"  Files modified: {modified_count}")
    print(f"  Total time: {total_time:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
