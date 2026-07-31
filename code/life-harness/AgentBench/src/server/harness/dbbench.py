"""
DBBench Harness — H0 / H_SCHEMA / H1 / H2 / H3 / H4 / H5  (v6)

H0       Task Parser            — one-time task classification per episode
H_SCHEMA Column-name mapping    — replicates DBBenchTask._sanitize_identifier
                                  (64-char truncation, dedupe) to build
                                  raw→sanitized name_map + Schema Card
H1       Session State          — per-round SQL history, error tracking,
                                  candidate answer, streaks
H2       Action Gate            — tool-call rescue + SQL auto-backtick +
                                  MySQL dialect fix + commit gate + answer
                                  normalisation (interface-boundary repair)
H3       Tool-description patch — static hints on execute_sql / commit_final_answer
H4       Post-step Monitor      — syntax / unknown-col / empty / loop +
                                  budget warn/force (runtime monitoring)
H5       Skill library + step   — BM25-ranked cold-start skills + per-step
                                  guidance (round-0 templates + lint)

Merge policy (per user feedback):
  - Budget management lives inside H4 (runtime monitoring).
  - Answer normalisation lives inside H2 commit gate (interface-boundary repair).
  - Only four top-level switches are exposed: h2 / h3 / h4 / h5.

H4 false-positive avoidance notes (per user warning):
  - Error hints only on specific MySQL error tokens (Unknown column / doesn't
    exist / near 'X' / syntax error / NULL aggregate).
  - Loop detection requires N identical normalised SQL in a row (default N=3).
  - Empty-result hint only after empty_streak >= threshold (default 2).
  - Budget force only fires with a PLAUSIBLE candidate_answer sourced from
    a successful SQL execution whose shape matches H0's answer_shape.
  - DESCRIBE hint only fires when the last error explicitly said Unknown column
    (not on any arbitrary error).
  - Commit gate will only *block* a submission once per task to avoid dead-lock.
"""

import ast
import copy
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Task / answer shape constants
# ─────────────────────────────────────────────────────────────────────────────

TASK_SELECT        = "SELECT"
TASK_INSERT        = "INSERT"
TASK_UPDATE        = "UPDATE"
TASK_DELETE        = "DELETE"
TASK_COUNTING      = "counting"
TASK_RANKING       = "ranking"
TASK_AGG_MAX       = "aggregation-MAX"
TASK_AGG_MIN       = "aggregation-MIN"
TASK_AGG_SUM       = "aggregation-SUM"
TASK_AGG_AVG       = "aggregation-AVG"
TASK_AGG_COUNT     = "aggregation-COUNT"
TASK_COMPARISON    = "comparison"
TASK_OTHER         = "other"

_MUTATION_TYPES = {TASK_INSERT, TASK_UPDATE, TASK_DELETE}
_AGGREGATION_TYPES = {TASK_AGG_MAX, TASK_AGG_MIN, TASK_AGG_SUM, TASK_AGG_AVG, TASK_AGG_COUNT}

SHAPE_SCALAR_INT   = "scalar_int"
SHAPE_SCALAR_FLOAT = "scalar_float"
SHAPE_SCALAR_STR   = "scalar_str"
SHAPE_MULTI_SINGLE = "multi_row_single_col"
SHAPE_MULTI_MULTI  = "multi_row_multi_col"
SHAPE_HASH         = "hash"  # INSERT/UPDATE/DELETE — answer ignored by evaluator


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DBBenchHarnessConfig:
    enabled: bool = False
    # Only h2/h3/h4/h5 — budget management lives in H4, answer normalisation in H2.
    h2_enabled: bool = True
    h3_enabled: bool = True
    h4_enabled: bool = True
    h5_enabled: bool = True

    # H2
    h2_repeat_sql_block_after: int = 2   # same SQL N times → auto-commit candidate
    h2_commit_block_limit: int = 1       # max block attempts for non-mutation tasks
    # Mutation tasks always block until mutation_attempted; this is the max for their
    # "commit before mutation" gate (separate from the non-mutation limit).
    h2_mutation_commit_block_limit: int = 3

    # H4
    h4_stall_window: int = 3             # turns of same SQL → loop alert
    h4_empty_threshold: int = 2          # empty SELECT rows in a row → broaden hint
    h4_budget_warn_threshold: int = 3    # remaining <= N → soft warn
    h4_budget_force_threshold: int = 2   # remaining <= N + candidate plausible → force

    # H4-E hint
    h4_hint_max_words: int = 40

    # H5
    h5_top_k: int = 2
    h5_cold_start_max_words: int = 50


# ─────────────────────────────────────────────────────────────────────────────
# Identifier sanitisation — mirrors DBBenchTask._sanitize_identifier exactly.
# If the task impl changes, keep this in sync.
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_identifier(name: str, fallback_prefix: str = "col") -> str:
    """Mirror of DBBenchTask._sanitize_identifier.

    Replace escape sequences, strip, truncate to MySQL's 64-char identifier
    limit.  Kept here so the harness doesn't import the task class.
    """
    name = (name or "").replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    name = name.strip()
    if not name:
        name = fallback_prefix
    return name[:64]


def _sanitize_columns(cols: List[Dict[str, Any]]) -> List[str]:
    seen: Dict[str, bool] = {}
    out: List[str] = []
    for col_idx, c in enumerate(cols):
        sname = sanitize_identifier(c.get("name", ""), fallback_prefix=f"col_{col_idx}")
        base = sname
        counter = 1
        while sname in seen:
            sname = f"{base[:61]}_{counter}"
            counter += 1
        seen[sname] = True
        out.append(sname)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BM25 retrieval (for H5 cold-start)
# ─────────────────────────────────────────────────────────────────────────────

def _bm25_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _skill_doc_tokens(skill: Dict[str, Any]) -> List[str]:
    parts = [skill.get("text", "")] + list(skill.get("keywords", []))
    return _bm25_tokenize(" ".join(parts))


def _bm25_scores(
    query_tokens: List[str],
    docs: List[List[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    if not docs:
        return []
    n_docs = len(docs)
    avgdl = sum(len(d) for d in docs) / n_docs if n_docs else 1
    df: Dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    scores: List[float] = []
    unique_q = list(dict.fromkeys(query_tokens))
    for doc in docs:
        dl = len(doc)
        tf: Dict[str, int] = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for t in unique_q:
            if t not in tf:
                continue
            idf = math.log(1.0 + (n_docs - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            f = tf[t]
            denom = f + k1 * (1.0 - b + b * dl / avgdl)
            score += idf * f * (k1 + 1.0) / denom
        scores.append(score)
    return scores


def _truncate_to_word_budget(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Skill library (H5)
# ─────────────────────────────────────────────────────────────────────────────

_ALL_TASK_TYPES_LIST = [
    TASK_SELECT, TASK_INSERT, TASK_UPDATE, TASK_DELETE,
    TASK_COUNTING, TASK_RANKING,
    TASK_AGG_MAX, TASK_AGG_MIN, TASK_AGG_SUM, TASK_AGG_AVG, TASK_AGG_COUNT,
    TASK_COMPARISON, TASK_OTHER,
]

DB_SKILLS: List[Dict[str, Any]] = [
    {
        "id": "backtick_identifiers",
        "task_types": list(_ALL_TASK_TYPES_LIST),
        "keywords": ["column", "name", "space", "dot", "backtick", "identifier", "quote"],
        "text": (
            "Any table or column name with a space, dot, slash, or punctuation must be "
            "wrapped in backticks. Examples: `Race Name`, `No.`, `Olympic Medal Table`. "
            "Unquoted identifiers with spaces cause MySQL syntax errors."
        ),
    },
    {
        "id": "mysql_dialect_concat",
        "task_types": [TASK_SELECT, TASK_COMPARISON, TASK_OTHER, TASK_COUNTING],
        "keywords": ["concat", "concatenate", "string", "combine", "mysql", "dialect"],
        "text": (
            "This is MySQL, not SQLite. Use `CONCAT(a, b, c)` to concatenate strings — "
            "`a || b` is SQLite syntax and MySQL will treat `||` as a boolean OR, "
            "producing wrong/NULL results."
        ),
    },
    {
        "id": "mysql_cast_numeric",
        "task_types": [TASK_AGG_SUM, TASK_AGG_AVG, TASK_AGG_MAX, TASK_AGG_MIN, TASK_RANKING, TASK_COMPARISON],
        "keywords": ["sum", "avg", "average", "max", "min", "numeric", "text", "cast", "order"],
        "text": (
            "All columns in this DB are TEXT type. Before SUM/AVG/MAX/MIN or numeric "
            "ORDER BY, cast: `CAST(`col` AS DECIMAL(20,6))` or `AS SIGNED`. Otherwise "
            "you get lexicographic order (e.g. '10' < '2') or SUM returns NULL."
        ),
    },
    {
        "id": "describe_first_on_error",
        "task_types": list(_ALL_TASK_TYPES_LIST),
        "keywords": ["describe", "schema", "columns", "unknown", "error"],
        "text": (
            "If you get 'Unknown column' or 'Table doesn't exist', run "
            "`DESCRIBE `table_name`;` or `SHOW TABLES;` first to see the real names. "
            "Column names were truncated to 64 characters on import."
        ),
    },
    {
        "id": "like_contains_word",
        "task_types": [TASK_SELECT, TASK_COUNTING, TASK_COMPARISON],
        "keywords": ["contains", "mentions", "includes", "like", "substring", "partial"],
        "text": (
            "When the question says 'contains', 'mentions', or 'includes', use "
            "`WHERE `col` LIKE '%word%'` (not `= 'word'`). For case-insensitive, wrap "
            "both sides in LOWER(): `LOWER(`col`) LIKE LOWER('%word%')`."
        ),
    },
    {
        "id": "or_any_of",
        "task_types": [TASK_SELECT, TASK_COUNTING],
        "keywords": ["any", "or", "either", "one of", "multiple"],
        "text": (
            "For 'any of A, B, C' use OR: `col LIKE '%A%' OR col LIKE '%B%' OR col LIKE '%C%'`. "
            "IN-list only works when you match exact values, not substrings."
        ),
    },
    {
        "id": "select_star_preserves_shape",
        "task_types": [TASK_SELECT],
        "keywords": ["list", "all", "show", "information", "rows", "every"],
        "text": (
            "For 'list all …' / 'show all the …' that expect multiple columns, prefer "
            "`SELECT * FROM `t` WHERE …;` and submit the DB output exactly as returned. "
            "Do not reformat rows into sentences."
        ),
    },
    {
        "id": "count_rows_basic",
        "task_types": [TASK_COUNTING, TASK_AGG_COUNT],
        "keywords": ["count", "number of", "how many"],
        "text": (
            "For counting: `SELECT COUNT(*) FROM `t` WHERE …;`. Submit only the integer "
            "number (e.g. '5'), never '5 rows' or 'count: 5'."
        ),
    },
    {
        "id": "ranking_order_by_limit",
        "task_types": [TASK_RANKING],
        "keywords": ["highest", "lowest", "top", "bottom", "first", "most", "least", "rank"],
        "text": (
            "For ranking: `SELECT `col` FROM `t` ORDER BY CAST(`num_col` AS SIGNED) DESC LIMIT 1;`. "
            "Cast the ORDER BY expression when the column is TEXT-typed numeric, otherwise "
            "you get '9' > '10' lexicographic order."
        ),
    },
    {
        "id": "rank_by_one_return_another",
        "task_types": [TASK_RANKING, TASK_SELECT],
        "keywords": ["where", "from", "which team", "which country", "name of", "first", "last", "earliest", "latest"],
        "text": (
            "When the question asks for an attribute of the top/first row (e.g., "
            "'where was the first player transferred from'), sort by the ranking/date column "
            "but SELECT the requested target column. Example: "
            "`SELECT `From` FROM `t` ORDER BY `Date` ASC LIMIT 1;` "
            "(not `SELECT `Date` ...`)."
        ),
    },
    {
        "id": "aggregation_cast_sum",
        "task_types": [TASK_AGG_SUM, TASK_AGG_AVG],
        "keywords": ["sum", "average", "total", "add", "mean"],
        "text": (
            "`SELECT SUM(CAST(`col` AS DECIMAL(20,6))) FROM `t` WHERE …;`. "
            "If SUM returns NULL, there are no matching rows — submit '0' (the "
            "evaluator maps None/null → '0')."
        ),
    },
    {
        "id": "aggregation_avg_denominator",
        "task_types": [TASK_AGG_AVG],
        "keywords": ["average", "mean", "per", "avg"],
        "text": (
            "For AVG, exclude empty / NULL values: `WHERE `col` != '' AND `col` IS NOT NULL`. "
            "Otherwise the denominator may include empty rows and the mean is wrong."
        ),
    },
    {
        "id": "insert_value_order_matches_columns",
        "task_types": [TASK_INSERT],
        "keywords": ["insert", "add", "new row"],
        "text": (
            "`INSERT INTO `t` (`col1`, `col2`, …) VALUES ('v1', 'v2', …);`. "
            "All values should be quoted as strings (all columns are TEXT). "
            "Use the exact values from the question — do not guess or compute them. "
            "Commit only AFTER the INSERT executes without error."
        ),
    },
    {
        "id": "insert_request_means_execute_insert",
        "task_types": [TASK_INSERT],
        "keywords": ["needs to be recorded", "add this incident", "add this entry", "new record", "record this", "append row"],
        "text": (
            "If the task asks to add/record a new entry, you must execute an INSERT. "
            "A SELECT-only workflow is not enough. At most do one quick duplicate check, "
            "then run the INSERT and verify the new row exists."
        ),
    },
    {
        "id": "update_needs_where",
        "task_types": [TASK_UPDATE, TASK_DELETE],
        "keywords": ["update", "change", "set", "delete", "remove"],
        "text": (
            "ALWAYS include a WHERE clause on UPDATE/DELETE. A bare `UPDATE `t` SET …` "
            "modifies every row and the grading hash will never match."
        ),
    },
    {
        "id": "update_after_preview_same_filter",
        "task_types": [TASK_UPDATE],
        "keywords": ["for all", "update", "set to", "whose", "where", "consistency", "correct", "replace"],
        "text": (
            "For UPDATE tasks, a preview SELECT is optional, but you must execute UPDATE "
            "with the same filter condition afterward. Do not commit after preview only. "
            "Flow: preview target rows -> UPDATE ... WHERE same condition -> SELECT to verify."
        ),
    },
    {
        "id": "commit_raw_db_output_for_multirow",
        "task_types": [TASK_SELECT],
        "keywords": ["multiple", "rows", "list", "table", "output"],
        "text": (
            "For multi-column SELECT (≥2 columns per row), submit each ROW as one "
            "element keeping the Python tuple repr: "
            "answers=[\"('v1', 'v2')\", \"('v3', 'v4')\"]. "
            "For single-column results, submit bare values: answers=['v1', 'v2']. "
            "Never flatten columns or reformat into sentences."
        ),
    },
    {
        "id": "commit_numeric_bare",
        "task_types": [TASK_COUNTING, TASK_AGG_MAX, TASK_AGG_MIN, TASK_AGG_SUM, TASK_AGG_AVG, TASK_AGG_COUNT, TASK_RANKING],
        "keywords": ["number", "integer", "count", "numeric", "value"],
        "text": (
            "When the answer is a number, commit ONLY the bare number: '5', not "
            "'5 games', '5 records', or 'The answer is 5'. Trailing unit words cause "
            "set-comparison mismatches."
        ),
    },
    {
        "id": "none_or_null_means_zero",
        "task_types": [TASK_AGG_SUM, TASK_AGG_AVG, TASK_AGG_MAX, TASK_AGG_MIN, TASK_AGG_COUNT, TASK_COUNTING],
        "keywords": ["none", "null", "empty", "zero", "missing"],
        "text": (
            "If SUM/AVG/MAX/MIN returns NULL/None, submit '0'. The evaluator maps "
            "None/null/'' → '0', so '0' matches when the true answer is 'no rows'."
        ),
    },
    {
        "id": "mutation_must_execute",
        "task_types": [TASK_INSERT, TASK_UPDATE, TASK_DELETE],
        "keywords": ["insert", "update", "delete", "modify", "change", "commit"],
        "text": (
            "INSERT/UPDATE/DELETE is scored by a table hash. You must actually run "
            "the mutation SQL successfully before calling commit_final_answer — "
            "committing without running the SQL fails the hash check."
        ),
    },
    {
        "id": "no_repeat_when_done",
        "task_types": list(_ALL_TASK_TYPES_LIST),
        "keywords": ["already", "have", "answer", "submit", "result", "done"],
        "text": (
            "If your last query already returned the exact answer, do not re-run the "
            "same SQL. Call commit_final_answer immediately with that value."
        ),
    },
    {
        "id": "truncation_handling",
        "task_types": list(_ALL_TASK_TYPES_LIST),
        "keywords": ["truncated", "truncation", "64", "long", "name"],
        "text": (
            "Long column names were truncated to 64 characters during table creation. "
            "Use the names shown in the schema card below — NOT the raw descriptive "
            "names from the question — as MySQL identifiers."
        ),
    },
    {
        "id": "date_format_passthrough",
        "task_types": [TASK_SELECT, TASK_RANKING, TASK_COMPARISON],
        "keywords": ["date", "time", "year", "month", "day"],
        "text": (
            "Date values are stored as strings. Match them with `=` or `LIKE` exactly "
            "as they appear (e.g. 'February 1' or '2023-01-05'). When submitting dates, "
            "use the exact string the DB returned."
        ),
    },
    {
        "id": "distinct_unique_count",
        "task_types": [TASK_COUNTING, TASK_SELECT, TASK_AGG_COUNT],
        "keywords": ["unique", "distinct", "different", "how many types", "separate", "variety"],
        "text": (
            "For 'how many unique/distinct X' use `SELECT COUNT(DISTINCT `col`) FROM `t`;`. "
            "For listing unique values use `SELECT DISTINCT `col` FROM `t` WHERE …;`. "
            "Without DISTINCT, COUNT(*) includes duplicates."
        ),
    },
    {
        "id": "group_by_having_filter",
        "task_types": [TASK_AGG_SUM, TASK_AGG_AVG, TASK_AGG_COUNT, TASK_COUNTING],
        "keywords": ["group", "having", "per", "each", "total per", "filter aggregate", "at least", "more than"],
        "text": (
            "To filter on an aggregated value use HAVING (not WHERE): "
            "`SELECT col, COUNT(*) FROM `t` GROUP BY col HAVING COUNT(*) > 2;`. "
            "WHERE filters individual rows before grouping; HAVING filters groups after. "
            "Never put an aggregate function inside a WHERE clause."
        ),
    },
    {
        "id": "scalar_expect_one_row",
        "task_types": [TASK_SELECT, TASK_COUNTING, TASK_COMPARISON],
        "keywords": ["how many", "total", "count", "number of", "single value", "overall"],
        "text": (
            "When the answer is a single total (e.g. 'how many X'), use a global aggregate "
            "WITHOUT GROUP BY: `SELECT COUNT(*) FROM `t` WHERE …;`. "
            "Adding GROUP BY returns one row per group, not a single total — you will "
            "get many rows instead of one number. Do not use this pattern when the question "
            "explicitly says 'for each', 'grouped by', or 'in each <category>'."
        ),
    },
    {
        "id": "between_range_filter",
        "task_types": [TASK_SELECT, TASK_COUNTING, TASK_COMPARISON, TASK_RANKING],
        "keywords": ["between", "range", "from", "to", "above", "below", "over", "under", "greater", "less"],
        "text": (
            "For numeric range filters: `WHERE CAST(`col` AS SIGNED) BETWEEN 10 AND 20` "
            "(inclusive on both ends). Always CAST TEXT-typed numeric columns first. "
            "For date ranges: `WHERE `col` >= '2020-01-01' AND `col` <= '2020-12-31'`."
        ),
    },
    {
        "id": "text_currency_numeric_compare",
        "task_types": [TASK_SELECT, TASK_COUNTING, TASK_COMPARISON, TASK_RANKING],
        "keywords": ["£", "$", "currency", "toll", "price", "above", "greater than", "at least"],
        "text": (
            "If numeric values contain currency symbols or commas (e.g., '£4.00', '$1,200'), "
            "strip symbols before numeric comparison: "
            "`CAST(REPLACE(REPLACE(`col`, '£', ''), ',', '') AS DECIMAL(20,6)) > 4.0`. "
            "Do not compare such values as raw strings."
        ),
    },
    {
        "id": "subquery_in_list",
        "task_types": [TASK_SELECT, TASK_COUNTING, TASK_COMPARISON],
        "keywords": ["not in", "exclude", "except", "without", "never", "no", "don't have"],
        "text": (
            "For 'rows that DON'T appear in another set' use NOT IN: "
            "`SELECT * FROM `t` WHERE `col` NOT IN (SELECT `col` FROM `t2` WHERE …);`. "
            "For 'rows that DO appear' use IN. "
            "Watch out for NULL: `NOT IN (…)` returns no rows if the subquery contains NULL."
        ),
    },
    {
        "id": "next_previous_row_lookup",
        "task_types": [TASK_SELECT, TASK_COMPARISON],
        "keywords": ["next to", "next", "after", "following", "previous", "before", "adjacent", "neighbor"],
        "text": (
            "For 'next/previous row' questions, first find the anchor row key, then query the "
            "adjacent row by an ordering column (id/rank/date). Example: "
            "`SELECT ... FROM t WHERE id = (SELECT id FROM t WHERE name='X') + 1`. "
            "Do not stop at returning the anchor row itself."
        ),
    },
    {
        "id": "sample_values_before_filter",
        "task_types": list(_ALL_TASK_TYPES_LIST),
        "keywords": ["empty", "no rows", "not found", "zero", "cannot find", "exact", "stored"],
        "text": (
            "If your WHERE clause returns zero rows, the stored value may differ from what "
            "you expect (different case, extra spaces, em-dash vs hyphen, or extra text). "
            "Run `SELECT DISTINCT `col` FROM `t` LIMIT 10;` to see actual stored values, "
            "then match them exactly in your WHERE clause."
        ),
    },
    {
        "id": "insert_all_columns_required",
        "task_types": [TASK_INSERT],
        "keywords": ["insert", "add row", "all columns", "hash", "missing"],
        "text": (
            "Your INSERT must include EVERY column in the table. "
            "The grader checks the table hash across ALL columns — if any column is "
            "omitted it defaults to NULL and the hash will not match. "
            "List all column names from the schema card in your INSERT … column list."
        ),
    },
    {
        "id": "quote_apostrophe_in_value",
        "task_types": list(_ALL_TASK_TYPES_LIST),
        "keywords": ["apostrophe", "quote", "single quote", "string", "value", "escape"],
        "text": (
            "If a WHERE/VALUES string contains a single-quote (apostrophe), escape it by "
            "doubling it: `WHERE `col` = 'O''Brien'` or use a backslash: `'O\\'Brien'`. "
            "An unescaped apostrophe breaks the SQL string and causes a syntax error."
        ),
    },
    {
        "id": "update_preview_then_mutate",
        "task_types": [TASK_UPDATE, TASK_DELETE],
        "keywords": ["update", "delete", "where", "find", "row", "target", "verify"],
        "text": (
            "Before running UPDATE/DELETE, confirm the target row exists: "
            "`SELECT * FROM `t` WHERE …;` — if it returns empty, relax the filter or "
            "check the exact stored values. Once you find the row, run the mutation "
            "using the EXACT same WHERE clause."
        ),
    },
    {
        "id": "text_column_exact_match_cast",
        "task_types": [TASK_AGG_MIN, TASK_AGG_MAX, TASK_AGG_SUM, TASK_AGG_AVG, TASK_COUNTING, TASK_COMPARISON, TASK_RANKING],
        "keywords": ["text", "numeric", "cast", "match", "filter", "null", "empty result"],
        "text": (
            "All columns are stored as TEXT. When filtering on a numeric-looking value "
            "(e.g. `WHERE score = -8`), implicit coercion usually works but fails if "
            "the stored value has extra text or a unicode minus (−). "
            "Use CAST for reliable numeric comparison: `WHERE CAST(`col` AS DECIMAL) = -8`, "
            "or check actual values first: `SELECT DISTINCT `col` FROM `t` LIMIT 10`."
        ),
    },
    {
        "id": "verify_mutation_with_select",
        "task_types": [TASK_INSERT, TASK_UPDATE, TASK_DELETE],
        "keywords": ["insert", "update", "delete", "verify", "check", "confirm", "success"],
        "text": (
            "After INSERT/UPDATE/DELETE returns `[]` (success), verify with "
            "`SELECT * FROM `tbl` WHERE …` that the change is correct. "
            "If the SELECT shows wrong data, run the corrected SQL. "
            "Only commit_final_answer once the data looks right."
        ),
    },
    {
        "id": "empty_result_wrong_filter",
        "task_types": [TASK_SELECT, TASK_COUNTING, TASK_AGG_COUNT, TASK_AGG_SUM, TASK_AGG_AVG, TASK_AGG_MAX, TASK_AGG_MIN, TASK_RANKING, TASK_COMPARISON],
        "keywords": ["empty", "no rows", "nothing", "zero", "null", "not found", "no result"],
        "text": (
            "If SELECT returns `[]`, your WHERE filter matched nothing. "
            "Check actual values: `SELECT * FROM `tbl` LIMIT 5`. "
            "Common causes: wrong column name, case mismatch, extra spaces, "
            "or numeric stored as text. Use LIKE for partial matches."
        ),
    },
    {
        "id": "aggregate_zero_check",
        "task_types": [TASK_COUNTING, TASK_AGG_COUNT, TASK_AGG_SUM, TASK_AGG_AVG, TASK_AGG_MAX, TASK_AGG_MIN],
        "keywords": ["count", "sum", "average", "max", "min", "aggregate", "zero", "null", "total"],
        "text": (
            "If COUNT/SUM/AVG returns 0 or NULL, the WHERE clause likely filtered out "
            "all rows. Run `SELECT COUNT(*) FROM `tbl`` (no WHERE) to check total rows. "
            "Then add filters step-by-step to find the issue."
        ),
    },
    {
        "id": "ranking_commit_value_not_rank",
        "task_types": [TASK_RANKING],
        "keywords": ["rank", "ranking", "position", "order", "who", "which", "name", "person"],
        "text": (
            "Ranking tasks ask for the NAME or VALUE at a specific rank — "
            "e.g. 'who ranked 3rd' → commit the person's name, not the number 3. "
            "Use ORDER BY … LIMIT 1 OFFSET N-1 or WHERE rank_col = N to get the entity."
        ),
    },
    {
        "id": "insert_values_exact_text_format",
        "task_types": [TASK_INSERT],
        "keywords": ["insert", "add", "new row", "values", "text", "string", "format", "exact"],
        "text": (
            "ALL columns are TEXT. INSERT values must be quoted strings matching "
            "the EXACT format in the table — including units ('100.0 m.'), ordinals ('2nd'), "
            "date strings ('October 23, 2005'), and score formats ('70-70-70=210'). "
            "Do not use bare numbers (30) — use '30'. Check sample rows in the schema card."
        ),
    },
    {
        "id": "grouped_by_multi_row_answer",
        "task_types": [TASK_SELECT, TASK_COUNTING],
        "keywords": ["group", "grouped by", "for each", "per", "count by", "how many each"],
        "text": (
            "Questions with 'grouped by X' / 'for each X' expect MULTI-ROW answers — "
            "one row per group. Use `SELECT X, COUNT(*) FROM tbl GROUP BY X` and commit "
            "ALL rows as tuple strings: answers=[\"('X1', 3)\", \"('X2', 1)\", ...]."
        ),
    },
    {
        "id": "unique_entity_with_related_values",
        "task_types": [TASK_SELECT, TASK_COUNTING],
        "keywords": ["unique", "appears only once", "each driver once", "list each", "with car numbers", "with values"],
        "text": (
            "If the question asks for one row per entity plus related values (e.g. "
            "'each driver once with car numbers'), group by the entity and aggregate the "
            "related field: `SELECT Driver, GROUP_CONCAT(DISTINCT `Car #`) FROM t ... "
            "GROUP BY Driver ORDER BY Driver`. Do not return duplicated entity rows."
        ),
    },
    {
        "id": "update_with_dataset_average",
        "task_types": [TASK_UPDATE],
        "keywords": ["replace with average", "set to average", "update all rows where", "incorrectly recorded as 0", "average of all other"],
        "text": (
            "For UPDATE tasks that set bad rows to a dataset average, compute the average "
            "from valid rows in a subquery and use it in UPDATE. Example: "
            "`UPDATE t SET col=(SELECT ROUND(AVG(CAST(col AS DECIMAL(20,6)))) FROM t WHERE col!='0') "
            "WHERE col='0';` then verify with SELECT."
        ),
    },
]


def retrieve_db_skills(
    task_type: str,
    query: str,
    top_k: int = 2,
) -> List[Dict[str, Any]]:
    candidates = [s for s in DB_SKILLS if task_type in s.get("task_types", [])]
    if not candidates:
        candidates = [s for s in DB_SKILLS if TASK_SELECT in s.get("task_types", [])]
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates
    q = (query or "").lower()

    # Hard preferences for top-1 mode on high-conflict phrasings.
    unique_entity_phrase = (
        ("unique" in q and ("once" in q or "only once" in q))
        or "each driver" in q
        or "car number" in q
    )
    if unique_entity_phrase:
        unique_skill = next((s for s in candidates if s.get("id") == "unique_entity_with_related_values"), None)
        if unique_skill is not None and top_k == 1:
            return [unique_skill]

    # Grouped/per-category phrasing usually expects multi-row grouped outputs.
    grouped_phrase = any(p in q for p in ["grouped by", "for each", "in each", "count by"])
    if grouped_phrase:
        grouped_skill = next((s for s in candidates if s.get("id") == "grouped_by_multi_row_answer"), None)
        if grouped_skill is not None and top_k == 1:
            return [grouped_skill]

    query_tokens = _bm25_tokenize(query)
    if not query_tokens:
        return candidates[:top_k]
    docs = [_skill_doc_tokens(s) for s in candidates]
    scores = _bm25_scores(query_tokens, docs)

    # Top-1 mode needs stronger disambiguation on high-conflict phrasings.
    def heuristic_boost(skill_id: str) -> float:
        boost = 0.0
        if grouped_phrase and skill_id == "grouped_by_multi_row_answer":
            boost += 6.0
        if grouped_phrase and skill_id == "scalar_expect_one_row":
            boost -= 3.0

        if unique_entity_phrase:
            if skill_id == "unique_entity_with_related_values":
                boost += 2.0

        if "update" in q and "average" in q and ("incorrectly" in q or "0" in q):
            if skill_id == "update_with_dataset_average":
                boost += 2.0

        if any(p in q for p in ["next to", "next ", "previous", "before", "after", "adjacent"]):
            if skill_id == "next_previous_row_lookup":
                boost += 2.0

        if any(sym in q for sym in ["£", "$", "currency", "toll", "price"]) and any(
            p in q for p in ["above", "greater than", "at least", "over"]
        ):
            if skill_id == "text_currency_numeric_compare":
                boost += 2.0

        if any(p in q for p in ["needs to be recorded", "add this incident", "add this entry", "new record"]):
            if skill_id == "insert_request_means_execute_insert":
                boost += 2.0

        return boost

    adjusted = []
    for score, skill in zip(scores, candidates):
        sid = skill.get("id", "")
        adjusted.append((score + heuristic_boost(sid), skill))

    ranked = sorted(adjusted, key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# H0 — Task Parser
# ─────────────────────────────────────────────────────────────────────────────

_LIKE_SIGNALS = [
    "contains", "containing", "mentions", "mentioned", "includes", "including",
    "has the word", "with the word",
]
_CASE_INSENSITIVE_SIGNALS = [
    "ignoring case", "regardless of case", "case-insensitive", "case insensitive",
]


@dataclass
class DBTaskContext:
    raw_description: str = ""
    task_type: str = TASK_OTHER
    answer_shape: Optional[str] = None
    target_table_raw: Optional[str] = None
    target_table_sanitized: Optional[str] = None
    target_cols_raw: List[str] = field(default_factory=list)
    target_cols_sanitized: List[str] = field(default_factory=list)
    mentions_like: bool = False
    case_insensitive: bool = False
    expected_insert_cols: Optional[int] = None   # for INSERT: number of columns in std_sql shape


def _detect_answer_shape(task_type: str, description: str) -> Optional[str]:
    t = (description or "").lower()
    if task_type in _MUTATION_TYPES:
        return SHAPE_HASH
    if task_type == TASK_COUNTING or task_type == TASK_AGG_COUNT:
        # "how many X of each Y" / "for each Y" → GROUP BY → multi-row answer
        if re.search(r"\bfor each\b|\bof each\b|\bper \w+\b|\bgrouped? by\b", t):
            return SHAPE_MULTI_MULTI
        return SHAPE_SCALAR_INT
    if task_type == TASK_AGG_SUM or task_type == TASK_AGG_AVG:
        return SHAPE_SCALAR_FLOAT
    if task_type in (TASK_AGG_MAX, TASK_AGG_MIN):
        # Max/min could be numeric or string — infer from keywords.
        if re.search(r"\b(highest|lowest|most|least|maximum|minimum|number|count|score|points|value)\b", t):
            return SHAPE_SCALAR_FLOAT
        return SHAPE_SCALAR_STR
    if task_type == TASK_RANKING:
        # Ranking usually asks for a name/row at top/bottom.
        return SHAPE_SCALAR_STR
    if task_type == TASK_SELECT:
        # GROUP BY / "for each" queries always return multi-row — check first so
        # "how many ... grouped by X" doesn't get collapsed to scalar.
        _grouped = bool(re.search(
            r"\bgrouped? by\b|\bfor each\b|\bper \w+\b|\beach \w+ (has|have|had|shows?|lists?)\b",
            t,
        ))
        if _grouped:
            return SHAPE_MULTI_MULTI
        # Prefer multi-row multi-col unless description strongly implies single value.
        if re.search(r"\bhow many\b|\bnumber of\b|\btotal\b|\bcount\b", t):
            return SHAPE_SCALAR_INT
        return SHAPE_MULTI_MULTI
    if task_type == TASK_COMPARISON:
        return SHAPE_SCALAR_STR
    return None


def parse_task_context(entry: Dict[str, Any]) -> DBTaskContext:
    """Build DBTaskContext from a dataset entry.

    Reads only `type`, `description`, `table`, and `sql.query` structure (the
    latter for INSERT-column-count only — never values).
    """
    ctx = DBTaskContext()
    desc = entry.get("description", "") or ""
    ctx.raw_description = desc

    # Task type
    raw_type_list = entry.get("type", []) or []
    raw_type = raw_type_list[0] if raw_type_list else TASK_OTHER
    if raw_type in _ALL_TASK_TYPES_LIST:
        ctx.task_type = raw_type
    else:
        ctx.task_type = TASK_OTHER

    ctx.answer_shape = _detect_answer_shape(ctx.task_type, desc)

    # Target table — first table in entry['table'] (single or list)
    tables = entry.get("table")
    if isinstance(tables, list) and tables:
        first = tables[0]
    elif isinstance(tables, dict):
        first = tables
    else:
        first = None
    if first:
        raw = first.get("table_name", "") or ""
        ctx.target_table_raw = raw
        ctx.target_table_sanitized = sanitize_identifier(raw, fallback_prefix="table_0")
        cols = first.get("table_info", {}).get("columns", []) or []
        ctx.target_cols_raw = [c.get("name", "") for c in cols]
        ctx.target_cols_sanitized = _sanitize_columns(cols)

    # Intent signals (lexical only; no label inspection)
    d_low = desc.lower()
    ctx.mentions_like = any(sig in d_low for sig in _LIKE_SIGNALS)
    ctx.case_insensitive = any(sig in d_low for sig in _CASE_INSENSITIVE_SIGNALS)

    # INSERT column-count hint (from std_sql structure only — NOT values)
    if ctx.task_type == TASK_INSERT:
        std_sql = (entry.get("sql") or {}).get("query") or ""
        m = re.search(r"INSERT\s+INTO\s+`?[^`\s]+`?\s*\(([^)]+)\)", std_sql, re.IGNORECASE)
        if m:
            cols_part = m.group(1)
            ctx.expected_insert_cols = len([c for c in cols_part.split(",") if c.strip()])
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# H_SCHEMA — Schema Card
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SchemaMap:
    tables: List[Dict[str, Any]] = field(default_factory=list)
    # name_map: lowercase raw name → sanitized name (for case-insensitive fuzzy match)
    name_map: Dict[str, str] = field(default_factory=dict)
    has_truncation: bool = False


def build_schema_map(entry: Dict[str, Any]) -> SchemaMap:
    sm = SchemaMap()
    tables = entry.get("table")
    if isinstance(tables, dict):
        tables = [tables]
    if not isinstance(tables, list):
        return sm

    for t_idx, t in enumerate(tables):
        raw_tname = t.get("table_name", "") or ""
        s_tname = sanitize_identifier(raw_tname, fallback_prefix=f"table_{t_idx}")
        cols = t.get("table_info", {}).get("columns", []) or []
        raw_col_names = [c.get("name", "") for c in cols]
        s_col_names = _sanitize_columns(cols)

        cols_detail = []
        for raw_c, s_c in zip(raw_col_names, s_col_names):
            truncated = (raw_c != s_c)
            if truncated:
                sm.has_truncation = True
            cols_detail.append({"raw": raw_c, "sanitized": s_c, "truncated": truncated})
            # name_map for H2 auto-backtick (lowercase key for case-insensitive)
            if s_c:
                sm.name_map[s_c.lower()] = s_c

        # Store up to 2 sample rows (raw values) for INSERT task guidance.
        sample_rows = t.get("table_info", {}).get("rows", [])[:2]

        sm.tables.append({
            "raw": raw_tname,
            "sanitized": s_tname,
            "truncated": raw_tname != s_tname,
            "columns": cols_detail,
            "sample_rows": sample_rows,
        })
        if s_tname:
            sm.name_map[s_tname.lower()] = s_tname

    return sm


def build_schema_card(schema_map: SchemaMap, max_lines: int = 20, task_type: str = "") -> str:
    """Render a compact readable schema card for prompt injection."""
    if not schema_map.tables:
        return ""
    show_samples = task_type in (TASK_INSERT, TASK_UPDATE, TASK_DELETE)
    lines: List[str] = [
        "[SCHEMA HINT] MySQL tables (always wrap names with spaces/punct in backticks):",
    ]
    for t in schema_map.tables:
        s_tname = t["sanitized"]
        cols = t["columns"]
        col_parts = []
        for c in cols:
            entry = f"`{c['sanitized']}`"
            if c["truncated"]:
                entry += f" /* truncated from {len(c['raw'])} chars */"
            col_parts.append(entry)
        line = f"- `{s_tname}` ({', '.join(col_parts)})"
        if t["truncated"]:
            line += f"  /* table name truncated from {len(t['raw'])} chars */"
        lines.append(line)
        # For INSERT/UPDATE/DELETE: show 2 sample rows so agent knows exact value formats.
        if show_samples and t.get("sample_rows"):
            s_cols = [c["sanitized"] for c in cols]
            lines.append(
                "  Sample rows (values are TEXT — use these exact formats in INSERT/UPDATE):"
            )
            for row in t["sample_rows"]:
                pairs = ", ".join(
                    f"`{sc}`: {repr(str(v))}" for sc, v in zip(s_cols, row)
                )
                lines.append(f"    {{{pairs}}}")
        if len(lines) >= max_lines:
            break
    if schema_map.has_truncation:
        lines.append(
            "Note: names longer than 64 chars were truncated on import. Reference columns EXACTLY as shown above."
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# H1 — Session State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DBSessionState:
    sql_history: List[str] = field(default_factory=list)  # normalised (lowercase, whitespace-collapsed)
    sql_history_raw: List[str] = field(default_factory=list)
    last_sql: Optional[str] = None
    last_result_raw: str = ""
    last_result_was_error: bool = False
    last_error_kind: Optional[str] = None   # syntax / unknown_col / unknown_table / timeout / empty / null_agg / ok
    last_error_text: str = ""
    discovered_columns: Dict[str, List[str]] = field(default_factory=dict)
    candidate_answer: Optional[Any] = None        # raw DB output string or extracted shape
    candidate_answer_shape: Optional[str] = None  # SHAPE_* tag for the candidate
    candidate_implausible: bool = False
    error_streak: int = 0
    empty_streak: int = 0
    text_only_streak: int = 0
    loop_streak: int = 0
    mutation_attempted: bool = False
    commit_blocks_used: int = 0
    mutation_commit_blocks_used: int = 0
    scalar_multi_answer_blocks_used: int = 0  # dedicated counter for scalar>1 blocks
    last_result_col_count: Optional[int] = None  # column count of last multi-col result
    last_result_row_count: Optional[int] = None  # row count of last SQL result
    h4_fired_last_round: bool = False


def _normalize_sql(sql: str) -> str:
    s = (sql or "").strip().rstrip(";")
    s = re.sub(r"\s+", " ", s).lower()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# H2 — Rescue parser for text-embedded tool calls
# ─────────────────────────────────────────────────────────────────────────────

_RESCUE_EXECUTE_JSON_RE = re.compile(
    r'execute_sql\s*[\({]\s*\{?\s*[\"\']?query[\"\']?\s*:\s*[\"\']((?:[^\"\'\\]|\\.)+)[\"\']',
    re.IGNORECASE | re.DOTALL,
)
_RESCUE_EXECUTE_KWARG_RE = re.compile(
    r"execute_sql\s*\(\s*query\s*=\s*['\"]((?:[^'\"\\]|\\.)+)['\"]",
    re.IGNORECASE | re.DOTALL,
)
_RESCUE_SQL_FENCE_RE = re.compile(r"```sql\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
_RESCUE_COMMIT_JSON_RE = re.compile(
    r'commit_final_answer\s*[\({]\s*\{?\s*[\"\']?answers[\"\']?\s*:\s*(\[[^\]]*\])',
    re.IGNORECASE | re.DOTALL,
)
_RESCUE_COMMIT_KWARG_RE = re.compile(
    r"commit_final_answer\s*\(\s*answers\s*=\s*(\[[^\]]*\])",
    re.IGNORECASE | re.DOTALL,
)
_RESCUE_TOOL_CALL_XML_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
# Tolerant XML: <tool_call> without closing tag — model output may be truncated.
_RESCUE_TOOL_CALL_XML_OPEN_RE = re.compile(
    r"<tool_call>\s*(\{[^<]{0,8000})",
    re.DOTALL | re.IGNORECASE,
)
# Partial answers list: capture all fully-quoted strings from a truncated JSON list.
_RESCUE_PARTIAL_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')

# xLAM-style bracket-list tool calls.
#   JSON form:    [{"name": "execute_sql", "arguments": {"query": "..."}}]
#                 (also accepts "args" / "parameters" as field aliases)
#   Python form:  [execute_sql(query='SELECT ... WHERE x="a"')]
#                 [commit_final_answer(answers=['a','b'])]
_RESCUE_PY_CALL_HEAD_RE = re.compile(
    r"^\s*\[\s*(execute_sql|commit_final_answer)\s*\(",
    re.IGNORECASE,
)
_RESCUE_PY_KWARG_RE = re.compile(r"\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", re.DOTALL)
_VALID_TOOL_NAMES = {"execute_sql", "commit_final_answer"}


def _find_matching_bracket(s: str, start: int) -> int:
    """Return index of ']' matching the '[' at position `start`, honouring
    string literals so brackets inside quotes don't unbalance the count.
    Returns -1 if no match.
    """
    if start >= len(s) or s[start] != "[":
        return -1
    depth = 0
    in_str: Optional[str] = None
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if esc:
            esc = False
            continue
        if in_str:
            if ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _find_matching_paren(s: str, start: int) -> int:
    """Return index of ')' matching the '(' at position `start`."""
    if start >= len(s) or s[start] != "(":
        return -1
    depth = 0
    in_str: Optional[str] = None
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if esc:
            esc = False
            continue
        if in_str:
            if ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _normalize_rescued_args(name: str, args: Any) -> Optional[Dict[str, Any]]:
    """Coerce parsed args dict/list into the canonical {query: ...} or
    {answers: [...]} shape expected by task.py."""
    if isinstance(args, str):
        # Some emitters double-encode arguments as a JSON string.
        try:
            args = json.loads(args)
        except Exception:
            pass
    if name == "execute_sql":
        if isinstance(args, dict):
            for key in ("query", "sql", "statement"):
                if key in args and isinstance(args[key], str):
                    return {"query": args[key]}
            # Fallback: take first string-valued field.
            for v in args.values():
                if isinstance(v, str):
                    return {"query": v}
        elif isinstance(args, str):
            return {"query": args}
    elif name == "commit_final_answer":
        if isinstance(args, dict):
            for key in ("answers", "answer", "result"):
                if key in args:
                    val = args[key]
                    if isinstance(val, list):
                        return {"answers": [str(x) for x in val]}
                    if val is not None:
                        return {"answers": [str(val)]}
        elif isinstance(args, list):
            return {"answers": [str(x) for x in args]}
        elif isinstance(args, str):
            return {"answers": [args]}
    return None


def _try_rescue_bracket_list(content: str) -> Optional[Dict[str, Any]]:
    """Rescue xLAM-style bracket-list tool calls (JSON or Python-call form).

    Field-name aliases: arguments / args / parameters. We only emit a rescue
    when the call name is one of the two real tools — guards against
    false-positives like `[{"Year": "1958", ...}]` (table-row echoing).
    """
    if not content:
        return None
    s = content.lstrip()
    if not s.startswith("["):
        return None

    # ── 1) JSON-list form ────────────────────────────────────────────────
    end = _find_matching_bracket(s, 0)
    if end > 0:
        try:
            arr = json.loads(s[: end + 1])
        except Exception:
            arr = None
        if isinstance(arr, list) and arr and isinstance(arr[0], dict):
            first = arr[0]
            name = first.get("name") or first.get("tool") or first.get("function")
            if isinstance(name, str) and name in _VALID_TOOL_NAMES:
                raw_args = (
                    first.get("arguments")
                    if "arguments" in first
                    else first.get("args")
                    if "args" in first
                    else first.get("parameters")
                )
                if raw_args is None:
                    raw_args = {}
                normalised = _normalize_rescued_args(name, raw_args)
                if normalised is not None:
                    return {"name": name, "arguments": normalised}

    # ── 1b) Malformed JSON-list fallback ─────────────────────────────────
    # xLAM frequently emits broken JSON like
    #   [{"name":"commit_final_answer","arguments":{"answers":["..."}}]
    # (forgets to close the inner `]`). Extract via tolerant regex when the
    # head clearly says it's a tool call.
    head = re.match(
        r'^\s*\[\s*\{\s*["\']name["\']\s*:\s*["\']([A-Za-z_]\w*)["\']',
        s,
    )
    if head:
        name = head.group(1)
        if name in _VALID_TOOL_NAMES:
            if name == "commit_final_answer":
                items = _try_rescue_partial_commit(s)
                if items:
                    return {
                        "name": name,
                        "arguments": {"answers": [str(x) for x in items]},
                    }
            elif name == "execute_sql":
                qm = re.search(
                    r'["\'](?:query|sql|statement)["\']\s*:\s*"((?:[^"\\]|\\.)*)"',
                    s,
                )
                if qm:
                    sql = qm.group(1).encode().decode(
                        "unicode_escape", errors="ignore"
                    )
                    return {"name": name, "arguments": {"query": sql}}

    # ── 2) Python-call form: [execute_sql(query=...)] ────────────────────
    m = _RESCUE_PY_CALL_HEAD_RE.match(s)
    if not m:
        return None
    name = m.group(1).lower()
    paren_open = m.end() - 1  # index of '('
    paren_close = _find_matching_paren(s, paren_open)
    if paren_close < 0:
        return None
    payload = s[paren_open + 1 : paren_close].strip()
    if not payload:
        return None

    # Try to parse as a single kwarg (kw=VALUE). xLAM nearly always emits
    # exactly one keyword arg here.
    kw_m = _RESCUE_PY_KWARG_RE.match(payload)
    val: Any = None
    if kw_m:
        raw_val = kw_m.group(2).strip().rstrip(",").strip()
        try:
            val = ast.literal_eval(raw_val)
        except Exception:
            # Last-ditch: strip outer matched quotes.
            if (
                len(raw_val) >= 2
                and raw_val[0] in ("'", '"')
                and raw_val[-1] == raw_val[0]
            ):
                val = raw_val[1:-1]
            else:
                val = None
    else:
        # No `=`: treat the whole payload as a positional literal.
        try:
            val = ast.literal_eval(payload)
        except Exception:
            val = None

    if val is None:
        return None
    normalised = _normalize_rescued_args(name, val)
    if normalised is None:
        return None
    return {"name": name, "arguments": normalised}


# xLAM-70B harness mode learns the harness's `[SCHEMA HINT]` bracket convention
# in-context and emits free-text section markers like
#   [Thoughts] reasoning ...
#   [SQL] UPDATE `tbl` SET ...
#   [Final Answer] X
# Recognise these and lift them into real tool calls.
_SECTION_SQL_MARKERS = frozenset({
    "execute_sql", "sql", "sql query", "sql code", "verification sql",
    "operation", "code", "execution", "tool call",
})
_SECTION_COMMIT_MARKERS = frozenset({
    "commit_final_answer", "commit",
    "final answer", "final_answer", "final",
    "answer", "answers",
})
_SECTION_HEAD_RE = re.compile(r"\[\s*([A-Za-z][A-Za-z _]{0,30}?)\s*\]\s*")
_SECTION_FENCE_RE = re.compile(
    r"^```(?:[A-Za-z0-9_+-]+)?\s*\n?(.*?)\n?```\s*$", re.DOTALL
)
_SECTION_COMMIT_CALL_RE = re.compile(
    r"^\s*commit_final_answer\s*\(\s*(.+?)\s*\)\s*$", re.DOTALL | re.IGNORECASE
)


def _strip_outer_code_fence(s: str) -> str:
    s = s.strip()
    m = _SECTION_FENCE_RE.match(s)
    return m.group(1).strip() if m else s


def _section_extract_sql(body: str) -> Optional[str]:
    body = _strip_outer_code_fence(body)
    if not body:
        return None
    # If body contains an inner ```sql ... ``` fence, prefer that.
    inner = re.search(r"```sql\s*\n(.*?)\n```", body, re.DOTALL | re.IGNORECASE)
    if inner:
        return inner.group(1).strip() or None
    return body


def _section_extract_answers(body: str) -> Optional[List[str]]:
    body = _strip_outer_code_fence(body)
    if not body:
        return None
    # commit_final_answer(...) call wrapped in section body.
    m = _SECTION_COMMIT_CALL_RE.match(body)
    if m:
        try:
            val = ast.literal_eval(m.group(1))
            if isinstance(val, list):
                return [str(x) for x in val]
            if isinstance(val, (str, int, float)):
                return [str(val)]
        except Exception:
            pass
    # Bare list literal.
    if body.startswith("["):
        try:
            val = ast.literal_eval(body)
            if isinstance(val, list):
                return [str(x) for x in val]
        except Exception:
            pass
    # Fallback: whole body is one answer.
    return [body]


def _try_rescue_section_marker(content: str) -> Optional[Dict[str, Any]]:
    """Rescue free-text section-marker tool calls (xLAM-70B harness pattern).

    Strategy: split content on `[marker]` heads, then take the LAST section
    whose marker is in our action whitelist. Commit and SQL markers compete
    on emission order — the latest one wins (matches model intent).
    """
    if not content or "[" not in content:
        return None
    matches = list(_SECTION_HEAD_RE.finditer(content))
    if not matches:
        return None

    # Build (marker, body) for every section so unknown markers still
    # delimit body extents correctly.
    sections: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        marker = m.group(1).strip().lower()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[body_start:body_end].strip()
        sections.append((marker, body))

    last_action: Optional[Tuple[str, str]] = None
    for marker, body in sections:
        if not body:
            continue
        if marker in _SECTION_SQL_MARKERS:
            last_action = ("sql", body)
        elif marker in _SECTION_COMMIT_MARKERS:
            last_action = ("commit", body)

    if last_action is None:
        return None
    kind, body = last_action

    if kind == "sql":
        sql = _section_extract_sql(body)
        if sql:
            return {"name": "execute_sql", "arguments": {"query": sql}}
    else:
        answers = _section_extract_answers(body)
        if answers is not None:
            return {"name": "commit_final_answer", "arguments": {"answers": answers}}
    return None


def _try_rescue_partial_commit(raw_json: str) -> Optional[List[str]]:
    """Extract committed answers from a truncated JSON commit_final_answer body.

    Handles the case where the model writes a very long answers list that gets
    cut off mid-string, causing json.loads to fail.  We extract all complete
    double-quoted strings and skip any truncated tail.
    """
    # Find "answers": [ ... up to where it's cut
    m = re.search(r'"?answers"?\s*:\s*\[', raw_json, re.IGNORECASE)
    if not m:
        return None
    list_body = raw_json[m.end():]
    items = []
    for sm in _RESCUE_PARTIAL_STRING_RE.finditer(list_body):
        # Stop if we hit past the end of a valid list element zone
        items.append(sm.group(1))
    return items if items else None


def rescue_tool_call_from_text(content: str) -> Optional[Dict[str, Any]]:
    """Lift a tool call out of plain assistant content, if one is embedded.

    Priority: XML <tool_call> (full) → XML <tool_call> (truncated/open) →
    commit_final_answer JSON/kwarg → partial commit rescue →
    execute_sql JSON/kwarg → ```sql fence```.
    """
    if not content:
        return None

    # xLAM-style bracket-list (JSON or Python-call form). Try first because
    # it's a strong signal — the head must literally start with `[` and a
    # known tool name, so it can't false-trigger on prose.
    bracket = _try_rescue_bracket_list(content)
    if bracket is not None:
        return bracket

    # XML form with closing tag — unambiguous.
    m = _RESCUE_TOOL_CALL_XML_RE.search(content)
    if m:
        try:
            obj = json.loads(m.group(1))
            name = obj.get("name", "")
            args = obj.get("arguments", {})
            if name in ("execute_sql", "commit_final_answer") and args:
                return {"name": name, "arguments": args}
        except Exception:
            pass

    # XML form WITHOUT closing tag — model output was truncated.
    m = _RESCUE_TOOL_CALL_XML_OPEN_RE.search(content)
    if m:
        raw = m.group(1)
        # Try as complete JSON first (might have been cut after closing brace)
        try:
            obj = json.loads(raw)
            name = obj.get("name", "")
            args = obj.get("arguments", {})
            if name in ("execute_sql", "commit_final_answer") and args:
                return {"name": name, "arguments": args}
        except Exception:
            pass
        # Try partial commit extraction: pull out complete quoted strings
        if "commit_final_answer" in raw:
            items = _try_rescue_partial_commit(raw)
            if items:
                return {"name": "commit_final_answer", "arguments": {"answers": items}}

    # commit_final_answer — strict JSON / kwarg
    for pat in (_RESCUE_COMMIT_JSON_RE, _RESCUE_COMMIT_KWARG_RE):
        m = pat.search(content)
        if m:
            raw_list = m.group(1)
            try:
                arr = json.loads(raw_list)
                if isinstance(arr, list):
                    return {"name": "commit_final_answer", "arguments": {"answers": [str(x) for x in arr]}}
            except Exception:
                pass

    # execute_sql — JSON / kwarg
    for pat in (_RESCUE_EXECUTE_JSON_RE, _RESCUE_EXECUTE_KWARG_RE):
        m = pat.search(content)
        if m:
            sql = m.group(1).encode().decode("unicode_escape", errors="ignore")
            return {"name": "execute_sql", "arguments": {"query": sql}}

    # Section-marker form: [SQL] UPDATE ..., [Final Answer] X, [Thoughts]
    # ... [SQL] ...  (xLAM-70B harness pattern). Run after the strict
    # JSON/kwarg parsers but before the generic ```sql fence``` fallback so
    # an explicit `[Final Answer]` section beats an incidental sql fence.
    section = _try_rescue_section_marker(content)
    if section is not None:
        return section

    # ```sql fence```
    m = _RESCUE_SQL_FENCE_RE.search(content)
    if m:
        return {"name": "execute_sql", "arguments": {"query": m.group(1).strip()}}

    return None


# ─────────────────────────────────────────────────────────────────────────────
# H2 — SQL auto-backtick + dialect fix
# ─────────────────────────────────────────────────────────────────────────────

_BACKTICK_IDENT_RE = re.compile(r"`[^`]+`")
_QUOTED_STR_RE = re.compile(r"'(?:[^'\\]|\\.)*'")
_NEEDS_BACKTICK_CHARS_RE = re.compile(r"[\s.\-+/#]")


def _mask_out_literals(sql: str) -> Tuple[str, List[Tuple[int, int, str]]]:
    """Return (masked_sql, regions). `regions` lists (start, end, original) of
    already-quoted / backticked substrings that should not be tinkered with.
    Masked regions are replaced with NUL-padded placeholders of equal length
    so offsets stay stable.
    """
    regions: List[Tuple[int, int, str]] = []
    spans: List[Tuple[int, int]] = []
    for m in _BACKTICK_IDENT_RE.finditer(sql):
        spans.append((m.start(), m.end()))
        regions.append((m.start(), m.end(), m.group(0)))
    for m in _QUOTED_STR_RE.finditer(sql):
        spans.append((m.start(), m.end()))
        regions.append((m.start(), m.end(), m.group(0)))
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if merged and s < merged[-1][1]:
            continue
        merged.append((s, e))
    out = list(sql)
    for s, e in merged:
        for i in range(s, e):
            out[i] = "\x01"  # sentinel — NOT a word character
    return "".join(out), regions


def auto_backtick_sql(sql: str, schema_map: SchemaMap) -> Tuple[str, List[str]]:
    """Wrap known identifiers (table / column names from schema_map) in backticks
    when they appear as unquoted, whitespace-separated tokens in the SQL AND
    contain a char that requires quoting (space, dot, dash, etc.).

    Rationale: the failure-mode analysis showed 71 syntax errors in baseline,
    most because agents copied descriptive column names like `Race Name` or
    `No.` without backticks. We only auto-fix names that:
      1. Are in the schema_map (safe — not making up identifiers).
      2. Appear outside existing quotes/backticks in the SQL.
      3. Contain at least one char that MYSQL would reject unquoted.

    Returns (patched_sql, hits) where hits lists the names we wrapped.
    """
    if not sql or not schema_map.name_map:
        return sql, []

    # Build candidate list — only names that would actually need quoting.
    candidates: List[Tuple[str, str]] = []  # (lowercase match key, real sanitized name)
    for low, real in schema_map.name_map.items():
        if not real or not _NEEDS_BACKTICK_CHARS_RE.search(real):
            continue
        candidates.append((low, real))
    # Prefer longer names first to avoid partial overlap (e.g. "Race Name" before "Name").
    candidates.sort(key=lambda x: -len(x[1]))
    if not candidates:
        return sql, []

    masked, _regions = _mask_out_literals(sql)
    patched = sql
    masked_patched = masked
    hits: List[str] = []

    for _low, real in candidates:
        # Word-ish boundary on the "masked" text, case-insensitive. Use lookarounds
        # for non-identifier boundary on both sides.
        pattern = re.compile(
            r"(?<![A-Za-z0-9_`])" + re.escape(real) + r"(?![A-Za-z0-9_`])",
            re.IGNORECASE,
        )
        # Find ALL matches in masked, then apply the offsets to the real sql;
        # we rebuild both simultaneously so later candidates can't double-wrap.
        new_patched_parts: List[str] = []
        new_masked_parts: List[str] = []
        last = 0
        match_count = 0
        for m in pattern.finditer(masked_patched):
            s, e = m.start(), m.end()
            if s < last:
                continue
            new_patched_parts.append(patched[last:s])
            new_masked_parts.append(masked_patched[last:s])
            original_slice = patched[s:e]
            replacement = f"`{original_slice}`"
            new_patched_parts.append(replacement)
            # Mask the new backticked region so later candidates won't touch.
            new_masked_parts.append("\x01" * len(replacement))
            last = e
            match_count += 1
        new_patched_parts.append(patched[last:])
        new_masked_parts.append(masked_patched[last:])
        if match_count > 0:
            patched = "".join(new_patched_parts)
            masked_patched = "".join(new_masked_parts)
            hits.append(real)
    return patched, hits


_DIALECT_CONCAT_RE = re.compile(
    # capture two operands separated by ||, each a reasonably atomic SQL expression
    # (identifier, backticked, string literal, or function call).  Stay conservative
    # to avoid breaking boolean-OR uses (which MySQL accepts syntactically but
    # semantically means OR — we only rewrite when BOTH sides are string-y).
    r"""(
        `[^`]+`                    # backticked ident
        |'(?:[^'\\]|\\.)*'         # string literal
        |\w+\s*\([^()]*\)          # simple function call
        |[A-Za-z_][A-Za-z_0-9]*    # bare identifier
    )
    \s*\|\|\s*
    (
        `[^`]+`
        |'(?:[^'\\]|\\.)*'
        |\w+\s*\([^()]*\)
        |[A-Za-z_][A-Za-z_0-9]*
    )
    """,
    re.VERBOSE,
)


def dialect_fix_sql(sql: str) -> Tuple[str, List[str]]:
    """Rewrite obvious SQLite-isms into MySQL equivalents.

    Currently handles only `a || b` → `CONCAT(a, b)`, and only in a conservative
    2-arg form (agent-written concats are almost always simple pairs). If an
    agent actually means boolean OR they would use `OR` rather than `||` on
    string operands, so false positives are very rare in practice.
    """
    if not sql or "||" not in sql:
        return sql, []
    hits: List[str] = []
    new_sql = sql

    def _sub(m: re.Match) -> str:
        hits.append("concat_rewrite")
        return f"CONCAT({m.group(1).strip()}, {m.group(2).strip()})"

    # Apply at most a few times — avoid runaway rewrites on truly malformed SQL.
    for _ in range(3):
        changed = _DIALECT_CONCAT_RE.sub(_sub, new_sql)
        if changed == new_sql:
            break
        new_sql = changed
    return new_sql, hits


# ─────────────────────────────────────────────────────────────────────────────
# H2 — SQL safety filter
# ─────────────────────────────────────────────────────────────────────────────

_DANGEROUS_SQL_PATTERNS = [
    re.compile(r"\bdrop\s+database\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\bgrant\s+all\b", re.IGNORECASE),
]


def _is_dangerous_sql(sql: str) -> Optional[str]:
    for pat in _DANGEROUS_SQL_PATTERNS:
        if pat.search(sql or ""):
            return pat.pattern
    return None


# ─────────────────────────────────────────────────────────────────────────────
# H2 — Answer normalisation (interface-boundary repair at commit)
# ─────────────────────────────────────────────────────────────────────────────

_ANSWER_UNIT_TOKENS = {
    "game", "games", "goal", "goals", "win", "wins", "season", "seasons",
    "record", "records", "album", "albums", "item", "items", "row", "rows",
    "point", "points", "score", "scores", "medal", "medals",
    "episode", "episodes", "title", "titles", "entry", "entries",
    "year", "years", "match", "matches", "race", "races",
}

_ANSWER_STRIP_PREFIXES = [
    "the answer is", "answer:", "answer is", "result:", "result is",
    "output:", "count:", "total:", "total is",
]


def _normalize_scalar_numeric(value: str) -> Tuple[str, bool]:
    s = str(value or "").strip()
    original = s
    low = s.lower()
    for p in _ANSWER_STRIP_PREFIXES:
        if low.startswith(p):
            s = s[len(p):].lstrip(" :\t")
            break
    s = s.strip().strip("'\"")
    # Drop trailing period
    s = s.rstrip(".")
    # Trailing unit tokens
    tokens = s.split()
    while len(tokens) > 1 and tokens[-1].lower().strip(".,") in _ANSWER_UNIT_TOKENS:
        tokens.pop()
    s = " ".join(tokens).strip()
    # Thousand separators
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    # None/null → "0"
    if s.lower() in {"none", "null", "nan", "", "undefined"}:
        s = "0"
    # If still multi-token (has spaces), extract last numeric token.
    # Guard: only apply when there is a space in s to avoid mangling "1900s" → "1900"
    # or ordinals like "3rd". Strings without spaces that fail the fullmatch are
    # non-numeric strings (years-with-suffix, ordinals, etc.) and must be kept as-is.
    if not re.fullmatch(r"-?\d+(\.\d+)?", s) and " " in s:
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
        if nums:
            s = nums[-1]
    return s, (s != original)


def _normalize_scalar_string(value: str) -> Tuple[str, bool]:
    s = str(value or "").strip()
    original = s
    # Replace non-breaking space (U+00A0) with regular space; evaluator does
    # plain string comparison so \xa0 ≠ space causes false mismatches.
    s = s.replace("\xa0", " ")
    low = s.lower()
    for p in _ANSWER_STRIP_PREFIXES:
        if low.startswith(p):
            s = s[len(p):].lstrip(" :\t")
            break
    s = s.strip().strip("'\"")
    s = s.rstrip(".")
    return s, (s != original)


def normalize_answers_list(
    answers: List[str],
    answer_shape: Optional[str],
) -> Tuple[List[str], Dict[str, Any]]:
    """Return (normalized_answers, audit).

    audit contains: mutated (bool), rule_hits (list), before, after.
    Passthrough when answer_shape is SHAPE_HASH or None.
    """
    audit = {"mutated": False, "rule_hits": [], "before": list(answers or []), "after": None}
    if answer_shape in (SHAPE_HASH, None):
        audit["after"] = list(answers or [])
        return list(answers or []), audit
    if not answers:
        audit["after"] = []
        return [], audit

    out = [str(a) if a is not None else "" for a in answers]
    if answer_shape in (SHAPE_SCALAR_INT, SHAPE_SCALAR_FLOAT):
        new_out = []
        for a in out:
            s, mutated = _normalize_scalar_numeric(a)
            if mutated:
                audit["rule_hits"].append("numeric_strip")
            new_out.append(s)
        out = new_out
    elif answer_shape == SHAPE_SCALAR_STR:
        new_out = []
        for a in out:
            s, mutated = _normalize_scalar_string(a)
            if mutated:
                audit["rule_hits"].append("string_strip")
            new_out.append(s)
        out = new_out
    elif answer_shape in (SHAPE_MULTI_SINGLE, SHAPE_MULTI_MULTI):
        if len(out) == 1 and isinstance(out[0], str):
            s = out[0].strip()
            if s.startswith("[") and s.endswith("]"):
                # Agent wrapped the whole list in ONE string like "[('1948',),('1992',)]"
                try:
                    arr = eval(s, {"__builtins__": {}}, {})
                    if isinstance(arr, list):
                        if answer_shape == SHAPE_MULTI_MULTI:
                            # Keep each row as its Python tuple repr-string so the
                            # evaluator's set-comparison matches the ground truth.
                            out = [str(item) if isinstance(item, tuple) else str(item) for item in arr]
                        else:
                            flat: List[str] = []
                            for item in arr:
                                if isinstance(item, tuple):
                                    for cell in item:
                                        flat.append(str(cell))
                                else:
                                    flat.append(str(item))
                            out = flat
                        audit["rule_hits"].append("unpack_stringified_list")
                except Exception:
                    pass
            elif answer_shape == SHAPE_MULTI_MULTI and s.startswith("("):
                # Agent crammed all rows into ONE string WITHOUT outer brackets:
                # "('a', 'b'), ('c', 'd')" — parse as a list by wrapping first.
                try:
                    arr = eval(f"[{s}]", {"__builtins__": {}}, {})
                    if isinstance(arr, list) and all(isinstance(x, tuple) for x in arr):
                        out = [str(item) for item in arr]
                        audit["rule_hits"].append("unpack_bare_tuple_sequence")
                except Exception:
                    pass

        # Per-element light cleanup (strip outer quotes only, not tuple parens)
        # Also replace non-breaking space (U+00A0) with regular space.
        out = [
            (a.replace("\xa0", " ").strip().strip("'\"") if not a.strip().startswith("(") else a.replace("\xa0", " ").strip())
            for a in out
        ]

    if out != audit["before"]:
        audit["mutated"] = True
    audit["after"] = out
    return out, audit


# ─────────────────────────────────────────────────────────────────────────────
# H1 helpers — parsing DB responses
# ─────────────────────────────────────────────────────────────────────────────

_MYSQL_UNKNOWN_COL_RE = re.compile(r"Unknown column '([^']+)'", re.IGNORECASE)
_MYSQL_UNKNOWN_TBL_RE = re.compile(r"Table '[^']*?([\w\s\.-]+)' doesn'?t exist", re.IGNORECASE)
_MYSQL_SYNTAX_NEAR_RE = re.compile(r"near '([^']{1,60})'", re.IGNORECASE)


def classify_db_response(response: str) -> Tuple[str, Optional[str]]:
    """Return (kind, error_text) where kind is one of:
    syntax / unknown_col / unknown_table / timeout / null_agg / empty / ok.
    """
    if not response:
        return ("empty", None)
    low = response.lower()
    if "error: sql execution timed out" in low or "timed out" in low:
        return ("timeout", response)
    if "unknown column" in low:
        return ("unknown_col", response)
    if "doesn't exist" in low or "doesn’t exist" in low:
        return ("unknown_table", response)
    if "syntax" in low and "error" in low:
        return ("syntax", response)
    if "error" in low:
        # Generic error — treat as 'syntax' for hint purposes but mark as error.
        if "near '" in low:
            return ("syntax", response)
        return ("syntax", response)
    # NULL aggregation result: "[(None,)]" or "[(null,)]"
    if re.match(r"^\s*\[\(\s*(none|null)\s*,\s*\)\s*\]\s*$", response.strip(), re.IGNORECASE):
        return ("null_agg", None)
    if response.strip() in ("[]", "()"):
        return ("empty", None)
    return ("ok", None)


def extract_candidate_from_response(response: str, answer_shape: Optional[str]) -> Tuple[Optional[Any], bool]:
    """Extract a candidate answer from a raw MySQL response string.

    Returns (candidate, implausible). The candidate can be either a str
    (for scalar shapes) or a list[str] (for multi-row shapes). If nothing
    reasonable can be extracted, candidate is None.
    """
    if not response:
        return None, False
    s = response.strip()
    # Try eval — the DB backend returns Python-repr strings like "[(1,)]".
    parsed = None
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = eval(s, {"__builtins__": {}}, {})
        except Exception:
            parsed = None

    if parsed is not None and isinstance(parsed, list):
        if not parsed:
            return None, False

        # Scalar shapes
        if answer_shape in (SHAPE_SCALAR_INT, SHAPE_SCALAR_FLOAT, SHAPE_SCALAR_STR, None):
            if len(parsed) == 1 and isinstance(parsed[0], tuple) and len(parsed[0]) == 1:
                v = parsed[0][0]
                if v is None:
                    # Treat None as implausible for scalar_int but let "0" be
                    # the candidate (evaluator maps None → "0").
                    return "0", True
                return str(v), False
            # Multi-row but shape says scalar → take the first cell.
            # Only flag implausible when the agent returned MULTIPLE rows for a
            # scalar-shape task; a single-row multi-col result is an exploratory
            # query and should not trigger the "multi-row returned" H4 warning.
            try:
                first_cell = parsed[0][0] if isinstance(parsed[0], tuple) else parsed[0]
                is_implausible = len(parsed) > 1
                return str(first_cell), is_implausible
            except Exception:
                return None, False

        # Multi-row single-col
        if answer_shape == SHAPE_MULTI_SINGLE:
            vals: List[str] = []
            for item in parsed:
                if isinstance(item, tuple):
                    if len(item) >= 1:
                        v = item[0]
                        vals.append("" if v is None else str(v))
                else:
                    vals.append(str(item))
            return vals, False

        # Multi-row multi-col — return as raw response string (safer for eval parity)
        if answer_shape == SHAPE_MULTI_MULTI:
            return s, False

    # Non-parseable response — return stripped string for scalar_str shape
    if answer_shape == SHAPE_SCALAR_STR and len(s) <= 200:
        return s, False
    return None, False


# ─────────────────────────────────────────────────────────────────────────────
# H3 — Tool description patching
# ─────────────────────────────────────────────────────────────────────────────

_H3_EXECUTE_HINT = (
    " This is MySQL. Wrap identifiers with spaces/dots/punctuation in backticks "
    "(e.g. `High points`, `No.`, `Olympic Medal Table`). Use `CONCAT(a, b)` "
    "not `a || b`. Prefer one SQL per turn. For TEXT-typed numeric columns, "
    "cast with `CAST(col AS DECIMAL(20,6))` before SUM/AVG or numeric ORDER BY."
)

_H3_COMMIT_HINT = (
    " For scalar results (single COUNT/SUM/AVG), submit ONE bare value: "
    "answers=['42'] — no tuple brackets. "
    "For GROUP BY / 'for each' queries, submit ALL rows as tuple strings: "
    "answers=[\"('GroupA', 3)\", \"('GroupB', 1)\", …]. "
    "For multi-column SELECT, submit each ROW as one tuple-repr element: "
    "answers=[\"('v1', 'v2')\", \"('v3', 'v4')\"]. "
    "For INSERT/UPDATE/DELETE, run the mutation SQL successfully BEFORE "
    "committing — the answer field is ignored but the table hash must match."
)

_H3_SYSTEM_APPEND = (
    "\n\nThis database is MySQL. Column names longer than 64 characters were "
    "truncated on import; use the schema card shown below as the source of "
    "truth. Always wrap identifiers containing spaces or punctuation in "
    "backticks. A syntax error near a column name is almost always an "
    "un-backticked identifier. "
    "IMPORTANT: ALL columns are stored as TEXT — there are no INT/FLOAT columns. "
    "For INSERT/UPDATE, values must be quoted strings. Check the sample rows in "
    "the schema card to determine the correct format for units (e.g. '100.0 m.'), "
    "ordinals (e.g. '2nd'), and dates (e.g. 'October 23, 2005'). For pure numeric "
    "values, use plain digits without thousands separators unless the sample rows "
    "explicitly show commas (e.g. use '63000' not '63,000')."
)


def patch_dbbench_tool_descriptions(
    tools: Optional[List[Dict[str, Any]]]
) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return tools
    patched = copy.deepcopy(tools)
    for tool in patched:
        fn = tool.get("function", {})
        name = fn.get("name", "")
        if name == "execute_sql":
            fn["description"] = fn.get("description", "") + _H3_EXECUTE_HINT
        elif name == "commit_final_answer":
            fn["description"] = fn.get("description", "") + _H3_COMMIT_HINT
        tool["function"] = fn
    return patched


def patch_dbbench_system_prompt(system_prompt: str) -> str:
    return (system_prompt or "") + _H3_SYSTEM_APPEND


# ─────────────────────────────────────────────────────────────────────────────
# Runtime — glue H0..H5 together
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DBBenchHarnessRuntime:
    config: DBBenchHarnessConfig
    task_ctx: Optional[DBTaskContext] = field(default=None)
    schema_map: Optional[SchemaMap] = field(default=None)
    state: DBSessionState = field(default_factory=DBSessionState)
    force_next_action: Optional[Dict[str, Any]] = field(default=None)
    _last_hint: Optional[str] = field(default=None)
    _h5_injected_cold: bool = field(default=False)

    # ── H0 + H_SCHEMA ───────────────────────────────────────────────────────

    def init_task(self, entry: Dict[str, Any]) -> None:
        self.task_ctx = parse_task_context(entry)
        self.schema_map = build_schema_map(entry)
        self.state = DBSessionState()
        self.force_next_action = None
        self._last_hint = None
        self._h5_injected_cold = False

    def schema_card(self) -> str:
        if self.schema_map is None:
            return ""
        task_type = self.task_ctx.task_type if self.task_ctx else ""
        return build_schema_card(self.schema_map, task_type=task_type)

    # ── H5 cold-start ───────────────────────────────────────────────────────

    def cold_start_skill_hints(self) -> List[Dict[str, str]]:
        if not self.config.h5_enabled or self.task_ctx is None:
            return []
        skills = retrieve_db_skills(
            task_type=self.task_ctx.task_type,
            query=self.task_ctx.raw_description,
            top_k=self.config.h5_top_k,
        )
        result: List[Dict[str, str]] = []
        for skill in skills:
            text = skill["text"]
            result.append({
                "id": skill["id"],
                "text": text,
                "trigger": "cold_start",
                "token_cost": str(len(text.split())),
            })
        self._h5_injected_cold = True
        return result

    # ── H2: SQL gate ────────────────────────────────────────────────────────

    def pre_validate_sql(self, sql: str) -> Dict[str, Any]:
        """Inspect + maybe rewrite an execute_sql query before it runs.

        Returns: {action: "run"|"block"|"force_commit", sql, blocked_reason,
                  rule_hits[], force_args}.
        """
        response: Dict[str, Any] = {
            "action": "run",
            "sql": sql,
            "blocked_reason": "",
            "rule_hits": [],
            "force_args": None,
        }
        if not self.config.h2_enabled or self.task_ctx is None or self.schema_map is None:
            return response

        # Safety filter.
        dpat = _is_dangerous_sql(sql)
        if dpat:
            response["action"] = "block"
            response["blocked_reason"] = f"dangerous_sql:{dpat}"
            return response

        # Auto-backtick known identifiers.
        patched, bt_hits = auto_backtick_sql(sql, self.schema_map)
        if bt_hits:
            response["rule_hits"].append({"rule": "auto_backtick", "identifiers": bt_hits})

        # Dialect fix.
        patched, dc_hits = dialect_fix_sql(patched)
        if dc_hits:
            response["rule_hits"].append({"rule": "dialect_concat", "count": len(dc_hits)})

        response["sql"] = patched

        # INSERT checks
        if self.task_ctx.task_type == TASK_INSERT:
            m_ins = re.search(
                r"INSERT\s+INTO\s+`?[^`\s(]+`?\s*\(([^)]+)\)",
                patched, re.IGNORECASE
            )
            if m_ins and self.schema_map.tables:
                # Column count check
                agent_col_count = len([c.strip() for c in m_ins.group(1).split(",") if c.strip()])
                schema_col_count = len(self.schema_map.tables[0].get("columns", []))
                if schema_col_count > 0 and agent_col_count < schema_col_count:
                    response["rule_hits"].append({
                        "rule": "insert_partial_columns",
                        "agent_cols": agent_col_count,
                        "schema_cols": schema_col_count,
                        "hint": (
                            f"Harness: your INSERT specifies {agent_col_count} columns but the "
                            f"table has {schema_col_count}. The table hash includes ALL columns — "
                            f"include every column in your INSERT or the hash will not match."
                        ),
                    })

            # Unquoted numeric literals check: all columns are TEXT, so VALUES
            # like (1996, 97.0, 2) must be ('1996', '97.0', '2') to match hash.
            # Match bare numbers that appear as VALUE tokens: preceded by ( or ,
            # with optional whitespace, and NOT immediately following a quote char.
            m_vals = re.search(r"\bVALUES\s*\((.+)\)\s*;?\s*$", patched, re.IGNORECASE | re.DOTALL)
            if m_vals:
                vals_text = m_vals.group(1)
                # A token is an unquoted value position if preceded by ( or ,
                # (with whitespace), and the value itself is a bare number or NULL.
                _unquoted_num = re.compile(
                    r"(?:^|(?<=,))\s*([-+]?\d+(?:\.\d+)?|NULL)\s*(?=,|\))",
                    re.IGNORECASE,
                )
                bare_nums = _unquoted_num.findall(vals_text)
                if bare_nums:
                    response["rule_hits"].append({
                        "rule": "insert_unquoted_numerics",
                        "hint": (
                            "Harness: every column is TEXT — quote ALL values as strings: "
                            "VALUES ('Harbor View Tower', '30', '100.0 m.', '2024', …). "
                            "Do NOT use bare numbers (30, 97.0) — they must match the exact "
                            "string stored in the table including any units or suffixes."
                        ),
                    })

        # Repeated-SQL commit shortcut: if the same normalised SQL has been run
        # 2 times already AND the last run was successful AND we have a
        # candidate matching the expected shape → convert this 3rd attempt into
        # a commit.  We DO NOT use candidate_implausible here.
        normalised = _normalize_sql(patched)
        hist = self.state.sql_history
        if (
            len(hist) >= self.config.h2_repeat_sql_block_after
            and all(h == normalised for h in hist[-self.config.h2_repeat_sql_block_after:])
            and self.state.candidate_answer is not None
            and not self.state.candidate_implausible
            and not self.state.last_result_was_error
            and self.task_ctx.answer_shape not in (SHAPE_HASH, None)
        ):
            answers = self._candidate_to_answers_list()
            if answers is not None:
                response["action"] = "force_commit"
                response["force_args"] = {"answers": answers}
                response["rule_hits"].append({"rule": "repeated_sql_force_commit"})
                return response

        return response

    def _candidate_to_answers_list(self) -> Optional[List[str]]:
        cand = self.state.candidate_answer
        if cand is None:
            return None
        if isinstance(cand, list):
            return [str(x) for x in cand]
        return [str(cand)]

    # ── H2: Commit gate + answer normalisation ──────────────────────────────

    def gate_commit(self, answers: List[str]) -> Dict[str, Any]:
        """Decide whether to allow, normalise, or block a commit_final_answer.

        Returns: {action: "allow"|"block", answers, blocked_reason,
                  normalize_audit, rule_hits[]}
        """
        response: Dict[str, Any] = {
            "action": "allow",
            "answers": list(answers or []),
            "blocked_reason": "",
            "normalize_audit": None,
            "rule_hits": [],
        }
        if not self.config.h2_enabled or self.task_ctx is None:
            return response
        ctx = self.task_ctx
        st = self.state

        is_mutation = ctx.task_type in _MUTATION_TYPES

        # Block explanatory / "unable to find" text answers on mutation tasks.
        # When agents can't locate the target row they sometimes give up and submit
        # a sentence like "Unable to update..." — this always fails the hash check.
        _GIVE_UP_PATTERNS = re.compile(
            r"\b(unable|cannot|can't|not found|no such|not exist|could not|doesn't exist)\b",
            re.IGNORECASE,
        )
        if is_mutation and answers and any(_GIVE_UP_PATTERNS.search(str(a)) for a in answers):
            if st.mutation_commit_blocks_used < self.config.h2_mutation_commit_block_limit:
                st.mutation_commit_blocks_used += 1
                response["action"] = "block"
                response["blocked_reason"] = "give_up_text_on_mutation"
                response["rule_hits"].append({"rule": "block_mutation_give_up_text"})
                return response

        # Block empty commits on non-mutation tasks (once).
        # Also catches all-whitespace or all-empty-string answers like [""].
        _answers_all_blank = (
            not answers or all(not str(a).strip() for a in answers)
        )
        if _answers_all_blank and not is_mutation:
            if st.commit_blocks_used < self.config.h2_commit_block_limit:
                st.commit_blocks_used += 1
                response["action"] = "block"
                response["blocked_reason"] = "empty_answers_before_query"
                response["recovery_prompt"] = (
                    "Harness: you committed an empty or blank answer. "
                    "Run execute_sql to retrieve the actual value from the database "
                    "and commit the numeric/string result — NOT an empty string."
                )
                response["rule_hits"].append({"rule": "block_empty_answers"})
                return response

        # Block scalar tasks where agent commits "0" or "none" immediately after
        # getting an empty SQL result (likely a bad WHERE condition, not a true zero).
        _trivially_wrong_scalar = (
            not is_mutation
            and ctx.answer_shape in (SHAPE_SCALAR_INT, SHAPE_SCALAR_FLOAT)
            and len(answers) == 1
            and str(answers[0]).strip().lower() in ("0", "0.0", "none", "null", "")
            and st.last_error_kind == "empty"
            and st.sql_history  # only after at least one SQL
        )
        if _trivially_wrong_scalar:
            if st.commit_blocks_used < self.config.h2_commit_block_limit:
                st.commit_blocks_used += 1
                response["action"] = "block"
                response["blocked_reason"] = "zero_after_empty_result"
                response["recovery_prompt"] = (
                    f"Harness: your last SQL returned no rows, so submitting "
                    f"{answers[0]!r} is likely wrong. "
                    "Your WHERE condition probably didn't match anything. "
                    "Run `SELECT * FROM `tbl` LIMIT 5` to see actual values, "
                    "then adjust your query."
                )
                response["rule_hits"].append({"rule": "block_zero_after_empty"})
                return response

        # Mutation task: agent must have actually executed a mutation.
        # Uses a separate, higher block counter to keep prompting the agent.
        if is_mutation and not st.mutation_attempted:
            if st.mutation_commit_blocks_used < self.config.h2_mutation_commit_block_limit:
                st.mutation_commit_blocks_used += 1
                response["action"] = "block"
                response["blocked_reason"] = "mutation_not_executed"
                response["rule_hits"].append({"rule": "block_commit_before_mutation"})
                return response

        # Non-mutation task: agent must have run at least one execute_sql.
        if not is_mutation and not st.sql_history:
            if st.commit_blocks_used < self.config.h2_commit_block_limit:
                st.commit_blocks_used += 1
                response["action"] = "block"
                response["blocked_reason"] = "commit_before_any_sql"
                response["rule_hits"].append({"rule": "block_commit_before_sql"})
                return response

        # Block scalar tasks where agent submits multiple answers (e.g. GROUP BY result).
        # This is the #1 cause of SELECT scalar_int failures: agent runs GROUP BY,
        # gets N rows, and tries to commit all N values when a single total is expected.
        if (
            ctx.answer_shape in (SHAPE_SCALAR_INT, SHAPE_SCALAR_FLOAT)
            and len(answers) > 1
            and st.scalar_multi_answer_blocks_used < 3  # dedicated higher limit
        ):
            st.scalar_multi_answer_blocks_used += 1
            response["action"] = "block"
            response["blocked_reason"] = "scalar_expected_but_multiple_answers"
            response["recovery_prompt"] = (
                f"Harness: this task expects ONE number but you submitted {len(answers)} values. "
                "You likely used GROUP BY which returns one row per group. "
                "Remove GROUP BY and use a global aggregate: "
                "`SELECT COUNT(*) FROM `t` WHERE …` or `SELECT SUM(…) FROM `t` WHERE …`. "
                "Then commit the single result."
            )
            response["rule_hits"].append({"rule": "block_scalar_multi_answer"})
            return response

        # H2 answer normalisation (shape-aware).
        normalised, audit = normalize_answers_list(answers, ctx.answer_shape)

        # For multi-col SELECT: if the agent submitted flat cells (e.g. ["a","b","c","d"…])
        # instead of row tuple-strings ("('a','b')","('c','d')"), try to regroup them
        # using the last known column count so they match the evaluator's ground-truth format.
        if (
            ctx.answer_shape == SHAPE_MULTI_MULTI
            and self.state.last_result_col_count is not None
            and self.state.last_result_col_count > 1
            and len(normalised) > 1
            and not any(a.strip().startswith("(") for a in normalised)
        ):
            n = self.state.last_result_col_count
            if len(normalised) % n == 0:
                rows: List[str] = []
                for i in range(0, len(normalised), n):
                    cells = normalised[i : i + n]
                    row_str = "(" + ", ".join(f"'{c}'" for c in cells) + ")"
                    rows.append(row_str)
                normalised = rows
                audit.setdefault("rule_hits", []).append("regroup_flat_to_tuples")
                audit["mutated"] = True

        # Single-col tuple unwrap: task classified as MULTI_MULTI but actual query
        # returned only 1 column.  Agent over-formats as ["('v1',)", "('v2',)"] but
        # the evaluator's _clean_mysql_result returns bare ["v1", "v2"] for single-col
        # tuples, so they would never match without this unwrap.
        if (
            ctx.answer_shape == SHAPE_MULTI_MULTI
            and self.state.last_result_col_count is not None
            and self.state.last_result_col_count == 1
            and normalised
        ):
            _single_tuple_re = re.compile(r"^\('([^'\\]*)'\s*,?\s*\)$")
            unwrapped: List[str] = []
            any_unwrapped = False
            for a in normalised:
                m = _single_tuple_re.match(a.strip())
                if m:
                    unwrapped.append(m.group(1))
                    any_unwrapped = True
                else:
                    unwrapped.append(a)
            if any_unwrapped:
                normalised = unwrapped
                audit.setdefault("rule_hits", []).append("unwrap_single_col_tuples")
                audit["mutated"] = True

        response["answers"] = normalised
        response["normalize_audit"] = audit
        if audit.get("mutated"):
            response["rule_hits"].append({"rule": "normalize_answers", "hits": audit.get("rule_hits", [])})

        return response

    # ── H1: update after tool result ────────────────────────────────────────

    def update_state_after_sql(self, sql: str, response: str) -> None:
        normalised = _normalize_sql(sql or "")
        self.state.sql_history.append(normalised)
        self.state.sql_history_raw.append(sql or "")
        self.state.last_sql = sql
        self.state.last_result_raw = response or ""

        kind, err_text = classify_db_response(response or "")
        self.state.last_error_kind = kind
        self.state.last_error_text = err_text or ""
        self.state.last_result_was_error = kind in ("syntax", "unknown_col", "unknown_table", "timeout")

        if self.state.last_result_was_error:
            self.state.error_streak += 1
        else:
            self.state.error_streak = 0

        if kind == "empty":
            self.state.empty_streak += 1
        else:
            self.state.empty_streak = 0

        # Loop streak (same normalised SQL)
        if (
            len(self.state.sql_history) >= 2
            and self.state.sql_history[-1] == self.state.sql_history[-2]
        ):
            self.state.loop_streak += 1
        else:
            self.state.loop_streak = 0

        # Track mutation attempts — structural check only (task type + verb +
        # target table). We deliberately do not compare values against std_sql.
        if self.task_ctx is not None and self.task_ctx.task_type in _MUTATION_TYPES:
            verb = self.task_ctx.task_type.lower()
            if re.search(rf"\b{verb}\b", sql or "", re.IGNORECASE):
                if self.task_ctx.target_table_sanitized:
                    tbl = self.task_ctx.target_table_sanitized.lower()
                    if tbl in (sql or "").lower() and not self.state.last_result_was_error:
                        self.state.mutation_attempted = True
                elif not self.state.last_result_was_error:
                    self.state.mutation_attempted = True

        # Discovered columns — from DESCRIBE output
        if re.match(r"^\s*describe\b", (sql or ""), re.IGNORECASE):
            # MySQL DESCRIBE output is repr-string of tuples — first field per row is the column name.
            try:
                parsed = eval((response or "").strip(), {"__builtins__": {}}, {})
                if isinstance(parsed, list):
                    tbl_match = re.search(r"describe\s+`?([^`\s;]+)`?", sql or "", re.IGNORECASE)
                    tbl_name = tbl_match.group(1) if tbl_match else "_"
                    cols = [str(row[0]) for row in parsed if isinstance(row, tuple) and row]
                    if cols:
                        self.state.discovered_columns[tbl_name] = cols
            except Exception:
                pass

        # Track column count for multi-col results (used for flat-cell regroup in gate_commit).
        if not self.state.last_result_was_error and self.task_ctx is not None:
            resp_s = (response or "").strip()
            if resp_s.startswith("[") and resp_s.endswith("]"):
                try:
                    parsed_r = eval(resp_s, {"__builtins__": {}}, {})
                    if isinstance(parsed_r, list) and parsed_r and isinstance(parsed_r[0], tuple):
                        self.state.last_result_col_count = len(parsed_r[0])
                        self.state.last_result_row_count = len(parsed_r)
                except Exception:
                    pass

        # Candidate extraction — only when the response is ok AND the query is not
        # a schema-inspection command (DESCRIBE / SHOW / EXPLAIN). Those return
        # metadata rows (column definitions) that look like valid results but are
        # not answers; promoting them as candidates causes H5 to tell the agent to
        # commit schema output as the answer.
        _is_schema_inspect = bool(re.match(
            r"\s*(describe|show\s+tables|show\s+columns|explain)\b",
            sql or "", re.IGNORECASE
        ))
        if not self.state.last_result_was_error and self.task_ctx is not None and not _is_schema_inspect:
            cand, implausible = extract_candidate_from_response(
                response or "", self.task_ctx.answer_shape
            )
            if cand is not None:
                self.state.candidate_answer = cand
                self.state.candidate_answer_shape = self.task_ctx.answer_shape
                self.state.candidate_implausible = bool(implausible)
            elif kind == "null_agg" and self.task_ctx.answer_shape in (SHAPE_SCALAR_INT, SHAPE_SCALAR_FLOAT):
                # NULL aggregate → "0" candidate (evaluator maps None → "0").
                self.state.candidate_answer = "0"
                self.state.candidate_answer_shape = self.task_ctx.answer_shape
                # Mark plausible — the evaluator treats None as "0" so this is a legit answer.
                self.state.candidate_implausible = False

    def note_text_only_turn(self) -> None:
        self.state.text_only_streak += 1

    def reset_text_only_streak(self) -> None:
        self.state.text_only_streak = 0

    # ── H4: Post-step monitor (with budget merge) ───────────────────────────

    def post_step_monitor(self, remaining_rounds: int = 99) -> Dict[str, Any]:
        """Return a recovery prompt + optional force action.

        remaining_rounds = max_round - completed_rounds_so_far.
        Careful on false positives (per user warning).
        """
        response: Dict[str, Any] = {
            "audit_reason": "",
            "recovery_prompt": None,
            "force_action": None,
            "budget_branch": None,
        }
        if not self.config.h4_enabled or self.task_ctx is None:
            self.state.h4_fired_last_round = False
            return response
        st = self.state
        ctx = self.task_ctx
        prior_h4 = st.h4_fired_last_round

        def _return(r: Dict[str, Any]) -> Dict[str, Any]:
            st.h4_fired_last_round = bool(r.get("recovery_prompt")) or (r.get("force_action") is not None)
            return r

        # ⓪-pre Text-only loop: agent keeps writing tool calls as plain text
        # and failing rescue for N consecutive turns. Force a commit or break the loop.
        if st.text_only_streak >= 3:
            response["audit_reason"] = "text_only_loop"
            if (
                st.candidate_answer is not None
                and not st.candidate_implausible
                and ctx.answer_shape not in (SHAPE_HASH, None)
            ):
                answers = self._candidate_to_answers_list()
                if answers is not None:
                    response["force_action"] = {
                        "name": "commit_final_answer",
                        "arguments": {"answers": answers},
                    }
                    response["recovery_prompt"] = (
                        "Harness: you have been writing tool calls as plain text. "
                        "Forcing commit_final_answer with last good candidate."
                    )
                    return _return(response)
            response["recovery_prompt"] = (
                "Harness: you must use the function-calling API, NOT plain text. "
                "Call execute_sql or commit_final_answer via the tool interface directly."
            )
            return _return(response)

        # ⓪ Syntax error — parse the near-token if possible.
        if st.last_error_kind == "syntax":
            near = None
            m = _MYSQL_SYNTAX_NEAR_RE.search(st.last_result_raw or "")
            if m:
                near = m.group(1).strip()
            extra = f" The error is near `{near}` — check if this is an un-backticked identifier." if near else ""
            response["audit_reason"] = "syntax_error"
            response["recovery_prompt"] = (
                "Harness: MySQL syntax error." + extra +
                " Wrap any identifier with spaces/punctuation in backticks (e.g. `Race Name`), "
                "and use `CONCAT(a, b)` instead of `a || b`."
            )
            return _return(response)

        # ① Unknown column
        if st.last_error_kind == "unknown_col":
            m = _MYSQL_UNKNOWN_COL_RE.search(st.last_result_raw or "")
            col = m.group(1) if m else None
            hint_tbl = ctx.target_table_sanitized or "target_table"
            detail = f" (column '{col}' not found)" if col else ""
            response["audit_reason"] = "unknown_column"
            response["recovery_prompt"] = (
                f"Harness: Unknown column{detail}. Run `DESCRIBE `{hint_tbl}`;` to see "
                f"the real column names — column names were truncated to 64 chars on import, "
                f"so the descriptive name in the question may not match the actual column."
            )
            return _return(response)

        # ② Unknown table
        if st.last_error_kind == "unknown_table":
            response["audit_reason"] = "unknown_table"
            response["recovery_prompt"] = (
                "Harness: Table not found. Run `SHOW TABLES;` to list actual table names — "
                "the table name may have been sanitized during import."
            )
            return _return(response)

        # ②.5 Mutation task but only SELECT queries executed — strong reminder.
        if (
            ctx.task_type in _MUTATION_TYPES
            and len(st.sql_history) >= 2
            and not st.mutation_attempted
            and all(s.strip().startswith("select") for s in st.sql_history)
        ):
            response["audit_reason"] = "mutation_only_select"
            verb = ctx.task_type
            example = (
                f"`INSERT INTO `tbl` (…) VALUES (…)`" if verb == "INSERT" else
                f"`UPDATE `tbl` SET col='…' WHERE …`" if verb == "UPDATE" else
                f"`DELETE FROM `tbl` WHERE …`"
            )
            response["recovery_prompt"] = (
                f"Harness: this is a {verb} task but you have only run SELECT queries. "
                f"You MUST execute a mutation SQL now, e.g. {example}. "
                f"SELECT queries do not modify the table."
            )
            return _return(response)

        # ③ NULL aggregate — for aggregation task types and SELECT tasks using agg functions.
        _uses_agg = bool(re.search(r"\b(sum|avg|count)\s*\(", st.last_sql or "", re.IGNORECASE))
        if st.last_error_kind == "null_agg" and (
            ctx.task_type in _AGGREGATION_TYPES
            or (ctx.task_type in (TASK_SELECT, TASK_COUNTING) and _uses_agg)
        ):
            response["audit_reason"] = "null_aggregate"
            response["recovery_prompt"] = (
                "Harness: aggregate returned NULL. The numeric column may be TEXT-typed — "
                "wrap it: `CAST(`col` AS DECIMAL(20,6))`. If truly no matching rows, "
                "submit '0' (the grader maps None/null → '0')."
            )
            return _return(response)

        # ③.5 Mutation task with persistent empty SELECT — suggest relaxing filter.
        # Agents often run SELECT to preview the target row before mutating; when
        # the SELECT returns empty they give up rather than trying a different filter.
        if (
            ctx.task_type in _MUTATION_TYPES
            and st.last_error_kind == "empty"
            and st.empty_streak >= 2
            and not st.mutation_attempted
        ):
            response["audit_reason"] = "mutation_empty_select"
            response["recovery_prompt"] = (
                "Harness: your SELECT returned no rows. The target row may use "
                "different capitalisation, spacing, or special characters. "
                "Try `SELECT * FROM `tbl` LIMIT 5;` to see actual values, then "
                "match exactly. Do NOT give up — run the mutation SQL once you "
                "find the correct WHERE clause."
            )
            return _return(response)

        # ④ Empty results for query-shape tasks — only after N consecutive empties.
        _query_task_types = (
            TASK_SELECT, TASK_COUNTING, TASK_COMPARISON,
            TASK_AGG_SUM, TASK_AGG_AVG, TASK_AGG_COUNT,
            TASK_AGG_MIN, TASK_AGG_MAX, TASK_RANKING, TASK_OTHER,
        )
        if (
            st.last_error_kind == "empty"
            and ctx.task_type in _query_task_types
            and st.empty_streak >= self.config.h4_empty_threshold
        ):
            response["audit_reason"] = "empty_result"
            response["recovery_prompt"] = (
                "Harness: zero rows for several queries. Filter may be too strict — try "
                "`LIKE '%X%'` for partial match, drop a WHERE clause, or use LOWER() for "
                "case-insensitive compare. Run `SELECT * FROM `tbl` LIMIT 5` to see actual values."
            )
            return _return(response)

        # ⑤ SQL loop — N identical SQL in a row. Budget force if we have a good candidate.
        if (
            len(st.sql_history) >= self.config.h4_stall_window
            and len(set(st.sql_history[-self.config.h4_stall_window:])) == 1
        ):
            response["audit_reason"] = "sql_loop"
            if (
                st.candidate_answer is not None
                and not st.candidate_implausible
                and ctx.answer_shape not in (SHAPE_HASH, None)
            ):
                answers = self._candidate_to_answers_list()
                if answers is not None:
                    response["force_action"] = {
                        "name": "commit_final_answer",
                        "arguments": {"answers": answers},
                    }
                    response["recovery_prompt"] = (
                        f"Harness: you ran the same SQL {self.config.h4_stall_window} times. "
                        "The output already contains the answer — forcing commit_final_answer."
                    )
                    return _return(response)
            response["recovery_prompt"] = (
                "Harness: you are repeating the same SQL. Try a different approach — "
                "DESCRIBE the table, relax the WHERE clause, or cast the numeric column."
            )
            return _return(response)

        # ⑥ Budget management (H4 sub-branch).  Only fires when NO earlier branch did.
        rem = remaining_rounds
        budget_force_threshold = self.config.h4_budget_force_threshold
        budget_warn_threshold = self.config.h4_budget_warn_threshold

        if (
            rem <= budget_force_threshold
            and st.candidate_answer is not None
            and not st.candidate_implausible
            and ctx.answer_shape not in (SHAPE_HASH, None)
        ):
            answers = self._candidate_to_answers_list()
            if answers is not None:
                response["audit_reason"] = "budget_force"
                response["budget_branch"] = "force"
                response["force_action"] = {
                    "name": "commit_final_answer",
                    "arguments": {"answers": answers},
                }
                response["recovery_prompt"] = (
                    f"[{rem} rounds left] Harness: forcing commit_final_answer with "
                    f"candidate from last successful query."
                )
                return _return(response)
        if rem <= budget_warn_threshold and not st.h4_fired_last_round:
            response["audit_reason"] = "budget_warn"
            response["budget_branch"] = "warn"
            if (
                st.candidate_answer is not None
                and not st.candidate_implausible
                and ctx.answer_shape not in (SHAPE_HASH, None)
            ):
                response["recovery_prompt"] = (
                    f"[{rem} rounds left] Harness: you have a valid candidate answer from "
                    "your last successful query. Call commit_final_answer now."
                )
            else:
                response["recovery_prompt"] = (
                    f"[{rem} rounds left] Harness: finalise your approach — only a few "
                    "rounds left. Prefer DESCRIBE / LIKE / CAST if the last query didn't work."
                )
            return _return(response)

        # No trigger — reset flag.
        st.h4_fired_last_round = False
        return response

    # ── H4-E: state-driven per-step guidance ────────────────────────────────

    def step_guidance(
        self,
        round_num: int,
        h4_audit_active: bool = False,
    ) -> Optional[str]:
        """H4-E: Per-step guidance driven by H1 runtime state (candidate answers, SQL history)."""
        if not self.config.h4_enabled or self.task_ctx is None:
            return None
        ctx = self.task_ctx
        st = self.state

        hint: Optional[str] = None

        # 0. Promote candidate answer on non-mutation tasks (suppressed when H4 just fired)
        if (
            not h4_audit_active
            and ctx.answer_shape not in (SHAPE_HASH, None)
            and st.candidate_answer is not None
            and not st.candidate_implausible
            and st.sql_history
        ):
            if ctx.answer_shape == SHAPE_MULTI_MULTI and st.last_result_col_count and st.last_result_col_count > 1:
                row_note = f"{st.last_result_row_count} rows " if st.last_result_row_count else "rows "
                hint = (
                    f"Harness hint: last query returned {row_note}with "
                    f"{st.last_result_col_count} columns. Submit ALL rows — each as "
                    f"one tuple-repr element: answers=[\"('v1','v2')\", …]. "
                    f"Call commit_final_answer now."
                )
            else:
                preview = str(st.candidate_answer)[:60]
                hint = (
                    f"Harness hint: last successful query returned `{preview}`. "
                    f"Submit the bare value(s) — no tuple brackets. "
                    f"Call commit_final_answer now."
                )

        # 1. Semantic-gap lint — only when no candidate yet.
        if hint is None and st.sql_history_raw and st.candidate_answer is None:
            last_sql = st.sql_history_raw[-1] or ""
            low = last_sql.lower()
            if ctx.mentions_like and " like " not in low and "=" in low:
                hint = (
                    "Harness lint: task mentions 'contains / includes' — use "
                    "`WHERE col LIKE '%word%'`, not `= 'word'`."
                )
            elif ctx.case_insensitive and "lower(" not in low and "collate" not in low:
                hint = (
                    "Harness lint: task says 'ignoring case' — wrap the compared "
                    "expression with LOWER() on both sides."
                )
            elif ctx.task_type == TASK_RANKING and " order by " not in low:
                hint = (
                    "Harness lint: ranking task needs `ORDER BY CAST(`col` AS SIGNED) [DESC] LIMIT 1`."
                )

        # 2. Round-0 template — only when no SQL has been run yet.
        if hint is None and round_num == 0 and not st.sql_history:
            hint = self._first_turn_hint()

        # 3. Implausibility warn — fires for counting/agg AND for SELECT/comparison
        #    tasks where a scalar is expected but multi-row was returned.
        if hint is None and st.candidate_implausible and not h4_audit_active:
            if ctx.task_type in (TASK_COUNTING, *_AGGREGATION_TYPES):
                hint = (
                    "Harness: the last numeric result looks suspicious (0 / None from an "
                    "over-strict filter). Relax the WHERE clause or cast the column before "
                    "committing."
                )
            elif (
                ctx.task_type in (TASK_SELECT, TASK_COMPARISON, TASK_RANKING)
                and ctx.answer_shape in (SHAPE_SCALAR_INT, SHAPE_SCALAR_FLOAT, SHAPE_SCALAR_STR)
            ):
                hint = (
                    "Harness: multiple rows returned but task expects a single value. "
                    "Use COUNT(*), MAX/MIN, or add a more specific WHERE / LIMIT 1."
                )

        if hint is None:
            return None

        words = hint.split()
        if len(words) > self.config.h4_hint_max_words:
            hint = " ".join(words[: self.config.h4_hint_max_words])

        if hint == self._last_hint:
            return None
        self._last_hint = hint
        return hint

    def _first_turn_hint(self) -> Optional[str]:
        ctx = self.task_ctx
        if ctx is None:
            return None
        tbl = f"`{ctx.target_table_sanitized}`" if ctx.target_table_sanitized else "`<table>`"

        if ctx.task_type in (TASK_COUNTING, TASK_AGG_COUNT):
            return f"Hint: try `SELECT COUNT(*) FROM {tbl} WHERE …;` — commit the bare integer."
        if ctx.task_type == TASK_RANKING:
            return f"Hint: `SELECT `name_col` FROM {tbl} ORDER BY CAST(`num_col` AS SIGNED) DESC LIMIT 1;` — cast if the column is TEXT."
        if ctx.task_type == TASK_SELECT:
            return (
                f"Hint: `SELECT * FROM {tbl} WHERE …;` — "
                f"for multi-column results submit each ROW as one tuple-repr element: "
                f"answers=[\"('v1','v2')\"]; for single-column submit bare values: answers=['v1','v2']."
            )
        if ctx.task_type == TASK_AGG_SUM:
            return f"Hint: `SELECT SUM(CAST(`col` AS DECIMAL(20,6))) FROM {tbl} WHERE …;` — submit '0' if NULL."
        if ctx.task_type == TASK_AGG_AVG:
            return f"Hint: `SELECT AVG(CAST(`col` AS DECIMAL(20,6))) FROM {tbl} WHERE `col` != '';`."
        if ctx.task_type in (TASK_AGG_MAX, TASK_AGG_MIN):
            fn = "MAX" if ctx.task_type == TASK_AGG_MAX else "MIN"
            return f"Hint: `SELECT {fn}(CAST(`col` AS DECIMAL(20,6))) FROM {tbl};` — cast for numeric TEXT columns."
        if ctx.task_type == TASK_INSERT:
            return (
                f"Hint: `INSERT INTO {tbl} (col1, …) VALUES ('val1', …);` — "
                "ALL values must be quoted strings matching the exact format in the sample rows "
                "(units, ordinals, date strings preserved). Then commit_final_answer."
            )
        if ctx.task_type == TASK_UPDATE:
            return f"Hint: `UPDATE {tbl} SET `col` = '…' WHERE …;` — ALWAYS include WHERE."
        if ctx.task_type == TASK_DELETE:
            return f"Hint: `DELETE FROM {tbl} WHERE …;` — ALWAYS include WHERE."
        if ctx.task_type == TASK_COMPARISON:
            return f"Hint: `SELECT … FROM {tbl} WHERE …;` — compare using `>`, `<`, or LIKE as appropriate."
        return None
