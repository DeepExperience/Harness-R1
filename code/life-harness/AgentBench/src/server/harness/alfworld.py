"""
ALFWorld Harness — H0/H2/H3/H4/H5

H0  Task Parser        — one-time structured task parsing per episode
H2  Action Gate        — verb-safe canonicalization + invalid blocking
H3  Tool Description   — task-type-aware tool hint embedding
H4  Process Monitor    — world model, subgoal state machine, stall detection,
                         step-budget urgency + forced completion (budget sub-branch)
H5  Goal-directed Hint — cold-start skill + per-step guided suggestions

Generalization notes:
  - _SEARCH_PRIORITY is based on household commonsense, NOT training statistics.
  - H5 step guidance uses "suggestion" language; hard-forcing only in the H4
    budget sub-branch when world-model evidence is unambiguous (item in hand +
    put action in admissible).
  - All observation/action patterns match the ALFWorld environment API, which is
    identical across train/test splits.
"""

import copy
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_to_word_budget(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]).strip()


def _verb_of(action: str) -> str:
    """Return the first word (verb) of an action string."""
    s = action.strip().lower()
    return s.split()[0] if s else ""


def _extract_objects(observation: str) -> List[str]:
    """Extract 'word number' object tokens from an observation string (fixed regex)."""
    if not observation:
        return []
    return re.findall(r"\b([a-z]+\s+\d+)\b", observation.lower())


# ─────────────────────────────────────────────────────────────────────────────
# BM25 retrieval helpers (used by H5 cold-start)
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _skill_document_tokens(skill: Dict[str, Any]) -> List[str]:
    parts = [skill.get("text", "")] + list(skill.get("keywords", []))
    return _tokenize(" ".join(parts))


def _bm25_scores(
    query_tokens: List[str],
    docs: List[List[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    if not docs:
        return []
    n_docs = len(docs)
    avgdl = sum(len(d) for d in docs) / n_docs
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


def retrieve_skills_for_task(
    task_type: str,
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Two-layer skill retrieval:
      1. Hard filter  — keep only skills whose task_types include task_type.
      2. BM25 ranking — rank filtered candidates by relevance to query.
    Returns up to top_k results.
    """
    candidates = [s for s in ALF_SKILLS if task_type in s.get("task_types", [])]
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates   # no need to rank if fits within budget
    query_tokens = _tokenize(query)
    if not query_tokens:
        return candidates[:top_k]
    docs = [_skill_document_tokens(s) for s in candidates]
    scores = _bm25_scores(query_tokens, docs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ALFWorldHarnessConfig:
    enabled: bool = False
    h2_enabled: bool = True
    h3_enabled: bool = True
    h4_enabled: bool = True
    h5_enabled: bool = True

    # H2
    action_similarity_threshold: float = 0.55   # raised from 0.20
    never_block_prefixes: Tuple[str, ...] = ("go to ", "open ", "close ")
    invalid_block_after: int = 2

    # H3
    h3_max_words: int = 20

    # H4
    h4_stall_window: int = 4
    h4_soft_intervention_rounds: int = 2
    h4_min_rounds_before_stall: int = 8
    h4_post_put_grace: int = 4

    # H5
    h5_top_k: int = 1          # max skills injected at cold-start
    h5_cold_start_max_words: int = 35     # word cap per cold-start skill
    h5_step_hint_max_words: int = 25      # word cap for per-step guidance

    # H4 — extra stall patterns
    h4_nothing_happens_window: int = 4      # obs window for "nothing happens" loop check
    h4_nothing_happens_threshold: int = 3   # how many "nothing happens" obs to trigger

    # H2 — empty-turn intervention
    h2_empty_turn_threshold: int = 2        # consecutive empty actions before reminder

    # H4 budget
    h4_budget_warn_threshold: int = 7       # remaining steps < N → inject urgency hint
    h4_budget_force_threshold: int = 4      # remaining steps < N → force PUT if possible
    h4_budget_warn_threshold_multistep: int = 12  # higher warn for pick_two_obj / pick_cool


# ─────────────────────────────────────────────────────────────────────────────
# H0 — Task Context
# ─────────────────────────────────────────────────────────────────────────────

# Subgoal labels
_SG_FIND      = "FIND"
_SG_TAKE      = "TAKE"
_SG_GOTO_SINK = "GOTO_SINK"
_SG_CLEAN     = "CLEAN"
_SG_GOTO_MCW  = "GOTO_MICROWAVE"
_SG_HEAT      = "HEAT"
_SG_GOTO_FRG  = "GOTO_FRIDGE"
_SG_COOL      = "COOL"
_SG_GOTO_DEST = "GOTO_DEST"
_SG_PUT       = "PUT"
_SG_GOTO_LAMP = "GOTO_LAMP"
_SG_USE_LAMP  = "USE_LAMP"
_SG_EXAMINE   = "EXAMINE"
_SG_DONE      = "DONE"

_SUBGOAL_CHAINS: Dict[str, List[str]] = {
    "pick_and_place":        [_SG_FIND, _SG_TAKE, _SG_GOTO_DEST, _SG_PUT],
    "pick_clean_then_place": [_SG_FIND, _SG_TAKE, _SG_GOTO_SINK, _SG_CLEAN, _SG_GOTO_DEST, _SG_PUT],
    "pick_heat_then_place":  [_SG_FIND, _SG_TAKE, _SG_GOTO_MCW,  _SG_HEAT,  _SG_GOTO_DEST, _SG_PUT],
    "pick_cool_then_place":  [_SG_FIND, _SG_TAKE, _SG_GOTO_FRG,  _SG_COOL,  _SG_GOTO_DEST, _SG_PUT],
    # look_at_obj: lamp must be ON before examine action appears; agent must also
    # TAKE the target (examine X with desklamp only shows in admissible when holding X).
    "look_at_obj":           [_SG_GOTO_LAMP, _SG_USE_LAMP, _SG_FIND, _SG_TAKE, _SG_EXAMINE],
    # pick_two_obj: two full pick-place cycles (agent can only hold one item at a time)
    "pick_two_obj":          [
        _SG_FIND, _SG_TAKE, _SG_GOTO_DEST, _SG_PUT,
        _SG_FIND, _SG_TAKE, _SG_GOTO_DEST, _SG_PUT,
    ],
}

# Transformation keyword → intermediate location name
_XFORM_LOCATION: Dict[str, str] = {
    "clean": "sinkbasin",
    "heat":  "microwave",
    "cool":  "fridge",
}

# Search priority: household commonsense, not derived from training statistics.
# Only used as a soft hint for exploration ordering, never as a hard force.
_SEARCH_PRIORITY: Dict[str, List[str]] = {
    "kettle":      ["stoveburner", "cabinet", "countertop", "microwave"],
    "pan":         ["stoveburner", "cabinet", "countertop"],
    "pot":         ["stoveburner", "cabinet", "countertop"],
    "plate":       ["cabinet", "countertop", "shelf", "drawer"],
    "bowl":        ["cabinet", "countertop", "shelf", "drawer"],
    "cup":         ["cabinet", "countertop", "shelf", "drawer"],
    "mug":         ["cabinet", "shelf", "countertop", "desk"],
    "fork":        ["drawer", "countertop", "diningtable"],
    "knife":       ["drawer", "countertop", "diningtable"],
    "spoon":       ["drawer", "countertop", "diningtable"],
    "butterknife": ["drawer", "countertop", "diningtable"],
    "spatula":     ["drawer", "countertop", "diningtable"],
    "newspaper":   ["coffeetable", "diningtable", "sidetable", "sofa", "ottoman"],
    "book":        ["shelf", "sidetable", "coffeetable"],
    "handtowel":   ["handtowelholder", "cabinet", "countertop"],
    "cloth":       ["handtowelholder", "countertop", "cabinet"],
    "soapbar":     ["bathtubbasin", "sinkbasin", "cabinet", "shelf"],
    "toiletpaper": ["toiletpaperhanger", "cabinet", "shelf"],
    "keychain":    ["sidetable", "drawer", "coffeetable"],
    "cellphone":   ["sidetable", "drawer", "coffeetable", "desk"],
    "creditcard":  ["sidetable", "drawer", "coffeetable"],
    "pencil":      ["desk", "drawer", "sidetable"],
    "pen":         ["desk", "drawer", "sidetable"],
    "cd":          ["desk", "shelf", "drawer"],
    "tomato":      ["fridge", "countertop", "diningtable"],
    "apple":       ["fridge", "countertop", "diningtable"],
    "bread":       ["cabinet", "countertop", "microwave"],
    "egg":         ["fridge", "countertop"],
    "lettuce":     ["fridge", "countertop", "diningtable"],
    "potato":      ["fridge", "countertop", "diningtable"],
    "dishsponge":  ["sinkbasin", "cabinet", "countertop"],
    "glassbottle": ["cabinet", "countertop", "fridge"],
    "remotecontrol": ["sofa", "sidetable", "coffeetable", "ottoman"],
    # Bedroom / living room objects missing from original list
    "pillow":       ["bed", "sofa", "armchair", "ottoman", "sidetable"],
    "vase":         ["shelf", "sidetable", "diningtable", "coffeetable"],
    "laptop":       ["desk", "sidetable", "coffeetable"],
    "watch":        ["sidetable", "drawer", "desk"],
    "statue":       ["shelf", "sidetable", "diningtable"],
    "candle":       ["sidetable", "shelf", "diningtable"],
    "tissuebox":    ["sidetable", "coffeetable", "shelf", "countertop"],
    "spraybottle":  ["cabinet", "countertop", "bathtubbasin"],
    "ladle":        ["drawer", "countertop", "cabinet"],
    "peppershaker": ["countertop", "cabinet", "diningtable"],
    "saltshaker":   ["countertop", "cabinet", "diningtable"],
    "scrubbrush":   ["sinkbasin", "bathtubbasin", "cabinet"],
    "box":          ["shelf", "cabinet", "desk", "sidetable"],
    "alarmclock":   ["sidetable", "desk", "shelf"],
    "baseball":     ["desk", "shelf", "ottoman", "sidetable"],
    "basketball":   ["ottoman", "sofa", "shelf"],
    "tennisracket": ["shelf", "ottoman", "sidetable"],
}
_SEARCH_PRIORITY_DEFAULT = [
    "countertop", "cabinet", "shelf", "drawer",
    "diningtable", "sidetable", "coffeetable",
    "desk", "fridge", "sofa",
]


@dataclass
class TaskContext:
    task_type: str
    target_type: str               # e.g. "kettle"
    destination_type: str          # e.g. "countertop"
    transformation: Optional[str]  # "clean" / "heat" / "cool" / None
    count: int                     # 1 or 2
    subgoal_chain: List[str]
    task_goal: str                 # raw task description, used as BM25 query


def _detect_task_type(desc: str) -> str:
    d = desc.lower()
    if re.search(r"\bclean\b", d):
        return "pick_clean_then_place"
    if re.search(r"\bheat\b|\bhot\b", d):
        return "pick_heat_then_place"
    if re.search(r"\bcool\b|\bcold\b", d):
        return "pick_cool_then_place"
    # ALFWorld uses two phrasings for look_at_obj:
    #   "examine the X with the desklamp"
    #   "look at X under the desklamp"
    if re.search(r"\bexamine\b|\blook\s+at\b", d):
        return "look_at_obj"
    if re.search(r"\btwo\b", d):
        return "pick_two_obj"
    return "pick_and_place"


def _extract_target_dest(task_desc: str) -> Tuple[str, str]:
    d = task_desc.lower()
    for pfx in ("your task is to:", "here is your task.", "here is your task:"):
        if pfx in d:
            d = d.split(pfx, 1)[-1].strip()
    d = d.rstrip(".").strip()

    # "examine the X [Y] with the desklamp"  — one phrasing used by ALFWorld
    m = re.search(r"examine\s+the\s+([\w]+(?:\s+[\w]+)?)\s+with", d)
    if m:
        return m.group(1).strip(), "desklamp"

    # "look at X under the desklamp"  — alternate phrasing used by ALFWorld
    m = re.search(r"look\s+at\s+([\w]+(?:\s+[\w]+)?)\s+under", d)
    if m:
        return m.group(1).strip(), "desklamp"

    # "find two/a/some X and put them/it in/on Y"
    m = re.search(
        r"find\s+(?:two|a|some)\s+(\w+)\s+and\s+put\s+(?:it|them)\s+(?:in|on)\s+(\w+)", d
    )
    if m:
        # strip plural 's' for pick_two_obj (e.g. "pillows" → "pillow")
        target = m.group(1)
        if target.endswith("s") and len(target) > 3:
            target = target[:-1]
        return target, m.group(2)

    # "clean/heat/cool some X and put it in/on Y"
    m = re.search(
        r"(?:clean|heat|cool)\s+some\s+(\w+)\s+and\s+put\s+it\s+(?:in|on)\s+(\w+)", d
    )
    if m:
        return m.group(1), m.group(2)

    # "put a/some/two [adjective] X in/on Y"  (adj = hot/cool/cold/clean/heated/cooled)
    m = re.search(
        r"put\s+(?:a|some|two)\s+(?:hot|cool|cold|clean|heated|cooled|cleaned)?\s*(\w+)\s+(?:in|on)\s+(\w+)",
        d,
    )
    if m:
        tgt, dst = m.group(1).strip(), m.group(2).strip()
        if tgt not in ("it", "them", "and"):
            return tgt, dst

    return "", ""


def parse_task_context(init_prompt: str) -> TaskContext:
    """H0: Parse the episode's init_prompt into a structured TaskContext."""
    m = re.search(r"[Yy]our task is to:\s*(.+?)(?:\.?\s*AVAILABLE|\.|$)", init_prompt)
    task_goal = m.group(1).strip() if m else init_prompt

    task_type = _detect_task_type(task_goal)
    target_type, destination_type = _extract_target_dest(task_goal)
    transformation = {
        "pick_clean_then_place": "clean",
        "pick_heat_then_place":  "heat",
        "pick_cool_then_place":  "cool",
    }.get(task_type)
    count = 2 if task_type == "pick_two_obj" else 1
    chain = list(_SUBGOAL_CHAINS.get(task_type, [_SG_FIND, _SG_TAKE, _SG_GOTO_DEST, _SG_PUT]))
    return TaskContext(
        task_type=task_type,
        target_type=target_type,
        destination_type=destination_type,
        transformation=transformation,
        count=count,
        subgoal_chain=chain,
        task_goal=task_goal,
    )


# ─────────────────────────────────────────────────────────────────────────────
# H4 — World Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorldModel:
    """Incremental spatial knowledge built from observations and actions."""
    object_at: Dict[str, str] = field(default_factory=dict)
    inventory: Optional[str] = None
    visited: set = field(default_factory=set)
    unvisited: List[str] = field(default_factory=list)
    current_location: Optional[str] = None
    target_found: bool = False
    target_location: Optional[str] = None
    placed_count: int = 0
    placed_locations: List[str] = field(default_factory=list)  # destination locs (for hint filter)
    placed_items: List[str] = field(default_factory=list)       # specific item names we placed
    lamp_location: Optional[str] = None  # location where desklamp was turned on (look_at_obj)

    def update(
        self,
        action: str,
        observation: str,
        admissible: List[str],
        target_type: str,
    ) -> None:
        obs_lower = (observation or "").lower()
        act_lower = (action or "").strip().lower()

        # Track current location from navigation actions
        goto_m = re.match(r"go to (.+)", act_lower)
        if goto_m:
            loc = goto_m.group(1).strip()
            self.current_location = loc
            self.visited.add(loc)
            if loc in self.unvisited:
                self.unvisited.remove(loc)

        # Inventory: "you pick up the X from Y"
        pick_m = re.search(r"you pick up the (.+?) from", obs_lower)
        if pick_m:
            self.inventory = pick_m.group(1).strip()

        # Inventory: "you put the X in/on the Y" → drop item, record what/where
        if act_lower.startswith("put ") and re.search(r"you put the .+? (?:in|on) ", obs_lower):
            if self.inventory and self.inventory not in self.placed_items:
                self.placed_items.append(self.inventory)   # remember this specific item
            self.inventory = None
            self.placed_count += 1
            if self.current_location and self.current_location not in self.placed_locations:
                self.placed_locations.append(self.current_location)

        # Parse objects visible at current location
        if self.current_location:
            for obj in _extract_objects(observation):
                if obj not in self.object_at:
                    self.object_at[obj] = self.current_location
                # P3 (revised): exclude specific items we have already placed —
                # not locations, so naturally-present items at the destination
                # can still be detected as new targets.
                if (target_type
                        and not self.target_found
                        and obj.split()[0] == target_type.lower()
                        and obj not in self.placed_items):
                    self.target_found = True
                    self.target_location = self.current_location

        # Extend unvisited from "go to X" entries in admissible
        for cmd in admissible:
            if cmd.startswith("go to "):
                loc = cmd[len("go to "):]
                if loc not in self.visited and loc not in self.unvisited:
                    self.unvisited.append(loc)

    # Locations that are very unlikely to contain any useful item.
    # Kept as absolute last resort for navigation hints.
    _DEPRIORITIZED_LOCS = ("garbagecan", "bathtubbasin", "toiletpaperhangerbox")

    def ordered_unvisited(self, target_type: str) -> List[str]:
        """Unvisited locations sorted by commonsense priority for target_type.

        Locations that are extremely unlikely to contain any household item
        (garbagecan, bathtubbasin) are ranked last so navigation hints
        never suggest them unless no other option remains.
        """
        priority = _SEARCH_PRIORITY.get(target_type.lower(), _SEARCH_PRIORITY_DEFAULT)
        n_priority = len(priority)

        def rank(loc: str) -> int:
            ll = loc.lower()
            for i, p in enumerate(priority):
                if ll.startswith(p):
                    return i
            # Push deprioritized locations past everything else
            if any(ll.startswith(d) for d in WorldModel._DEPRIORITIZED_LOCS):
                return n_priority + 100
            return n_priority

        return sorted(self.unvisited, key=rank)

    def find_take_action(self, target_type: str, admissible: List[str]) -> Optional[str]:
        """Return a take action for target_type, skipping already-placed items and locations.

        Two filters prevent the second-cycle loop in pick_two_obj:
          1. placed_items  — skip named items we already placed ("take soapbottle 2 from ...")
          2. placed_locations — skip any take action whose source is a destination we placed at
             ("take tissuebox 2 from shelf 2" where shelf 2 is where we deposited item 1).
             This prevents the agent from picking up an item that was already at the destination
             (either naturally or placed there by us) and re-placing it there.
        """
        tt_lower = target_type.lower()
        placed_lower = [p.lower() for p in self.placed_items]
        placed_loc_lower = [loc.lower() for loc in self.placed_locations]
        for a in admissible:
            if not a.startswith("take ") or tt_lower not in a.lower():
                continue
            # Skip if this action targets a specific item we've already placed
            if any(p in a.lower() for p in placed_lower):
                continue
            # Skip if the source location is somewhere we've already deposited an item.
            # Regex: "take X from <source>" — source is everything after " from "
            src_m = re.search(r" from (.+)$", a.lower())
            if src_m:
                src = src_m.group(1).strip()
                if any(src == loc or src.startswith(loc) for loc in placed_loc_lower):
                    continue
            return a
        return None

    def find_known_target_location(self, target_type: str) -> Optional[str]:
        """Return a previously-observed location of target_type, excluding placed_items.

        Uses object_at (which is cleared when the agent picks up an item) to
        give a direct FIND hint for items the agent has already seen but not yet taken.
        """
        tt_lower = target_type.lower()
        held = (self.inventory or "").lower()
        for obj_name, loc in self.object_at.items():
            if (tt_lower in obj_name.lower()
                    and obj_name not in self.placed_items
                    and obj_name.lower() != held):
                return loc
        return None

    def find_put_action(self, dest_type: str, admissible: List[str]) -> Optional[str]:
        """Return the put action for dest_type, or None.

        No fallback: returning a wrong-destination put action causes the agent
        to place the item in the wrong receptacle, which then can't be undone
        and triggers a "Nothing happens" loop at the correct destination.

        For pick_two_obj second cycle: prefer the exact same destination instance
        used in the first cycle (e.g., countertop 1 rather than countertop 2).
        """
        if not self.inventory:
            return None
        # Prefer previously-used destination locations (pick_two_obj consistency)
        if self.placed_locations:
            for prev_loc in self.placed_locations:
                for a in admissible:
                    if (a.startswith("put ")
                            and dest_type.lower() in a.lower()
                            and prev_loc.lower() in a.lower()):
                        return a
        for a in admissible:
            if a.startswith("put ") and dest_type.lower() in a.lower():
                return a
        return None

    def find_transform_action(self, transform: str, admissible: List[str]) -> Optional[str]:
        for a in admissible:
            if a.startswith(transform + " "):
                return a
        return None

    def find_examine_action(self, target_type: str, admissible: List[str]) -> Optional[str]:
        """Return the 'examine X with desklamp' action, or None.

        We explicitly require 'desklamp' in the action string: plain 'examine X'
        (without the lamp) gives 'There's nothing special' and never completes
        the task, so offering it as a hint would create an infinite loop.
        """
        for a in admissible:
            if "examine" in a and target_type.lower() in a.lower() and "desklamp" in a.lower():
                return a
        return None

    def find_use_lamp_action(self, admissible: List[str]) -> Optional[str]:
        for a in admissible:
            if a.startswith("use desklamp") or ("desklamp" in a and "use" in a):
                return a
        return None


# ─────────────────────────────────────────────────────────────────────────────
# H2 — Action Gate helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gate_action(
    action: str,
    admissible: List[str],
    threshold: float,
) -> Tuple[Optional[str], bool, float]:
    """
    Verb-safe similarity matching.

    Returns (best_candidate, canonicalized, score).
    Only considers candidates that share the same verb as `action`.
    If no same-verb candidates exist, returns (admissible[0], False, 0.0) —
    i.e., never substitutes across different verbs (e.g. take ↔ put).
    """
    verb = _verb_of(action)
    same_verb = [a for a in admissible if _verb_of(a) == verb]

    # Hard guard: if no candidate shares the verb, refuse to canonicalize.
    if not same_verb:
        return admissible[0], False, 0.0

    best_action = same_verb[0]
    best_score = -1.0
    for candidate in same_verb:
        score = SequenceMatcher(None, action, candidate).ratio()
        if score > best_score:
            best_score = score
            best_action = candidate

    canonicalized = best_score >= threshold
    return best_action, canonicalized, best_score


# ─────────────────────────────────────────────────────────────────────────────
# H4 — Subgoal advancement helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pick_forced_action(admissible: List[str], last_output: str) -> Optional[str]:
    last_output = (last_output or "").strip().lower()
    for action in admissible:
        low = action.lower()
        if low == last_output:
            continue
        if low.startswith("inventory") or low.startswith("look"):
            continue
        return action
    return admissible[0] if admissible else None


# H5 tool-contract mapping (module-level so it can't conflict with dataclass fields).
# Maps task_type → the single tool-contract skill id to inject at cold-start.
#
# Empirical findings (Qwen3-4B, 100 episodes):
#   - look_at_obj: "examine_plain_is_useless" caused small models to over-generalise
#     "examine X with desklamp" into the search phase, triggering the 3-repeated-action
#     terminator. H4 already has a reactive "examine_without_lamp" recovery prompt that
#     handles this correctly without cold-start injection.
#   - pick_cool/heat/clean: "take_before_visiting_appliance" changed agent search
#     behaviour (opening every container) and altered search order in ways that exposed
#     the world-model substring bug. H3 step sequence + H4 stall recovery cover both
#     failure modes reactively and more precisely.
# Both entries are removed; _H5_CONTRACT_MAP is kept as an extension point for future
# task types or larger models where cold-start injection proves beneficial.
_H5_CONTRACT_MAP: Dict[str, str] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Runtime
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ALFWorldHarnessRuntime:
    config: ALFWorldHarnessConfig

    # H2 state
    invalid_consecutive_count: int = 0
    force_next_action: Optional[str] = None
    empty_turn_count: int = 0          # P2-1: consecutive empty-action turns

    # H4 state
    last_outputs: List[str] = field(default_factory=list)
    last_observations: List[str] = field(default_factory=list)  # P1-1: for nothing-happens check
    last_actions: List[str] = field(default_factory=list)        # for non-advancing loop detection
    soft_intervention_count: int = 0
    post_put_grace: int = 0

    # H0 / H4 world state
    task_ctx: Optional[TaskContext] = field(default=None)
    world: WorldModel = field(default_factory=WorldModel)
    _subgoal_idx: int = field(default=0)

    # H5 step-guidance dedup state
    _last_step_hint: Optional[str] = field(default=None)
    _last_step_hint_sg_idx: int = field(default=-1)
    # H5 specific-action hint enforcement: track when agent ignores "use: X" hints
    _last_specific_action_hint: Optional[str] = field(default=None)  # the action string from "use: X"
    _specific_action_ignored_count: int = field(default=0)            # consecutive ignores

    # ── H0 ───────────────────────────────────────────────────────────────────

    def init_task(self, init_prompt: str, initial_admissible: List[str]) -> None:
        """Parse task context and seed the world model with initial admissible locations."""
        self.task_ctx = parse_task_context(init_prompt)
        self._subgoal_idx = 0
        self.world = WorldModel()
        # Seed unvisited from initial navigable locations
        for cmd in initial_admissible:
            if cmd.startswith("go to "):
                loc = cmd[len("go to "):]
                if loc not in self.world.unvisited:
                    self.world.unvisited.append(loc)

        # P2-2: If the target is immediately takeable from the start (visible in initial
        # admissible), mark it found so the first FIND hint is actionable right away.
        if self.task_ctx.target_type:
            for cmd in initial_admissible:
                if (cmd.startswith("take ")
                        and self.task_ctx.target_type.lower() in cmd.lower()):
                    self.world.target_found = True
                    self.world.target_location = "starting location"
                    break

    def _current_subgoal(self) -> str:
        if self.task_ctx is None:
            return _SG_DONE
        chain = self.task_ctx.subgoal_chain
        if self._subgoal_idx < len(chain):
            return chain[self._subgoal_idx]
        return _SG_DONE

    def _advance_subgoal(self, action: str, observation: str, admissible: List[str]) -> None:
        """Advance the subgoal state machine based on post-step evidence."""
        if self.task_ctx is None:
            return
        sg = self._current_subgoal()
        obs_lower = (observation or "").lower()
        tt = self.task_ctx.target_type
        dt = self.task_ctx.destination_type

        if sg == _SG_FIND:
            # Advance when "take target" is available — applies to ALL task types.
            # For look_at_obj, the agent must take the object before examine X with desklamp
            # appears in admissible (lamp can be on but examine only shows up when holding X).
            if self.world.find_take_action(tt, admissible):
                self._subgoal_idx += 1

        elif sg == _SG_TAKE:
            if "you pick up" in obs_lower:
                self._subgoal_idx += 1

        elif sg == _SG_GOTO_SINK:
            if self.world.find_transform_action("clean", admissible):
                self._subgoal_idx += 1

        elif sg == _SG_CLEAN:
            if "you clean" in obs_lower:
                self._subgoal_idx += 1

        elif sg == _SG_GOTO_MCW:
            if self.world.find_transform_action("heat", admissible):
                self._subgoal_idx += 1

        elif sg == _SG_HEAT:
            if "you heat" in obs_lower:
                self._subgoal_idx += 1

        elif sg == _SG_GOTO_FRG:
            if self.world.find_transform_action("cool", admissible):
                self._subgoal_idx += 1

        elif sg == _SG_COOL:
            if "you cool" in obs_lower:
                self._subgoal_idx += 1

        elif sg == _SG_GOTO_DEST:
            if "you put" in obs_lower:
                # PUT was already executed while we were in GOTO_DEST state.
                # Advance past both GOTO_DEST and PUT in one step.
                self._subgoal_idx += 2
                self.post_put_grace = self.config.h4_post_put_grace
                # P0-2 (same as _SG_PUT case): if next subgoal is FIND, reset target state.
                if self._current_subgoal() == _SG_FIND:
                    self.world.target_found = False
                    self.world.target_location = None
            elif self.world.find_put_action(dt, admissible):
                self._subgoal_idx += 1  # → PUT (destination reached, ready to put)

        elif sg == _SG_PUT:
            if "you put" in obs_lower:
                self._subgoal_idx += 1
                self.post_put_grace = self.config.h4_post_put_grace
                # P0-2: If the next subgoal is FIND (pick_two_obj second cycle),
                # reset target search state so we don't navigate back to the now-empty
                # source location. Generalises to any future task with multi-cycle PUT→FIND.
                if self._current_subgoal() == _SG_FIND:
                    self.world.target_found = False
                    self.world.target_location = None

        elif sg == _SG_GOTO_LAMP:
            if self.world.find_use_lamp_action(admissible):
                self._subgoal_idx += 1

        elif sg == _SG_USE_LAMP:
            if "you turn on" in obs_lower or "you switch on" in obs_lower:
                # Record where the lamp is so the EXAMINE hint can direct the agent back.
                self.world.lamp_location = self.world.current_location
                self._subgoal_idx += 1

        elif sg == _SG_EXAMINE:
            # Terminal — task completion detected by done flag in alfworld_run
            pass

    # ── H3 ───────────────────────────────────────────────────────────────────

    def build_h3_hint(self) -> str:
        if self.task_ctx is None:
            return _truncate_to_word_budget(
                "Always call take_action with exactly one action from AVAILABLE ACTIONS.",
                self.config.h3_max_words,
            )
        tt = self.task_ctx.target_type or "target"
        dt = self.task_ctx.destination_type or "destination"
        hints = {
            "pick_and_place":        f"Find the {tt} → take it → go to {dt} → put it there.",
            "pick_two_obj":          f"Carry one {tt} at a time. Put it in {dt}, then find the second {tt}.",
            "pick_clean_then_place": f"Find the {tt} → take it → go to sinkbasin → clean it → go to {dt} → put it.",
            "pick_heat_then_place":  f"Find the {tt} → take it → go to microwave → heat it → go to {dt} → put it.",
            "pick_cool_then_place":  f"Find the {tt} → take it → go to fridge → cool it → go to {dt} → put it.",
            "look_at_obj":           f"Find the desklamp → turn it on → find the {tt} → take it → examine {tt} with desklamp.",
        }
        hint = hints.get(self.task_ctx.task_type, hints["pick_and_place"])
        return _truncate_to_word_budget(hint, self.config.h3_max_words)

    # ── H2 ───────────────────────────────────────────────────────────────────

    def pre_validate_action(self, raw_action: str, admissible: List[str]) -> Dict[str, Any]:
        # Apply forced action from H4 hard intervention — but yield to task-critical agent actions
        if self.force_next_action:
            agent_verb = _verb_of((raw_action or "").strip().lower())
            critical = {"take", "put", "clean", "heat", "cool", "use", "examine"}
            agent_in_admissible = (raw_action or "").strip().lower() in admissible
            if agent_verb in critical and agent_in_admissible:
                # Agent is about to do something important; honour it and discard the force
                self.force_next_action = None
            else:
                forced = self.force_next_action
                self.force_next_action = None
                return {
                    "action": forced,
                    "canonicalized": True,
                    "blocked": False,
                    "reason": "h4_forced_action",
                    "raw_action": raw_action,
                }

        action = (raw_action or "").strip().lower().split("\n")[0]
        if not admissible:
            return {
                "action": action,
                "canonicalized": False,
                "blocked": False,
                "reason": "no_admissible_actions",
                "raw_action": raw_action,
            }

        # Exact match
        if action in admissible:
            self.invalid_consecutive_count = 0
            return {
                "action": action,
                "canonicalized": False,
                "blocked": False,
                "reason": "exact_match",
                "raw_action": raw_action,
            }

        # Verb-safe similarity match
        best_action, canonicalized, score = _gate_action(
            action, admissible, self.config.action_similarity_threshold
        )
        chosen = best_action if canonicalized else action
        is_invalid = chosen not in admissible

        if is_invalid:
            self.invalid_consecutive_count += 1
        else:
            self.invalid_consecutive_count = 0

        blockable = not any(action.startswith(p) for p in self.config.never_block_prefixes)
        blocked = (
            is_invalid
            and blockable
            and self.invalid_consecutive_count >= self.config.invalid_block_after
        )
        reason = "canonicalized" if canonicalized else "invalid_action"
        if blocked:
            reason = "blocked_invalid_action"
        return {
            "action": chosen,
            "canonicalized": canonicalized,
            "blocked": blocked,
            "reason": reason,
            "raw_action": raw_action,
        }

    # ── H4 ───────────────────────────────────────────────────────────────────

    def post_step_monitor(
        self,
        raw_output: str,
        final_action: str,
        observation: str,
        admissible: List[str],
    ) -> Dict[str, Any]:
        # Update world model
        if self.task_ctx:
            self.world.update(final_action, observation, admissible, self.task_ctx.target_type)
            self._advance_subgoal(final_action, observation, admissible)

        # H5 enforcement: if agent just executed the recommended specific action, reset tracker
        if (self._last_specific_action_hint
                and final_action
                and final_action.strip().lower() == self._last_specific_action_hint.lower()):
            self._last_specific_action_hint = None
            self._specific_action_ignored_count = 0

        # Decrement post-PUT grace period
        if self.post_put_grace > 0:
            self.post_put_grace -= 1

        # Track recent outputs/observations for stall detection
        self.last_outputs.append((raw_output or "").strip().lower())
        if len(self.last_outputs) > self.config.h4_stall_window:
            self.last_outputs = self.last_outputs[-self.config.h4_stall_window:]

        obs_lower_str = (observation or "").lower()
        self.last_observations.append(obs_lower_str)
        win = self.config.h4_nothing_happens_window
        if len(self.last_observations) > win:
            self.last_observations = self.last_observations[-win:]

        act_lower = (final_action or "").strip().lower()
        self.last_actions.append(act_lower)
        if len(self.last_actions) > 6:
            self.last_actions = self.last_actions[-6:]

        response: Dict[str, Any] = {
            "stall_score": 0,
            "intervention_level": "none",
            "audit_reason": "",
            "recovery_prompt": None,
            "forced_action": None,
        }

        # P2-1: Empty turn detection — agent produced no tool call / empty action.
        # Inject a nudge after consecutive empty turns to avoid wasting max_step budget.
        if not final_action or not final_action.strip():
            self.empty_turn_count += 1
            if self.empty_turn_count >= self.config.h2_empty_turn_threshold:
                sg_label = self._current_subgoal()
                tt = self.task_ctx.target_type if self.task_ctx else "target"
                response["intervention_level"] = "soft"
                response["audit_reason"] = "empty_turn"
                response["recovery_prompt"] = (
                    f"Harness: you must call take_action with an action from AVAILABLE ACTIONS. "
                    f"Current goal: {sg_label} ({tt}). "
                    f"Choose any listed action to move forward."
                )
                return response
        else:
            self.empty_turn_count = 0

        if not self.config.h4_enabled:
            return response

        # Stall detection: suppressed during grace period or when subgoal is DONE
        in_grace = self.post_put_grace > 0
        sg_done = self._current_subgoal() == _SG_DONE
        if in_grace or sg_done:
            return response

        # L1: "examine without lamp" loop — agent in EXAMINE subgoal is repeatedly doing
        # plain "examine X" (not "examine X with desklamp"), getting "nothing special".
        # This fires before the generic stall check so recovery is more specific.
        if self._current_subgoal() == _SG_EXAMINE and self.world.inventory is not None:
            nothing_special_count = sum(
                1 for obs in self.last_observations
                if "nothing special" in obs
            )
            if nothing_special_count >= 2:
                tt = self.task_ctx.target_type if self.task_ctx else "the object"
                lamp_loc = self.world.lamp_location
                response["stall_score"] = 2
                response["intervention_level"] = "soft"
                response["audit_reason"] = "examine_without_lamp"
                response["recovery_prompt"] = (
                    f"Harness: 'examine {tt}' without the desklamp does nothing. "
                    + (f"Go to {lamp_loc} — " if lamp_loc else "Find the lit desklamp — ")
                    + f"you must be AT the lit desklamp to examine {tt} with it."
                )
                self.last_observations.clear()
                return response

        # Navigation oscillation: agent is bouncing between ≤2 locations without
        # making progress (A→B→A→B…). Distinct from the 3-identical-action terminator
        # in task.py which only catches strictly repeated single actions.
        recent_navs = [a for a in self.last_actions[-6:] if a.startswith("go to ")]
        if len(recent_navs) >= 5:
            unique_dests = set(recent_navs)
            if len(unique_dests) <= 2 and self.task_ctx:
                tt = self.task_ctx.target_type or "target"
                sg_label = self._current_subgoal()
                next_hint = ""
                if sg_label in (_SG_FIND, _SG_TAKE) and self.world.unvisited:
                    ordered = self.world.ordered_unvisited(tt)
                    filtered = [loc for loc in ordered if loc not in self.world.placed_locations
                                and loc not in unique_dests]
                    if filtered:
                        next_hint = f" Try: go to {filtered[0]}."
                response["stall_score"] = 2
                response["intervention_level"] = "soft"
                response["audit_reason"] = "nav_oscillation"
                response["recovery_prompt"] = (
                    f"Harness: you are oscillating between the same locations. "
                    f"The {tt} is not there — explore somewhere new.{next_hint}"
                )
                self.last_actions.clear()
                return response

        # Container open/close oscillation: agent is repeatedly opening and closing the
        # same container without taking or placing anything. Common with fridge (pick_cool/
        # pick_heat) when the agent doesn't know what to do at the container.
        recent_oc = [a for a in self.last_actions[-6:]
                     if a.startswith("open ") or a.startswith("close ")]
        if len(recent_oc) >= 4 and self.task_ctx:
            # Check if these open/close actions target ≤ 2 containers
            targets = set(a.split(" ", 1)[1] for a in recent_oc)
            if len(targets) <= 2:
                tt = self.task_ctx.target_type or "target"
                sg_label = self._current_subgoal()
                xform = self.task_ctx.transformation
                container = next(iter(targets))
                response["stall_score"] = 2
                response["intervention_level"] = "soft"
                response["audit_reason"] = "container_oscillation"
                if xform and sg_label in (_SG_GOTO_MCW, _SG_GOTO_FRG, _SG_GOTO_SINK):
                    # Knows it needs the transformation but can't do it yet
                    response["recovery_prompt"] = (
                        f"Harness: you need to take the {tt} first before you can "
                        f"{xform} it here. Find and pick up the {tt}, then return."
                    )
                elif sg_label in (_SG_GOTO_MCW, _SG_HEAT):
                    response["recovery_prompt"] = (
                        f"Harness: opening/closing {container} repeatedly does nothing. "
                        f"Take {tt}, open {container}, then use 'heat {tt} with {container}'."
                    )
                elif sg_label in (_SG_GOTO_FRG, _SG_COOL):
                    response["recovery_prompt"] = (
                        f"Harness: opening/closing {container} repeatedly does nothing. "
                        f"Take {tt}, open {container}, then use 'cool {tt} with {container}'."
                    )
                else:
                    response["recovery_prompt"] = (
                        f"Harness: opening/closing {container} repeatedly is not advancing "
                        f"the task. Navigate elsewhere to find the {tt}."
                    )
                self.last_actions.clear()
                return response

        # Dead-end action loop: agent is cycling through examine / inventory / look
        # without navigating. This is a distinct failure from the "nothing happens" loop:
        # these actions DO return output but never advance the task. After 4 consecutive
        # non-advancing actions, force a directed navigation hint.
        _NON_ADVANCING = ("examine", "inventory", "look")
        recent_na = [a for a in self.last_actions[-5:] if any(a.startswith(p) for p in _NON_ADVANCING)]
        if len(recent_na) >= 4 and self.task_ctx:
            sg_label = self._current_subgoal()
            tt = self.task_ctx.target_type or "target"
            # Build a directed recovery prompt with a concrete next action
            next_hint = ""
            if sg_label == _SG_FIND and self.world.unvisited:
                ordered = self.world.ordered_unvisited(tt)
                filtered = [loc for loc in ordered if loc not in self.world.placed_locations]
                next_loc = (filtered[0] if filtered else ordered[0]) if ordered else self.world.unvisited[0]
                next_hint = f" Navigate: go to {next_loc}."
            elif sg_label == _SG_TAKE:
                loc = self.world.target_location or self.world.find_known_target_location(tt)
                if loc:
                    next_hint = f" Go to {loc} to pick up the {tt}."
            response["stall_score"] = 2
            response["intervention_level"] = "soft"
            response["audit_reason"] = "dead_end_loop"
            response["recovery_prompt"] = (
                f"Harness: examine/inventory/look are not advancing the task. "
                f"Stop checking and take a navigation action.{next_hint}"
            )
            self.last_actions.clear()
            return response

        # P1-1: "Nothing happens" loop — keep issuing put/take actions that have no effect.
        nothing_happens_count = sum(
            1 for obs in self.last_observations
            if "nothing happens" in obs
        )
        if nothing_happens_count >= self.config.h4_nothing_happens_threshold and self.task_ctx:
            tt = self.task_ctx.target_type or "the target object"
            xform = self.task_ctx.transformation
            sg_label = self._current_subgoal()
            if self.world.inventory is None:
                response["stall_score"] = 2
                response["intervention_level"] = "soft"
                response["audit_reason"] = "nothing_happens_loop"
                response["recovery_prompt"] = (
                    f"Harness: you are not holding anything — put actions have no effect. "
                    f"Find and pick up the {tt} before trying to place it."
                )
            elif xform and sg_label in (_SG_GOTO_DEST, _SG_PUT, _SG_GOTO_SINK,
                                        _SG_GOTO_MCW, _SG_GOTO_FRG):
                # Holding the item but put fails → likely skipped the transformation step
                xform_loc = _XFORM_LOCATION.get(xform, xform)
                response["stall_score"] = 2
                response["intervention_level"] = "soft"
                response["audit_reason"] = "put_skipped_transform"
                response["recovery_prompt"] = (
                    f"Harness: putting {self.world.inventory} is failing — "
                    f"you must {xform} it first. "
                    f"Go to {xform_loc} and use the '{xform}' action before placing it."
                )
            else:
                response["stall_score"] = 1
                response["intervention_level"] = "soft"
                response["audit_reason"] = "nothing_happens_loop"
                response["recovery_prompt"] = (
                    f"Harness: actions are not taking effect. "
                    f"Try a different approach or navigate elsewhere."
                )
            self.last_observations.clear()
            return response

        enough_rounds = len(self.last_outputs) >= self.config.h4_min_rounds_before_stall
        repeated_action = (
            len(self.last_outputs) >= self.config.h4_stall_window
            and len(set(self.last_outputs[-self.config.h4_stall_window:])) == 1
        )

        # Object novelty (now works correctly after regex fix)
        new_objects = set(_extract_objects(observation))
        no_new_objects = len(new_objects) == 0 or new_objects.issubset(
            set(self.world.object_at.keys())
        )

        stall_score = int(repeated_action) + int(no_new_objects)
        response["stall_score"] = stall_score
        stalled = enough_rounds and repeated_action and no_new_objects

        if not stalled:
            return response

        if self.soft_intervention_count < self.config.h4_soft_intervention_rounds:
            self.soft_intervention_count += 1
            response["intervention_level"] = "soft"
            response["audit_reason"] = "stall_detected_soft"
            response["recovery_prompt"] = (
                "Harness: you seem stuck. Choose a different, non-repeated action "
                "from AVAILABLE ACTIONS and avoid repeating look/inventory."
            )
        else:
            forced = _pick_forced_action(admissible, self.last_outputs[-1] if self.last_outputs else "")
            if forced:
                self.force_next_action = forced
                response["intervention_level"] = "hard"
                response["audit_reason"] = "stall_persists"
                response["forced_action"] = forced
                response["recovery_prompt"] = (
                    "Harness: breaking the loop — execute a different exploration action next turn."
                )
        return response

    # ── H5 ───────────────────────────────────────────────────────────────────

    def cold_start_skill_hints(self) -> List[Dict[str, str]]:
        """
        H5 cold-start: inject one tool-contract skill for task types where the
        model has a known interface-contract blind spot.

        Design principle: H5 is PREVENTIVE (front-loads non-obvious constraints
        before the agent can fail), not DUPLICATIVE (strategy already in H3/H4).
        Only task types with a genuinely non-obvious action precondition get an
        injection; simple placement tasks do not.
        """
        if not self.config.h5_enabled or self.task_ctx is None:
            return []

        skill_id = _H5_CONTRACT_MAP.get(self.task_ctx.task_type)
        if skill_id is None:
            return []

        skill = next((s for s in ALF_SKILLS if s["id"] == skill_id), None)
        if skill is None:
            return []

        # Skill texts are curated offline; do not truncate at runtime.
        # Keep full wording to avoid clipping key preconditions.
        text = skill["text"]
        return [{"id": skill_id, "text": text, "trigger": "cold_start",
                 "token_cost": str(len(text.split()))}]

    def step_guidance(self, current_round: int, max_step: int, admissible: List[str]) -> Optional[str]:
        """
        H4-E per-step guidance: generate a specific, actionable hint based on
        current subgoal state and world model.

        Returns None when no helpful guidance can be produced (to avoid noise).
        Guidance is always phrased as a suggestion, not a command, to preserve
        agent autonomy and generalisation across scenes.
        """
        if not self.config.h4_enabled or self.task_ctx is None:
            return None

        sg = self._current_subgoal()
        tt = self.task_ctx.target_type
        dt = self.task_ctx.destination_type
        xform = self.task_ctx.transformation
        world = self.world

        hint: Optional[str] = None

        if sg == _SG_FIND:
            if self.task_ctx.task_type == "look_at_obj":
                # P0-3: agent must TAKE the target object first (examine X with desklamp
                # only appears in admissible when holding X and the lamp is on).
                if world.target_found and world.target_location:
                    hint = (
                        f"Hint: {tt} was spotted at {world.target_location}. "
                        f"Navigate there and take it."
                    )
                elif world.unvisited:
                    ordered = world.ordered_unvisited(tt)
                    next_loc = ordered[0] if ordered else world.unvisited[0]
                    hint = f"Hint: searching for {tt}. Try: go to {next_loc}."
            else:
                if world.target_found and world.target_location:
                    hint = (
                        f"Hint: {tt} was spotted at {world.target_location}. "
                        f"Navigate there and take it."
                    )
                else:
                    # object_at: check if we have already observed target_type somewhere
                    # (e.g., pick_two_obj second cycle where we saw the second item earlier).
                    # Excludes placed_locations so we never point back to our own deposits.
                    known_loc = world.find_known_target_location(tt)
                    if known_loc:
                        hint = (
                            f"Hint: {tt} was previously seen at {known_loc}. "
                            f"Go there and take it."
                        )
                    elif world.unvisited:
                        ordered = world.ordered_unvisited(tt)
                        # P1-2: exclude already-placed destination locations from search hints
                        filtered = [loc for loc in ordered if loc not in world.placed_locations]
                        next_loc = (filtered[0] if filtered else ordered[0]) if ordered else world.unvisited[0]
                        hint = f"Hint: {tt} not found yet. Suggested next: go to {next_loc}."

        elif sg == _SG_TAKE:
            take_a = world.find_take_action(tt, admissible)
            if take_a:
                hint = f"Hint: pick up the {tt} — use: {take_a}."
            else:
                # Take action not yet in admissible — agent is not at target location.
                loc = world.target_location or world.find_known_target_location(tt)
                if loc:
                    hint = (
                        f"Hint: you are not at the {tt}'s location. "
                        f"Go to {loc} first, then pick up the {tt}."
                    )

        elif sg == _SG_GOTO_SINK:
            if not any(a.startswith("clean ") for a in admissible):
                hint = f"Hint: go to sinkbasin to clean the {world.inventory or tt}."

        elif sg == _SG_CLEAN:
            clean_a = world.find_transform_action("clean", admissible)
            if clean_a:
                hint = f"Hint: clean the {world.inventory or tt} — use: {clean_a}."

        elif sg == _SG_GOTO_MCW:
            if not any(a.startswith("heat ") for a in admissible):
                hint = f"Hint: go to microwave to heat the {world.inventory or tt}."

        elif sg == _SG_HEAT:
            heat_a = world.find_transform_action("heat", admissible)
            if heat_a:
                hint = f"Hint: heat the {world.inventory or tt} — use: {heat_a}."

        elif sg == _SG_GOTO_FRG:
            if not any(a.startswith("cool ") for a in admissible):
                hint = f"Hint: go to fridge to cool the {world.inventory or tt}."

        elif sg == _SG_COOL:
            cool_a = world.find_transform_action("cool", admissible)
            if cool_a:
                hint = f"Hint: cool the {world.inventory or tt} — use: {cool_a}."

        elif sg in (_SG_GOTO_DEST, _SG_PUT):
            put_a = world.find_put_action(dt, admissible)
            # For pick_two_obj: both items must go to the SAME destination instance.
            # If a put action is available but points to a different instance than
            # where we placed the first item, redirect rather than endorsing it.
            if (put_a and self.task_ctx
                    and self.task_ctx.task_type == "pick_two_obj"
                    and world.placed_locations):
                prev_loc = world.placed_locations[0]
                if prev_loc.lower() not in put_a.lower():
                    # Wrong destination instance — steer agent to the correct one
                    hint = (
                        f"Hint: go to {prev_loc} to place the {world.inventory or tt} "
                        f"— same spot as the first one."
                    )
                else:
                    hint = f"Hint: place the {world.inventory or tt} — use: {put_a}."
            elif put_a:
                # Already at destination (or PUT is immediately available)
                hint = f"Hint: place the {world.inventory or tt} — use: {put_a}."
            else:
                # For pick_two_obj second cycle: navigate back to the SAME specific
                # destination instance used for the first item, not just any matching type.
                if (self.task_ctx and self.task_ctx.task_type == "pick_two_obj"
                        and world.placed_locations):
                    target_loc = world.placed_locations[0]
                    hint = (
                        f"Hint: go to {target_loc} to place the {world.inventory or tt} "
                        f"— same spot as the first one."
                    )
                else:
                    hint = f"Hint: go to {dt} to place the {world.inventory or tt}."

        elif sg == _SG_GOTO_LAMP:
            hint = "Hint: find the desklamp in this room and go to it."

        elif sg == _SG_USE_LAMP:
            lamp_a = world.find_use_lamp_action(admissible)
            if lamp_a:
                hint = f"Hint: turn on the lamp — use: {lamp_a}."

        elif sg == _SG_EXAMINE:
            ex_a = world.find_examine_action(tt, admissible)
            if ex_a:
                # find_examine_action only returns the "with desklamp" form, so this is exact.
                hint = f"Hint: use exactly: {ex_a}."
            elif world.inventory:
                # Holding the target but "examine X with desklamp" is not in admissible —
                # agent is not at the lamp location. Direct it back.
                if world.lamp_location:
                    hint = (
                        f"Hint: go to {world.lamp_location} first — "
                        f"'examine {tt} with desklamp' only works when you are at the lit lamp."
                    )
                else:
                    hint = (
                        f"Hint: find the lit desklamp and go to it — "
                        f"then use 'examine {tt} with desklamp' to complete the task."
                    )

        if hint is None:
            # If we have a pending specific action, agent might have just done it
            self._last_specific_action_hint = None
            self._specific_action_ignored_count = 0
            return None
        hint = _truncate_to_word_budget(hint, self.config.h5_step_hint_max_words)

        # Dedup: suppress if the hint content and subgoal haven't changed since last injection.
        sg_changed = (self._subgoal_idx != self._last_step_hint_sg_idx)
        content_changed = (hint != self._last_step_hint)
        if not sg_changed and not content_changed:
            # Same hint as last turn — agent didn't follow it.
            # If this hint contains a specific "use: X" action, track ignores
            # and force via force_next_action on the 2nd consecutive ignore.
            if self._last_specific_action_hint:
                self._specific_action_ignored_count += 1
                if self._specific_action_ignored_count >= 2:
                    self.force_next_action = self._last_specific_action_hint
                    self._specific_action_ignored_count = 0
                    self._last_specific_action_hint = None
            return None

        # Extract specific action from "use: <action>" phrasing for enforcement tracking
        use_m = re.search(r"use: (.+?)\.?$", hint)
        if use_m:
            candidate = use_m.group(1).strip().rstrip(".")
            # Only enforce if action is actually in admissible (safety check)
            if any(candidate.lower() == a.lower() for a in admissible):
                self._last_specific_action_hint = candidate
                self._specific_action_ignored_count = 0
            else:
                self._last_specific_action_hint = None
        else:
            self._last_specific_action_hint = None
            self._specific_action_ignored_count = 0

        self._last_step_hint = hint
        self._last_step_hint_sg_idx = self._subgoal_idx
        return hint

    # ── H4 budget (sub-branch of post-step monitoring) ────────────────────────

    def budget_check(self, remaining_steps: int, admissible: List[str]) -> Dict[str, Any]:
        """
        H4-D: Step-budget management (H4 sub-branch).

        - Warn when budget is low and we're still searching.
        - Force a PUT action when budget is critical and conditions are unambiguous:
          agent holds the correct item AND the put action is in admissible.
          This is the only place where harness forces a non-exploration action.
        """
        response: Dict[str, Any] = {"hint": None, "force_action": None}
        if not self.config.h4_enabled or self.task_ctx is None:
            return response

        sg = self._current_subgoal()
        world = self.world
        dt = self.task_ctx.destination_type
        tt = self.task_ctx.target_type

        # Hard force: agent holds correct item, put action is available, steps critical
        if remaining_steps < self.config.h4_budget_force_threshold:
            if world.inventory and sg in (_SG_GOTO_DEST, _SG_PUT):
                put_a = world.find_put_action(dt, admissible)
                if put_a:
                    response["force_action"] = put_a
                    response["hint"] = (
                        f"[{remaining_steps} steps left] Harness: placing {world.inventory} now."
                    )
                    return response

        # Soft warn: still searching with few steps left.
        # Multi-step tasks (pick_two_obj: 2 full cycles; pick_cool: extra fridge detour)
        # need more buffer, so their effective warn threshold is raised.
        warn_threshold = self.config.h4_budget_warn_threshold
        if self.task_ctx.task_type in ("pick_two_obj", "pick_cool_then_place"):
            warn_threshold = max(warn_threshold, self.config.h4_budget_warn_threshold_multistep)
        if remaining_steps < warn_threshold and sg == _SG_FIND:
            ordered = world.ordered_unvisited(tt)
            top = ordered[:2] if ordered else []
            if top:
                locs = " or ".join(top)
                response["hint"] = (
                    f"[{remaining_steps} steps left] Urgently check: {locs}."
                )

        return response


# ─────────────────────────────────────────────────────────────────────────────
# H3 helper — tool description patching
# ─────────────────────────────────────────────────────────────────────────────

def patch_take_action_tool_description(
    tools: Optional[List[Dict[str, Any]]], hint: str
) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return tools
    patched = copy.deepcopy(tools)
    for tool in patched:
        fn = tool.get("function", {})
        if fn.get("name") != "take_action":
            continue
        description = fn.get("description", "").strip()
        if hint in description:
            break
        fn["description"] = f"{description}\n\n{hint}".strip()
        break
    return patched


# ─────────────────────────────────────────────────────────────────────────────
# Skill definitions (used by H5 cold-start)
# ─────────────────────────────────────────────────────────────────────────────

ALF_SKILLS: List[Dict[str, Any]] = [
    # ── Tool-contract skills (injectable via H5) ──────────────────────────────
    # These encode non-obvious preconditions that models miss from pretraining.
    # H5 injects one of these at cold-start for task types with a known blind spot.
    {
        "id": "take_before_visiting_appliance",
        "task_types": ["pick_cool_then_place", "pick_heat_then_place", "pick_clean_then_place"],
        "keywords": ["carry", "hold", "inventory", "pick up", "before", "appliance"],
        "text": (
            "You cannot clean, heat, or cool an item remotely. First pick up the exact target "
            "object from the task (for example, do not confuse 'pot' with 'potato'), then carry "
            "it to the required appliance (sink/microwave/fridge) and apply the transform action. "
            "After transforming, go to the task destination and put that same transformed object there."
        ),
    },
    {
        "id": "examine_plain_is_useless",
        "task_types": ["look_at_obj"],
        "keywords": ["examine", "desklamp", "nothing special", "lamp", "with"],
        "text": (
            "Plain 'examine X' only gives 'nothing special' and does not finish the task. "
            "Use exactly 'examine X with desklamp' while standing next to a lit desklamp. "
            "Turn on the lamp, take the target object, then run that exact examine action."
        ),
    },

    # ── Strategy skills (NOT injectable via H5) ───────────────────────────────
    # task_types=[] means retrieve_skills_for_task never returns them.
    # H3 covers step sequence; H4 per-step guidance covers exploration and stall
    # recovery reactively. Injecting these at cold-start duplicates H3/H4 and
    # adds noise for small models — experiments showed no-H5 outperforms H5
    # when only strategy skills are injected.
    {
        "id": "pickup_then_deliver",
        "task_types": [],  # strategy — covered by H3 instance-specific hint
        "keywords": ["put", "place", "locate", "navigate", "deliver"],
        "text": (
            "For placement tasks, find the target, take it, go directly to the destination, "
            "and place it there."
        ),
    },
    {
        "id": "state_changing_actions_only",
        "task_types": [],  # strategy — covered by H3 instance-specific hint
        "keywords": ["clean", "heat", "cool", "transform", "appliance"],
        "text": (
            "For transformation tasks, carry the item to the required appliance "
            "(sinkbasin, microwave, or fridge), apply the transform action, then deliver it "
            "to the final destination."
        ),
    },
    {
        "id": "two_object_staging",
        "task_types": [],  # strategy — covered by H3 instance-specific hint
        "keywords": ["two", "both", "second", "pair", "another"],
        "text": (
            "You can carry only one item at a time. Deliver the first object, then return to "
            "find and deliver the second."
        ),
    },
    {
        "id": "examine_with_lamp",
        "task_types": [],  # strategy — superseded by examine_plain_is_useless (tool contract)
        "keywords": ["examine", "lamp", "light", "desklamp", "look"],
        "text": (
            "Go to the desklamp and turn it on first. Then find and take the target object, "
            "and examine it with the desklamp."
        ),
    },
    {
        "id": "explore_unseen_first",
        "task_types": [],  # strategy — covered by H4 navigation hints
        "keywords": ["find", "locate", "search", "explore", "unvisited"],
        "text": (
            "When searching, prioritize unvisited locations and containers before revisiting "
            "places you already checked."
        ),
    },
    {
        "id": "appliance_is_midpoint_not_destination",
        "task_types": [],  # strategy — covered by H3; also wrong when dest IS an appliance
        "keywords": ["fridge", "microwave", "sinkbasin", "destination", "deliver", "after", "then"],
        "text": (
            "The appliance (fridge, microwave, sinkbasin) is usually a midpoint, not the final "
            "destination. After transforming the object, pick it up again and carry it to the "
            "target receptacle."
        ),
    },
    {
        "id": "nothing_happens_means_wrong_state",
        "task_types": [],  # strategy — H4 handles this reactively after the fact
        "keywords": ["nothing happens", "wrong", "state", "repeat", "fail", "stuck"],
        "text": (
            "'Nothing happens' means the action is invalid in the current state. You may need "
            "to hold the item, transform it first, or open a receptacle. Do not repeat the same "
            "action; change strategy."
        ),
    },
    {
        "id": "two_obj_second_item_different_location",
        "task_types": [],  # strategy — covered by H4 world model placed_items filter
        "keywords": ["second", "another", "two", "different", "source", "destination"],
        "text": (
            "After placing the first object, search unexplored locations for the second. Do not "
            "pick up an item you already placed at the destination. If unsure, verify it is a "
            "new instance before taking it."
        ),
    },
    {
        "id": "break_loops_early",
        "task_types": [],  # not injectable; injected by H4 recovery only
        "keywords": ["loop", "repeat", "stuck", "stall", "revisit"],
        "text": (
            "If recent actions reveal no new objects or progress, break the loop and switch to "
            "a different unexplored location immediately."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat shim (used by task.py import)
# ─────────────────────────────────────────────────────────────────────────────

def first_sentence_query(user_prompt: str) -> str:
    """
    Legacy helper kept for import compatibility.
    No longer used by the harness itself (BM25 retrieval removed).
    """
    text = (user_prompt or "").strip()
    prefix = "Here is your task."
    if text.startswith(prefix):
        text = text[len(prefix):].strip()
    parts = re.split(r"(?<=[.!?])\s+|\n+", text, maxsplit=1)
    return parts[0].strip() if parts and parts[0].strip() else text
