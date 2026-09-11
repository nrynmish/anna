"""Deterministic heuristic labeler for ANNA intents (Uber).

This module implements a conservative, deterministic heuristic labeler that
produces weak labels using keyword-based rules. It uses the taxonomy defined in
`src.intent.taxonomy` and intentionally avoids any external API or LLM calls.

Key functions:
- normalize_customer_text(text)
- label_case(case)
- label_cases(cases)
- explain_label(case)

Label representation (dict):
 - case_id
 - customer_id
 - intent
 - labeling_method
 - labeling_confidence
 - taxonomy_version
 - original_text

Labeling methods enumerated: keyword_rules, heuristic, manual, llm, unresolved
"""

from __future__ import annotations

import re
from typing import List, Dict, Any

from src.intent import taxonomy

# Labeling methods
LABELING_METHODS = ["keyword_rules", "heuristic", "manual", "llm", "unresolved"]


URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHANUMERIC = re.compile(r"[^\w\s]")


def normalize_customer_text(text: str) -> str:
    """Normalize customer text deterministically: lowercase, remove URLs and mentions, collapse whitespace."""
    if text is None:
        return ""
    t = text
    t = URL_RE.sub("", t)
    t = MENTION_RE.sub("", t)
    t = t.replace("\n", " ")
    # collapse punctuation to spaces
    t = NON_ALPHANUMERIC.sub(" ", t)
    t = t.lower()
    t = " ".join(t.split())
    return t


# Build simple keyword rules from the taxonomy (conservative, deterministic)
_INTENT_PATTERNS = {
    # intent: list of (pattern, strength)
    # strength: 'strong', 'medium', 'weak'
    "account_access": [
        ("acct", "weak"),
        ("disabled", "medium"),
        ("account disabled", "strong"),
        ("can't sign in", "strong"),
        ("cant sign in", "strong"),
        ("cant sign in to", "strong"),
        ("cannot sign in", "strong"),
        ("cannot sign in to", "strong"),
        ("cannot access account", "strong"),
        ("cant access", "medium"),
        ("can't sign in to", "strong"),
        ("cannot access account", "strong"),
        ("account is disabled", "strong"),
        ("disabled account", "strong"),
        ("account suspended", "strong"),
        ("account locked", "strong"),
        ("locked out", "strong"),
        ("can't login", "strong"),
        ("cannot login", "strong"),
        ("unable to login", "strong"),
        ("reset password", "strong"),
        ("forgot password", "medium"),
        ("password reset", "strong"),
        ("verify email", "medium"),
        ("verify number", "medium"),
        ("can't get my receipts", "medium"),
        ("receipts", "weak"),
        ("login", "weak"),
        ("log in", "weak"),
    ],

    "payments_charges": [
        ("was charged", "strong"),
        ("charged twice", "strong"),
        ("charged for", "strong"),
        ("cancellation fee", "strong"),
        ("charged me", "strong"),
        ("charge me", "strong"),
        ("chargeback", "medium"),
        ("charged", "weak"),
        ("credit card", "medium"),
        ("wallet", "medium"),
        ("balance", "weak"),
    ],

    "refunds_adjustments": [
        ("refund", "strong"),
        ("reimburse", "strong"),
        ("money back", "strong"),
        ("fare adjusted", "medium"),
        ("refund please", "strong"),
        ("please refund", "strong"),
        ("overcharged", "strong"),
    ],

    "rides_and_trips": [
        ("airport", "medium"),
        ("scheduled", "medium"),
        ("pickup window", "medium"),
        ("no show", "medium"),
        ("no-show", "medium"),
        # generic context words are weak
        ("trip", "weak"),
        ("ride", "weak"),
        ("pickup", "weak"),
        ("drop", "weak"),
        ("disappears", "weak"),
    ],

    "driver_behavior": [
        ("rude driver", "strong"),
        ("report the driver", "strong"),
        ("driver cancelled", "strong"),
        ("driver canceled", "strong"),
        ("cancelled on me", "strong"),
        ("cancelled on", "strong"),
        ("drivers cancelled", "strong"),
        ("drivers canceled", "strong"),
        ("three cars have canceled", "strong"),
        ("unprofessional", "strong"),
        ("cancelled my trip", "strong"),
        ("multiple drivers", "medium"),
        ("driver was rude", "strong"),
        ("driver was", "weak"),
        ("rude", "weak"),
    ],

    "lost_and_found": [
        ("left my bag", "strong"),
        ("left my phone", "strong"),
        ("lost my", "strong"),
        ("lost item", "strong"),
        ("lost and found", "strong"),
        ("left my", "medium"),
        ("left her bag", "strong"),
        ("left his bag", "strong"),
        ("left their bag", "strong"),
        ("left bag", "strong"),
    ],

    "promotions_discounts": [
        ("promo code", "strong"),
        ("promo", "weak"),
        ("ride pass", "medium"),
        ("discount", "medium"),
        ("coupon", "medium"),
    ],

    "app_technical": [
        ("app not working", "strong"),
        ("app isn't working", "strong"),
        ("app is not working", "strong"),
        ("app crashed", "strong"),
        ("app crash", "strong"),
        ("app error", "strong"),
        ("something went wrong", "strong"),
        ("not working", "medium"),
        ("error", "medium"),
        ("app", "weak"),
    ],

    "signup_onboarding": [
        ("sign up", "medium"),
        ("signup", "medium"),
        ("applying", "medium"),
        ("background check", "strong"),
        ("apply to be a driver", "medium"),
    ],

    "payouts_and_earnings": [
        ("cash out", "strong"),
        ("cashout", "strong"),
        ("missing earnings", "strong"),
        ("earnings", "medium"),
        ("payout", "medium"),
    ],

    "safety_incident": [
        ("attack", "strong"),
        ("attacked", "strong"),
        ("assault", "strong"),
        ("injured", "strong"),
        ("pain and suffering", "strong"),
        ("pulled from the road", "medium"),
    ],
}

# strength weights
_STRENGTH_WEIGHT = {"strong": 5, "medium": 3, "weak": 1}


def _match_keywords(text: str) -> Dict[str, Dict[str, Any]]:
    """Return a map of intent -> details with matched patterns, strengths and score.

    The returned dict maps intent -> {matches: [(pattern,strength),...], score: int, max_strength: str}
    """
    matches: Dict[str, Dict[str, Any]] = {}
    for intent_id, patterns in _INTENT_PATTERNS.items():
        found = []
        score = 0
        max_strength = None
        for pat, strength in patterns:
            # normalize the pattern the same way we normalize customer text
            pat_norm = normalize_customer_text(pat)
            if pat_norm and pat_norm in text:
                found.append((pat, strength))
                w = _STRENGTH_WEIGHT.get(strength, 1)
                score += w
                if max_strength is None or _STRENGTH_WEIGHT.get(strength, 0) > _STRENGTH_WEIGHT.get(max_strength, 0):
                    max_strength = strength
        if found:
            matches[intent_id] = {"matches": found, "score": score, "max_strength": max_strength}
    # special-case: detect numeric charge evidence (e.g., charged $140 or was charged 140)
    if re.search(r"\bcharg(?:e|ed|es)\b", text) and re.search(r"\b\d{2,}\b", text):
        pid = "payments_charges"
        if pid in matches:
            matches[pid]["matches"].append(("charged_amount", "strong"))
            matches[pid]["score"] += _STRENGTH_WEIGHT["strong"]
            matches[pid]["max_strength"] = "strong"
        else:
            matches[pid] = {"matches": [("charged_amount", "strong")], "score": _STRENGTH_WEIGHT["strong"], "max_strength": "strong"}
    return matches


def _valid_taxonomy_id(intent_id: str) -> bool:
    return intent_id in {i["id"] for i in taxonomy.TAXONOMY}


def label_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Label a single support case deterministically using heuristic rules.

    Input `case` must contain at least: `case_id`, `customer_id`, and
    `customer_messages` (list of texts or dicts with 'text').

    Returns a label dict with the required fields.
    """
    # gather text
    messages = case.get("customer_messages") or []
    # messages may be list of strings or dicts
    texts: List[str] = []
    for m in messages:
        if isinstance(m, str):
            texts.append(m)
        elif isinstance(m, dict):
            texts.append(m.get("text", ""))
    # combine in chronological order (assume input order is chronological)
    combined = " ".join(texts)
    norm = normalize_customer_text(combined)

    matches = _match_keywords(norm)

    label: Dict[str, Any] = {
        "case_id": case.get("case_id"),
        "customer_id": case.get("customer_id"),
        "original_text": combined,
        "taxonomy_version": taxonomy.TAXONOMY_VERSION,
        "labeling_method": "heuristic",
        "labeling_confidence": 0.0,
        "intent": "unclear",
    }

    # No matches -> if there is any substantive text, mark other; else unclear
    if not norm.strip():
        label["intent"] = "unclear"
        label["labeling_confidence"] = 0.0
        label["explain"] = {"reason": "no_text", "matches": matches}
        return label

    if not matches:
        # if text contains question words or verbs but no keyword matched, label other
        # conservative: detect presence of verbs/nouns by length
        if len(norm.split()) >= 3:
            label["intent"] = "other"
            label["labeling_confidence"] = 0.5
            label["explain"] = {"reason": "no_intent_match", "matches": matches}
        else:
            label["intent"] = "unclear"
            label["labeling_confidence"] = 0.0
            label["explain"] = {"reason": "too_short", "matches": matches}
        return label

    # If exactly one intent matched -> assign with confidence based on max_strength
    if len(matches) == 1:
        intent_id = next(iter(matches.keys()))
        det = matches[intent_id]
        max_strength = det.get("max_strength")
        # Avoid assigning on single weak generic matches (single-word triggers)
        if max_strength == "weak" and det.get("score", 0) <= 1:
            label["intent"] = "unclear"
            label["labeling_confidence"] = 0.0
            label["explain"] = {"reason": "weak_generic_single", "matches": matches, "details": matches}
            return label
        if max_strength == "strong":
            conf = 0.9
        elif max_strength == "medium":
            conf = 0.7
        else:
            conf = 0.5
        label["intent"] = intent_id if _valid_taxonomy_id(intent_id) else "other"
        label["labeling_confidence"] = conf
        label["explain"] = {"reason": "single_intent_match", "matches": matches, "details": matches}
        return label

    # Multiple intents matched -> use weighted scores and match strength
    scores = {k: v.get("score", 0) for k, v in matches.items()}
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_intent, top_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0

    top_max_strength = matches[top_intent].get("max_strength")
    second_max_strength = matches[sorted_scores[1][0]].get("max_strength") if len(sorted_scores) > 1 else None

    # First: detect multiple strong semantic intents and resolve or mark unclear
    strong_intents = {i for i, d in matches.items() if d.get("max_strength") == "strong"}
    if len(strong_intents) >= 2:
        # If both payments and lost_and_found are strongly signaled, treat as multi-issue -> unclear
        if "payments_charges" in strong_intents and "lost_and_found" in strong_intents:
            label["intent"] = "unclear"
            label["labeling_confidence"] = 0.0
            label["explain"] = {"reason": "conflicting_strong_payment_and_lost", "matches": matches, "details": matches}
            return label

        # If driver behavior is one of the strong intents and contains driver-priority phrases, prefer driver
        if "driver_behavior" in strong_intents:
            drv = matches.get("driver_behavior", {})
            drv_matched = {p for p, _ in drv.get("matches", [])}
            driver_priority_triggers = {"unprofessional", "cancelled on me", "cancelled on", "cancelled my trip", "three cars have canceled", "rude driver", "driver was rude"}
            if any(t in drv_matched for t in driver_priority_triggers):
                label["intent"] = "driver_behavior"
                label["labeling_confidence"] = 0.85
                label["explain"] = {"reason": "strong_driver_over_other_strong", "matches": matches, "details": matches}
                return label

        # If refund and payment both strong, and refund has explicit refund phrase, prefer refund
        if "refunds_adjustments" in strong_intents and "payments_charges" in strong_intents:
            ref = matches.get("refunds_adjustments", {})
            ref_pats = [p for p, _ in ref.get("matches", [])]
            if any(x in ref_pats for x in ("please refund", "refund please", "refund", "refund me")):
                label["intent"] = "refunds_adjustments"
                label["labeling_confidence"] = 0.9
                label["explain"] = {"reason": "explicit_refund_over_payment", "matches": matches, "details": matches}
                return label

        # Otherwise when multiple strong intents exist and none of the above precedence rules apply, mark unclear
        label["intent"] = "unclear"
        label["labeling_confidence"] = 0.0
        label["explain"] = {"reason": "multiple_competing_strong_intents", "matches": matches, "details": matches}
        return label

    # Intent-specific precedence and tie-break rules (explicit, auditable)
    # Account access should win over app technical when account-specific language exists
    if "account_access" in matches and "app_technical" in matches:
        acc = matches["account_access"]
        if _STRENGTH_WEIGHT.get(acc.get("max_strength"), 0) >= _STRENGTH_WEIGHT.get("medium"):
            label["intent"] = "account_access"
            label["labeling_confidence"] = 0.85
            label["explain"] = {"reason": "precedence_account_over_app", "matches": matches, "details": matches}
            return label

    # payments_charges vs driver_behavior: prefer driver when driver evidence is stronger or equal
    if "payments_charges" in matches and "driver_behavior" in matches:
        pay = matches["payments_charges"]
        drv = matches["driver_behavior"]
        # if driver has stronger max strength and payment is not strong -> driver
        if drv.get("max_strength") == "strong" and pay.get("max_strength") != "strong":
            label["intent"] = "driver_behavior"
            label["labeling_confidence"] = 0.8
            label["explain"] = {"reason": "precedence_driver_over_payment", "matches": matches, "details": matches}
            return label
        # if both strong, use score tie-breaker: prefer driver when driver score >= payment score
        if drv.get("max_strength") == "strong" and pay.get("max_strength") == "strong":
            if drv.get("score", 0) >= pay.get("score", 0):
                label["intent"] = "driver_behavior"
                label["labeling_confidence"] = 0.85
                label["explain"] = {"reason": "tie_break_driver_over_payment", "matches": matches, "details": matches}
                return label
            else:
                label["intent"] = "payments_charges"
                label["labeling_confidence"] = 0.85
                label["explain"] = {"reason": "tie_break_payment_over_driver", "matches": matches, "details": matches}
                return label
        # if payment strong and driver not strong -> payments
        if pay.get("max_strength") == "strong" and drv.get("max_strength") != "strong":
            label["intent"] = "payments_charges"
            label["labeling_confidence"] = 0.85
            label["explain"] = {"reason": "precedence_payment_over_driver", "matches": matches, "details": matches}
            return label

    # refunds vs payments: explicit refund requests should map to refunds_adjustments, tie-break by score
    if "refunds_adjustments" in matches and "payments_charges" in matches:
        ref = matches["refunds_adjustments"]
        pay = matches["payments_charges"]
        if ref.get("max_strength") == "strong" and pay.get("max_strength") != "strong":
            label["intent"] = "refunds_adjustments"
            label["labeling_confidence"] = 0.85
            label["explain"] = {"reason": "precedence_refund_over_payment", "matches": matches, "details": matches}
            return label
    

    # account access vs app technical: account-specific access language wins
    if "account_access" in matches and "app_technical" in matches:
        acc = matches["account_access"]
        app = matches["app_technical"]
        if _STRENGTH_WEIGHT.get(acc.get("max_strength"), 0) >= _STRENGTH_WEIGHT.get("medium"):
            label["intent"] = "account_access"
            label["labeling_confidence"] = 0.85
            label["explain"] = {"reason": "precedence_account_over_app", "matches": matches, "details": matches}
            return label

    # lost_and_found vs rides_and_trips: explicit lost-and-found language wins
    if "lost_and_found" in matches and "rides_and_trips" in matches:
        lost = matches["lost_and_found"]
        if _STRENGTH_WEIGHT.get(lost.get("max_strength"), 0) >= _STRENGTH_WEIGHT.get("medium"):
            label["intent"] = "lost_and_found"
            label["labeling_confidence"] = 0.85
            label["explain"] = {"reason": "precedence_lost_over_trip", "matches": matches, "details": matches}
            return label

    # payments vs lost_and_found: if both strong, mark unclear (multi-issue)
    if "payments_charges" in matches and "lost_and_found" in matches:
        pay = matches["payments_charges"]
        lost = matches["lost_and_found"]
        if pay.get("max_strength") == "strong" and lost.get("max_strength") == "strong":
            label["intent"] = "unclear"
            label["labeling_confidence"] = 0.0
            label["explain"] = {"reason": "conflicting_strong_payment_and_lost", "matches": matches, "details": matches}
            return label

    # driver_behavior vs rides_and_trips: prefer driver_behavior when driver evidence is strong
    if "driver_behavior" in matches and "rides_and_trips" in matches:
        drv = matches["driver_behavior"]
        drv_pats = [p for p, s in drv.get("matches", [])]
        # prefer driver when strong driver phrases present
        if drv.get("max_strength") == "strong" and any(x in drv_pats for x in ("unprofessional", "cancelled my trip", "cancelled on me", "cancelled on", "driver canceled", "driver cancelled")):
            label["intent"] = "driver_behavior"
            label["labeling_confidence"] = 0.85
            label["explain"] = {"reason": "precedence_driver_over_trip", "matches": matches, "details": matches}
            return label

    # Decision heuristics:
    # - If top score is at least 2x second score and top_score > 0 -> choose top
    # - If top has a strong match and second only weak -> choose top
    # - Otherwise ambiguous -> unclear
    if second_score == 0:
        # only one intent actually matched (shouldn't happen due to earlier check), but handle defensively
        conf = 0.8 if top_max_strength == "strong" else 0.6
        label["intent"] = top_intent
        label["labeling_confidence"] = conf
        label["explain"] = {"reason": "only_one_scored", "matches": matches, "details": matches}
        return label

    if top_score >= 2 * second_score and top_score > 0:
        conf = 0.8 if top_max_strength == "strong" else 0.65
        label["intent"] = top_intent
        label["labeling_confidence"] = conf
        label["explain"] = {"reason": "dominant_weight", "matches": matches, "details": matches}
        return label

    # strong vs weak rule: if top has a strong match and second only weak, pick top
    if top_max_strength == "strong" and (second_max_strength is None or second_max_strength == "weak") and top_score > second_score:
        label["intent"] = top_intent
        label["labeling_confidence"] = 0.8
        label["explain"] = {"reason": "strong_over_weak", "matches": matches, "details": matches}
        return label

    # Otherwise ambiguous -> unclear
    label["intent"] = "unclear"
    label["labeling_confidence"] = 0.0
    label["explain"] = {"reason": "conflicting_specific_matches", "matches": matches, "details": matches}
    return label


def label_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [label_case(c) for c in cases]


def explain_label(case: Dict[str, Any]) -> Dict[str, Any]:
    """Return the explanation for the heuristic label on a case.

    If `label_case` hasn't been run, run it and return the `explain` structure.
    """
    lbl = label_case(case)
    return lbl.get("explain", {})
