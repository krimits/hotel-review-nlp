"""Find what guests praise and complain about, clause by clause.

1. Each review is split into clauses with regular expressions. Only whitespace
   is ever inserted, so every quote shown to the owner is verbatim review text.
2. A lexicon names the hotel aspects in each clause ("breakfast" -> food).
3. An aspect-based sentiment model reads the clause once per word that named an
   aspect: "The room was clean, not much of a view" is positive about "room"
   and negative about "view". A neutral answer produces no finding.

Measured on hand-labelled hotel reviews: see README.md.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable, Sequence

ASPECTS = ("cleanliness", "staff", "location", "room", "food", "noise", "value", "facilities")
MODEL_ID = "yangheng/deberta-v3-base-absa-v1.1"
MAX_TERMS_PER_ASPECT = 3

# (clause, word that named the aspect) -> "positive", "negative" or None (neutral).
Scorer = Callable[[list[tuple[str, str]]], list[str | None]]

# Sentence boundaries. Besides ". " they include boundaries whose whitespace was
# lost when line breaks were stripped ("outside.Room", "helpfulMost"), a closing
# bracket followed by a capital, and ellipses.
_SENTENCE = re.compile(
    r"(?<!\bSt\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bMrs\.)(?<!\bAve\.)(?<!\bvs\.)(?<=[.!?;])\s+"
    r"|(?<=[a-z0-9)][.!?])(?=[A-Z])"
    r"|(?<=[a-z]{3})(?=[A-Z][a-z])"
    r"|(?<=\))\s*(?=[A-Z])"
    r"|\s*(?:\.{2,}|…)\s*"
)
# Contrast flips sentiment, so it always starts a new clause.
_CONTRAST = re.compile(
    r"(?:,\s*|\s+)(?:but|however|although|though|whereas|except that|except|yet)\s+", re.I
)
# "and" starts a new clause only when a new subject or predicate follows:
# "spotless and the staff were kind" splits, "clean and tidy" does not.
_AND_CLAUSE = re.compile(
    r"(?:,\s*|\s+)and\s+(?=(?:the|our|my|we|they|it|its|their|there|he|she|i|was|were|only)\b)",
    re.I,
)
_STRIP = " ,;:-"

LEXICON = {
    "cleanliness": r"clean\w*|dirt\w*|filth\w*|dust\w*|(?:blood)?stain(?!e?less)\w*|spotless|hygien\w*"
                   r"|unhygien\w*|smell\w*|odou?r\w*|mou?ld\w*|musty|mildew\w*|smoke|smoky|grim\w*"
                   r"|bed ?bugs?|cockroach\w*|roach\w*|tidy|untidy|messy|debris|litter\w*|trash\w*"
                   r"|garbage|rubbish|hairs? in",
    "staff": r"staff\w*|reception\w*|front desk|concierge|manager\w*|management|service"
             r"|employee\w*|porter\w*|bell ?(?:man|men|boy|boys|desk|hop)\w*|doorm[ae]n|housekeep\w*"
             r"|maid\w*|waiter\w*|waitress\w*|bartender\w*|check[- ]?in|check[- ]?out"
             r"|helpful|unhelpful|friendly|unfriendly|rude\w*|polite\w*|courteous\w*"
             r"|welcom\w*|attentive|accommodat\w*|apologetic",
    "location": r"locat\w*|neighbou?rhood|walk(?:ing)? distance|walkable|the walk"
                r"|(?:short|easy|quick|long|little|nice|brisk|minutes?|mins?|blocks?|\d+)[\s-]+walk\w*"
                r"|walk(?:s|ed|ing)? (?:to|from|back)|subway\w*|metro|(?:train|bus|penn) station"
                r"|close to|closer to|near|nearby|nearest|distance|central|downtown|times square"
                r"|broadway|central park|blocks? (?:away|from|off|walk)|blk|minutes? (?:from|away)"
                r"|convenient\w*|transport\w*",
    "room": r"rooms?|bed|beds|bedrooms?|bathrooms?|bath|bathtub|tub|shower\w*|toilet\w*"
            r"|sink|pillow\w*|mattress\w*|sheets?|linens?|towel\w*|views?|windows?|balcon\w*"
            r"|suite\w*|furniture|d[eé]cor\w*|tv|television|air[- ]?condition\w*|heating"
            r"|temperature|lamps?|closet\w*|wardrobe\w*|fridge|kettle|hair ?dryer\w*"
            r"|robes?|carpet\w*|walls?|wall ?paper|curtains?|blinds?|duvets?|headboards?"
            r"|tin(?:y|ier|iest)|small(?:er|est)?|cramped|space|spaces|spacious|compact|snug",
    "food": r"food|breakfast\w*|brunch|lunch|dinner\w*|meals?|restaurant\w*|caf[eé]\w*"
            r"|coffee|tea|buffet\w*|menu\w*|dish\w*|drinks?|bars?|snacks?"
            r"|fruit|cake|bakery|eat|ate|eating|dining|cuisine|tast\w*|delicious|salty"
            r"|bland|cooked|undercooked|starbucks",
    "noise": r"noise\w*|noisy|loud\w*|quiet\w*|silen\w*|soundproof\w*|sounds?|thin walls"
             r"|hear|heard|hearing|earplugs?|music|sirens?|construction|peaceful|throbbing",
    "value": r"prices?|priced|pricey|price ?tags?|value|money|expensive|inexpensive|cheap\w*"
             r"|affordab\w*|overpriced|costs?|costly|rates?|fees?|charg\w*|paid|pay|paying|worth"
             r"|deals?|bargain|budget|deposit\w*|\$\s?\d+|\d+\s?\$|\d+\s?(?:usd|eur|euros?)|€\s?\d+"
             r"|\d+\s?€|dollars?|tax(?:es)?|credit",
    "facilities": r"facilit\w*|amenit\w*|lobby|gym|fitness|pool|spa|sauna|wi-?fi|internet"
                  r"|parking|garage|elevators?|lifts?|escalators?|hallways?|corridors?"
                  r"|lounges?|business cent\w*|rooftop|terrace|security|keys?|key ?cards?"
                  r"|lighting|signs|signage|laundry|luggage|entrance|exterior"
                  r"|public spaces|button",
}


def _phrase(pattern: str) -> re.Pattern:
    return re.compile(rf"(?<![\w-])(?:{pattern})(?![\w-])", re.I)


_PATTERNS = {aspect: _phrase(pattern) for aspect, pattern in LEXICON.items()}

# Phrases whose words would otherwise be read as another aspect. Each one is
# tagged as its real aspect (or none) and blanked before the lexicon runs.
_OVERRIDES = [
    (_phrase(r"room service"), "food"),
    (_phrase(r"(?:meeting|conference|function|business|fitness|exercise|computer|luggage|storage"
             r"|board|ball|game|laundry) ?rooms?"), "facilities"),
    (_phrase(r"(?:public|common|parking|office|event|meeting) spaces?"), "facilities"),
    (_phrase(r"(?:reception|check[- ]?in|front desk|lobby|seating|waiting) areas?"), "facilities"),
    (_phrase(r"(?:breakfast|dining) ?rooms?"), "food"),
    (_phrase(r"internet (?:rate|deal|price|booking|special|fare)s?"), "value"),
    (_phrase(r"(?:no ?where|not anywhere) near"), None),
    (_phrase(r"tiny bit|small (?:group|fee|price|charge|thing|detail|touch|issue|problem|matter"
             r"|complaint|gripe|point)s?"), None),
    (_phrase(r"deal(?:s|t|ing)? with"), None),
    # "check out the spa", "check those out": the verb, not the front desk.
    (_phrase(r"check(?:ed|ing)? (?:it|them|those|these|this|that) out"
             r"|check out (?:the|those|these|this|that|our|their|your|some|a|an|what)"), None),
]

# Words that name where something happened rather than what was good or bad.
_ROOM_NOUNS = _phrase(r"(?:bed|bath)?rooms?")
_ROOM_AS_PLACE = _phrase(
    r"(?:in|into|to|of|from|inside|at|for)\s+(?:(?:the|our|my|your|their|his|her|a)\s+)?(?:bed|bath)?rooms?"
)
_FOOD_VENUES = _phrase(r"restaurants?|bars?|caf[eé]s?|diners?|delis?|pubs?|eateries|clubs?")

_ENGLISH = re.compile(
    r"\b(?:the|and|was|were|is|are|we|our|it|to|of|in|for|with|but|not|very|room|hotel|staff)\b", re.I
)


def normalize(text: str) -> str:
    """Decode HTML entities and collapse whitespace: quotes are taken from this text."""
    return " ".join(html.unescape(str(text or "")).split())


def looks_english(text: str) -> bool:
    """Cheap guard: the lexicon and the model only understand English reviews."""
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) < 0.9:
        return False
    words = len(text.split())
    return words < 4 or len(_ENGLISH.findall(text)) / words >= 0.1


def _only_via(aspect: str, clause: str, nouns: re.Pattern) -> bool:
    """True when the aspect is named only through the given nouns."""
    return not _PATTERNS[aspect].search(nouns.sub(" ", clause))


def lexicon_matches(clause: str) -> list[tuple[str, list[str]]]:
    """(aspect, the distinct words that named it) for every aspect in the clause."""
    found: dict[str, list[str]] = {}
    for pattern, aspect in _OVERRIDES:
        match = pattern.search(clause)
        if match:
            if aspect:
                found.setdefault(aspect, []).append(match.group())
            clause = pattern.sub(lambda m: " " * len(m.group()), clause)
    for aspect, pattern in _PATTERNS.items():
        for match in pattern.finditer(clause):
            terms = found.setdefault(aspect, [])
            if match.group().lower() not in (t.lower() for t in terms):
                terms.append(match.group())
    if len(found) > 1 and "room" in found and _only_via("room", clause, _ROOM_NOUNS):
        # "locked out of our room" (a place), "the bathroom was dirty" (cleanliness).
        only_a_place = len(_ROOM_AS_PLACE.findall(clause)) == len(_ROOM_NOUNS.findall(clause))
        if only_a_place or "cleanliness" in found or "noise" in found:
            del found["room"]
    if len(found) > 1 and "food" in found and _only_via("food", clause, _FOOD_VENUES):
        # "loads of restaurants nearby" is location, "noise from the bars" is noise.
        if "location" in found or "noise" in found:
            del found["food"]
    return [(aspect, found[aspect]) for aspect in ASPECTS if aspect in found]


def _pieces(text: str, start: int, end: int, separator: re.Pattern) -> list[tuple[int, int]]:
    spans, cursor = [], start
    for match in separator.finditer(text, start, end):
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, end))
    return spans


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in _STRIP:
        start += 1
    while end > start and text[end - 1] in _STRIP:
        end -= 1
    return start, end


def clause_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the clauses of a normalised text.

    A fragment of fewer than two words ("closed.", "Horrible.") is merged into
    the clause before it rather than dropped, so every opinion stays attached
    to something the owner can read.
    """
    spans: list[tuple[int, int]] = []
    for s_start, s_end in _pieces(text, 0, len(text), _SENTENCE):
        for c_start, c_end in _pieces(text, s_start, s_end, _CONTRAST):
            for start, end in _pieces(text, c_start, c_end, _AND_CLAUSE):
                start, end = _strip(text, start, end)
                if start == end:
                    continue
                if len(text[start:end].split()) >= 2 or not spans:
                    spans.append((start, end))
                else:
                    spans[-1] = (spans[-1][0], end)
    if len(spans) > 1 and len(text[slice(*spans[0])].split()) < 2:
        spans[1] = (spans[0][0], spans[1][1])
        del spans[0]
    return spans


def split_clauses(text: str) -> list[str]:
    text = normalize(text)
    return [text[start:end] for start, end in clause_spans(text)]


def analyze(reviews: Sequence[str], scorer: Scorer) -> list[list[dict]]:
    """One list of findings per review: {aspect, sentiment, term, quote}.

    All (clause, word) pairs of all reviews go to the scorer in one call, so
    the model batches them. A clause can praise and criticise the same aspect;
    that yields two findings with the same quote.
    """
    jobs = []
    for index, text in enumerate(reviews):
        for clause in split_clauses(text):
            for aspect, terms in lexicon_matches(clause):
                jobs += [(index, clause, aspect, term) for term in terms[:MAX_TERMS_PER_ASPECT]]
    labels = scorer([(clause, term) for _, clause, _, term in jobs]) if jobs else []
    findings: list[list[dict]] = [[] for _ in reviews]
    seen = set()
    for (index, clause, aspect, term), sentiment in zip(jobs, labels, strict=True):
        if sentiment not in ("positive", "negative") or (index, clause, aspect, sentiment) in seen:
            continue
        seen.add((index, clause, aspect, sentiment))
        findings[index].append({"aspect": aspect, "sentiment": sentiment, "term": term, "quote": clause})
    return findings


class AbsaModel:
    """DeBERTa-v3 aspect sentiment: the argmax of negative / neutral / positive.

    Loaded once per process; pairs are sorted by length before batching so a
    batch pads to similar lengths, then returned in the caller's order.
    """

    def __init__(self, model_id: str = MODEL_ID, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(min(torch.get_num_threads(), 4))
        self._torch = torch
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        names = {int(i): label.lower() for i, label in self.model.config.id2label.items()}
        if set(names.values()) != {"negative", "neutral", "positive"}:
            raise ValueError(f"{model_id} labels are {names}, expected negative/neutral/positive")
        self.names = [None if names[i] == "neutral" else names[i] for i in range(len(names))]

    def __call__(self, pairs: list[tuple[str, str]],
                 progress: Callable[[float], None] | None = None) -> list[str | None]:
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]))
        labels: list[str | None] = [None] * len(pairs)
        for start in range(0, len(order), self.batch_size):
            batch = order[start:start + self.batch_size]
            encoded = self.tokenizer(
                [pairs[i][0] for i in batch], [pairs[i][1] for i in batch],
                padding=True, truncation=True, max_length=160, return_tensors="pt",
            )
            with self._torch.inference_mode():
                best = self.model(**encoded).logits.argmax(-1).tolist()
            for i, label_index in zip(batch, best, strict=True):
                labels[i] = self.names[label_index]
            if progress:
                progress(min(1.0, (start + len(batch)) / len(order)))
        return labels
