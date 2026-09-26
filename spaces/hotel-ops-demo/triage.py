"""Find what guests praise and complain about, clause by clause and topic by topic.

1. Each review is split into sentences, and a sentence into clauses at "but",
   "although" or "and the …", but only where the new clause names something of
   its own: "Airconds provided but all not cold" stays one clause. Only
   whitespace is ever inserted, so every quote is verbatim review text.
2. A lexicon names 30 topics an owner can act on (bed, air conditioning,
   check-in, lift and stairs …) in 8 categories. Rules keep a word from naming
   the wrong topic: "welcoming" about an apartment is not the staff, "moving
   furniture" upstairs is noise, "got the room at 3rd floor" is a place.
3. An aspect-based sentiment model reads the clause once per word that named a
   topic. A neutral answer produces no finding, a suggestion ("should be
   serviced") is a complaint, and a problem that was reported but not solved
   is a finding of its own.

Measured on reviews labelled by people: see README.md.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable, Sequence
from typing import NamedTuple

ASPECTS = ("cleanliness", "staff", "location", "room", "food", "noise", "value", "facilities")
TOPICS = (
    "cleanliness.general", "cleanliness.linen", "cleanliness.odour", "cleanliness.pests",
    "staff.people", "staff.response", "staff.resolution", "staff.checkin",
    "location.general",
    "room.general", "room.comfort", "room.size", "room.bed", "room.climate", "room.bathroom",
    "room.storage", "room.equipment", "room.view",
    "food.breakfast", "food.dining",
    "noise.general",
    "value.price", "value.charges",
    "facilities.access", "facilities.wifi", "facilities.parking", "facilities.kitchen",
    "facilities.family", "facilities.leisure", "facilities.common",
)
MODEL_ID = "yangheng/deberta-v3-base-absa-v1.1"
MAX_TERMS_PER_TOPIC = 3
# A finding the model is less sure of than this is marked for the owner to check.
CHECK_BELOW = 0.7

# (clause, word that named the topic) -> (label, probability). The label is
# "positive", "negative" or None (neutral).
Score = tuple[str | None, float]
Scorer = Callable[[list[tuple[str, str]]], list[Score]]

# Sentence boundaries. Besides ". " and line breaks they include boundaries
# whose whitespace was lost when line breaks were stripped ("outside.Room",
# "helpfulMost"), a closing bracket followed by a capital, and ellipses.
_SENTENCE = re.compile(
    r"(?<!\bSt\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bMrs\.)(?<!\bAve\.)(?<!\bvs\.)(?<=[.!?;])\s+"
    r"|\s*\n\s*"
    r"|(?<=[a-z0-9)][.!?])(?=[A-Z])"
    r"|(?<=[a-z]{3})(?=[A-Z][a-z])"
    r"|(?<=\))\s*(?=[A-Z])"
    r"|\s*(?:\.{2,}|…)\s*"
)
# Contrast flips sentiment, so it can start a new clause.
_CONTRAST = re.compile(
    r"(?:,\s*|\s+)(?:but|however|although|though|whereas|except that|except|yet)\s+", re.I
)
# "and" can start a new clause only when a new subject or predicate follows:
# "spotless and the staff were kind" splits, "clean and tidy" does not.
_AND_CLAUSE = re.compile(
    r"(?:,\s*|\s+)and\s+(?=(?:the|our|my|we|they|it|its|their|there|he|she|i|was|were|only)\b)",
    re.I,
)
_STRIP = " ,;:-"

# Words that name a topic by what they are.
NOUNS = {
    "cleanliness.general": r"housekeep\w*|cleaning\s+(?:service|staff|lad(?:y|ies)|crew|team)|cleaners?",
    "cleanliness.linen": r"(?:bed\s?)?linens?|(?:bed\s?)?sheets?|bedding|(?:bath|beach)\s?towels?|towels?"
                         r"|pillow\s?cases?|duvet\s?covers?|blankets?|bath\s?(?:mats?|robes?)",
    "cleanliness.odour": r"smell\w*|odou?r\w*|stink\w*|stank|stench|musty|mou?ld\w*|mildew\w*|damp\w*|sewage"
                         r"|sewer\w*|smok(?:e|ed|y)|cigarettes?|fumes",
    "cleanliness.pests": r"bed\s?bugs?|bugs?|insects?|cockroach\w*|roach(?:es)?|ants|mosquito(?:e?s)?|mice|mouse"
                         r"|rats?|flies|fleas?|spiders?|pests?|bites?|bitten",
    "staff.people": r"staff\w*|hosts?|hostess\w*|owners?|landlord\w*|landlad(?:y|ies)|manager\w*|management"
                    r"|reception\w*|front\s+desk|concierge|porters?|bell\s?(?:man|men|boys?|desk|hops?)|doorm[ae]n"
                    r"|maids?|waiter\w*|waitress\w*|bartenders?|employees?|personnel|team|service|hospitality",
    "staff.response": r"respon(?:d|ds|ded|ding|se|ses|sive|siveness)|unresponsive|repl(?:y|ies|ied|ying)"
                      r"|answer\w*|communicat\w*|contact(?:ed|ing|able)?|reachable|unreachable"
                      r"|messag(?:e|es|ed|ing)|whatsapp"
                      r"|came\s+(?:\w+\s+)?(?:quickly|immediately|straight\s+away|right\s+away|at\s+once|promptly)"
                      r"|came\s+(?:\w+\s+)?with\s+(?:tools|a\s+technician|someone|a\s+plumber)",
    "staff.checkin": r"(?:self[-\s]?)?check[-\s]?ins?|check[-\s]?outs?|key\s?box(?:es)?|lock\s?box(?:es)?"
                     r"|key\s?safe|key\s?cards?|keycards?|keys?|registration\w*|register(?:ed|ing)?|passports?"
                     r"|id\s+cards?|(?:early|late)\s+arrival|luggage\s+storage"
                     r"|(?:left|leave|store|stored|keep|kept)\s+(?:our|the|my)\s+(?:bags|luggage|suitcases?)",
    "location.general": r"locat\w*|position(?:ed)?|neighbou?rhood|area|district|surroundings|distance"
                        r"|walk(?:ing)?\s+distance|walkable|the\s+walk|(?:a\s+)?bit\s+of\s+a\s+walk|quite\s+a\s+walk"
                        r"|(?:too|quite|very|a\s+bit|a\s+little|rather)\s+far|far\s+(?:from|away|out)|long\s+way"
                        r"|(?:short|easy|quick|long|little|nice|brisk|minutes?|mins?|blocks?)[\s-]+walk\w*"
                        r"|\d+[\s-]*(?:min(?:ute)?s?|blocks?)?[\s-]+walk\w*|walk(?:s|ed|ing)?\s+(?:to|from|back)"
                        r"|minutes?\s+(?:from|away)|blocks?\s+(?:away|from|off)|subway\w*|metro|trams?|bus(?:es)?"
                        r"|(?:train|bus|railway|metro)\s+stations?|stations?|airport|taxis?|transport\w*"
                        r"|downtown|old\s+town|cent(?:er|re)|city\s+cent(?:er|re)|beach(?:es)?|attractions?"
                        r"|sights|sightseeing|museums?|shops|shopping|supermarkets?|grocer\w*|safe|safety"
                        r"|unsafe|times\s+square|broadway|central\s+park",
    "room.general": r"(?:bed)?rooms?|apartments?|flats?|studios?|suites?|accommodation|property|premises",
    "room.bed": r"beds?|mattress\w*|pillows?|sofa\s?beds?|bunk\s?beds?|(?:king|queen)[\s-]?sized?",
    "room.climate": r"air[\s-]?con\w*|aircon\w*|a/c|ac|air[\s-]?condition\w*|heating|heaters?|radiators?"
                    r"|temperature|thermostat|fans?|ventilat\w*|stuffy|freezing|chilly",
    "room.bathroom": r"bathrooms?|wash\s?rooms?|baths?|bathtubs?|tubs?|showers?|showering|toilets?|wc|sinks?"
                     r"|wash\s?basins?"
                     r"|basins?|taps?|faucets?|(?:hot|cold|no)\s+water|water\s+pressure|drains?|drainage"
                     r"|plumbing|flush\w*|toiletries|shampoo|soap|bidet|hair\s?dryers?",
    "room.storage": r"wardrobes?|closets?|cupboards?|drawers?|shel(?:f|ves)|shelving|hangers?|(?:coat\s+)?hooks"
                    r"|luggage\s?racks?|storage|(?:place|space|room)\s+(?:to|for)\s+(?:put|hang|store|leave|keep)",
    "room.equipment": r"furniture|tvs?|televisions?|fridges?|refrigerators?|mini[\s-]?bars?|kettles?"
                      r"|coffee\s+(?:machines?|makers?|pods?|capsules?)|nespresso|iron(?:ing)?|lamps?|lights?"
                      r"|bulbs?|curtains?|blinds?|shutters?|windows?|doors?|locks?|sockets?|plugs?|outlets?"
                      r"|chargers?|usb|desks?|sofas?|couch(?:es)?|mirrors?|carpets?|rugs?|floors?|walls?"
                      r"|wall\s?paper|ceilings?|appliances?|equipment|amenit(?:y|ies)|electricity|power"
                      r"|safe\s+(?:box|deposit)|water\s+bottles?|bottles?\s+of\s+water|bottled\s+water"
                      r"|(?:the|in-room|room|electronic)\s+safe"
                      r"(?!\s+(?:area|neighbou?rhood|place|district|city|street|location|bet|to))",
    "room.view": r"views?|balcon(?:y|ies)|terraces?|patio|garden|courtyard|panoram\w*|overlook\w*|rooftop"
                 r"|scenery",
    "food.breakfast": r"breakfast\w*|brunch|buffets?|croissants?|pastr(?:y|ies)|cereals?|yogh?urts?|continental",
    "food.dining": r"food|lunch\w*|dinners?|meals?|restaurants?|caf[eé]s?|coffee|tea|menus?|dish|dishes|drinks?"
                   r"|bars?|snacks?|fruit|cakes?|bakery|eat|ate|eating|dining|cuisine|wines?|cocktails?|beers?"
                   r"|room\s+service|welcome\s+drinks?",
    "noise.general": r"noise\w*|noisy|loud\w*|quiet\w*|silen\w*|sound\s?proof\w*|sounds?|thin\s+walls|hear|heard"
                     r"|hearing|ear\s?plugs?|music|sirens?|construction|peaceful|throbbing|traffic|snor\w*"
                     r"|barking|bells?",
    "value.charges": r"fees?|charg(?:e|es|ed)|surcharges?|deposits?|tax(?:es)?|(?:city|tourist)\s+tax(?:es)?"
                     r"|(?:extra|hidden)\s+(?:costs?|charges?|fees?)|pre-?authori[sz]ation|refund(?:s|ed)?"
                     r"|overcharg\w*|bills?|invoices?|service\s+charges?",
    "facilities.access": r"lifts?|elevators?|escalators?|stairs|stair\s?cases?|stairways?|steps"
                         r"|flights?\s+of\s+stairs|walk[\s-]?up|luggage|baggages?|bags|suitcases?|wheelchairs?"
                         r"|mobility|disabled|ramps?|(?:\d+(?:st|nd|rd|th)|ground|first|second|third|fourth|fifth"
                         r"|sixth|seventh|top|highest|upper|last)\s+floors?",
    "facilities.wifi": r"wi-?fi|wifi|internet|wireless|broadband|router",
    "facilities.parking": r"parking|car\s?parks?|garages?|valet|ztl",
    "facilities.kitchen": r"kitchen\w*|cook(?:ing|er)?|stoves?|hobs?|ovens?|pans|pots|utensils|cutlery"
                          r"|dishwashers?|washing\s+machines?|washers?|dryers?|laundry|detergent|toasters?"
                          r"|microwaves?",
    "facilities.family": r"cots?|cribs?|travel\s+cots?|high\s?chairs?|bab(?:y|ies)|infants?|toddlers?|children"
                         r"|kids?|famil(?:y|ies)|strollers?|prams?|pushchairs?",
    "facilities.leisure": r"pools?|swimming|spas?|saunas?|jacuzzis?|hot\s?tubs?|gym|fitness|wellness|massages?"
                          r"|steam\s?rooms?|hammam",
    "facilities.common": r"lobby|hallways?|corridors?|lounges?|entrances?|security|cctv|signs|signage|exterior"
                         r"|fa[cç]ade|business\s+cent(?:er|re)|facilit(?:y|ies)|common\s+(?:areas?|rooms?|spaces?)",
}
# Words that name a topic by an opinion about it. The rules below give them the
# topic of the thing they describe ("the bed was too small" is about the bed),
# or drop them when they describe something else ("a welcoming apartment").
OPINIONS = {
    "cleanliness.general": r"clean\w*|unclean|dirt\w*|filth\w*|dust\w*|(?:blood)?stain(?!e?less)\w*|spotless"
                           r"|hygien\w*|unhygien\w*|grim(?:e|y)|tidy|untidy|messy|debris|litter\w*|trash\w*"
                           r"|garbage|rubbish|hairs?|immaculate",
    "staff.people": r"helpful|unhelpful|friendly|unfriendly|rude\w*|polite\w*|impolite|courteous\w*|discourteous"
                    r"|welcom\w*|attentive|inattentive|accommodating|apologetic|hospitable|kind|caring"
                    r"|professional\w*|unprofessional",
    "location.general": r"convenient\w*|central\w*|close\s+to|closer\s+to|near|nearby|nearest",
    "room.comfort": r"comfort\w*|uncomfortable|comfy|cos(?:y|ier)|coz(?:y|ier)|homely|homey|relax\w*|charming"
                    r"|stylish|atmosphere|ambi[ae]nce|d[eé]cor\w*|furnished|feels?\s+like\s+home|at\s+home",
    "room.size": r"tin(?:y|ier|iest)|small(?:er|est)?|cramped|spacious|compact|snug|roomy|narrow|space|spaces"
                 r"|square\s+met(?:er|re)s|m2|sq\s?m",
    "room.equipment": r"outdated|dated|old[-\s]fashioned|worn(?:[-\s]out)?|shabby|run[-\s]down|tired[-\s]looking"
                      r"|dilapidated|needs?\s+(?:a\s+)?(?:refresh|refurbishment|renovation|makeover|update)",
    "food.dining": r"tast(?:y|ed|ing|es|eless)|delicious|salty|bland|cooked|undercooked|overcooked|yummy",
    "value.price": r"prices?|priced|pricey|value|money|expensive|inexpensive|cheap\w*|affordab\w*|overpriced"
                   r"|costs?|costly|rates?|worth\w*|deals?|bargain|budget|paid|pay|paying|\$\s?\d+|\d+\s?\$"
                   r"|\d+\s?(?:usd|eur|euros?)|€\s?\d+|\d+\s?€|dollars?|euros?",
}
# Phrases that would otherwise name the wrong topic. Each names its real topic,
# or none (None); where phrases overlap, the longest one wins.
_OVERRIDES = [
    (r"(?:be|been|being|get|gets|got|well|regularly|properly|to)\s+servic(?:e|ed|ing)"
     r"|servic(?:ed|ing)\s+(?:the|them|it)", None),
    (r"(?:customer|guest)\s+service", "staff.people"),
    (r"(?<=tools\s)to\s+clean|(?<=came\s)to\s+clean|(?<=come\s)to\s+clean", None),
    (r"(?:never|did\s*n['’]?t|did\s+not|without)\s+(?:\w+\s+)?(?:met|meet|meeting|see|saw|seen)\s+"
     r"(?:the\s+|our\s+|a\s+|any\s+)?(?:hosts?|owners?|staff|landlord|landlady|anyone|anybody)(?:\s+in\s+person)?"
     r"|in\s+person", None),
    (r"(?:tea|trash|garbage|rubbish|bin|plastic|paper|shopping)\s+bags?", None),
    (r"(?:sad|sorry|disappointed|happy|glad|pleased|surprised|shocked|upset)\s+to\s+hear"
     r"|hear(?:d)?\s+(?:that|about|from|back)", None),
    (r"(?:keep|kept|keeping|leave|left)\s+(?:the\s+|all\s+the\s+)?(?:windows?|doors?)\s+open"
     r"|open(?:ed|ing)?\s+(?:the\s+|all\s+the\s+)?(?:windows?|doors?)", None),
    (r"hair\s?dryers?", "room.bathroom"),
    (r"(?:meeting|conference|function|business|computer|games?)\s?rooms?", "facilities.common"),
    (r"(?:fitness|exercise)\s?rooms?", "facilities.leisure"),
    (r"(?:luggage|storage)\s?rooms?", "staff.checkin"),
    (r"laundry\s?rooms?", "facilities.kitchen"),
    (r"(?:breakfast|dining)\s?(?:rooms?|areas?|halls?)", "food.breakfast"),
    (r"(?:public|common|office|event|meeting|shared)\s+(?:spaces?|areas?)"
     r"|(?:reception|lobby|waiting|smoking|seating|lounge)\s+areas?", "facilities.common"),
    (r"(?:parking|car)\s+(?:spaces?|spots?|areas?|lots?)", "facilities.parking"),
    (r"(?:storage|wardrobe|closet|cupboard|hanging)\s+(?:space|room)", "room.storage"),
    (r"(?:shower|bathroom|toilet)\s+(?:areas?|space|cubicles?|cabins?|trays?)", "room.bathroom"),
    (r"(?:kitchen|cooking)\s+(?:areas?|space|corners?)", "facilities.kitchen"),
    (r"(?:living|sitting)\s+(?:areas?|rooms?|spaces?)", "room.general"),
    (r"(?:pool|spa)\s+areas?", "facilities.leisure"),
    (r"play\s?(?:areas?|grounds?|rooms?)|kids['’]?\s+club", "facilities.family"),
    (r"family\s+(?:rooms?|suites?|apartments?)", "room.general"),
    (r"family[-\s](?:run|owned)|family\s+business", "staff.people"),
    (r"(?:a\s+)?(?:few|couple\s+of|\d+|two|three|four|five|ten)\s+steps\s+(?:from|away|to)", "location.general"),
    (r"internet\s+(?:rates?|deals?|prices?|booking|specials?|fares?)", "value.price"),
    (r"security\s+deposits?", "value.charges"),
    (r"(?:no\s?where|not\s+anywhere)\s+near", None),
    (r"(?:a\s+)?tiny\s+bit|little\s+bit|small\s+(?:group|fee|price|charge|thing|detail|touch|issue|problem|matter"
     r"|complaint|gripe|point|downside|negative|minus|con)s?", None),
    (r"deal(?:s|t|ing)?\s+with", None),
    (r"check(?:ed|ing)?\s+(?:it|them|those|these|this|that)\s+out|check\s+out\s+(?:the|those|these|this|that|our"
     r"|their|your|some|a|an|what|all)", None),
    (r"(?:some|any|this|that|what|all)\s+kinds?\s+of|kind\s+of", None),
    (r"(?:not|never\s+been)\s+(?:a|the)\s+(?:big\s+|huge\s+)?fans?\s+of|(?:big|huge)\s+fans?\s+of", None),
    (r"located\s+(?:in|inside)\s+(?:an?|the)\s+(?:\w+\s+)?(?:building|palazzo|palace|villa|tower|house)", None),
    (r"charg(?:e|ed|ing)\s+(?:my|our|the|your|a)\s+(?:phones?|laptops?|devices?|car)", "room.equipment"),
    (r"in\s+charge", None),
    (r"welcome\s+(?:drinks?|gifts?|baskets?|packs?)", "food.dining"),
    (r"(?:tea|coffee)\s+(?:and|&)\s+(?:tea|coffee)\s+(?:facilities|making|maker)", "room.equipment"),
]


def _phrase(pattern: str) -> re.Pattern:
    return re.compile(rf"(?<![\w-])(?:{pattern})(?![\w-])", re.I)


# (pattern, topic or None, priority when two matches are equally long, names by opinion)
_MATCHERS = (
    [(_phrase(pattern), topic, 0, False) for pattern, topic in _OVERRIDES]
    + [(_phrase(pattern), topic, 1, False) for topic, pattern in NOUNS.items()]
    + [(_phrase(pattern), topic, 2, True) for topic, pattern in OPINIONS.items()]
)

# Someone the staff words can describe.
_PERSON = _phrase(r"he|she|him|her|his|they|them|everyone|everybody|people|guys?|lad(?:y|ies)|m[ae]n|wom[ae]n"
                  r"|gentlem[ae]n|girls?|boys?|hotel|b&b|guest\s?house")
# A room word after these is where something happened, not what was judged:
# "got the room at 3rd floor", "sell your rooms", "not a place to put clothes in the rooms".
_PLACE_BEFORE = re.compile(
    r"\b(?:got|get|gave\s+us|given|booked|book|reserved|assigned|allocated|sell|sold|selling|rent(?:ed|ing)?"
    r"|upgraded\s+to|moved\s+to|changed\s+to|to|into|in|inside|of|from|at|for|out\s+of|enter(?:ed|ing)?"
    r"|back\s+to|reach(?:ed|ing)?|access(?:ed|ing)?|find|found|leave|left)\s+"
    r"(?:(?:the|our|my|your|their|his|her|a|an|another|this|that|each|every|both|all)\s+)?$",
    re.I,
)
_PLACEABLE = re.compile(r"(?:bed|bath)?rooms?|apartments?|flats?|studios?|suites?|accommodation|property"
                        r"|premises", re.I)
# Something broken is a fault of the thing, even in a clause about dirt.
_FAULT = _phrase(r"broken|not\s+working|(?:did|does|do)\s*n['’]?t\s+work|(?:did|does|do)\s+not\s+work"
                 r"|out\s+of\s+order|leak\w*|faulty|damaged|cracked|missing|blocked|clogged|stopped\s+working"
                 r"|no\s+hot\s+water")
# Room words that are where a noise comes from: "moving furniture", "thin walls".
_NOISE_SOURCE = re.compile(r"furniture|walls?|floors?|ceilings?|doors?|windows?", re.I)
# Food words that are places to go: "restaurants nearby" is about the location.
_VENUE = re.compile(r"restaurants?|bars?|caf[eé]s?|diners?|delis?|pubs?|eateries|clubs?|eat|eating|dining", re.I)
_CHEAP = re.compile(r"cheap\w*", re.I)
_PRICEY = re.compile(r"expensive|overpriced|pricey|costly", re.I)
# "Cheap" as a judgement of quality, not of price.
_CHEAP_QUALITY = _phrase(r"(?:poor|tacky|nasty|shabby|flimsy)\s+and\s+cheap"
                         r"|cheap\s+and\s+(?:nasty|poor|tacky|shabby|flimsy)"
                         r"|cheap(?:ly)?\s+(?:made|looking|quality|materials?|feel\w*)"
                         r"|(?:feels?|felt|looks?|looked)\s+cheap")
# "No cod for the baby" is a cot.
_COD = _phrase(r"cods?")
_BABY = _phrase(r"bab(?:y|ies)|infants?|toddlers?|child|children|kids?|son|daughter")

# A suggestion about the clause's subject: "Airconds should be well service(d)".
_SUGGEST_WHOLE = re.compile(
    r"\b(?:should|must|needs?\s+to|ought\s+to|has\s+to|have\s+to)\s+"
    r"(?:really\s+|definitely\s+|seriously\s+|urgently\s+)?(?:be|been|get|gets|have\s+been)\s+(?:\w+\s+)?"
    r"(?:servic\w*|fix\w*|repair\w*|replac\w*|clean\w*|chang\w*|updat\w*|upgrad\w*|renovat\w*|improv\w*"
    r"|check\w*|maintain\w*|provid\w*|add\w*|install\w*|paint\w*|refresh\w*|remov\w*|mention\w*|stated?"
    r"|disclos\w*|more|better|bigger|cleaner|quieter|warmer|cooler|faster|friendlier|nicer|clearer)\b"
    r"|\bneeds?\s+(?:a\s+|an\s+|some\s+|more\s+|urgent\s+|serious\s+|real\s+)?(?:\w+\s+){0,2}"
    r"(?:repair\w*|fix\w*|replac\w*|clean\w*|renovat\w*|updat\w*|upgrad\w*|attention|maintenance|improv\w*"
    r"|refurbish\w*|refresh\w*|paint\w*|servic\w*|work)\b"
    r"|\b(?:would|could|might)\s+(?:have\s+)?(?:been|be)\s+(?:so\s+|a\s+(?:little|bit)\s+|much\s+)?better\b"
    r"|\b(?:would|will)\s+be\s+(?:a\s+)?(?:good|great|nice|better)\s+idea\b|\bwould\s+(?:really\s+)?help\b"
    r"|\b(?:updating|renovating|refurbishing|replacing|fixing|repainting|repairing|cleaning|improving)\b"
    r"[^.!?]{0,60}?\b(?:is|are|would\s+be)\s+(?:crucial|essential|necessary|needed|important|a\s+must|overdue)\b",
    re.I,
)
# A suggestion about what follows: "it would be nice to have a kettle", "just get a 160cm bed".
_SUGGEST_AFTER = re.compile(
    r"\b(?:would|could|might)\s+(?:have\s+)?(?:been|be)\s+(?:so\s+|really\s+|very\s+)?"
    r"(?:nice|good|great|helpful|useful|appreciated|ideal|perfect|lovely)\b"
    r"|\b(?:please|kindly)\s+(?:fix|add|provide|change|replace|clean|improve|consider|install|put|get|buy|update"
    r"|repair|offer)\b"
    r"|\b(?:i|we)\s+(?:would\s+)?(?:suggest|recommend)\s+(?:that\s+|to\s+)?(?:you\s+|the\s+(?:hotel|owners?"
    r"|hosts?|management|staff)\s+)?(?:fix|add|provide|change|replace|clean|improve|install|invest|consider"
    r"|get|put|update|upgrade|renovate|buy|offer)\b"
    r"|\b(?:could|can)\s+(?:do\s+with|use)\s+(?:a|an|some|more|better|new)\b"
    r"|\bthere\s+should\s+be\b"
    r"|\bshould\s+(?:provide|offer|add|include|put|install|fix|clean|replace|consider|invest\s+in|buy|get)\b"
    r"|\bshould\s+have\s+(?:a|an|some|more|better|proper)\b"
    r"|\b(?:i|we)\s+wish\s+(?:there|they|the|it|we|you)\b|\bif\s+only\b"
    r"|^(?:just\s+|please\s+)?(?:put|add|get|buy|provide|install|fix|replace)\s+(?:some|a|an|more|new|better)\b"
    r"|\bjust\s+(?:get|buy|put|add|install|fix|replace)\s+(?:a|an|some|more|new|better)\b"
    r"|\bwould\s+have\s+been\s+to\s+have\b|\bto\s+make\s+(?:the|our|your|a)\s+stay\s+(?:better|more\b)",
    re.I,
)
# Advice to other guests is not a complaint.
_RECOMMEND = re.compile(
    r"\byou\s+(?:should|must)\s+(?:definitely\s+|really\s+|absolutely\s+|totally\s+)?(?:stay|visit|book|go"
    r"|come|try|see|choose|pick|eat|have)\b"
    r"|\b(?:would|will|highly|definitely|strongly|absolutely|totally|fully|thoroughly|can)\s+recommend\b"
    r"(?!\s+(?:that|to)\s+(?:you|the|they)\b)"
    r"|\bmust[-\s](?:stay|visit|see|try|do)\b"
    r"|\bwould\s+(?:definitely\s+|happily\s+|gladly\s+|certainly\s+)?(?:stay|come|return|book|go)\b",
    re.I,
)
# A problem the guest reported, and what came of it.
_INTERVENE = re.compile(
    r"\b(?:report(?:ed|ing)?|complain(?:ed|ing)?|told\s+(?:the|our|him|her|them|reception|staff)"
    r"|asked\s+(?:the|him|her|them|for|reception|staff|about|if|whether|to)|inform(?:ed)?|call(?:ed)?\s+(?:the|them|him|her"
    r"|reception)|contact(?:ed)?|messag(?:ed|ing)|wrote\s+to|came\s+(?:to|with|and|up|over|quickly"
    r"|immediately|by)|sent|tried|attempted|changed|replaced|moved\s+us|gave\s+us|brought|fixed|repaired"
    r"|respond(?:ed)?|react(?:ed)?)\b",
    re.I,
)
_UNRESOLVED = re.compile(
    r"\b(?:did\s*n['’]?t|did\s+not|does\s*n['’]?t|does\s+not|never|was\s*n['’]?t|was\s+not|were\s*n['’]?t"
    r"|were\s+not|could\s*n['’]?t|could\s+not|would\s*n['’]?t|would\s+not|wo\s*n['’]?t|will\s+not)\s+"
    r"(?:\w+\s+){0,2}?(?:help(?:ed)?|work(?:ed)?|fix(?:ed)?|solved?|resolved?|improved?|changed?"
    r"|make\s+(?:any\s+|a\s+|much\s+|the\s+)?(?:real\s+|big\s+|much\s+|significant\s+)?difference"
    r"|go\s+away|disappear(?:ed)?|get\s+better|show\s+up|turn\s+up)\b"
    r"|\b(?:made|makes|making)\s+(?:no|little)\s+(?:real\s+|big\s+)?difference\b"
    r"|\bno\s+(?:real\s+|big\s+)?(?:difference|improvement|change|solution)\b"
    r"|\bnothing\s+(?:was|could\s+be|has\s+been|got|changed|happened|improved|helped)\b"
    r"|\bnot\s+(?:fixed|resolved|solved|sorted|repaired|replaced)\b"
    r"|\bstill\s+(?:smell\w*|stank|stinks?|broken|dirty|noisy|loud|cold|hot|warm|leak\w*|not\s+(?:working"
    r"|fixed|clean|cold|warm|hot)|did\s*n['’]?t|had\s+no|no\s|there|the\s+same|a\s+problem|an\s+issue|waiting)"
    r"|\b(?:problem|issue|smell|noise|leak|odou?r)s?\s+(?:persisted|remained|continued|came\s+back|returned)\b"
    r"|\bno\s*(?:one|body)\s+(?:answered|replied|responded|came|fixed|helped|showed\s+up|called\s+back)\b",
    re.I,
)
_RESOLVED = re.compile(
    r"\b(?:fixed|solved|resolved|sorted|repaired|replaced|changed|took\s+care\s+of|dealt\s+with|handled)\b"
    r"[^.!?]{0,40}?\b(?:quickly|immediately|promptly|right\s+away|straight\s+away|at\s+once|in\s+no\s+time"
    r"|within\s+(?:minutes|an\s+hour|the\s+hour|\d+\s+minutes)|responsibly|efficiently|professionally)\b"
    r"|\b(?:quickly|immediately|promptly|swiftly|efficiently)\s+(?:fixed|solved|resolved|sorted|repaired"
    r"|replaced|changed|took\s+care|dealt|handled|moved\s+us)\b"
    r"|\btook\s+care\s+of\s+(?:it|everything|the\s+(?:problem|issue|situation)|us)\b"
    r"|\b(?:problem|issue)\s+(?:was\s+)?(?:fixed|solved|resolved|sorted)\b",
    re.I,
)

# What a general-purpose sentiment model reads the wrong way round: an air
# conditioner that is "not cold", a shower that is "not hot", a host "saving on
# hangers". Each pattern makes every word of its topics in the clause a complaint.
_EXPECTED = [
    (("room.climate",), _phrase(r"(?:not|n['’]t|never|barely|hardly)\s+(?!too\b)(?:\w+\s+){0,2}?(?:cold|cool\w*)"
                                r"|(?:warm|hot)\s+air|no\s+cool\w*")),
    (("room.bathroom",), _phrase(r"(?:not|n['’]t|never|barely|hardly)\s+(?!too\b)(?:\w+\s+){0,2}?(?:hot|warm)"
                                 r"|luke\s?warm|(?:only|just)\s+cold\s+(?:water|showers?)"
                                 r"|(?:water|showers?)\s+(?:was|were|is|are|always)\s+(?:\w+\s+)?cold")),
    (TOPICS, _phrase(r"sav(?:e|es|ed|ing)\s+on|skimp\w*|cut(?:s|ting)?\s+(?:corners|costs)|stingy|cheapskate")),
]
# Pests are a complaint unless the guest says there were none.
_NO_PESTS = _phrase(r"(?:no|without|never\s+(?:saw|seen|had|found)|not\s+(?:a|any|one|single))\s+(?:\w+\s+){0,2}?"
                    r"(?:bed\s?bugs?|bugs?|insects?|cockroach\w*|roach(?:es)?|ants|mosquito(?:e?s)?|mice|mouse|rats?"
                    r"|flies|fleas?|spiders?|pests?|bites?)")
# "No kettle", "without lift", "no eggs or freshly prepared food": the missing
# thing is a complaint. Only for things a guest expects to find, and not after
# "no problem with", "no noise" or "no extra charge".
_ABSENT = re.compile(
    r"\b(?:no|without|lack\s+of|lacking|missing|not\s+a\s+single|not\s+any|nor)\s+"
    r"(?!(?:problems?|issues?|complaints?|trouble|doubt|worries|need|wait\w*|charges?|extra|hidden|noise|smell\w*"
    r"|dirt\w*|bugs?|stains?|hairs?|other|more|longer)\b)",
    re.I,
)
_ABSENCE_TOPICS = ("room.equipment", "room.storage", "room.climate", "room.bathroom", "room.view", "room.bed",
                   "cleanliness.linen", "food.", "facilities.", "staff.")
_ENGLISH = re.compile(
    r"\b(?:the|and|was|were|is|are|we|our|it|to|of|in|for|with|but|not|very|room|hotel|staff)\b", re.I
)


class Term(NamedTuple):
    """A word of a clause that names a topic."""

    text: str
    start: int
    topic: str
    opinion: bool = False  # names the topic by an opinion about it ("spacious")
    typo: bool = False

    @property
    def aspect(self) -> str:
        return self.topic.split(".", 1)[0]


def normalize(text: str) -> str:
    """Decode HTML entities and collapse whitespace, keeping line breaks: quotes are taken from this text."""
    lines = (" ".join(line.split()) for line in html.unescape(str(text or "")).splitlines())
    return "\n".join(line for line in lines if line)


def looks_english(text: str) -> bool:
    """Cheap guard: the lexicon and the model only understand English reviews."""
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) < 0.9:
        return False
    words = len(text.split())
    return words < 4 or len(_ENGLISH.findall(text)) / words >= 0.1


def _raw_terms(clause: str) -> list[Term]:
    """Every lexicon match; where two overlap the longer wins ("sofa bed" over "sofa")."""
    found = []
    for order, (pattern, topic, priority, opinion) in enumerate(_MATCHERS):
        for match in pattern.finditer(clause):
            found.append((match.start(), match.end(), priority, order, topic, opinion, match.group()))
    found.sort(key=lambda item: (item[0] - item[1], item[2], item[3], item[0]))
    taken: list[tuple[int, int]] = []
    terms = []
    for start, end, _, _, topic, opinion, text in found:
        if any(start < other_end and other_start < end for other_start, other_end in taken):
            continue
        taken.append((start, end))
        if topic:
            terms.append(Term(text, start, topic, opinion))
    return sorted(terms, key=lambda term: term.start)


def _gap(clause: str, a: Term, b: Term) -> int:
    """Words strictly between two terms."""
    first, second = sorted((a, b), key=lambda term: term.start)
    return len(clause[first.start + len(first.text):second.start].split())


def _nearest(term: Term, candidates: list[Term]) -> Term:
    return min(candidates, key=lambda other: abs(other.start - term.start))


def _apply_rules(clause: str, terms: list[Term]) -> list[Term]:
    """Keep each word only for the topic it is about in this clause."""
    nouns = [t for t in terms if not t.opinion]
    things = [t for t in nouns if t.topic.startswith(("room.", "food.", "facilities.", "cleanliness.linen"))]
    room_nouns = [t for t in nouns if t.aspect == "room"]
    specific_room = [t for t in room_nouns if t.topic != "room.general"]
    person = any(t.topic == "staff.people" for t in nouns) or _PERSON.search(clause)
    location_noun = any(t.aspect == "location" or _VENUE.fullmatch(t.text) for t in nouns)
    judged_clean = [t for t in terms if t.topic in ("cleanliness.odour", "cleanliness.pests")
                    or (t.topic == "cleanliness.general" and t.opinion)]
    noise = any(t.aspect == "noise" for t in terms)
    location = any(t.aspect == "location" for t in terms)
    fault = _FAULT.search(clause)
    other_value = any(t.aspect == "value" and not _CHEAP.fullmatch(t.text) for t in terms)
    kept = []
    for term in terms:
        if term.opinion:
            if term.topic == "staff.people" and not person and things:
                continue  # "the apartment was … very welcoming"
            if term.topic == "location.general" and not location_noun and any(
                    t.topic.startswith(("room.", "facilities.", "staff.checkin"))
                    or (t.aspect == "food" and not _VENUE.fullmatch(t.text)) for t in nouns):
                continue  # "convenient check-in"
            if term.topic in ("room.size", "room.comfort"):
                if not room_nouns and any(t.topic.startswith(("food.", "facilities.", "cleanliness.linen"))
                                          for t in nouns):
                    continue  # "a small pool"
                if specific_room:
                    term = term._replace(topic=_nearest(term, specific_room).topic)  # "the bed was too small"
            elif term.topic == "cleanliness.general":
                linen = [t for t in nouns if t.topic == "cleanliness.linen" and _gap(clause, t, term) <= 3]
                if linen:
                    term = term._replace(topic="cleanliness.linen")  # "hair in my towel"
            elif term.topic == "food.dining":
                food = [t for t in nouns if t.aspect == "food"]
                if food:
                    term = term._replace(topic=_nearest(term, food).topic)  # "delicious breakfast"
            elif term.topic == "room.equipment":
                worn = [t for t in nouns if t.topic.startswith(("room.", "facilities.", "location."))
                        and t.topic != "room.general"]
                if worn:
                    term = term._replace(topic=_nearest(term, worn).topic)  # "the bathroom is outdated"
            elif _CHEAP.fullmatch(term.text) and not other_value and (_CHEAP_QUALITY.search(clause) or things):
                continue  # "very poor and cheap", "cheap hangers": quality, not price
            elif _PRICEY.fullmatch(term.text) and not room_nouns and any(
                    t.topic.startswith(("food.", "facilities.")) for t in nouns):
                term = term._replace(topic="value.charges")  # "parking was expensive"
        else:
            if (_PLACEABLE.fullmatch(term.text) and _PLACE_BEFORE.search(clause[:term.start])
                    and any(not _PLACEABLE.fullmatch(t.text) for t in terms)):
                continue  # "got the room at 3rd floor without lift"
            if term.aspect == "room" and judged_clean and not fault and (
                    term.topic == "room.general" or min(_gap(clause, term, t) for t in judged_clean) <= 3):
                continue  # "the bathroom was dirty" is about cleaning, not the bathroom
            if term.topic == "facilities.kitchen" and judged_clean and not fault:
                continue
            if noise and (term.topic == "room.general"
                          or (term.aspect == "room" and _NOISE_SOURCE.fullmatch(term.text))):
                continue  # "moving furniture upstairs" is noise
            if term.aspect == "food" and _VENUE.fullmatch(term.text) and (location or noise):
                continue  # "loads of restaurants nearby", "noise from the bars"
        kept.append(term)
    return kept


def clause_terms(clause: str) -> list[Term]:
    """The words of a clause that name a topic, each under the topic it is about here."""
    terms = _raw_terms(clause)
    if _BABY.search(clause):
        terms += [Term(m.group(), m.start(), "facilities.family", typo=True) for m in _COD.finditer(clause)]
    kept: list[Term] = []
    for term in _apply_rules(clause, sorted(terms, key=lambda t: t.start)):
        if not any(t.topic == term.topic and t.text.lower() == term.text.lower() for t in kept):
            kept.append(term)
    return kept


def clause_topics(clause: str) -> list[tuple[str, list[str]]]:
    """(topic, the words that named it) for a clause, in TOPICS order."""
    found: dict[str, list[str]] = {}
    for term in clause_terms(clause):
        found.setdefault(term.topic, []).append(term.text)
    return [(topic, found[topic]) for topic in TOPICS if topic in found]


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


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for start, end in _pieces(text, 0, len(text), _SENTENCE):
        start, end = _strip(text, start, end)
        if start < end:
            spans.append((start, end))
    return spans


def clause_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the clauses of a normalised text.

    A fragment of fewer than two words ("closed.", "Horrible.") is merged into
    the clause before it rather than dropped, and so is a clause after "but" or
    "and" that names nothing of its own ("… but all not cold"), so every opinion
    stays attached to the thing it is about.
    """
    spans: list[tuple[int, int]] = []
    for s_start, s_end in _pieces(text, 0, len(text), _SENTENCE):
        first = len(spans)
        for c_start, c_end in _pieces(text, s_start, s_end, _CONTRAST):
            for start, end in _pieces(text, c_start, c_end, _AND_CLAUSE):
                start, end = _strip(text, start, end)
                if start == end:
                    continue
                piece = text[start:end]
                short = len(piece.split()) < 2
                if (spans and short) or (len(spans) > first and not any(not t.opinion for t in _raw_terms(piece))):
                    spans[-1] = (spans[-1][0], end)
                else:
                    spans.append((start, end))
    if len(spans) > 1 and len(text[slice(*spans[0])].split()) < 2:
        spans[1] = (spans[0][0], spans[1][1])
        del spans[0]
    return spans


def split_clauses(text: str) -> list[str]:
    text = normalize(text)
    return [text[start:end] for start, end in clause_spans(text)]


def suggestion_start(clause: str) -> int | None:
    """Where the words a suggestion is about begin, or None if the clause suggests nothing."""
    if _RECOMMEND.search(clause):
        return None
    if _SUGGEST_WHOLE.search(clause):
        return 0
    match = _SUGGEST_AFTER.search(clause)
    return match.start() if match else None


def _mention(text: str, start: int, end: int, topic: str, sentiment: str, term: str,
             confidence: float, flags) -> dict:
    return {"aspect": topic.split(".", 1)[0], "topic": topic, "sentiment": sentiment, "term": term,
            "quote": text[start:end], "start": start, "end": end, "confidence": round(confidence, 3),
            "flags": sorted(flags)}


def resolution_mentions(text: str) -> list[dict]:
    """Problems the guest reported: solved (positive) or not solved (negative, "unresolved")."""
    sentences = sentence_spans(text)
    found = []
    for k, (start, end) in enumerate(sentences):
        solved = _RESOLVED.search(text, start, end)
        if solved:
            found.append(_mention(text, start, end, "staff.resolution", "positive", solved.group(), 1.0, ()))
        failed = _UNRESOLVED.search(text, start, end)
        if not failed:
            continue
        if _INTERVENE.search(text, start, failed.start()):
            span = (start, end)
        elif k and _INTERVENE.search(text, *sentences[k - 1]):
            span = (sentences[k - 1][0], end)
        else:
            continue
        found.append(_mention(text, *span, "staff.resolution", "negative", failed.group(), 1.0, ("unresolved",)))
    return found


def _absence_spans(clause: str) -> list[tuple[int, int]]:
    """What "no", "without", "lack of" … cover: up to the next punctuation or conjunction, six words at most."""
    spans = []
    for match in _ABSENT.finditer(clause):
        rest = clause[match.end():]
        stop = re.search(r"[.,;:!?]|\s(?:and|but|so|because|which|that|as|if)\s", rest)
        words = list(re.finditer(r"\S+", rest[:stop.start() if stop else len(rest)]))[:6]
        if words:
            spans.append((match.end(), match.end() + words[-1].end()))
    return spans


def _forced_negative(clause: str, term: Term) -> bool:
    """A complaint the model reads the wrong way round: see _EXPECTED, _NO_PESTS and _ABSENT."""
    if term.topic == "cleanliness.pests" and not _NO_PESTS.search(clause):
        return True
    if any(term.topic.startswith(topics) and pattern.search(clause) for topics, pattern in _EXPECTED):
        return True
    return (not term.opinion and term.topic.startswith(_ABSENCE_TOPICS)
            and any(start <= term.start < end for start, end in _absence_spans(clause)))


def analyze(reviews: Sequence[str], scorer: Scorer) -> list[list[dict]]:
    """One list of mentions per review, in reading order.

    A mention is {aspect, topic, sentiment, term, quote, start, end, confidence,
    flags}, and `quote` is `normalize(review)[start:end]`. All (clause, word)
    pairs of all reviews go to the scorer in one call, so the model batches them.
    Flags: "suggestion" (a complaint worded as a suggestion), "unresolved",
    "check" (the model was unsure) and "typo" ("cod" read as "cot").

    The model reads each topic through its nouns ("the bed was too small" is
    scored on "bed"), and through the opinion word only when there is no noun.
    """
    texts = [normalize(review) for review in reviews]
    jobs = []
    for index, text in enumerate(texts):
        for start, end in clause_spans(text):
            clause = text[start:end]
            cue = suggestion_start(clause)
            # "He came with tools, but it made no difference": the outcome is judged by
            # resolution_mentions, and the effort alone says nothing about the reply.
            failed = _UNRESOLVED.search(clause)
            outcome_only = failed is not None and not re.match(r"no\s*(?:one|body)", failed.group(), re.I)
            by_topic: dict[str, list[Term]] = {}
            for term in clause_terms(clause):
                if not (outcome_only and term.topic == "staff.response"):
                    by_topic.setdefault(term.topic, []).append(term)
            for terms in by_topic.values():
                judged = any(t.opinion for t in terms)
                for term in ([t for t in terms if not t.opinion] or terms)[:MAX_TERMS_PER_TOPIC]:
                    jobs.append((index, start, end, term, judged, cue is not None and term.start >= cue,
                                 _forced_negative(clause, term)))
    scores = scorer([(texts[job[0]][job[1]:job[2]], job[3].text) for job in jobs]) if jobs else []

    found: dict[tuple, dict] = {}
    for (index, start, end, term, judged, suggested, forced), (label, probability) in zip(jobs, scores, strict=True):
        if suggested or forced:
            label = "negative"
        elif label not in ("positive", "negative"):
            continue
        key = (index, start, end, term.topic, label)
        entry = found.setdefault(key, {"term": term.text, "confidence": 0.0, "suggested": False, "ruled": False,
                                       "typo": False, "nouns_only": True})
        entry["confidence"] = max(entry["confidence"], probability)
        entry["suggested"] |= suggested
        entry["ruled"] |= forced
        entry["typo"] |= term.typo
        entry["nouns_only"] &= not judged

    for key, entry in list(found.items()):
        # A specific topic covers the general one it came with: "the room was spacious"
        # is about size, "the host asked us to check in ourselves" about check-in.
        index, start, end, topic, label = key
        siblings = {k[3] for k in found if k[:3] == key[:3] and k[4] == label and k != key}
        if topic == "room.general" and any(t.startswith("room.") for t in siblings):
            del found[key]
        elif topic == "staff.people" and entry["nouns_only"] and siblings & {"staff.checkin", "staff.response"}:
            del found[key]

    mentions: list[list[dict]] = [resolution_mentions(text) for text in texts]
    for (index, start, end, topic, label), entry in found.items():
        flags = set()
        if entry["suggested"]:
            flags.add("suggestion")
        elif not entry["ruled"] and entry["confidence"] < CHECK_BELOW:
            flags.add("check")
        if entry["typo"]:
            flags |= {"typo", "check"}
        # A label a rule decided is as sure as the rule; only the model's own answers carry its probability.
        confidence = 1.0 if entry["ruled"] or entry["suggested"] else entry["confidence"]
        mentions[index].append(_mention(texts[index], start, end, topic, label, entry["term"], confidence, flags))
    return [sorted(items, key=lambda m: (m["start"], TOPICS.index(m["topic"]), m["sentiment"]))
            for items in mentions]


class AbsaModel:
    """DeBERTa-v3 aspect sentiment: the most probable of negative / neutral / positive.

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
                 progress: Callable[[float], None] | None = None) -> list[Score]:
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]))
        scores: list[Score] = [(None, 0.0)] * len(pairs)
        for start in range(0, len(order), self.batch_size):
            batch = order[start:start + self.batch_size]
            encoded = self.tokenizer(
                [pairs[i][0] for i in batch], [pairs[i][1] for i in batch],
                padding=True, truncation=True, max_length=160, return_tensors="pt",
            )
            with self._torch.inference_mode():
                best = self.model(**encoded).logits.softmax(-1).max(-1)
            for i, probability, label_index in zip(batch, best.values.tolist(), best.indices.tolist(),
                                                   strict=True):
                scores[i] = (self.names[label_index], probability)
            if progress:
                progress(min(1.0, (start + len(batch)) / len(order)))
        return scores
