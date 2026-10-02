"""The questions Jev is asked about a whole review, with their own version.

The benchmark's questions (scripts/benchmark_jev_topics.py) tell the model that the text is what a guest
wrote when Booking.com asked what they did not like. That is wrong for a whole review, which can mix
praise and complaints, so these are separate questions with a separate version and hash.

The rules for each of the five topics are the labelling guideline's, in the benchmark's words. A test
checks that they have not drifted. The wording around them is new, and nothing here has been measured on
whole or mixed reviews: the pilot and its confirmation used the negative field alone.
"""

from __future__ import annotations

import hashlib
import json

from reviewnlp.triage.schemas import OTHER, TOPICS

QUESTIONS_VERSION = "full-review-v1"
OTHER_QUESTION = "other_complaint"
LABELS = ("1", "0", "unsure")

CONTEXT = ("The text is a whole hotel guest review. It may praise some things and complain about others, "
           "and it may be long.")
RULES = ("A complaint is a problem, a lack or a negative judgement; a suggestion or wish that implies a lack; "
         "a problem anywhere in the hotel, not only in the room; or a mild complaint. Not complaints: praise, a "
         "neutral mention, and saying nothing was wrong. A complaint about this topic counts even when the rest "
         "of the review is positive, and praise for other things does not cancel it. A topic that is only the "
         "setting of another complaint is not a complaint about it: 'kept the windows closed and the air con on "
         "because of the street noise' is a noise complaint, so air conditioning is 0. Each topic is judged on "
         "its own.")
UNSURE = ("Only when the text supports both readings, for example 'bathroom disgusting' with no reason given, "
          "or 'the fan' with no hint of which fan. Not for mild complaints or rare topics: a mild complaint "
          "is still 1.")

# (the question, what is a 1, what is a 0). The 1 and 0 texts are the benchmark's, from the guideline.
TOPIC_RULES = {
    "bathroom": (
        "Does the guest complain about the bathroom or shower?",
        "The bathroom, the shower, the bath or tub, the toilet, the sink and taps, or how they work or are "
        "built: size, layout, privacy, doors and locks, fittings, leaks, drainage, water pressure, hot water; "
        "or a missing bath or shower.",
        "No such complaint. These do not count: dirt, hair, stains or mould in the bathroom, which are "
        "cleanliness (mark bathroom 1 only if something else about the bathroom is also criticised); towels, "
        "bathrobes and bath mats, which are linen; a hair dryer, toiletries and slippers, which are amenities; "
        "hearing other rooms' toilets, which is noise.",
    ),
    "cleanliness": (
        "Does the guest complain about cleanliness?",
        "Dirt, dust, stains, hair, mould or rubbish; a room or area that was not cleaned or was cleaned badly; "
        "cleaners who did not clean.",
        "No such complaint. These do not count: praise ('clean', 'spotless'); housekeeping knocking, entering "
        "or waking the guest, which is about staff; smells alone, which is another topic; worn or shabby "
        "furniture with no dirt described; missing supplies (milk, toilet paper) with no cleaning problem; "
        "dirt outside the hotel, such as dirty streets.",
    ),
    "air_conditioning": (
        "Does the guest complain about air conditioning, heating or ventilation?",
        "Air conditioning, heating, fans or ventilation; the room's temperature or its control; stuffy air; "
        "noise made by the air conditioner itself.",
        "No such complaint. These do not count: the water temperature in the shower, which is bathroom; the "
        "weather outside; the temperature of a pool or other facilities; a view of air-conditioning units; "
        "outside noise that made the guest close the windows.",
    ),
    "pests": (
        "Does the guest complain about pests?",
        "Insects, bed bugs or their bites, cockroaches, ants, flies, mosquitoes, spiders, mice or rats, "
        "anywhere in the hotel.",
        "No such complaint. These do not count: 'bite' meaning food ('a quick bite', 'bite-size'); a computer "
        "mouse; a 'flea market'; a place named after an animal; a figure of speech ('we felt like rats').",
    ),
    "responsiveness": (
        "Does the guest complain about the hotel's responsiveness?",
        "The hotel or its staff did not answer, reply, call back or act on a request or complaint, or did so "
        "very late: unanswered phone calls or emails, ignored requests, promises to follow up that were not "
        "kept, a reported problem left for days, a change the guest was not told about.",
        "No such complaint. These do not count: staff who did answer but were rude, unfriendly or unhelpful, "
        "which is about attitude; a language barrier; an answer the guest did not like ('the answer was no'); "
        "a device that is 'unresponsive'; praise ('they responded quickly'); Booking.com, not the hotel, "
        "failing to pass something on.",
    ),
}

OTHER_RULES = (
    "Does the guest complain about anything other than the bathroom, cleanliness, air conditioning or heating, "
    "pests and responsiveness?",
    "A problem, a lack or a negative judgement about anything else: for example noise, breakfast and food, "
    "staff attitude, price, location, parking, wifi, the bed, the size of the room, the view or the facilities.",
    "No other complaint: only praise, neutral mentions, or complaints on the five topics already asked about.",
)


def _question(ask: str, complaint: str, other: str) -> dict:
    return {"type": "choice", "instructions": f"{ask} {CONTEXT} {RULES}",
            "criteria": {"1": complaint, "0": other, "unsure": UNSURE}}


QUESTIONS = {**{topic: _question(*TOPIC_RULES[topic]) for topic in TOPICS},
             OTHER_QUESTION: _question(*OTHER_RULES)}
QUESTIONS_SHA256 = hashlib.sha256(
    json.dumps({"version": QUESTIONS_VERSION, "questions": QUESTIONS}, sort_keys=True).encode("utf-8")).hexdigest()
TOPIC_OF_QUESTION = {**{topic: topic for topic in TOPICS}, OTHER_QUESTION: OTHER}


def request_body(text: str, model: str) -> dict:
    """The System One request for one review: the review as the state, and every question."""
    return {"state": text, "model": model, "questions": QUESTIONS}
