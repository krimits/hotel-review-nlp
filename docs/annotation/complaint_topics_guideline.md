# Labelling guideline: five complaint topics

This guideline is for the two annotators of the
[pilot evaluation](pilot_protocol.md). It is fixed before the sample is drawn.

## The task

Each row is what a guest wrote when Booking.com asked what they did not like.
The source data has no punctuation, so sentences run together. Read the whole
text, then fill in every cell:

| Column | What to write |
|---|---|
| `bathroom`, `cleanliness`, `air_conditioning`, `pests`, `responsiveness` | `1` if the guest complains about this topic, `0` if not, `unsure` if you cannot decide after reading the text twice |
| `done` | `yes` when you have finished the row |
| `note` | anything worth telling us (optional) |

- **Fill every cell.** Never leave a topic cell blank. A blank is not read as 0:
  the whole sheet is sent back.
- **Work alone.** Do not discuss texts with the other annotator until you have
  both handed in your sheets.
- **Keep the key closed.** Do not open `key.csv`.

## What counts as a complaint

**Counts (1):**
- **A problem, a lack or a negative judgement.** "the shower was cold", "no air
  conditioning", "bathroom tiny".
- **A suggestion or wish that implies a lack.** "would like a bath tub", "air
  conditioning would be good".
- **A problem anywhere in the hotel, not only in the room.** "a mouse in the bar".
- **A mild complaint.** "shower a bit small".
- **A text with several topics.** Each topic gets its own 1.

**Does not count (0):**
- **Praise,** even in this field. "very clean", "the air con worked well".
- **Neutral mentions.** "we had a room with a bath".
- **Saying nothing was wrong.**
- **A topic that is only the setting of another complaint.** Judge what the
  guest complains about. "kept the windows closed and the air con on because of
  the street noise" is a noise complaint, so air conditioning is 0.

## The five topics

### `bathroom`: bathroom and shower

**1:**
- the bathroom, the shower, the bath or tub, the toilet, the sink and taps;
- how they work or are built: size, layout, privacy, doors and locks, fittings,
  leaks, drainage, water pressure, hot water;
- a missing bath or shower.

**0**, because these belong to other topics:
- dirt, hair, stains or mould in the bathroom are **cleanliness**. Mark bathroom
  1 only if something else about the bathroom is also criticised;
- towels, bathrobes and bath mats are linen;
- a hair dryer, toiletries and slippers are amenities;
- hearing other rooms' toilets is noise.

### `cleanliness`: cleanliness

**1:**
- dirt, dust, stains, hair, mould or rubbish;
- a room or area that was not cleaned or was cleaned badly;
- cleaners who did not clean.

**0:**
- praise ("clean", "spotless");
- housekeeping knocking, entering or waking the guest, which is about staff;
- smells alone, which is another topic;
- worn or shabby furniture with no dirt described;
- missing supplies (milk, toilet paper) with no cleaning problem;
- dirt outside the hotel, such as dirty streets.

### `air_conditioning`: air conditioning, heating and ventilation

**1:**
- air conditioning, heating, fans or ventilation;
- the room's temperature or its control;
- stuffy air;
- noise made by the air conditioner itself.

**0:**
- the water temperature in the shower, which is bathroom;
- the weather outside;
- the temperature of a pool or other facilities;
- a view of air-conditioning units;
- outside noise that made the guest close the windows.

### `pests`: pests

**1:** insects, bed bugs or their bites, cockroaches, ants, flies, mosquitoes,
spiders, mice or rats, anywhere in the hotel.

**0:**
- "bite" meaning food ("a quick bite", "bite-size");
- a computer mouse;
- a "flea market";
- a place named after an animal;
- a figure of speech ("we felt like rats").

### `responsiveness`: responsiveness

**1:** the hotel or its staff did not answer, reply, call back or act on a
request or complaint, or did so very late. Examples:
- unanswered phone calls or emails;
- ignored requests;
- promises to follow up that were not kept;
- a reported problem left for days;
- a change the guest was not told about.

**0:**
- staff who did answer but were rude, unfriendly or unhelpful, which is about
  attitude;
- a language barrier;
- an answer the guest did not like ("the answer was no");
- a device that is "unresponsive";
- praise ("they responded quickly");
- Booking.com, not the hotel, failing to pass something on.

## When to use `unsure`

- **Only after two readings,** and only when the text supports both readings.
  Examples:
  - "bathroom disgusting" with no reason given;
  - "the fan" with no hint of which fan.
- **Not for mild complaints or rare topics.** A mild complaint is still 1.
- **One topic at a time.** `unsure` applies to one topic. The other topics of
  the row still get `1` or `0`.

## Examples

These texts are made up, not taken from the sample.

| Text | bathroom | cleanliness | air_conditioning | pests | responsiveness |
|---|:-:|:-:|:-:|:-:|:-:|
| shower was lukewarm and the room too hot at night | 1 | 0 | 1 | 0 | 0 |
| bathroom dirty hair in the sink | 0 | 1 | 0 | 0 | 0 |
| nothing really very clean hotel | 0 | 0 | 0 | 0 | 0 |
| called reception three times about the heating nobody came | 0 | 0 | 1 | 0 | 1 |
| nowhere nearby for a quick bite | 0 | 0 | 0 | 0 | 0 |
| saw a mouse in the restaurant | 0 | 0 | 0 | 1 | 0 |
| receptionist was rude when we asked for a late checkout | 0 | 0 | 0 | 0 | 0 |
| would be nice to have air conditioning | 0 | 0 | 1 | 0 | 0 |
| no hair dryer and only one towel | 0 | 0 | 0 | 0 | 0 |
| housekeeping knocked at 8am | 0 | 0 | 0 | 0 | 0 |
| the toilet door does not lock and the walls are dusty | 1 | 1 | 0 | 0 | 0 |
| emailed twice about a refund and never got a reply | 0 | 0 | 0 | 0 | 1 |
