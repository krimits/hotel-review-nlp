# Labelling hotel guest reviews

You label hotel guest reviews for the problems the guest reports. You will receive the reviews as CSV
files of 50 rows, one batch at a time. Label each batch and return it before the next one.

## How to work

- Read every review yourself, in full: both the `Disliked:` and the `Liked:` part. A complaint can be in
  either part.
- Decide every row from what the guest wrote. Do not write or run code, keyword rules or scripts to assign
  labels, and do not label from single words: "room" alone is not a problem, and "clean" is usually
  praise.
- Use only the problem types below, as they are defined here, including what each one excludes.
- Do not skip rows. If a review is unclear, decide as well as you can from its text.

## What to put in the file

- Keep every column and every row, in the same order. Do not change `item` or `text`.
- Put `1` in the column of each problem type the review reports. Leave the other type columns empty.
- If the review reports no problem, put `1` in `none` and leave every type column empty.
- Put `1` in `done` on every row. Leave `note` empty.
- Return the completed batch as a downloadable UTF-8 CSV file named like the batch with `_filled`, for
  example `batch_01_filled.csv`.

## Labelling rules

- Label a type when the guest reports the problem as having happened during the stay, in either field.
- A problem the hotel fixed during the stay is still labelled; the department still needs to know.
- Do not label praise, hypotheticals ("if the AC broke..."), or wishes ("a kettle would be nice").
- One review can carry several types; label each distinct problem once. The empty set is allowed.
- When two types fit, label the more specific one (e.g. a leaking shower is plumbing_major, not equipment_fault).
- If no type fits a real operational problem, use other_problem.

## Problem types

- `cleanliness`: Dirt, stains, dust, hair, mould, or a bad smell in the room, bathroom or bedding.
  Not this type: A smell from a drain or leak (plumbing_major); worn or old decor with no dirt (room_and_facilities).
- `pests`: Insects, bed bugs, cockroaches, ants, mice or rats seen in the room or the hotel.
  Not this type: Mosquitoes coming in from outside through an open window with no screen (room_and_facilities).
- `security_fault`: A door, lock, window or in-room safe that does not close, lock or open as it should.
  Not this type: Theft, injury or fire risk (safety_hazard).
- `safety_hazard`: Theft, an injury or an injury risk (loose wiring, broken glass, slippery or broken stairs), or a fire-safety problem (blocked exit, smoke, missing alarm).
  Not this type: A lock or safe that does not work with no theft (security_fault).
- `hvac_failure`: The air conditioning or heating is present but does not work, or cannot bring the room to a usable temperature.
  Not this type: A room with no AC or heating at all (room_and_facilities); AC that works but is noisy (noise_internal).
- `plumbing_major`: No hot water, an active leak or flood, a blocked or broken toilet, or a sewage smell.
  Not this type: A slow drain or weak pressure (plumbing_minor).
- `plumbing_minor`: A slow drain, weak water pressure, a shower that is hard to control, or lukewarm water.
  Not this type: No hot water, leaks or a broken toilet (plumbing_major).
- `equipment_fault`: Broken or faulty equipment or furniture other than plumbing, AC or locks (TV, lights, lift, kettle, fridge, hairdryer, curtains).
  Not this type: Wi-Fi (wifi); AC or heating (hvac_failure); locks and safes (security_fault).
- `wifi`: Wi-Fi that is slow, drops, is unavailable in the room, or is charged for unexpectedly.
- `noise_internal`: Noise from inside the building (neighbours, corridor, staff, events, lift, pipes, AC or other equipment).
  Not this type: Street, traffic, nightlife or construction outside (noise_external).
- `linen_amenities`: Towels, bedding or toiletries missing or not replaced, or the room not serviced as expected.
  Not this type: Stained or dirty linen (cleanliness).
- `bed_comfort`: An uncomfortable, sagging or noisy bed or mattress, or poor pillows.
  Not this type: Bed bugs (pests); dirty bedding (cleanliness).
- `food_beverage`: Poor quality, variety, temperature or service of breakfast, restaurant or bar food and drink.
  Not this type: Breakfast price alone (price_value); breakfast hours as a policy (room_and_facilities).
- `staff_service`: Staff who were rude, unhelpful or slow, or a request or complaint left unanswered.
  Not this type: Billing or check-in errors (checkin_billing).
- `checkin_billing`: Check-in or check-out delays, a wrong or downgraded room, a lost booking, or an incorrect charge or deposit.
  Not this type: The price being high in general (price_value).
- `other_problem`: A real operational problem that fits no other type.
  Not this type: Structural complaints (the four structural types).
- `noise_external`: Noise from outside the building (street, traffic, nightlife, trains, construction).
  Not this type: Noise from inside the building (noise_internal).
- `room_and_facilities`: Room size or layout, dated decor, no lift, no AC, no parking or pool, breakfast hours or other policies.
  Not this type: Anything broken or dirty (the matching operational type).
- `location_view`: The location, neighbourhood, distance from the centre or transport, or the view.
- `price_value`: The price, or extras such as breakfast or parking, judged too high for what was offered.
  Not this type: An incorrect charge (checkin_billing).

(Catalogue catalogue_v1, 20 types.)
