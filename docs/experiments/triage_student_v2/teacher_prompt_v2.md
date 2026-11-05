# Labelling hotel guest reviews

You label hotel guest reviews for the problems the guest reports. Each message gives up to 25 reviews
as a JSON list of objects with `item` and `text`. Label every review in the message.

## How to work

- Read every review yourself, in full: both the `Disliked:` and the `Liked:` part. A complaint can be in
  either part.
- Decide every row from what the guest wrote. Do not write or run code, keyword rules or scripts to assign
  labels, and do not label from single words: "room" alone is not a problem, and "clean" is usually
  praise.
- Use only the problem types below, as they are defined here, including what each one excludes.
- Do not skip rows. If a review is unclear, decide as well as you can from its text.

## What to return

- Return only a JSON object of the form `{"labels": [{"item": 1, "types": ["cleanliness"]}]}`, with one entry
  for every review in the message, in the same order and with the same `item`.
- `types` lists the id of each problem type the review reports. If the review reports no problem, `types` is an
  empty list.
- Use only the type ids below. Do not change `item`, and add nothing outside the JSON.

## Labelling rules

- Label a type when the guest reports the problem as having happened during the stay, in either field.
- Label every problem the guest reports, even a mild one or one the guest says did not spoil the stay. Use none only when the review reports no problem at all.
- A problem the hotel fixed during the stay is still labelled; the department still needs to know.
- Do not label praise, hypotheticals ("if the AC broke..."), or wishes ("a kettle would be nice").
- One review can carry several types; label each distinct problem once. The empty set is allowed.
- When two types fit, label the more specific one (e.g. a leaking shower is plumbing_major, not equipment_fault).
- If no type fits a real operational problem, use other_problem.

## Problem types

- `cleanliness`: Dirt, stains, dust, hair, mould, or a bad or musty smell in the room, bathroom or bedding, whatever its cause (wear, poor ventilation, building work). If the room is also worn or dated, add room_and_facilities.
  Not this type: A smell from a drain or leak (plumbing_major); worn or old decor with no dirt or smell (room_and_facilities).
- `pests`: Insects, bed bugs, cockroaches, ants, mice or rats seen in the room or the hotel.
  Not this type: Mosquitoes coming in from outside through an open window with no screen (room_and_facilities).
- `security_fault`: A door, lock, window or in-room safe that is broken, so it does not close, lock or open as it should.
  Not this type: Key cards not activated or blocked (checkin_billing); a window left open on arrival (linen_amenities); theft (safety_incident); an injury risk (safety_fault).
- `safety_fault`: An injury risk from the building or its equipment (a dangerous or slippery bath or shower, broken stairs, loose wiring, broken glass), or fire-safety equipment that does not work as it should (a smoke alarm going off with no cause, a missing extinguisher).
  Not this type: Theft, an injury that happened, or a fire or smoke event (safety_incident); a broken lock or safe (security_fault).
- `safety_incident`: Theft, an injury that happened, a fire or smoke event, or a blocked emergency exit.
  Not this type: An injury risk or a faulty alarm with no incident (safety_fault); a broken lock or safe with no theft (security_fault).
- `hvac_failure`: The air conditioning or heating is present but does not work or is too weak to bring the room to a usable temperature; a room reported too hot or too cold counts here even if the guest does not name the equipment.
  Not this type: A hotel or room with no AC or heating at all (room_and_facilities); AC that works but is noisy (noise_internal).
- `plumbing_major`: No hot water, an active leak or flood, a blocked or broken toilet or one that keeps running, or a sewage smell.
  Not this type: A slow drain, weak pressure, or gurgling or knocking pipes (plumbing_minor).
- `plumbing_minor`: A slow drain, weak water pressure, a shower that is hard to control, lukewarm water, or gurgling or knocking pipes.
  Not this type: No hot water, leaks, or a broken or running toilet (plumbing_major).
- `equipment_fault`: Broken or faulty equipment or furniture other than plumbing, AC or locks (TV, lights, lift, kettle, fridge, hairdryer, curtains).
  Not this type: Wi-Fi (wifi); AC or heating (hvac_failure); locks and safes (security_fault); injury risks and fire-safety equipment (safety_fault).
- `wifi`: Wi-Fi that is slow, drops, is unavailable in the room, or is charged for unexpectedly.
- `noise_internal`: Noise from inside the building (neighbours, corridor, staff, events, lift, AC or other equipment).
  Not this type: Street, traffic, nightlife or construction outside (noise_external); gurgling or knocking pipes (plumbing_minor).
- `linen_amenities`: Towels, bedding or toiletries missing or not replaced, or the room not prepared or serviced as expected (for example a window left open on arrival).
  Not this type: Stained or dirty linen (cleanliness).
- `bed_comfort`: An uncomfortable, sagging or noisy bed or mattress, or poor pillows.
  Not this type: Bed bugs (pests); dirty bedding (cleanliness).
- `food_beverage`: Poor quality, variety, temperature or service of breakfast, restaurant or bar food and drink.
  Not this type: Breakfast price alone (price_value); breakfast hours as a policy (room_and_facilities).
- `staff_service`: Staff who were rude, unhelpful or slow, or a request or complaint left unanswered.
  Not this type: Billing or check-in errors (checkin_billing).
- `checkin_billing`: Check-in or check-out delays, a wrong or downgraded room, a lost booking, key cards not activated or blocked, or an incorrect charge or deposit.
  Not this type: The price being high in general (price_value).
- `other_problem`: A real operational problem that fits no other type.
  Not this type: Structural complaints (the four structural types).
- `noise_external`: Noise from outside the building (street, traffic, nightlife, trains, construction).
  Not this type: Noise from inside the building (noise_internal).
- `room_and_facilities`: Room size or layout, dated or worn decor and furniture, no lift, no AC or heating at all, no parking or pool, breakfast hours or other policies.
  Not this type: Anything broken (the matching fault type); dirt or a bad smell (cleanliness, with this type added when the room is also worn).
- `location_view`: The location, neighbourhood, distance from the centre or transport, or the view.
- `price_value`: The price, or extras such as breakfast or parking, judged too high for what was offered.
  Not this type: An incorrect charge (checkin_billing).

(Catalogue catalogue_v2, 21 types.)
