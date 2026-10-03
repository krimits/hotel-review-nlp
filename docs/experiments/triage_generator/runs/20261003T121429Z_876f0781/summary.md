# Generator comparison

Synthetic cases; structural counts are not human usefulness.

| Candidate | Attempted | Parser accepted JSON | Whole JSON, unique keys | Reviews with accepted actions | Budget hits | Errors |
|---|---:|---:|---:|---:|---:|---:|
| C | 24 | 24 | 17 | 18 | 0 | 0 |
| E | 24 | 24 | 12 | 5 | 0 | 0 |

Human quality: pending. Fill human_review.csv using the protocol.
All cases are queried for generator diagnosis; production_routed preserves the actual routing.
Case latency excludes warmup; local GPU billing is unknown.
API cost: {"attempts": 24, "attempts_with_reported_cost": 24, "reported_cost_sum": 0.002526804, "complete_cost_available": true, "cost_unit": "provider usage.cost; no currency inferred", "local_gpu_charge": null}
Whole JSON requires one complete JSON object without duplicate keys; it is not semantic validation.
Reused upstream: no new Jev calls. The upstream API cost above belongs to the reference run.
Current API attempts: 0. Local GPU billing remains unknown.
