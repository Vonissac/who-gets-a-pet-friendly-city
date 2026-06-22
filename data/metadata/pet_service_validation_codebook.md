# Pet-Service Validation Codebook

Locked title: Pet-Service Ecologies Outpace Urban Readiness in Shenzhen: Rule-Liminal Venues and the Uneven Emergence of Companion-Animal Urban Capability

## Labels

- `core_medical`: veterinary hospital, clinic, animal diagnosis/treatment.
- `core_retail`: pet shop, pet supplies, pet food, pet medicine.
- `core_grooming`: pet grooming, washing, care.
- `core_boarding`: pet boarding, hotel, daycare, hosting.
- `core_training`: dog/pet training.
- `extended_leisure`: pet cafe, pet park, pet camp, animal-themed co-presence venue.
- `extended_aquarium_flower_bird_fish`: aquarium, flower-bird-fish market; sensitivity only unless clearly companion-animal service.
- `false_positive`: non-pet use of cat/dog/animal terms, brands, food names, entertainment titles, unrelated organizations.
- `uncertain`: insufficient evidence from available fields.

## Rules

Use name, type, category, address, source category, and match snippet. Do not infer from phone or private data. If a record is a company with pet-supply wording but no consumer-facing venue evidence, mark core only if the project later includes producer/supply-chain services; otherwise mark uncertain or exclude from public-facing service layer.

## Required Review Fields

- `review_label`
- `review_reason`
- `is_core_service`
- `is_extended_service`
- `is_false_positive`
- `reviewer`
- `review_date`
