# Outpost defence validation

Validated on 4 October 2026 in headless Edge with WebGPU, on the available NVIDIA Blackwell adapter. No browser or GPU validation errors were reported by the successful runs. All new geometry, landing and walking, mission state and pirate AI are generated or simulated in CUDA. JavaScript supplies input, DOM dialogue, dispatch and persistence.

The desktop scenario registered a new pilot, disembarked, walked 7.6 metres to the commander, accepted the contract, returned to the parked craft and took off. The test aimed using flight controls and fired the actual mounted lasers against moving AI until all three first-wave pirates and all four reinforcements were destroyed. It then landed, collected exactly 750 credits and continued the saved game on foot. Separate deterministic fixtures verified base damage, failure, remaining-wave reconstruction and rejection of unsafe speed and height; no forced enemy deaths were used to clear the mission.

Additional checks covered crew collision, an unchanged parked navigation frame while walking, engineer repairs/rearming, physical flight away from the apron followed by landing and walking on the actual terrain, rejection of landing over water, dialogue Escape handling, and peaceful Free Roam preserving normal progress. The ship BVH oracle retained 36 checked rays, with maximum hit-distance error 0.00000394 metres against the brute-force reference.

| Ground-level outpost view | Median GPU time | p95 GPU time |
| --- | ---: | ---: |
| 2560 × 1440, PC quality | 23.68 ms | 23.94 ms |
| 3840 × 2160, PC quality | 44.32 ms | 45.44 ms |

These are 16 measured samples after warmup, using a fixed scene. They include scene compute/render dispatches but exclude the canvas texture copy, browser presentation and startup compilation. They are not a claim of 60 FPS at 4K or a worst-case combat benchmark.

Touch emulation used the actual mobile graphics profile with a Pixel 7 browser configuration. Joystick walking, drag look, touch conversation/acceptance, portrait dialogue bounds and landscape overflow checks passed. A separate input fixture held a first finger while a second activated Talk, verifying one command only and continued keyboard activation. The mobile-profile landscape render measured 3.97 ms median at 960 × 440 **on the desktop GPU**; this is not a Samsung A34 or iPhone performance result. Physical Android and iPhone testing remains to be done.

A dedicated GPU migration fixture tested both 512-wide mobile and 1024-wide desktop geology maps. Starting over an 11,000-metre-deep ocean basin at orbital altitude, the global fallback found land at 12.89 m and 12.48 m respectively, 394 km and 255 km away. The saved ship record and navigation pairs stayed unchanged. An already-stamped invalid underwater anchor was also repaired. Run `npm run test:outpost:migration` for this isolated check.

The art and mission scope is a first playable slice: two exterior crew conversations, one two-wave contract, a saved credit reward and ship servicing. Crew use simple procedural helmeted suits. Interiors, animated character rigs, spending credits, a broader campaign and on-foot weapons are not part of this stage.

Commands: `npm run build`, `npm run test:outpost`, `npm run test:outpost:mobile`, `npm run test:outpost:input`. Machine-readable results are in [outpost-validation.json](outpost-validation.json).
