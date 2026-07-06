// Copyright AI-nimator.
//
// GENERATED-BY-COPY. Do not hand-edit the JSON payload below: it is a
// VERBATIM copy of the canonical B6 keyword table,
// `apps/spec/text_to_control.json` (design doc: `apps/spec/text_to_control.md`).
// Any evolution of the table happens THERE first, then this string is
// re-synced — never the reverse. Kept as an embedded string (rather than
// a runtime-loaded asset) so `FTextToControlResolver` has zero Content/
// or file-I/O dependency and works identically in every build
// configuration, including a game with no `Content/text_to_control.json`
// deployed at all.

#pragma once

/** Verbatim copy of apps/spec/text_to_control.json (see file comment). */
static const TCHAR* GAInimatorTextToControlTableJson = TEXT(R"JSON(
{
  "version": "1.0",
  "comment": "Canonical text-to-control keyword table (B6). Single source of truth: engines embed a verbatim copy and point back here. Speeds are RAW meters/frame in the root-local ground frame (engine z-normalizes). Directions are (x, z) in the local frame, +z = forward.",
  "directions": {
    "forward": [0.0, 1.0],
    "ahead": [0.0, 1.0],
    "straight": [0.0, 1.0],
    "avance": [0.0, 1.0],
    "avant": [0.0, 1.0],
    "devant": [0.0, 1.0],
    "droit": [0.0, 1.0],
    "back": [0.0, -1.0],
    "backward": [0.0, -1.0],
    "backwards": [0.0, -1.0],
    "recule": [0.0, -1.0],
    "arriere": [0.0, -1.0],
    "derriere": [0.0, -1.0],
    "left": [-1.0, 0.0],
    "gauche": [-1.0, 0.0],
    "right": [1.0, 0.0],
    "droite": [1.0, 0.0]
  },
  "speeds": {
    "sprint": 0.15,
    "dash": 0.15,
    "fonce": 0.15,
    "run": 0.1,
    "running": 0.1,
    "cours": 0.1,
    "course": 0.1,
    "vite": 0.1,
    "quickly": 0.1,
    "fast": 0.1,
    "rapidement": 0.1,
    "jog": 0.066,
    "trotte": 0.066,
    "walk": 0.033,
    "walking": 0.033,
    "marche": 0.033,
    "slow": 0.015,
    "slowly": 0.015,
    "lentement": 0.015,
    "doucement": 0.015,
    "stop": 0.0,
    "halt": 0.0,
    "idle": 0.0,
    "stand": 0.0,
    "arrete": 0.0,
    "immobile": 0.0
  },
  "defaults": {
    "speed_when_direction_only": 0.033,
    "direction_when_speed_only": [0.0, 1.0]
  }
}
)JSON");
