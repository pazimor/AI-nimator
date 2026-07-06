"""Build orchestrator for the engine plugins (Goal B phase B5).

Single-command pipeline (ROADMAP_PLUGINS §3.4): export a fresh bundle
from a mandatory checkpoint, validate it against the spec schemas,
deliver it into the target plugin's gitignored resource folder, then
invoke the engine packaging step. Lives outside the ``ainimator``
package; only invokes the export CLI and engine tools.
"""
