// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"

/**
 * Dedicated log category for the AInimator plugin.
 *
 * All bundle-loading, normalization and runtime errors go through this
 * category so a game team can isolate plugin diagnostics from the rest
 * of the engine log — important for the fail-fast contract-validation
 * rule (never fail silently).
 */
DECLARE_LOG_CATEGORY_EXTERN(LogAInimator, Log, All);
