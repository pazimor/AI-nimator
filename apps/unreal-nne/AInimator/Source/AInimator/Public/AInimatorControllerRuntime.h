// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "AInimatorManifest.h"
#include "AInimatorNormStats.h"
#include "AInimatorNormalizer.h"
#include "AInimatorStateBuffer.h"
#include "AInimatorControlPreset.h"
#include "AInimatorControllerRuntime.generated.h"

namespace UE::NNE
{
	class IModelInstanceCPU;
}
class UNNEModelData;

/**
 * Owns one loaded controller bundle and runs exactly one NNE forward
 * per Tick() call — no autoregressive loop inside the model graph
 * (ROADMAP_PLUGINS.md §2 vérité #2; inference_contract.md §2).
 *
 * Responsibilities (single class, no gameplay logic):
 * - Load the bundle (manifest + norm_stats + presets) via
 *   FBundleLoader / FPresetLoader and the .onnx via NNE.
 * - Maintain the FStateBuffer autoregressive window.
 * - Normalize inputs / denormalize the output delta via FNormalizer.
 * - Integrate Δstate → state (root-local motion → world yaw/position,
 *   matching ainimator/model/controller_rollout.py's
 *   _integrateOneStep exactly, so Unreal and the Python/Unity
 *   reference reproduce the same trajectory).
 * - Expose SetControl / SetPreset / Tick to Blueprint.
 *
 * Foot-lock IK and physics blending (B4) are explicitly NOT this
 * class's job — Tick() returns a plausible unlocked pose, downstream
 * post-processing is the caller's responsibility.
 *
 * NNE API note: this targets UE 5.3+ NNE (`UE::NNE::GetRuntime<
 * INNERuntimeCPU>`, `IModelCPU::CreateModelInstanceCPU()`,
 * `IModelInstanceCPU::RunSync(Inputs, Outputs)` with
 * `FTensorBindingCPU`). NNE's public surface changed across 5.2-5.4;
 * VERIFY exact type/method names against the installed engine's
 * `NNE`/`NNERuntimeCPU` module headers before compiling — this file
 * cannot be compiled or engine-tested in this environment.
 */
UCLASS(BlueprintType)
class AINIMATOR_API UAInimatorControllerRuntime : public UObject
{
	GENERATED_BODY()

public:
	/**
	 * Loads a bundle directory (manifest.json, norm_stats.json,
	 * controller.onnx, presets/) and prepares the runtime for ticking.
	 *
	 * Fails loudly (returns false, logs to LogAInimator) on any
	 * contract violation — never a silent partial initialization
	 * (ROADMAP_PLUGINS.md §2 vérité #4).
	 */
	UFUNCTION(BlueprintCallable, Category = "AInimator|Runtime")
	bool LoadBundle(const FString& BundleDirectory);

	/** Sets the raw (unnormalized) control vector applied on the next
	 *  Tick(); length must equal the manifest's control_channels. */
	UFUNCTION(BlueprintCallable, Category = "AInimator|Runtime")
	bool SetControl(const TArray<float>& RawControlVector);

	/** Convenience wrapper around SetControl that builds the vector
	 *  from a ControlPreset asset (§1.1: presets are a convenience,
	 *  never a mandatory entry point). */
	UFUNCTION(BlueprintCallable, Category = "AInimator|Runtime")
	bool SetPreset(UAInimatorControlPreset* Preset);

	/**
	 * B6 (`apps/spec/text_to_control.md`): resolves free-text Command
	 * ("cours vers la gauche", "run left"...) into a raw control vector
	 * via `FTextToControlResolver` and applies it through the exact same
	 * `SetControl` path as `SetPreset` — zero impact on the preset
	 * machinery, both remain interchangeable frame by frame
	 * (ROADMAP_PLUGINS.md §4 B6 acceptance).
	 *
	 * ⚠ NOT the prompt channel (rig_binding.md §3): this only writes the
	 * low-level (vx, vz[, aim_x, aim_z]) control vector, never the text
	 * encoder / prompt_emb path.
	 *
	 * Returns false (logged by the resolver, not this function) when the
	 * command is unrecognized or ambiguous — the CURRENT control is left
	 * untouched in that case (spec §2 step 6: never a silent fallback).
	 */
	UFUNCTION(BlueprintCallable, Category = "AInimator|Runtime")
	bool SetTextCommand(const FString& Command);

	/** Sets the active prompt embedding; ignored (with a warning) if
	 *  the bundle has PromptEmbChannels == 0. Passing an empty array
	 *  reverts to the learned null embedding (never zeros). */
	UFUNCTION(BlueprintCallable, Category = "AInimator|Runtime")
	bool SetPromptEmbedding(const TArray<float>& PromptEmbedding);

	/**
	 * B7 (`apps/spec/text_encoding.md` §3): encodes free-text Text with
	 * the bundle's in-engine text encoder (`text_encoder.onnx` +
	 * `tokenizer/`, A7.1 bundles) into OutEmbedding — the value-parity
	 * twin of Unity's PromptTextEncoder. Does NOT apply the embedding:
	 * callers (UAInimatorCharacterComponent::SetPromptText) feed it
	 * through the normal SetPromptEmbedding / cross-fade path.
	 *
	 * ⚠ Distinct from SetTextCommand (B6, low-level control vector):
	 * this drives the high-level prompt channel.
	 *
	 * Returns false (logged, never a silent fallback) when the bundle
	 * ships no text encoder (A7.0 / exported without --encoder-artifact)
	 * or the encode fails; OutEmbedding is untouched then.
	 */
	bool EncodePromptText(const FString& Text, TArray<float>& OutEmbedding);

	/** The prompt embedding currently fed to the model every Tick() —
	 *  either an explicit embedding set via SetPromptEmbedding/SetPreset,
	 *  or the bundle's learned null embedding by default (never zeros,
	 *  inference_contract.md §4). Empty when PromptEmbChannels == 0.
	 *  Added for B3-bis (rig_binding.md §3): the character component's
	 *  prompt cross-fade needs to read back the resolved "current"
	 *  embedding (including the resolved null embedding) as its fade
	 *  start point, without duplicating the null-embedding fallback
	 *  logic that already lives here. */
	UFUNCTION(BlueprintPure, Category = "AInimator|Runtime")
	const TArray<float>& GetActivePromptEmbedding() const { return CurrentPromptEmb; }

	/**
	 * Runs exactly one forward and integrates the resulting Δstate
	 * into the running world-space state. Call this once per game
	 * tick (or at a fixed animation-update cadence).
	 *
	 * Returns
	 * -------
	 * bool
	 *     False if the runtime has no bundle loaded or the forward
	 *     failed (logged); callers should stop driving the pawn rather
	 *     than apply a stale/garbage pose.
	 */
	UFUNCTION(BlueprintCallable, Category = "AInimator|Runtime")
	bool Tick();

	/** Current world-space root position (meters, Unreal Y-up
	 *  convention matches manifest.coord_system "Y-up right-handed"). */
	UFUNCTION(BlueprintPure, Category = "AInimator|Runtime")
	FVector GetWorldRootPosition() const { return CurrentWorldPosition; }

	/** Current world-space yaw in radians. */
	UFUNCTION(BlueprintPure, Category = "AInimator|Runtime")
	float GetWorldYaw() const { return CurrentWorldYaw; }

	/** Latest raw (denormalized) bone rotation6d frame, flattened
	 *  bone-major/channel-minor (NumBones * RotationChannelsPerBone). */
	UFUNCTION(BlueprintPure, Category = "AInimator|Runtime")
	const TArray<float>& GetLatestBoneFrame() const { return LatestRawBoneFrame; }

	UFUNCTION(BlueprintPure, Category = "AInimator|Runtime")
	const FAInimatorManifest& GetManifest() const { return Manifest; }

	UFUNCTION(BlueprintPure, Category = "AInimator|Runtime")
	bool IsLoaded() const { return bIsLoaded; }

	/** Presets discovered under the loaded bundle's presets/ directory. */
	UFUNCTION(BlueprintPure, Category = "AInimator|Runtime")
	const TArray<UAInimatorControlPreset*>& GetBundledPresets() const
	{
		return BundledPresets;
	}

private:
	bool InitializeModel(const FString& OnnxPath);
	bool InitializeSeedState();
	void ResetRuntimeState();

	/** Builds this frame's ONNX inputs from the state buffer, current
	 *  control and prompt/phase, runs RunSync, and writes the raw
	 *  (denormalized) deltas into OutRawBoneDelta / OutRawGlobalDelta. */
	bool RunOneForward(
		TArray<float>& OutRawBoneDelta,
		TArray<float>& OutRawGlobalDelta);

	/** Integrates one root-local motion delta into world position/yaw.
	 *  Mirrors controller_rollout.py::_integrateOneStep exactly:
	 *      worldDx = cos(yaw) * dFwd - sin(yaw) * dLat
	 *      worldDz = sin(yaw) * dFwd + cos(yaw) * dLat
	 *      newPos  = pos + (worldDx, dHeight, worldDz)
	 *      newYaw  = yaw + dYaw
	 *  (Unreal FVector is (X, Y, Z) with Y-up per the contract's
	 *  coord_system: worldDx -> X, dHeight -> Y, worldDz -> Z.) */
	void IntegrateRootLocalDelta(const TArray<float>& RawGlobalDelta);

	FAInimatorManifest Manifest;
	FAInimatorNormStats NormStats;
	TUniquePtr<FNormalizer> Normalizer;
	TUniquePtr<FStateBuffer> StateBuffer;

	/** Bundle directory of the last successful LoadBundle — needed to
	 *  lazily initialize the B7 prompt text encoder. */
	FString LoadedBundleDirectory;

	/** Lazily-built in-engine prompt encoder (B7) — only when the
	 *  bundle ships text_encoder.onnx + tokenizer/ (A7.1+). */
	TUniquePtr<class FAInimatorPromptTextEncoder> PromptTextEncoder;

	/** Model asset + instance. TObjectPtr keeps the model data alive
	 *  for the lifetime of this runtime; the instance is created once
	 *  in InitializeModel and reused every Tick (no per-frame
	 *  allocation, ROADMAP_PLUGINS.md NNE-specifics guidance). */
	UPROPERTY()
	TObjectPtr<UNNEModelData> ModelData;

	TSharedPtr<UE::NNE::IModelInstanceCPU> ModelInstance;

	/** Pre-allocated, reused every Tick — normalized state/control
	 *  scratch buffers feeding the ONNX input tensors. */
	TArray<float> ScratchNormalizedBoneWindow;
	TArray<float> ScratchNormalizedGlobalWindow;
	TArray<float> ScratchNormalizedControl;
	TArray<float> ScratchBoneDeltaOutput;
	TArray<float> ScratchGlobalDeltaOutput;

	TArray<float> CurrentRawControl;
	TArray<float> CurrentPromptEmb;

	TArray<float> LatestRawBoneFrame;

	FVector CurrentWorldPosition = FVector::ZeroVector;
	float CurrentWorldYaw = 0.0f;

	UPROPERTY()
	TArray<TObjectPtr<UAInimatorControlPreset>> BundledPresets;

	bool bIsLoaded = false;
};
