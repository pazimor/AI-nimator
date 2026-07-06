// Copyright AI-nimator.

#include "AInimatorControllerRuntime.h"
#include "AInimatorLog.h"
#include "AInimatorContractConstants.h"
#include "AInimatorBundleLoader.h"
#include "AInimatorForwardKinematics.h"
#include "AInimatorPresetLoader.h"
#include "AInimatorPromptTextEncoder.h"
#include "AInimatorRootLocalMath.h"
#include "AInimatorTextToControlResolver.h"
#include "Misc/Paths.h"
#include "Misc/FileHelper.h"

// --- NNE (verify exact API against the installed engine — see header
// comment in AInimatorControllerRuntime.h). Targeting UE 5.3+ NNE:
// UE::NNE::GetRuntime<INNERuntimeCPU>(), CreateModelCPU(ModelData),
// IModelCPU::CreateModelInstanceCPU(), IModelInstanceCPU::SetInputTensorShapes
// + RunSync(TConstArrayView<FTensorBindingCPU>, TConstArrayView<FTensorBindingCPU>).
#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "NNETypes.h"
#include "UObject/WeakInterfacePtr.h"

using namespace AInimatorContract;

bool UAInimatorControllerRuntime::LoadBundle(const FString& BundleDirectory)
{
	ResetRuntimeState();

	if (!FBundleLoader::LoadFromDirectory(BundleDirectory, Manifest, NormStats))
	{
		// FBundleLoader already logged the specific reason.
		return false;
	}

	Normalizer = MakeUnique<FNormalizer>(Manifest, NormStats);

	const int32 BoneFrameWidth = Manifest.NumBones * Manifest.RotationChannelsPerBone;
	StateBuffer = MakeUnique<FStateBuffer>(
		Manifest.ContextFrames, BoneFrameWidth, Manifest.RootLocalMotionChannels);

	const FString OnnxPath =
		FPaths::Combine(BundleDirectory, TEXT("controller.onnx"));
	if (!InitializeModel(OnnxPath))
	{
		return false;
	}

	if (!InitializeSeedState())
	{
		return false;
	}

	const FString PresetsDirectory =
		FPaths::Combine(BundleDirectory, TEXT("presets"));
	BundledPresets.Reset();
	TArray<UAInimatorControlPreset*> LoadedPresets;
	FPresetLoader::LoadPresetsFromDirectory(PresetsDirectory, this, LoadedPresets);
	for (UAInimatorControlPreset* Preset : LoadedPresets)
	{
		BundledPresets.Add(Preset);
	}

	// Default control: silence (all zero), reasonable until the caller
	// picks a preset or drives control directly.
	CurrentRawControl.SetNumZeroed(Manifest.ControlChannels);

	// Default prompt: the learned null embedding, never zeros
	// (inference_contract.md §4) — copied once here so RunOneForward
	// never has to special-case "no prompt selected yet".
	CurrentPromptEmb = NormStats.PromptNullEmb;

	// Remembered for the lazy B7 prompt-encoder init (EncodePromptText).
	LoadedBundleDirectory = BundleDirectory;

	bIsLoaded = true;
	return true;
}

bool UAInimatorControllerRuntime::EncodePromptText(
	const FString& Text,
	TArray<float>& OutEmbedding)
{
	if (!bIsLoaded)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: EncodePromptText called with no loaded bundle."));
		return false;
	}
	if (Manifest.PromptEmbChannels <= 0)
	{
		UE_LOG(LogAInimator, Warning,
			TEXT("AInimator: EncodePromptText — bundle declares ")
			TEXT("prompt_emb_channels == 0; prompt-at-runtime is unavailable."));
		return false;
	}
	if (!Manifest.HasTextEncoder())
	{
		UE_LOG(LogAInimator, Warning,
			TEXT("AInimator: EncodePromptText — bundle ships no ")
			TEXT("text_encoder.onnx (A7.0 bundle, or exported without ")
			TEXT("--encoder-artifact). Re-export with the encoder artifact ")
			TEXT("(text_encoding.md §1) or use SetPromptEmbedding with a ")
			TEXT("precomputed embedding; keeping the current prompt."));
		return false;
	}

	if (!PromptTextEncoder.IsValid())
	{
		TUniquePtr<FAInimatorPromptTextEncoder> Encoder =
			MakeUnique<FAInimatorPromptTextEncoder>();
		if (!Encoder->Init(LoadedBundleDirectory, Manifest.TextEncoder))
		{
			return false; // Encoder already logged the specific reason.
		}
		PromptTextEncoder = MoveTemp(Encoder);
	}

	return PromptTextEncoder->EncodePrompt(Text, OutEmbedding);
}

bool UAInimatorControllerRuntime::InitializeModel(const FString& OnnxPath)
{
	TArray<uint8> OnnxBytes;
	if (!FFileHelper::LoadFileToArray(OnnxBytes, *OnnxPath))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: failed to read controller.onnx at '%s'."),
			*OnnxPath);
		return false;
	}

	// UNNEModelData normally comes from an imported editor asset
	// (right-click .onnx -> Create NNE Model Data). Loading raw bytes
	// at runtime from a build-artifact bundle (never a versioned
	// asset, ROADMAP_PLUGINS.md §3.4) requires either:
	//   (a) an editor-time import step baked into the build
	//       orchestrator (apps/build/) that produces a UNNEModelData
	//       asset alongside the bundle before packaging, or
	//   (b) UNNEModelData::Init(TEXT("Onnx"), OnnxBytes) if the
	//       installed NNE version exposes a runtime-side init path.
	// VERIFY against the installed engine: this call must be replaced
	// with whichever path the target UE/NNE version actually supports;
	// this file cannot be compiled/tested in this environment.
	ModelData = NewObject<UNNEModelData>(this);
	if (!ModelData || !ModelData->Init(TEXT("onnx"), OnnxBytes))
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: UNNEModelData failed to init from '%s'. If ")
			TEXT("this NNE version has no runtime Init(), see the comment ")
			TEXT("in InitializeModel() for the editor-import alternative."),
			*OnnxPath);
		return false;
	}

	// CPU runtime chosen deliberately for determinism/parity across
	// platforms (ROADMAP_PLUGINS.md NNE-specifics guidance); a GPU/RDG
	// runtime remains a valid future perf path but risks breaking
	// bit-exact parity with the Unity/Sentis and Python reference.
	TWeakInterfacePtr<INNERuntimeCPU> Runtime =
		UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeORTCpu"));
	if (!Runtime.IsValid())
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: no NNE CPU runtime available (expected ")
			TEXT("'NNERuntimeORTCpu' or equivalent — verify the runtime ")
			TEXT("name registered by the installed NNE plugin)."));
		return false;
	}

	TSharedPtr<UE::NNE::IModelCPU> Model =
		Runtime->CreateModelCPU(ModelData);
	if (!Model.IsValid())
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: CreateModelCPU failed for '%s'."), *OnnxPath);
		return false;
	}

	ModelInstance = Model->CreateModelInstanceCPU();
	if (!ModelInstance.IsValid())
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: CreateModelInstanceCPU failed for '%s'."),
			*OnnxPath);
		return false;
	}

	return true;
}

bool UAInimatorControllerRuntime::InitializeSeedState()
{
	const int32 BoneFrameWidth = Manifest.NumBones * Manifest.RotationChannelsPerBone;

	// Seed with the canonical rest pose: rot6d identity
	// [1,0,0, 0,1,0] repeated per bone (column 0 = X axis, column 1 =
	// Y axis of an identity rotation matrix), global motion at zero.
	// A game may override this via a dedicated "seed from animation"
	// path later (out of scope for B2); this guarantees Tick() is safe
	// to call immediately after a successful LoadBundle().
	TArray<float> SeedBoneFrame;
	SeedBoneFrame.Reserve(BoneFrameWidth);
	for (int32 Bone = 0; Bone < Manifest.NumBones; ++Bone)
	{
		SeedBoneFrame.Append({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
	}
	TArray<float> SeedGlobalFrame;
	SeedGlobalFrame.SetNumZeroed(Manifest.RootLocalMotionChannels);

	StateBuffer->SeedWithFrame(SeedBoneFrame, SeedGlobalFrame);
	LatestRawBoneFrame = SeedBoneFrame;

	CurrentWorldPosition = FVector::ZeroVector;
	float PelvisRot6d[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
	CurrentWorldYaw = AInimatorRootLocalMath::PelvisYawFromRot6d(PelvisRot6d);
	return true;
}

void UAInimatorControllerRuntime::ResetRuntimeState()
{
	bIsLoaded = false;
	Normalizer.Reset();
	StateBuffer.Reset();
	ModelInstance.Reset();
	ModelData = nullptr;
	PromptTextEncoder.Reset();
	LoadedBundleDirectory.Reset();
	BundledPresets.Reset();
	CurrentRawControl.Reset();
	CurrentPromptEmb.Reset();
	LatestRawBoneFrame.Reset();
	CurrentWorldPosition = FVector::ZeroVector;
	CurrentWorldYaw = 0.0f;
}

bool UAInimatorControllerRuntime::SetControl(const TArray<float>& RawControlVector)
{
	if (!bIsLoaded)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: SetControl called before a bundle was loaded."));
		return false;
	}
	if (RawControlVector.Num() != Manifest.ControlChannels)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: SetControl received %d values, expected %d ")
			TEXT("(manifest.control_channels)."),
			RawControlVector.Num(), Manifest.ControlChannels);
		return false;
	}
	CurrentRawControl = RawControlVector;
	return true;
}

bool UAInimatorControllerRuntime::SetPreset(UAInimatorControlPreset* Preset)
{
	if (!bIsLoaded)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: SetPreset called before a bundle was loaded."));
		return false;
	}
	if (!Preset)
	{
		UE_LOG(LogAInimator, Error, TEXT("AInimator: SetPreset given null preset."));
		return false;
	}

	TArray<float> RawControlVector;
	if (!Preset->BuildControlVector(Manifest.ControlLayout, RawControlVector))
	{
		return false;
	}
	CurrentRawControl = RawControlVector;

	if (Preset->bHasPromptEmb)
	{
		return SetPromptEmbedding(Preset->PromptEmb);
	}
	return true;
}

bool UAInimatorControllerRuntime::SetTextCommand(const FString& Command)
{
	if (!bIsLoaded)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: SetTextCommand called before a bundle was loaded."));
		return false;
	}

	const TOptional<FAInimatorResolvedControl> Resolved = FTextToControlResolver::Resolve(Command);
	if (!Resolved.IsSet())
	{
		// FTextToControlResolver::Resolve already logged the specific
		// reason (unrecognized or ambiguous, text_to_control.md §2 step
		// 6) — current control is left untouched (never a silent
		// fallback to some movement).
		return false;
	}

	// Map the resolved (vx, vz[, aim_x, aim_z]) onto the manifest's
	// declared control_layout order, exactly like
	// UAInimatorControlPreset::BuildControlVector does for presets —
	// same SetControl path, zero impact on the preset machinery
	// (ROADMAP_PLUGINS.md §4 B6 acceptance).
	TArray<float> RawControlVector;
	RawControlVector.SetNumZeroed(Manifest.ControlLayout.Num());
	for (int32 Index = 0; Index < Manifest.ControlLayout.Num(); ++Index)
	{
		const FString& ChannelName = Manifest.ControlLayout[Index];
		if (ChannelName == TEXT("vx"))
		{
			RawControlVector[Index] = Resolved->Vx;
		}
		else if (ChannelName == TEXT("vz"))
		{
			RawControlVector[Index] = Resolved->Vz;
		}
		else if (ChannelName == TEXT("aim_x"))
		{
			RawControlVector[Index] = Resolved->AimX;
		}
		else if (ChannelName == TEXT("aim_z"))
		{
			RawControlVector[Index] = Resolved->AimZ;
		}
		else
		{
			UE_LOG(LogAInimator, Error,
				TEXT("AInimator: unknown control_layout channel '%s' in manifest ")
				TEXT("— this plugin build does not recognise it (SetTextCommand)."),
				*ChannelName);
			return false;
		}
	}
	return SetControl(RawControlVector);
}

bool UAInimatorControllerRuntime::SetPromptEmbedding(
	const TArray<float>& PromptEmbedding)
{
	if (Manifest.PromptEmbChannels == 0)
	{
		if (PromptEmbedding.Num() > 0)
		{
			UE_LOG(LogAInimator, Warning,
				TEXT("AInimator: SetPromptEmbedding given %d values but ")
				TEXT("this bundle has prompt_emb_channels=0; ignoring."),
				PromptEmbedding.Num());
		}
		return true;
	}

	if (PromptEmbedding.Num() == 0)
	{
		// No prompt active: fall back to the learned null embedding —
		// never a zero vector (inference_contract.md §4).
		CurrentPromptEmb = NormStats.PromptNullEmb;
		return true;
	}

	if (PromptEmbedding.Num() != Manifest.PromptEmbChannels)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: SetPromptEmbedding received %d values, ")
			TEXT("expected %d (manifest.prompt_emb_channels)."),
			PromptEmbedding.Num(), Manifest.PromptEmbChannels);
		return false;
	}
	CurrentPromptEmb = PromptEmbedding;
	return true;
}

bool UAInimatorControllerRuntime::Tick()
{
	if (!bIsLoaded)
	{
		UE_LOG(LogAInimator, Error, TEXT("AInimator: Tick called before LoadBundle."));
		return false;
	}

	TArray<float> RawBoneDelta;
	TArray<float> RawGlobalDelta;
	if (!RunOneForward(RawBoneDelta, RawGlobalDelta))
	{
		return false;
	}

	// state_{t+1} = state_t + rawDelta, matching
	// controller_rollout.py::_stepOnce exactly (delta added to the RAW
	// last bone frame, not the normalized one).
	TArray<float> LastBoneFrame;
	StateBuffer->GetLastBoneFrame(LastBoneFrame);
	TArray<float> NextBoneFrame;
	NextBoneFrame.SetNumUninitialized(LastBoneFrame.Num());
	for (int32 Index = 0; Index < LastBoneFrame.Num(); ++Index)
	{
		NextBoneFrame[Index] = LastBoneFrame[Index] + RawBoneDelta[Index];
	}

	// Normative re-orthonormalization (inference_contract.md §3.6): keep
	// the accumulated state on the rotation manifold before it re-enters
	// the window (parity with the Python reference loop and Unity).
	AInimatorForwardKinematics::OrthonormalizeFrame(NextBoneFrame);

	IntegrateRootLocalDelta(RawGlobalDelta);

	StateBuffer->PushFrame(NextBoneFrame, RawGlobalDelta);
	LatestRawBoneFrame = NextBoneFrame;
	return true;
}

void UAInimatorControllerRuntime::IntegrateRootLocalDelta(
	const TArray<float>& RawGlobalDelta)
{
	check(RawGlobalDelta.Num() == RootLocalMotionChannels);

	const float DeltaForward = RawGlobalDelta[0];
	const float DeltaLateral = RawGlobalDelta[1];
	const float DeltaHeight = RawGlobalDelta[2];
	const float DeltaYaw = RawGlobalDelta[3];

	const float CosYaw = FMath::Cos(CurrentWorldYaw);
	const float SinYaw = FMath::Sin(CurrentWorldYaw);

	// Mirrors controller_rollout.py::_integrateOneStep. coord_system is
	// "Y-up right-handed" (manifest); Unreal's FVector is (X, Y, Z)
	// with Y as the second axis, so worldDx -> X, dHeight -> Y,
	// worldDz -> Z, matching the Python (worldDx, dHeight, worldDz)
	// stacking order exactly.
	const float WorldDeltaX = CosYaw * DeltaForward - SinYaw * DeltaLateral;
	const float WorldDeltaZ = SinYaw * DeltaForward + CosYaw * DeltaLateral;

	CurrentWorldPosition += FVector(WorldDeltaX, DeltaHeight, WorldDeltaZ);
	CurrentWorldYaw += DeltaYaw;
}

bool UAInimatorControllerRuntime::RunOneForward(
	TArray<float>& OutRawBoneDelta,
	TArray<float>& OutRawGlobalDelta)
{
	if (!ModelInstance.IsValid())
	{
		UE_LOG(LogAInimator, Error, TEXT("AInimator: no model instance bound."));
		return false;
	}

	const int32 BoneFrameWidth = Manifest.NumBones * Manifest.RotationChannelsPerBone;
	const int32 GlobalFrameWidth = Manifest.RootLocalMotionChannels;

	// Normalize the state window frame-by-frame (the buffer stores raw
	// history; ONNX expects a normalized window every forward).
	ScratchNormalizedBoneWindow.SetNumUninitialized(
		Manifest.ContextFrames * BoneFrameWidth);
	ScratchNormalizedGlobalWindow.SetNumUninitialized(
		Manifest.ContextFrames * GlobalFrameWidth);

	const TArray<float>& RawBoneWindow = StateBuffer->GetBoneWindow();
	const TArray<float>& RawGlobalWindow = StateBuffer->GetGlobalWindow();

	TArray<float> FrameScratch;
	TArray<float> FrameNormalizedScratch;
	for (int32 Frame = 0; Frame < Manifest.ContextFrames; ++Frame)
	{
		FrameScratch.SetNumUninitialized(BoneFrameWidth);
		FMemory::Memcpy(
			FrameScratch.GetData(),
			RawBoneWindow.GetData() + Frame * BoneFrameWidth,
			BoneFrameWidth * sizeof(float));
		Normalizer->NormalizeBoneFrame(FrameScratch, FrameNormalizedScratch);
		FMemory::Memcpy(
			ScratchNormalizedBoneWindow.GetData() + Frame * BoneFrameWidth,
			FrameNormalizedScratch.GetData(),
			BoneFrameWidth * sizeof(float));

		FrameScratch.SetNumUninitialized(GlobalFrameWidth);
		FMemory::Memcpy(
			FrameScratch.GetData(),
			RawGlobalWindow.GetData() + Frame * GlobalFrameWidth,
			GlobalFrameWidth * sizeof(float));
		Normalizer->NormalizeGlobalFrame(FrameScratch, FrameNormalizedScratch);
		FMemory::Memcpy(
			ScratchNormalizedGlobalWindow.GetData() + Frame * GlobalFrameWidth,
			FrameNormalizedScratch.GetData(),
			GlobalFrameWidth * sizeof(float));
	}

	Normalizer->NormalizeControlVector(CurrentRawControl, ScratchNormalizedControl);

	// --- Bind NNE tensors and run one forward. ---
	// VERIFY against the installed NNE version: FTensorBindingCPU
	// layout, UE::NNE::FTensorShape construction and RunSync's exact
	// signature vary between 5.2/5.3/5.4. The shapes below follow
	// inference_contract.md §2 (B, K, 22, 6) / (B, C) / (B, K, 4)
	// [/ (B, D)] [/ (B, 2)] -> (B, 22, 6) / (B, 4), batch size 1.
	using namespace UE::NNE;

	TArray<FTensorBindingCPU> Inputs;
	FTensorBindingCPU BoneWindowBinding;
	BoneWindowBinding.Data = ScratchNormalizedBoneWindow.GetData();
	BoneWindowBinding.SizeInBytes =
		ScratchNormalizedBoneWindow.Num() * sizeof(float);
	Inputs.Add(BoneWindowBinding);

	FTensorBindingCPU ControlBinding;
	ControlBinding.Data = ScratchNormalizedControl.GetData();
	ControlBinding.SizeInBytes = ScratchNormalizedControl.Num() * sizeof(float);
	Inputs.Add(ControlBinding);

	FTensorBindingCPU GlobalWindowBinding;
	GlobalWindowBinding.Data = ScratchNormalizedGlobalWindow.GetData();
	GlobalWindowBinding.SizeInBytes =
		ScratchNormalizedGlobalWindow.Num() * sizeof(float);
	Inputs.Add(GlobalWindowBinding);

	if (Manifest.PromptEmbChannels > 0)
	{
		FTensorBindingCPU PromptEmbBinding;
		PromptEmbBinding.Data = CurrentPromptEmb.GetData();
		PromptEmbBinding.SizeInBytes = CurrentPromptEmb.Num() * sizeof(float);
		Inputs.Add(PromptEmbBinding);
	}

	// Phase input omitted: B2 scope carries phase_channels==0 bundles
	// (the shipped reference bundle has none). When phase_channels==2,
	// a (cos, sin) binding must be appended here in the same position
	// export_onnx.py uses (after prompt_emb, or directly after
	// global_window when there is no prompt) — TODO before shipping a
	// phase-conditioned bundle; see open questions in the session report.

	ScratchBoneDeltaOutput.SetNumUninitialized(BoneFrameWidth);
	ScratchGlobalDeltaOutput.SetNumUninitialized(GlobalFrameWidth);

	TArray<FTensorBindingCPU> Outputs;
	FTensorBindingCPU BoneDeltaBinding;
	BoneDeltaBinding.Data = ScratchBoneDeltaOutput.GetData();
	BoneDeltaBinding.SizeInBytes = ScratchBoneDeltaOutput.Num() * sizeof(float);
	Outputs.Add(BoneDeltaBinding);

	FTensorBindingCPU GlobalDeltaBinding;
	GlobalDeltaBinding.Data = ScratchGlobalDeltaOutput.GetData();
	GlobalDeltaBinding.SizeInBytes = ScratchGlobalDeltaOutput.Num() * sizeof(float);
	Outputs.Add(GlobalDeltaBinding);

	const int32 RunStatus = ModelInstance->RunSync(Inputs, Outputs);
	if (RunStatus != 0)
	{
		UE_LOG(LogAInimator, Error,
			TEXT("AInimator: NNE RunSync failed with status %d."), RunStatus);
		return false;
	}

	Normalizer->DenormalizeBoneDelta(ScratchBoneDeltaOutput, OutRawBoneDelta);
	Normalizer->DenormalizeGlobalDelta(ScratchGlobalDeltaOutput, OutRawGlobalDelta);
	return true;
}
