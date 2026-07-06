// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include "AInimatorControllerRuntime.h"
#include "AInimatorDemoPawn.generated.h"

class UAInimatorControlPreset;
class UInputAction;
class UInputMappingContext;
struct FInputActionValue;

/**
 * Minimal keyboard-driven demo Pawn for phase B2.
 *
 * WASD selects the matching bundled preset (forward/backward/
 * strafe_left/strafe_right) each tick; idle is used when no movement
 * key is held. This is intentionally the simplest possible authoring
 * — the real "button -> action + ControlPreset" binding UX is B3's
 * job (UAInimatorActionComponent); this pawn only proves the runtime
 * end-to-end (ROADMAP_PLUGINS.md B2 acceptance: "un pawn piloté au
 * clavier").
 *
 * No foot-lock IK, no skeletal mesh binding here (B4): the pawn only
 * exposes the runtime's world position/yaw/bone frame so a Blueprint
 * or a later AnimGraph node can drive an actual SkeletalMeshComponent.
 */
UCLASS(Blueprintable)
class AINIMATOR_API AAInimatorDemoPawn : public APawn
{
	GENERATED_BODY()

public:
	AAInimatorDemoPawn();

	/** Absolute path to the bundle directory (controller.onnx,
	 *  manifest.json, norm_stats.json, presets/) — typically the
	 *  plugin's own Content/ once the build orchestrator (apps/build/)
	 *  has copied a fresh bundle there (gitignored, never committed). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|Demo")
	FString BundleDirectory;

	UPROPERTY(BlueprintReadOnly, Category = "AInimator|Demo")
	TObjectPtr<UAInimatorControllerRuntime> Runtime;

	UPROPERTY(EditDefaultsOnly, Category = "AInimator|Demo|Input")
	TObjectPtr<UInputMappingContext> MappingContext;

	UPROPERTY(EditDefaultsOnly, Category = "AInimator|Demo|Input")
	TObjectPtr<UInputAction> MoveForwardAction;

	UPROPERTY(EditDefaultsOnly, Category = "AInimator|Demo|Input")
	TObjectPtr<UInputAction> MoveBackwardAction;

	UPROPERTY(EditDefaultsOnly, Category = "AInimator|Demo|Input")
	TObjectPtr<UInputAction> StrafeLeftAction;

	UPROPERTY(EditDefaultsOnly, Category = "AInimator|Demo|Input")
	TObjectPtr<UInputAction> StrafeRightAction;

protected:
	virtual void BeginPlay() override;
	virtual void Tick(float DeltaSeconds) override;
	virtual void SetupPlayerInputComponent(
		UInputComponent* PlayerInputComponent) override;

private:
	void OnMoveForwardPressed(const FInputActionValue& Value);
	void OnMoveBackwardPressed(const FInputActionValue& Value);
	void OnStrafeLeftPressed(const FInputActionValue& Value);
	void OnStrafeRightPressed(const FInputActionValue& Value);
	void OnMoveReleased(const FInputActionValue& Value);

	/** Selects a bundled preset by name (idle/forward/backward/
	 *  strafe_left/strafe_right) and applies it to Runtime. */
	void SelectPresetByName(const FString& PresetName);

	UAInimatorControlPreset* FindBundledPreset(const FString& PresetName) const;
};
