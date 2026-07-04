// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "IDetailCustomization.h"

class IDetailLayoutBuilder;
class UAInimatorRigMap;

/**
 * Details panel customization for `UAInimatorRigMap`
 * (`apps/spec/rig_binding.md` deliverable: "Editor: Details/factory du
 * RigMap (auto-map)"). Adds an **Auto-Map From Skeletal Mesh...** button
 * above the default `Entries` array editor: prompts for a
 * `USkeletalMesh` asset, reads its reference skeleton's bone names, and
 * calls `UAInimatorRigMap::AutoMapByCommonNames` against them.
 *
 * The default array UI already lets a user hand-edit every
 * `FAInimatorRigMapEntry::TargetBoneName` — this customization only adds
 * the best-effort authoring shortcut, never replaces manual control.
 */
class FAInimatorRigMapDetails : public IDetailCustomization
{
public:
	static TSharedRef<IDetailCustomization> MakeInstance();

	virtual void CustomizeDetails(IDetailLayoutBuilder& DetailBuilder) override;

private:
	FReply OnAutoMapClicked();

	TWeakObjectPtr<UAInimatorRigMap> EditedRigMap;
	IDetailLayoutBuilder* CachedDetailBuilder = nullptr;
};
