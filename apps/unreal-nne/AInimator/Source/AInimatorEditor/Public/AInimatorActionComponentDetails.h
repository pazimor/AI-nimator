// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "IDetailCustomization.h"

class IDetailLayoutBuilder;
class UAInimatorActionComponent;

/**
 * Details panel customization for `UAInimatorActionComponent`
 * (ROADMAP_PLUGINS.md §4 B3): surfaces the `Bindings` array as an
 * editable `[Input] -> [Preset]` list (the default array UI already
 * works — this customization adds the two authoring shortcuts the
 * acceptance criteria call for) plus two buttons:
 *
 * - **Create Preset** — spawns a new `UAInimatorControlPreset` asset
 *   via `UAInimatorControlPresetFactory` (Content Browser save dialog)
 *   and appends a new binding row pointing at it, so "add a binding"
 *   never requires leaving the Details panel to go create an asset
 *   first.
 * - **Import Presets From Bundle...** — picks a bundle directory and
 *   calls `FPresetLoader::LoadPresetsFromDirectory`, then offers to
 *   save each loaded preset as a project asset (via the same factory
 *   path) so bundle-provided presets (idle/forward/backward/
 *   strafe_left/strafe_right) can be promoted to versioned assets
 *   without hand re-entering their values.
 * - **Test Text Command** (B6, `apps/spec/text_to_control.md`) — resolves
 *   the component's `TextCommand` field via
 *   `UAInimatorActionComponent::ActivateTextCommand` for a quick PIE
 *   test ("cours vers la gauche" -> control applied immediately). Only
 *   meaningful while the game is running (Runtime must be loaded); a
 *   no-op warning is logged otherwise.
 *
 * Kept in the editor-only module (`AInimatorEditor`) so none of this
 * — nor its `UnrealEd`/`PropertyEditor` dependencies — ships in a
 * cooked build (ROADMAP_PLUGINS.md architecture note: "keep
 * editor-only code out of the runtime module").
 */
class FAInimatorActionComponentDetails : public IDetailCustomization
{
public:
	static TSharedRef<IDetailCustomization> MakeInstance();

	virtual void CustomizeDetails(IDetailLayoutBuilder& DetailBuilder) override;

private:
	FReply OnCreatePresetClicked();
	FReply OnImportFromBundleClicked();
	FReply OnTestTextCommandClicked();

	/** The single component instance being customized (Details panels
	 *  support multi-edit, but bundle import / preset creation only
	 *  make sense targeting exactly one component at a time — this
	 *  customization disables both buttons otherwise). */
	TWeakObjectPtr<UAInimatorActionComponent> EditedComponent;

	IDetailLayoutBuilder* CachedDetailBuilder = nullptr;
};
