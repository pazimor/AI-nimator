// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "IDetailCustomization.h"

class IDetailLayoutBuilder;
class UAInimatorControlPreset;

/**
 * Details panel customization for `UAInimatorControlPreset` (B3-bis,
 * `apps/spec/rig_binding.md` §4 / deliverable list: "Editor: ...
 * import d'un JSON encode_prompt dans un UAInimatorControlPreset").
 *
 * Adds an **Import Prompt Embedding (JSON)...** button that opens a file
 * picker for the JSON produced by
 * `python -m ainimator.cli.encode_prompt --prompt "..." --encoder-artifact
 * <dir> --output <file>` (shape `{"prompt": str, "prompt_emb": [float,
 * ...]}`) and hydrates this preset's `Prompt`/`PromptEmb`/`bHasPromptEmb`
 * fields from it — the authoring path documented in rig_binding.md §4.1
 * ("encoder la phrase côté Python ... coller dans un preset").
 */
class FAInimatorControlPresetDetails : public IDetailCustomization
{
public:
	static TSharedRef<IDetailCustomization> MakeInstance();

	virtual void CustomizeDetails(IDetailLayoutBuilder& DetailBuilder) override;

private:
	FReply OnImportPromptEmbeddingClicked();

	TWeakObjectPtr<UAInimatorControlPreset> EditedPreset;
	IDetailLayoutBuilder* CachedDetailBuilder = nullptr;
};
