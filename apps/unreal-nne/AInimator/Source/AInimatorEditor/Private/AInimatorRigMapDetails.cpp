// Copyright AI-nimator.

#include "AInimatorRigMapDetails.h"
#include "AInimatorRigMap.h"
#include "AInimatorLog.h"

#include "DetailLayoutBuilder.h"
#include "DetailCategoryBuilder.h"
#include "DetailWidgetRow.h"
#include "ContentBrowserModule.h"
#include "IContentBrowserSingleton.h"
#include "Engine/SkeletalMesh.h"
#include "Animation/Skeleton.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SComboButton.h"
#include "Widgets/Text/STextBlock.h"
#include "Framework/Application/SlateApplication.h"

#define LOCTEXT_NAMESPACE "AInimatorRigMapDetails"

TSharedRef<IDetailCustomization> FAInimatorRigMapDetails::MakeInstance()
{
	return MakeShared<FAInimatorRigMapDetails>();
}

void FAInimatorRigMapDetails::CustomizeDetails(IDetailLayoutBuilder& DetailBuilder)
{
	CachedDetailBuilder = &DetailBuilder;

	TArray<TWeakObjectPtr<UObject>> Objects;
	DetailBuilder.GetObjectsBeingCustomized(Objects);
	if (Objects.Num() == 1)
	{
		EditedRigMap = Cast<UAInimatorRigMap>(Objects[0].Get());
	}

	IDetailCategoryBuilder& RigMapCategory =
		DetailBuilder.EditCategory(TEXT("AInimator|RigMap"));

	TSharedPtr<SComboButton> PickerButton;

	FContentBrowserModule& ContentBrowserModule =
		FModuleManager::LoadModuleChecked<FContentBrowserModule>("ContentBrowser");

	FAssetPickerConfig PickerConfig;
	PickerConfig.Filter.ClassPaths.Add(USkeletalMesh::StaticClass()->GetClassPathName());
	PickerConfig.SelectionMode = ESelectionMode::Single;
	PickerConfig.OnAssetSelected = FOnAssetSelected::CreateLambda(
		[this](const FAssetData& AssetData)
		{
			FSlateApplication::Get().DismissAllMenus();

			UAInimatorRigMap* RigMap = EditedRigMap.Get();
			USkeletalMesh* SkeletalMesh = Cast<USkeletalMesh>(AssetData.GetAsset());
			if (!RigMap || !SkeletalMesh)
			{
				return;
			}

			const USkeleton* Skeleton = SkeletalMesh->GetSkeleton();
			if (!Skeleton)
			{
				UE_LOG(LogAInimator, Error,
					TEXT("AInimator: selected SkeletalMesh '%s' has no Skeleton asset."),
					*SkeletalMesh->GetName());
				return;
			}

			const FReferenceSkeleton& RefSkeleton = Skeleton->GetReferenceSkeleton();
			TArray<FName> CandidateBoneNames;
			CandidateBoneNames.Reserve(RefSkeleton.GetNum());
			for (int32 BoneIndex = 0; BoneIndex < RefSkeleton.GetNum(); ++BoneIndex)
			{
				CandidateBoneNames.Add(RefSkeleton.GetBoneName(BoneIndex));
			}

			RigMap->Modify();
			RigMap->AutoMapByCommonNames(CandidateBoneNames, /*bOverwriteExisting=*/false);

			if (CachedDetailBuilder)
			{
				CachedDetailBuilder->ForceRefreshDetails();
			}
		});

	RigMapCategory.AddCustomRow(LOCTEXT("AutoMapRowFilter", "Auto-Map From Skeletal Mesh"))
		.WholeRowContent()
		[
			SAssignNew(PickerButton, SComboButton)
			.ButtonContent()
			[
				SNew(STextBlock)
				.Text(LOCTEXT("AutoMapButton", "Auto-Map From Skeletal Mesh..."))
			]
			.OnGetMenuContent_Lambda([ContentBrowserModule_ptr = &ContentBrowserModule, PickerConfig]() -> TSharedRef<SWidget>
			{
				return SNew(SBox)
					.WidthOverride(400.0f)
					.HeightOverride(400.0f)
					[
						ContentBrowserModule_ptr->Get().CreateAssetPicker(PickerConfig)
					];
			})
		];
}

FReply FAInimatorRigMapDetails::OnAutoMapClicked()
{
	// Unused: auto-map is triggered from the asset picker's
	// OnAssetSelected lambda above (a combo-button menu, not a plain
	// button, since the workflow needs an asset PICK step, not just a
	// click) — kept as a declared method for interface symmetry with
	// FAInimatorActionComponentDetails and in case a future revision
	// wants a simpler "use last-set SkeletalMeshComponent" shortcut.
	return FReply::Handled();
}

#undef LOCTEXT_NAMESPACE
