// Copyright AI-nimator. Runtime module: consumes an NNE model + the
// AI-nimator controller bundle contract (apps/spec/inference_contract.md).
// Defines no model — pure engine-side inference + integration.

using UnrealBuildTool;

public class AInimator : ModuleRules
{
	public AInimator(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"NNE",
			"Json",
			"JsonUtilities",
			"InputCore",
			"EnhancedInput",
		});

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"Projects",
		});
	}
}
