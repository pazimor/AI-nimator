// Copyright AI-nimator. Editor-only module: authoring niceties for
// ControlPreset assets. Never linked into cooked/shipping builds.

using UnrealBuildTool;

public class AInimatorEditor : ModuleRules
{
	public AInimatorEditor(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"AInimator",
		});

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"UnrealEd",
			"Slate",
			"SlateCore",
			"PropertyEditor",
		});
	}
}
