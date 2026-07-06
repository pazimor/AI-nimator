// Copyright AI-nimator.

#include "AInimatorStateBuffer.h"

FStateBuffer::FStateBuffer(
	int32 InContextFrames,
	int32 InBoneFrameWidth,
	int32 InGlobalFrameWidth)
	: ContextFrames(InContextFrames)
	, BoneFrameWidth(InBoneFrameWidth)
	, GlobalFrameWidth(InGlobalFrameWidth)
{
	check(ContextFrames > 0);
	check(BoneFrameWidth > 0);
	check(GlobalFrameWidth > 0);

	BoneWindow.SetNumZeroed(ContextFrames * BoneFrameWidth);
	GlobalWindow.SetNumZeroed(ContextFrames * GlobalFrameWidth);
}

void FStateBuffer::SeedWithFrame(
	const TArray<float>& SeedBoneFrame,
	const TArray<float>& SeedGlobalFrame)
{
	check(SeedBoneFrame.Num() == BoneFrameWidth);
	check(SeedGlobalFrame.Num() == GlobalFrameWidth);

	for (int32 Slot = 0; Slot < ContextFrames; ++Slot)
	{
		FMemory::Memcpy(
			BoneWindow.GetData() + Slot * BoneFrameWidth,
			SeedBoneFrame.GetData(),
			BoneFrameWidth * sizeof(float));
		FMemory::Memcpy(
			GlobalWindow.GetData() + Slot * GlobalFrameWidth,
			SeedGlobalFrame.GetData(),
			GlobalFrameWidth * sizeof(float));
	}
}

void FStateBuffer::PushFrame(
	const TArray<float>& NewBoneFrame,
	const TArray<float>& NewGlobalFrame)
{
	check(NewBoneFrame.Num() == BoneFrameWidth);
	check(NewGlobalFrame.Num() == GlobalFrameWidth);

	// Shift left by one frame, then write the newest frame into the
	// last slot. ContextFrames is small (contract default: 8), so a
	// memmove here is cheap and keeps the buffer as a simple flat
	// array (no modulo indexing, no wraparound bugs, easy to hand the
	// whole thing to the ONNX input tensor as-is).
	if (ContextFrames > 1)
	{
		FMemory::Memmove(
			BoneWindow.GetData(),
			BoneWindow.GetData() + BoneFrameWidth,
			(ContextFrames - 1) * BoneFrameWidth * sizeof(float));
		FMemory::Memmove(
			GlobalWindow.GetData(),
			GlobalWindow.GetData() + GlobalFrameWidth,
			(ContextFrames - 1) * GlobalFrameWidth * sizeof(float));
	}
	FMemory::Memcpy(
		BoneWindow.GetData() + (ContextFrames - 1) * BoneFrameWidth,
		NewBoneFrame.GetData(),
		BoneFrameWidth * sizeof(float));
	FMemory::Memcpy(
		GlobalWindow.GetData() + (ContextFrames - 1) * GlobalFrameWidth,
		NewGlobalFrame.GetData(),
		GlobalFrameWidth * sizeof(float));
}

void FStateBuffer::GetLastBoneFrame(TArray<float>& OutFrame) const
{
	OutFrame.SetNumUninitialized(BoneFrameWidth);
	FMemory::Memcpy(
		OutFrame.GetData(),
		BoneWindow.GetData() + (ContextFrames - 1) * BoneFrameWidth,
		BoneFrameWidth * sizeof(float));
}
