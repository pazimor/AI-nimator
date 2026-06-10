"""End-to-end integration tests for the AI-nimator v2 pipeline.

These tests exercise the full text-to-motion stack at the level of
shape contracts and gradient flow:

    tokens (CustomTokenizer)
        → text hidden states (CustomTextEncoder)
        → cross-attention conditioning (MotionDenoiserV2)
        → diffusion target / loss (NoiseSchedule + losses_v2)
        → DDIM sampling (DDIMSamplerV2)

The point of these tests is **not** to verify learning convergence —
that is the job of the overfit-1-sample sanity check the user runs
manually after Phase B.  The point is to catch interface drift between
the v2 modules: any mismatch in dtype, shape, mask convention, or
prediction-mode handling will fail loudly here.
"""

from __future__ import annotations

import torch

from src.shared.model.generation.denoiser_v2 import (
    MotionDenoiserV2,
    MotionDenoiserV2Config,
)
from src.shared.model.generation.losses_v2 import (
    diffusionLossV2,
    velocityXyzLossV2,
)
from src.shared.model.generation.noise_schedule import (
    NoiseSchedule,
    NoiseScheduleConfig,
    PREDICTION_V,
)
from src.shared.model.generation.sampler_v2 import DDIMSamplerV2
from src.shared.model.text import (
    CustomTextEncoder,
    CustomTextEncoderConfig,
    CustomTokenizer,
    CustomTokenizerConfig,
)


# ---------------------------------------------------------------------
# Shared mini stack — small enough to run on CPU in milliseconds
# ---------------------------------------------------------------------
EMBED_DIM = 64
TEXT_TOKENS = 16
NUM_LAYERS = 2
NUM_HEADS = 4
NUM_BONES = 22
MOTION_CHANNELS = 6
GLOBAL_CHANNELS = 3
FRAMES = 8
NUM_TIMESTEPS = 100

CORPUS = [
    "a person walks forward.",
    "a person walks backward slowly.",
    "the person crawls on the ground.",
    "someone runs and jumps high.",
    "the woman dances gracefully in a circle.",
    "a man sits down on a chair.",
    "the person waves their right hand.",
    "someone stands up then walks.",
] * 4


def _buildTokenizerAndEncoder() -> tuple[CustomTokenizer, CustomTextEncoder]:
    tokenizer = CustomTokenizer.train(
        CORPUS,
        config=CustomTokenizerConfig(
            vocabSize=128, maxLength=TEXT_TOKENS, minFrequency=1
        ),
    )
    encoder = CustomTextEncoder(
        CustomTextEncoderConfig(
            vocabSize=tokenizer.vocabSize,
            maxLength=TEXT_TOKENS,
            hiddenDim=EMBED_DIM,
            numLayers=NUM_LAYERS,
            numHeads=NUM_HEADS,
            outputDim=EMBED_DIM,  # match denoiser embedDim, no projection
            padTokenId=tokenizer.padTokenId,
            dropout=0.0,
        )
    )
    return tokenizer, encoder


def _buildDenoiser() -> MotionDenoiserV2:
    return MotionDenoiserV2(
        MotionDenoiserV2Config(
            embedDim=EMBED_DIM,
            numHeads=NUM_HEADS,
            numLayers=NUM_LAYERS,
            numBones=NUM_BONES,
            motionChannels=MOTION_CHANNELS,
            globalChannels=GLOBAL_CHANNELS,
            textEmbedDim=EMBED_DIM,
            maxFrames=32,
            dropout=0.0,
        )
    )


def _buildSchedule() -> NoiseSchedule:
    return NoiseSchedule(NoiseScheduleConfig(numSteps=NUM_TIMESTEPS))


# ---------------------------------------------------------------------
# Forward pass: encoder → denoiser
# ---------------------------------------------------------------------
def test_encoder_to_denoiser_forward_pass_shapes() -> None:
    tokenizer, encoder = _buildTokenizerAndEncoder()
    denoiser = _buildDenoiser()

    encoder.eval()
    denoiser.eval()

    encoded = tokenizer.encode(
        ["a person walks forward.", "the woman dances."]
    )
    textOut = encoder(encoded.inputIds, encoded.attentionMask)

    motion = torch.randn(2, FRAMES, NUM_BONES, MOTION_CHANNELS)
    timesteps = torch.tensor([10, 80])
    globals_ = torch.randn(2, FRAMES, GLOBAL_CHANNELS)

    out = denoiser(
        noisyMotion=motion,
        timesteps=timesteps,
        textHiddenStates=textOut.hiddenStates,
        textKeyPaddingMask=textOut.keyPaddingMask,
        noisyGlobalFeatures=globals_,
    )
    assert out.boneOutput.shape == motion.shape
    assert out.globalOutput is not None
    assert out.globalOutput.shape == globals_.shape


# ---------------------------------------------------------------------
# Training step: schedule → denoiser → loss → backward
# ---------------------------------------------------------------------
def test_full_training_step_backward() -> None:
    tokenizer, encoder = _buildTokenizerAndEncoder()
    denoiser = _buildDenoiser()
    schedule = _buildSchedule()

    encoder.train()
    denoiser.train()

    encoded = tokenizer.encode(
        ["a person walks forward.", "the woman dances."]
    )
    textOut = encoder(encoded.inputIds, encoded.attentionMask)

    x0Bone = torch.randn(2, FRAMES, NUM_BONES, MOTION_CHANNELS)
    x0Global = torch.randn(2, FRAMES, GLOBAL_CHANNELS)
    timesteps = torch.tensor([20, 80])

    xtBone, noiseBone = schedule.qSample(x0Bone, timesteps)
    xtGlobal, noiseGlobal = schedule.qSample(x0Global, timesteps)
    targetBone = schedule.predictionTarget(
        x0Bone, noiseBone, timesteps, PREDICTION_V
    )
    targetGlobal = schedule.predictionTarget(
        x0Global, noiseGlobal, timesteps, PREDICTION_V
    )

    out = denoiser(
        noisyMotion=xtBone,
        timesteps=timesteps,
        textHiddenStates=textOut.hiddenStates,
        textKeyPaddingMask=textOut.keyPaddingMask,
        noisyGlobalFeatures=xtGlobal,
    )

    boneLoss = diffusionLossV2(
        prediction=out.boneOutput,
        target=targetBone,
        timesteps=timesteps,
        alphasCumprod=schedule.alphasCumprod,
        predictionMode=PREDICTION_V,
    )
    assert out.globalOutput is not None
    globalLoss = diffusionLossV2(
        prediction=out.globalOutput,
        target=targetGlobal,
        timesteps=timesteps,
        alphasCumprod=schedule.alphasCumprod,
        predictionMode=PREDICTION_V,
    )
    velLoss = velocityXyzLossV2(
        predictedRotation6d=out.boneOutput,
        targetRotation6d=x0Bone,
        timesteps=timesteps,
        alphasCumprod=schedule.alphasCumprod,
    )
    total = boneLoss + globalLoss + 0.5 * velLoss

    total.backward()

    # All trainable parameters in encoder + denoiser should have a
    # finite gradient; this catches any disconnected sub-graph.
    # Phase F — the encoder's nullEmbedding parameter is only reached
    # by gradient when CFG dropout fires; this pipeline test uses the
    # encoder forward path directly (no dropout), so the null token
    # legitimately receives no gradient and is excluded here.
    for module in (encoder, denoiser):
        for name, parameter in module.named_parameters():
            if isinstance(module, CustomTextEncoder) and name == "nullEmbedding":
                continue
            assert parameter.grad is not None, (
                f"Missing grad on {module.__class__.__name__}.{name}."
            )
            assert torch.isfinite(parameter.grad).all(), (
                f"Non-finite grad on {module.__class__.__name__}.{name}."
            )


# ---------------------------------------------------------------------
# Sampling: encoder → sampler → motion
# ---------------------------------------------------------------------
def test_sampler_pipeline_with_text_encoder() -> None:
    tokenizer, encoder = _buildTokenizerAndEncoder()
    denoiser = _buildDenoiser()
    schedule = _buildSchedule()
    sampler = DDIMSamplerV2(schedule, predictionMode=PREDICTION_V)

    encoder.eval()
    denoiser.eval()

    cond = tokenizer.encode(["a person walks forward."])
    uncond = tokenizer.encode([""])
    condOut = encoder(cond.inputIds, cond.attentionMask)
    uncondOut = encoder(uncond.inputIds, uncond.attentionMask)

    output = sampler.sample(
        denoiser=denoiser,
        textHiddenStates=condOut.hiddenStates,
        textKeyPaddingMask=condOut.keyPaddingMask,
        unconditionalTextHiddenStates=uncondOut.hiddenStates,
        unconditionalTextKeyPaddingMask=uncondOut.keyPaddingMask,
        frames=FRAMES,
        numSteps=20,
        cfgScale=3.5,
        seed=42,
    )

    assert output.boneMotion.shape == (
        1, FRAMES, NUM_BONES, MOTION_CHANNELS
    )
    assert output.globalMotion is not None
    assert output.globalMotion.shape == (1, FRAMES, GLOBAL_CHANNELS)
    assert torch.isfinite(output.boneMotion).all()
    assert torch.isfinite(output.globalMotion).all()


# ---------------------------------------------------------------------
# CFG behaviour at the pipeline level
# ---------------------------------------------------------------------
def test_cfg_scale_controls_text_influence() -> None:
    """Different CFG scales must produce visibly different outputs.

    The exact value of cfgScale that "works" depends on the trained
    denoiser; here we only assert that the output is sensitive to the
    scale knob — silent ignorance of cfgScale would indicate a bug
    upstream of the sampler.
    """
    tokenizer, encoder = _buildTokenizerAndEncoder()
    denoiser = _buildDenoiser()
    schedule = _buildSchedule()
    sampler = DDIMSamplerV2(schedule, predictionMode=PREDICTION_V)
    encoder.eval()
    denoiser.eval()

    cond = tokenizer.encode(["a person walks forward."])
    uncond = tokenizer.encode([""])
    condOut = encoder(cond.inputIds, cond.attentionMask)
    uncondOut = encoder(uncond.inputIds, uncond.attentionMask)

    common = dict(
        denoiser=denoiser,
        textHiddenStates=condOut.hiddenStates,
        textKeyPaddingMask=condOut.keyPaddingMask,
        unconditionalTextHiddenStates=uncondOut.hiddenStates,
        unconditionalTextKeyPaddingMask=uncondOut.keyPaddingMask,
        frames=FRAMES,
        numSteps=10,
        seed=0,
    )

    out_low = sampler.sample(**common, cfgScale=1.0)
    out_high = sampler.sample(**common, cfgScale=5.0)
    diff = (out_low.boneMotion - out_high.boneMotion).abs().max().item()
    assert diff > 1e-4, (
        f"cfgScale must influence output (max diff={diff:.2e})."
    )
