import torch

from src.shared.temporal_diffusion import CausalDiffusion, TemporalUNetMoE


def _build_model(bone_count: int = 3, hidden_dim: int = 16) -> TemporalUNetMoE:
    return TemporalUNetMoE(
        rotationInputDim=bone_count * 6,
        hiddenDim=hidden_dim,
        layerCount=2,
        moeConfiguration={"expertCount": 2, "topK": 1},
        textDim=hidden_dim,
    )


def test_temporal_unet_forward_shapes():
    torch.manual_seed(0)
    batch_size = 2
    frame_count = 4
    bone_count = 3
    model = _build_model(bone_count)
    rotation6d = torch.randn(batch_size, frame_count, bone_count, 6)
    time_vector = torch.rand(batch_size, frame_count, 1)
    text_embedding = torch.randn(batch_size, model.conditionProjection.in_features)
    tag_embedding = torch.randn(batch_size, model.conditionProjection.in_features)
    residual, aux_loss = model(
        rotation6d,
        time_vector,
        text_embedding,
        tag_embedding,
        contextSequence=None,
        causalMask=None,
    )
    assert residual.shape == rotation6d.shape
    assert aux_loss.numel() == 1


def test_causal_diffusion_loss_and_sampling():
    torch.manual_seed(1)
    bone_count = 2
    frame_count = 3
    model = _build_model(bone_count)
    diffusion = CausalDiffusion(model, trainingSteps=5)
    batch_size = 2
    rotation6d = torch.randn(batch_size, frame_count, bone_count, 6)
    text_embedding = torch.randn(batch_size, model.conditionProjection.in_features)
    tag_embedding = torch.randn(batch_size, model.conditionProjection.in_features)
    loss = diffusion.loss(
        rotation6d,
        text_embedding,
        tag_embedding,
        contextSequence=None,
        causalMask=None,
    )
    assert loss.shape == ()
    sampled = diffusion.sample(
        frameCount=frame_count,
        boneCount=bone_count,
        textEmbedding=text_embedding[:1],
        tagEmbedding=tag_embedding[:1],
        contextSequence=None,
        steps=3,
        guidanceScale=1.5,
        causalMask=None,
        device="cpu",
    )
    assert sampled.shape == (1, frame_count, bone_count, 6)


def test_causal_diffusion_with_context_and_eta():
    torch.manual_seed(2)
    model = _build_model(bone_count=2)
    diffusion = CausalDiffusion(model, trainingSteps=10)
    rotation6d = torch.randn(1, 4, 2, 6)
    context = torch.randn(1, 4, 2, 6)
    text_embedding = torch.randn(1, model.conditionProjection.in_features)
    tag_embedding = torch.randn(1, model.conditionProjection.in_features)
    mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    loss = diffusion.loss(rotation6d, text_embedding, tag_embedding, contextSequence=context, causalMask=mask)
    assert loss.item() > 0
    sampled = diffusion.sample(
        frameCount=4,
        boneCount=2,
        textEmbedding=text_embedding,
        tagEmbedding=tag_embedding,
        contextSequence=context,
        steps=4,
        guidanceScale=1.0,
        causalMask=mask,
        device="cpu",
        eta=0.5,
    )
    assert sampled.shape == (1, 4, 2, 6)
