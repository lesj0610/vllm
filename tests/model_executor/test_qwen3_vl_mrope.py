# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import random
from dataclasses import dataclass

import pytest
import torch

from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from vllm.model_executor.models.qwen2_vl import Qwen2VLForConditionalGeneration
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)


@pytest.fixture(autouse=True, scope="module")
def _force_cpu_default_device():
    # _get_mrope_input_positions returns CPU tensors (via torch.from_numpy).
    # Ensure the default device is CPU so the rest of the test tensors match.
    original = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(original)


IMAGE_TOKEN_ID = 999
VIDEO_TOKEN_ID = 888
VISION_START_TOKEN_ID = 777
VISION_END_TOKEN_ID = 778


@dataclass
class DummyVisionConfig:
    spatial_merge_size: int = 1
    tokens_per_second: float = 1.0


@dataclass
class DummyConfig:
    image_token_id: int = IMAGE_TOKEN_ID
    video_token_id: int = VIDEO_TOKEN_ID
    vision_start_token_id: int = VISION_START_TOKEN_ID
    vision_end_token_id: int = VISION_END_TOKEN_ID
    vision_config: DummyVisionConfig = dataclasses.field(
        default_factory=DummyVisionConfig
    )


def make_video_embedding(
    t, h, w, interleave_text_tokens: tuple[int, int], video_pruning_rate: float = 0.0
):
    """
    Helper function to make a video embedding for a given video size and pruning rate.

    Args:
        t: Number of frames.
        h: Number of rows.
        w: Number of columns.
        interleave_text_tokens: Tuple of minimum and maximum number of text tokens to
            interleave with the video.
        video_pruning_rate: Pruning rate for the video.

    Returns:
        Tuple of (unpruned_tokens_sequence, pruned_tokens_sequence, retention_mask)
    """
    unpruned_tokens_sequence = []
    population = list(range(1, 100))

    for _ in range(t):
        num_prefix_tokens = random.randint(
            interleave_text_tokens[0], interleave_text_tokens[1]
        )

        prefix_tokens = random.choices(population, k=num_prefix_tokens)
        vision_tokens = (
            [VISION_START_TOKEN_ID] + [VIDEO_TOKEN_ID] * h * w + [VISION_END_TOKEN_ID]
        )

        unpruned_tokens_sequence.extend(prefix_tokens)
        unpruned_tokens_sequence.extend(vision_tokens)

    unpruned_tokens_sequence = torch.tensor(unpruned_tokens_sequence, dtype=torch.long)
    video_token_mask = unpruned_tokens_sequence == VIDEO_TOKEN_ID

    pruning_mask = torch.bernoulli(video_token_mask.float() * video_pruning_rate).bool()  # type: ignore[attr-defined]
    # Sanity check that we don't prune what should not be pruned.
    assert not pruning_mask[~video_token_mask].any()

    retention_mask = ~pruning_mask
    pruned_tokens_sequence = unpruned_tokens_sequence[retention_mask]
    return unpruned_tokens_sequence, pruned_tokens_sequence, retention_mask


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
@pytest.mark.parametrize("grid_thw", [[3, 8, 7], [128, 10, 12]])
@pytest.mark.parametrize("num_prefix_tokens", [1, 11])
@pytest.mark.parametrize("num_suffix_tokens", [0, 7])
@pytest.mark.parametrize("video_pruning_rate", [0, 0.25, 0.75])
@pytest.mark.parametrize("interleave_text_tokens", [(0, 0), (1, 4)])
def test_match_qwen3vl_mrope_evs_on(
    spatial_merge_size: int,
    num_prefix_tokens: int,
    grid_thw: tuple[int, int, int],
    num_suffix_tokens: int,
    video_pruning_rate: float,
    interleave_text_tokens: tuple[int, int],
):
    hf_config = DummyConfig()
    hf_config.vision_config.spatial_merge_size = spatial_merge_size

    t, h, w = grid_thw
    population = list(range(1, 100))
    prefix_tokens = random.choices(population, k=num_prefix_tokens)
    suffix_tokens = random.choices(population, k=num_suffix_tokens)

    video_tokens, video_tokens_pruned, retention_mask = make_video_embedding(
        t,
        h // spatial_merge_size,
        w // spatial_merge_size,
        interleave_text_tokens=interleave_text_tokens,
        video_pruning_rate=video_pruning_rate,
    )
    assert len(video_tokens) == len(retention_mask)

    input_tokens = prefix_tokens + video_tokens.tolist() + suffix_tokens
    input_tokens_pruned = prefix_tokens + video_tokens_pruned.tolist() + suffix_tokens

    whole_sequence_retention_mask = torch.cat(
        [
            torch.ones(len(prefix_tokens), dtype=torch.bool),
            retention_mask,
            torch.ones(len(suffix_tokens), dtype=torch.bool),
        ],
        dim=0,
    )

    # Build the GT mrope for unpruned input.
    mm_feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                "video_grid_thw": MultiModalFieldElem(
                    data=torch.tensor(grid_thw),
                    field=None,  # HACK.
                ),
            }
        ),
        modality="video",
        identifier="DUMMY",
        mm_position=PlaceholderRange(offset=0, length=len(input_tokens)),
    )
    expected_mrope, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=input_tokens,
        mm_features=[mm_feature],
        config=hf_config,
    )

    # Compute mrope for a video-only media (unpruned).
    mm_feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                "video_grid_thw": MultiModalFieldElem(
                    data=torch.tensor(grid_thw),
                    field=None,  # HACK.
                ),
            }
        ),
        modality="video",
        identifier="DUMMY",
        mm_position=PlaceholderRange(offset=0, length=video_tokens.numel()),
    )
    video_mrope, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=video_tokens.tolist(),
        mm_features=[mm_feature],
        config=hf_config,
    )
    video_mrope = video_mrope.permute(1, 0)  # [N, 3]
    hidden_size = 16

    is_video_embed = torch.isin(
        video_tokens_pruned, torch.tensor([VIDEO_TOKEN_ID], dtype=torch.long)
    )

    expanded_positions = torch.full(
        (len(video_tokens_pruned), 5),
        fill_value=-100,
        device=video_mrope.device,
        dtype=torch.long,
    )
    expanded_positions[is_video_embed, :3] = video_mrope[retention_mask][is_video_embed]
    expanded_positions[~is_video_embed, :3] = video_mrope[retention_mask][
        ~is_video_embed
    ]

    is_vision_start = video_tokens_pruned == VISION_START_TOKEN_ID
    expanded_positions[..., 3] = is_vision_start
    expanded_positions[..., 4] = is_video_embed

    # Check that all positions were filled, since we initialized them as negative.
    assert (expanded_positions >= 0).all()

    video_embeddings = torch.empty(
        (len(video_tokens_pruned), hidden_size), device=video_mrope.device
    )

    video_embeddings = torch.cat(
        [
            video_embeddings,
            expanded_positions.float(),
        ],
        dim=1,
    )
    multimodal_embeddings = [video_embeddings]

    expected_mrope_masked = expected_mrope[:, whole_sequence_retention_mask]

    # Initialize computed_mrope with sequential positions for all prefix tokens
    computed_mrope = torch.empty((3, len(input_tokens_pruned)), dtype=torch.long)
    computed_mrope[:, 0 : len(prefix_tokens)] = expected_mrope[
        :, 0 : len(prefix_tokens)
    ]

    # Paranoia check that computed_mrope is wrong.
    assert not torch.equal(computed_mrope, expected_mrope_masked)

    _, actual_mrope, _ = Qwen3VLForConditionalGeneration._recompute_mrope_positions(
        input_ids=input_tokens_pruned,
        multimodal_embeddings=multimodal_embeddings,
        mrope_positions=computed_mrope,
        num_computed_tokens=len(prefix_tokens),
        vision_start_token_id=hf_config.vision_start_token_id,
        image_token_id=hf_config.image_token_id,
        video_token_id=hf_config.video_token_id,
    )

    assert torch.equal(actual_mrope, expected_mrope_masked)


def _make_mm_feature(
    modality: str,
    grid_thw: tuple[int, int, int],
    offset: int,
    length: int,
) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                f"{modality}_grid_thw": MultiModalFieldElem(
                    data=torch.tensor(grid_thw),
                    field=None,
                ),
            }
        ),
        modality=modality,
        identifier="DUMMY",
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def _make_normal_video_case(
    grid_thw: tuple[int, int, int],
    *,
    spatial_merge_size: int = 1,
    inter_frame_text_tokens: int = 0,
) -> tuple[list[int], list[MultiModalFeatureSpec]]:
    t, h, w = grid_thw
    tokens_per_frame = (h // spatial_merge_size) * (w // spatial_merge_size)
    tokens = [11, 12]
    offset = len(tokens)
    for frame_idx in range(t):
        if frame_idx > 0:
            tokens.extend([100 + frame_idx] * inter_frame_text_tokens)
        tokens.extend(
            [VISION_START_TOKEN_ID]
            + [VIDEO_TOKEN_ID] * tokens_per_frame
            + [VISION_END_TOKEN_ID]
        )
    video_length = len(tokens) - offset
    tokens.extend([13, 14, 15])
    return tokens, [_make_mm_feature("video", grid_thw, offset, video_length)]


def _make_lumped_video_case(
    grid_thw: tuple[int, int, int],
    num_video_tokens: int,
) -> tuple[list[int], list[MultiModalFeatureSpec]]:
    t, _, _ = grid_thw
    tokens = [21, 22, VISION_START_TOKEN_ID]
    offset = 2
    tokens.extend([VIDEO_TOKEN_ID] * num_video_tokens)
    tokens.append(VISION_END_TOKEN_ID)
    for _ in range(t - 1):
        tokens.extend([VISION_START_TOKEN_ID, VISION_END_TOKEN_ID])
    video_length = len(tokens) - offset
    tokens.extend([23, 24])
    return tokens, [_make_mm_feature("video", grid_thw, offset, video_length)]


def _make_multiple_image_case(
    grids: list[tuple[int, int, int]],
) -> tuple[list[int], list[MultiModalFeatureSpec]]:
    tokens = list(range(30, 94))
    features = []
    for index, grid_thw in enumerate(grids):
        _, h, w = grid_thw
        tokens.extend([100 + index] * (index + 1))
        offset = len(tokens)
        num_image_tokens = h * w
        tokens.extend([IMAGE_TOKEN_ID] * num_image_tokens)
        features.append(_make_mm_feature("image", grid_thw, offset, num_image_tokens))
    tokens.extend(range(200, 264))
    return tokens, features


def test_qwen3_5_mrope_positions_are_sequence_bounded():
    cases = [
        ([1, 2, 3, 4, 5], [], DummyConfig()),
        (*_make_multiple_image_case([(1, 2, 3)]), DummyConfig()),
        (
            *_make_multiple_image_case([(1, 1, 7), (1, 3, 4), (1, 5, 2), (1, 4, 4)]),
            DummyConfig(),
        ),
        (*_make_normal_video_case((3, 2, 3)), DummyConfig()),
        (*_make_normal_video_case((128, 10, 12)), DummyConfig()),
        (
            *_make_normal_video_case(
                (3, 4, 6),
                spatial_merge_size=2,
                inter_frame_text_tokens=5,
            ),
            DummyConfig(vision_config=DummyVisionConfig(spatial_merge_size=2)),
        ),
        # Three complete logical frames are stored in the first placeholder.
        (*_make_lumped_video_case((3, 2, 2), num_video_tokens=12), DummyConfig()),
        # Two complete frames and a partial-frame remainder are stored together.
        (*_make_lumped_video_case((3, 2, 2), num_video_tokens=10), DummyConfig()),
    ]

    rng = random.Random(0)
    for _ in range(100):
        grids = [
            (1, rng.randint(1, 8), rng.randint(1, 8)) for _ in range(rng.randint(1, 4))
        ]
        cases.append((*_make_multiple_image_case(grids), DummyConfig()))
    for _ in range(50):
        spatial_merge_size = rng.choice((1, 2))
        grid_thw = (
            rng.randint(1, 8),
            spatial_merge_size * rng.randint(1, 8),
            spatial_merge_size * rng.randint(1, 8),
        )
        cases.append(
            (
                *_make_normal_video_case(
                    grid_thw,
                    spatial_merge_size=spatial_merge_size,
                    inter_frame_text_tokens=rng.randint(0, 5),
                ),
                DummyConfig(
                    vision_config=DummyVisionConfig(
                        spatial_merge_size=spatial_merge_size
                    )
                ),
            )
        )

    for input_tokens, mm_features, config in cases:
        positions, mrope_position_delta = (
            Qwen3_5ForConditionalGeneration._get_mrope_input_positions(
                input_tokens=input_tokens,
                mm_features=mm_features,
                config=config,
            )
        )

        assert positions.shape == (3, len(input_tokens))
        assert positions.min().item() >= 0
        assert positions.max().item() < len(input_tokens)
        assert mrope_position_delta <= 0


def test_qwen3_vl_pruned_positions_can_exceed_pruned_sequence_length():
    input_tokens, mm_features = _make_normal_video_case((128, 10, 10))
    positions, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=input_tokens,
        mm_features=mm_features,
        config=DummyConfig(),
    )

    input_tensor = torch.tensor(input_tokens)
    retention_mask = input_tensor != VIDEO_TOKEN_ID
    video_indices = torch.where(input_tensor == VIDEO_TOKEN_ID)[0]
    retention_mask[video_indices[:100]] = True

    retained_positions = positions[:, retention_mask]
    pruned_len = int(retention_mask.sum())

    assert retained_positions.max().item() >= pruned_len
    assert retained_positions.max().item() + 1 - pruned_len > 0


@pytest.mark.parametrize(
    "model_cls",
    [Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration],
)
def test_qwen2_vl_video_temporal_positions_require_legacy_capacity(model_cls):
    config = DummyConfig(
        vision_config=DummyVisionConfig(
            spatial_merge_size=1,
            tokens_per_second=25,
        )
    )
    model = object.__new__(model_cls)
    torch.nn.Module.__init__(model)
    model.config = config

    mm_feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                "video_grid_thw": MultiModalFieldElem(
                    data=torch.tensor((2, 1, 1)),
                    field=None,
                ),
                "second_per_grid_ts": MultiModalFieldElem(
                    data=torch.tensor(4.0),
                    field=None,
                ),
            }
        ),
        modality="video",
        identifier="DUMMY",
        mm_position=PlaceholderRange(offset=0, length=2),
    )
    positions, mrope_position_delta = model.get_mrope_input_positions(
        input_tokens=[VIDEO_TOKEN_ID, VIDEO_TOKEN_ID],
        mm_features=[mm_feature],
    )

    assert positions[0].tolist() == [0, 100]
    assert positions.max().item() == 100
    assert mrope_position_delta == 99
    assert positions.max().item() >= 2
