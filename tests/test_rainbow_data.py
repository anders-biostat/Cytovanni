"""cytovanni.rainbow.rbim_data: subsampler and peak-share table (the dataset itself is tested in
test_rainbow_calibrator.py, where the fitted rainbow measurements are)."""
import numpy as np
import pytest
import torch

from cytovanni.rainbow.rbim_data import Sampler, DataLoader, _intype_GMM_crosstab_shares


GROUPS = [torch.tensor([7, 3, 9]), torch.tensor([], dtype=torch.long), torch.tensor([10, 1, 12, 13, 20, 21])]


@pytest.fixture
def sampler(device):
    return Sampler([g.clone() for g in GROUPS], Nsubsample=2, device=device)


def test_sample_sizes_and_membership(sampler):
    inds = sampler.sample()
    assert inds.device.type == "cpu" and inds.dtype == torch.int64
    assert len(inds) == 2 + 0 + 2
    assert set(inds[:2].tolist()) <= {7, 3, 9} and set(inds[2:].tolist()) <= {10, 1, 12, 13, 20, 21}
    assert len(set(inds[:2].tolist())) == 2 and len(set(inds[2:].tolist())) == 2     # without replacement


def test_small_groups_contribute_all(device):
    s = Sampler([g.clone() for g in GROUPS], Nsubsample=10, device=device)
    inds = s.sample()
    assert sorted(inds[:3].tolist()) == [3, 7, 9] and sorted(inds[3:].tolist()) == [1, 10, 12, 13, 20, 21]


def test_every_call_draws_anew(sampler):
    draws = {tuple(sorted(sampler.sample()[2:].tolist())) for _ in range(50)}
    assert len(draws) > 3
    counts = np.bincount(torch.cat([sampler.sample()[2:] for _ in range(600)]).numpy(), minlength=22)
    np.testing.assert_allclose(counts[[10, 1, 12, 13, 20, 21]] / 1200, 1 / 6, atol=.05)    # uniform within a group


def test_split_non_contiguous_and_empty_groups_R2_E14(sampler):
    """R2-E14: split=True assigns the drawn indices to their groups by the group counts (index values need not be
    contiguous ranges); empty groups are left out together with their numbers."""
    parts, groups = sampler.sample(split=True)
    assert groups.tolist() == [0, 2] and len(parts) == 2
    assert set(parts[0].tolist()) <= {7, 3, 9} and set(parts[1].tolist()) <= {10, 1, 12, 13, 20, 21}
    assert [len(p) for p in parts] == [2, 2]


def test_crosstab_shares_rows_sum_to_one():
    index = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    label = torch.tensor([0, 1, 1, 2, 2, 2, 0])
    shares = _intype_GMM_crosstab_shares(index, label, 2, 3)
    torch.testing.assert_close(shares, torch.tensor([[1 / 3, 2 / 3, 0.], [.25, 0., .75]]))


def test_crosstab_shares_sized_by_type_R2_E13():
    """R2-E13: the table has shape (N_sample, N_peak) even if the brightest peak or the last sample has no beads,
    and empty samples get uniform shares (not NaN)."""
    index = torch.tensor([0, 0, 2, 2])
    label = torch.tensor([0, 1, 1, 1])
    shares = _intype_GMM_crosstab_shares(index, label, 4, 3)
    assert shares.shape == (4, 3) and torch.isfinite(shares).all()
    torch.testing.assert_close(shares[0], torch.tensor([.5, .5, 0.]))
    torch.testing.assert_close(shares[1], torch.full((3,), 1 / 3))
    torch.testing.assert_close(shares[3], torch.full((3,), 1 / 3))
    torch.testing.assert_close(shares.sum(-1), torch.ones(4))


class _FakeDataset:
    def get_batch(self, tp, indices):
        return {"tp": tp, "indices": indices}


def test_dataloader_batches_and_length():
    loader = DataLoader(_FakeDataset(), "T", torch.arange(25), batch_size=10)
    batches = list(loader)
    assert len(loader) == 25                                      # events, not batches
    assert [len(b["indices"]) for b in batches] == [10, 10, 5]
    assert torch.equal(torch.cat([b["indices"] for b in batches]), torch.arange(25))
    assert all(b["tp"] == "T" for b in batches)
