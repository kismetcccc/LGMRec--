import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.lgmrecopt import LGMRec, LGMRecOpt
from run_experiment import (ROOT, VARIANTS, build_model, evaluate, evaluate_frozen,
                            load_data, sample_negatives, seed_everything, train_one)


@pytest.fixture
def config():
    value = yaml.safe_load((ROOT / 'configs/model/LGMRecOpt.yaml').read_text(encoding='utf-8'))
    value.update(embedding_size=4, hyper_num=3, n_ui_layers=1, n_mm_layers=1,
                 keep_rate=1.0, epochs=2, patience=2, batch_size=4,
                 eval_batch_size=2, item_chunk_size=3, topk=[1, 2],
                 selection_metric='recall@2', seed=999, **VARIANTS['full'])
    return value


@pytest.fixture
def data(tmp_path):
    directory = tmp_path / 'baby'
    directory.mkdir()
    pairs = [(0, 0, 0), (0, 1, 0), (0, 2, 1), (0, 3, 2),
             (1, 2, 0), (1, 3, 0), (1, 4, 1), (1, 5, 2),
             (2, 1, 0), (2, 4, 0), (2, 0, 1), (2, 2, 2)]
    pd.DataFrame(pairs, columns=['userID', 'itemID', 'x_label']).to_csv(
        directory / 'baby.inter', sep='\t', index=False)
    rng = np.random.default_rng(42)
    np.save(directory / 'image_feat.npy', rng.normal(size=(8, 5)).astype('float32'))
    np.save(directory / 'text_feat.npy', rng.normal(size=(8, 6)).astype('float32'))
    return load_data(directory, 'baby')


@pytest.mark.parametrize('variant', list(VARIANTS))
def test_forward_backward_and_pair_block_agreement(data, config, variant):
    config.update(VARIANTS[variant])
    seed_everything(999)
    model = build_model(data, config, 'cpu')
    u, pos, neg = torch.tensor([0, 1, 2]), torch.tensor([0, 2, 1]), torch.tensor([6, 7, 5])
    model.train()
    loss, metrics = model.loss(u, pos, neg)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    state = model.encode()
    pairs = model.pair_score(state, u, pos)
    all_scores = model.score_block(state, u, 0, model.n_items)
    torch.testing.assert_close(pairs, all_scores[torch.arange(3), pos], atol=1e-6, rtol=1e-5)
    # Evaluation consumes no RNG and agrees across item chunks.
    before = torch.get_rng_state().clone()
    state2 = model.encode()
    torch.testing.assert_close(before, torch.get_rng_state())
    torch.testing.assert_close(model.score_block(state2, u, 0, 8), all_scores)
    chunked = torch.cat([model.score_block(state, u, 0, 3), model.score_block(state, u, 3, 8)], 1)
    torch.testing.assert_close(chunked, all_scores)


def test_support_removes_query_from_every_graph_path(data, config):
    model = build_model(data, config, 'cpu').eval()
    u, pos = torch.tensor([0, 0, 1]), torch.tensor([0, 1, 2])
    edges, indices = model.support_view(u, pos)
    assert indices.tolist() == [0, 2]
    assert set(map(tuple, edges.T.tolist())) == {(0, 1), (1, 3), (2, 1), (2, 4)}
    state = model.encode(edges)
    adj = state.adjacency.to_dense()
    assert adj[0, 0] == 0 and adj[1, 2] == 0
    other = LGMRecOpt(data['n_users'], data['visual'], data['text'], edges, config).eval()
    other.load_state_dict(model.state_dict())
    independent = other.encode()
    torch.testing.assert_close(state.local, independent.local)
    torch.testing.assert_close(state.final, independent.final)
    for a, b in zip(state.slots, independent.slots):
        torch.testing.assert_close(a, b)


def test_degree_one_users_skip_auxiliary(data, config):
    edges = torch.tensor([[0, 1, 2], [0, 2, 4]])
    model = LGMRecOpt(3, data['visual'], data['text'], edges, config)
    support, selected = model.support_view(torch.tensor([0, 1]), torch.tensor([0, 2]))
    assert len(selected) == 0
    torch.testing.assert_close(support, edges)
    loss, details = model.loss(torch.tensor([0]), torch.tensor([0]), torch.tensor([7]))
    assert torch.isfinite(loss) and details['auxiliary'] == 0


def test_negatives_only_exclude_train(data):
    users = np.zeros(300, dtype=np.int64)
    result = sample_negatives(users, data['train_history'], 8, np.random.default_rng(9))
    assert not set(result) & {0, 1}
    assert {2, 3}.issubset(set(result))  # VALID/TEST never filter training samples.
    with pytest.raises(ValueError, match='no TRAIN-unobserved'):
        sample_negatives([0], {0: set(range(8))}, 8, np.random.default_rng(9))


def test_no_new_parameters_or_large_persistent_buffers(data, config):
    baseline = LGMRec(data['n_users'], data['visual'], data['text'], data['train'].T, config)
    optimized = build_model(data, config, 'cpu')
    assert set(baseline.state_dict()) == set(optimized.state_dict())
    assert all('visual' not in k and 'train_' not in k for k in optimized.state_dict())
    assert sum(x.numel() for x in baseline.parameters()) == sum(x.numel() for x in optimized.parameters())


def test_reference_matches_lgmrec_equations(data, config):
    model = LGMRec(data['n_users'], data['visual'], data['text'], data['train'].T, config).eval()
    adj, norm = model.train_adj.to_dense(), model.train_norm.to_dense()
    embedding = torch.cat((model.user_embedding.weight, model.item_embedding.weight))
    collaborative = (embedding + norm @ embedding) / 2
    local, global_parts = collaborative, []
    for features, projection, hp in zip((model.visual, model.text), model.projections, model.hyper_projections):
        logits = features @ hp
        hi = (logits / model.tau).softmax(1)
        hu = (adj @ logits / model.tau).softmax(1)
        item = features @ projection
        user = adj @ item / (adj.sum(1, keepdim=True) + 1e-7)
        modal = norm @ torch.cat((user, item))
        local = local + torch.nn.functional.normalize(modal, dim=1)
        prototype = hi.T @ collaborative[model.n_users:]
        global_parts.append(torch.cat((hu @ prototype, hi @ prototype)))
    expected = local + model.alpha * torch.nn.functional.normalize(sum(global_parts), dim=1)
    torch.testing.assert_close(model.encode().final, expected)


def test_random_structure_preserves_membership_multiset(data, config):
    learned = build_model(data, config, 'cpu').eval()
    shuffled = build_model(data, {**config, 'incidence_mode': 'random'}, 'cpu').eval()
    shuffled.load_state_dict(learned.state_dict())
    a, b = learned.encode(), shuffled.encode()
    for left, right in zip(a.incidence, b.incidence):
        torch.testing.assert_close(left[shuffled.structure_permutation], right)


def test_zero_alpha_recovers_local_scoring(data, config):
    model = build_model(data, {**config, 'alpha': 0.0}, 'cpu').eval()
    state = model.encode()
    users = torch.tensor([0, 1])
    expected = state.local[users] @ state.local[model.n_users:].T
    torch.testing.assert_close(model.score_block(state, users, 0, model.n_items), expected)


def test_evaluation_chunk_size_invariance(data, config):
    model = build_model(data, config, 'cpu')
    first = evaluate(model, data['valid'], data['train_history'], config)
    second = evaluate(model, data['valid'], data['train_history'], {**config, 'item_chunk_size': 8})
    assert first == second
    assert all(0 <= value <= 1 for value in first.values())


def test_validation_only_training_and_frozen_test_once(data, config, tmp_path):
    output = tmp_path / 'experiment'
    status = train_one(data, config, output, 'cpu')
    assert status['state'] == 'complete' and status['test_evaluated'] is False
    assert data['test'] is None and not (output / 'test.json').exists()
    assert (output / 'best.pt').stat().st_size < 100_000
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    expected = max(records, key=lambda row: row['valid'][config['selection_metric']])
    assert status['best_epoch'] == expected['epoch']
    assert status['best_valid'] == expected['valid']
    with pytest.raises(FileExistsError):
        train_one(data, config, output, 'cpu')
    evaluate_frozen(output, 'cpu')
    assert (output / 'test.json').exists()
    assert json.loads((output / 'status.json').read_text())['test_evaluated'] is True
    with pytest.raises(ValueError, match='not-yet-tested'):
        evaluate_frozen(output, 'cpu')


def test_changed_frozen_checkpoint_rejected(data, config, tmp_path):
    output = tmp_path / 'freeze'
    train_one(data, {**config, 'epochs': 1}, output, 'cpu')
    with (output / 'best.pt').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='checkpoint was changed'):
        evaluate_frozen(output, 'cpu')
    assert not (output / 'test.json').exists()


def test_changed_frozen_configuration_rejected(data, config, tmp_path):
    output = tmp_path / 'freeze-config'
    train_one(data, {**config, 'epochs': 1}, output, 'cpu')
    path = output / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['config']['alpha'] += 0.1
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='configuration was changed'):
        evaluate_frozen(output, 'cpu')
    assert not (output / 'test.json').exists()


def test_test_labels_do_not_change_training_inputs(data):
    directory = Path(data['directory'])
    path = directory / 'baby.inter'
    frame = pd.read_csv(path, sep='\t')
    frame.loc[frame.x_label == 2, 'itemID'] = 7
    frame.to_csv(path, sep='\t', index=False)
    changed = load_data(directory, 'baby')
    np.testing.assert_array_equal(data['train'], changed['train'])
    assert data['n_users'] == changed['n_users']
    assert data['valid'] == changed['valid']
    assert changed['test'] is None


def test_overlapping_splits_rejected(data):
    path = Path(data['directory']) / 'baby.inter'
    frame = pd.read_csv(path, sep='\t')
    frame.loc[(frame.userID == 0) & (frame.x_label == 1), 'itemID'] = 0
    frame.to_csv(path, sep='\t', index=False)
    with pytest.raises(ValueError, match='overlapping splits'):
        load_data(path.parent, 'baby')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_micro_training(data, config):
    model = build_model(data, config, 'cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    tensors = [torch.tensor(x, device='cuda') for x in ([0, 1], [0, 2], [6, 7])]
    loss, _ = model.loss(*tensors)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    result = evaluate(model, data['valid'], data['train_history'], config)
    assert all(math_value >= 0 for math_value in result.values())
