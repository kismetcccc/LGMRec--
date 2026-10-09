"""Isolated, TRAIN-only research entry point; legacy main.py is untouched."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch
import yaml

from models.lgmrecopt import LGMRec, LGMRecOpt


ROOT = Path(__file__).resolve().parent
VARIANTS = {
    'baseline': {'model': 'LGMRec'},
    'baseline_no_hcl': {'model': 'LGMRec', 'cl_weight': 0.0},
    'slots': {'model': 'LGMRecOpt', 'cl_weight': 0.0, 'auxiliary': 'none'},
    'slots_hcl': {'model': 'LGMRecOpt', 'auxiliary': 'none'},
    'full': {'model': 'LGMRecOpt', 'cl_weight': 0.0, 'auxiliary': 'role'},
    'fixed': {'model': 'LGMRecOpt', 'cl_weight': 0.0, 'auxiliary': 'none', 'readout': 'fixed'},
    'masked': {'model': 'LGMRecOpt', 'cl_weight': 0.0, 'auxiliary': 'masked_bpr'},
    'uniform': {'model': 'LGMRecOpt', 'cl_weight': 0.0, 'auxiliary': 'uniform_role'},
    'no_global': {'model': 'LGMRecOpt', 'cl_weight': 0.0, 'auxiliary': 'role', 'prototype_weight': 0.0},
    'random_structure': {'model': 'LGMRecOpt', 'cl_weight': 0.0, 'auxiliary': 'role', 'incidence_mode': 'random'},
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def code_fingerprint():
    # Git CRLF/LF conversion does not change the source semantics.
    return {name: hashlib.sha256((ROOT / name).read_text(encoding='utf-8').encode('utf-8')).hexdigest()
            for name in ('run_experiment.py', 'models/lgmrecopt.py')}


def config_fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Sparse CUDA kernels may not be bitwise deterministic; record this policy.
    torch.use_deterministic_algorithms(True, warn_only=True)


def histories(pairs):
    result = {}
    for user, item in pairs:
        result.setdefault(int(user), set()).add(int(item))
    return result


def load_data(directory, dataset, include_test=False):
    directory = Path(directory).resolve()
    paths = [directory / f'{dataset}.inter', directory / 'image_feat.npy', directory / 'text_feat.npy']
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Missing data file: {path}')
    frame = pd.read_csv(paths[0], sep='\t', usecols=['userID', 'itemID', 'x_label'])
    for column in ('userID', 'itemID', 'x_label'):
        if frame[column].isna().any() or not (frame[column] == frame[column].astype(np.int64)).all():
            raise ValueError(f'Invalid integer column: {column}')
    if not frame.x_label.isin([0, 1, 2]).all():
        raise ValueError('Expected x_label: TRAIN=0, validation=1, TEST=2')
    if frame.duplicated(['userID', 'itemID']).any():
        raise ValueError('Duplicate user-item pairs, possibly overlapping splits')
    train = frame.loc[frame.x_label == 0, ['userID', 'itemID']].to_numpy(dtype=np.int64)
    if len(train) == 0:
        raise ValueError('Empty TRAIN split')
    visual, text = [torch.from_numpy(np.load(path, allow_pickle=False)).float() for path in paths[1:]]
    if visual.ndim != 2 or text.ndim != 2 or len(visual) != len(text):
        raise ValueError('Feature row counts do not match')
    if (frame.userID < 0).any() or (frame.itemID < 0).any() or frame.itemID.max() >= len(visual):
        raise ValueError('Interaction IDs do not match feature rows')
    seen_users = set(train[:, 0].tolist())
    def partition(label):
        selected = frame[(frame.x_label == label) & frame.userID.isin(seen_users)]
        return histories(selected[['userID', 'itemID']].to_numpy(dtype=np.int64))
    return {
        'directory': str(directory), 'dataset': dataset, 'train': train,
        'train_history': histories(train), 'valid': partition(1),
        'test': partition(2) if include_test else None,
        'visual': visual, 'text': text, 'n_users': int(train[:, 0].max()) + 1,
        'fingerprint': {path.name: digest(path) for path in paths},
    }


def sample_negatives(users, train_history, n_items, rng):
    """Uniform rejection sampling; NEVER consult validation/TEST positives."""
    result = rng.integers(0, n_items, size=len(users), dtype=np.int64)
    for index, user in enumerate(users):
        blocked = train_history[int(user)]
        if len(blocked) >= n_items:
            raise ValueError(f'User {user} has no TRAIN-unobserved negatives')
        attempts = 0
        while int(result[index]) in blocked:
            attempts += 1
            if attempts == 64:
                available = np.array(sorted(set(range(n_items)) - blocked), dtype=np.int64)
                result[index] = rng.choice(available)
                break
            result[index] = rng.integers(n_items)
    return result


@torch.no_grad()
def evaluate(model, targets, excluded, config):
    if not targets:
        raise ValueError('No warm-start evaluation users')
    model.eval()
    state = model.encode()  # One full-graph encoding per evaluation, not per user batch.
    users = sorted(targets)
    device = model.user_embedding.weight.device
    ks = config['topk']
    take = min(max(ks), model.n_items)
    totals = {f'{metric}@{k}': 0.0 for k in ks for metric in ('recall', 'ndcg')}
    for offset in range(0, len(users), config['eval_batch_size']):
        group = users[offset:offset + config['eval_batch_size']]
        tensor_users = torch.tensor(group, dtype=torch.long, device=device)
        best_scores = torch.empty((len(group), 0), device=device)
        best_ids = torch.empty((len(group), 0), device=device, dtype=torch.long)
        for start in range(0, model.n_items, config['item_chunk_size']):
            end = min(model.n_items, start + config['item_chunk_size'])
            scores = model.score_block(state, tensor_users, start, end)
            if not torch.isfinite(scores).all():
                raise FloatingPointError('Non-finite prediction scores')
            for row, user in enumerate(group):
                blocked = [i - start for i in excluded.get(user, ()) if start <= i < end]
                scores[row, blocked] = -torch.inf
            ids = torch.arange(start, end, device=device).expand(len(group), -1)
            combined_scores = torch.cat((best_scores, scores), dim=1)
            combined_ids = torch.cat((best_ids, ids), dim=1)
            best_scores, chosen = combined_scores.topk(min(take, combined_scores.shape[1]), dim=1)
            best_ids = combined_ids.gather(1, chosen)
        for user, ranking, values in zip(group, best_ids.cpu().tolist(), best_scores.cpu().tolist()):
            truth = targets[user]
            hits = [int(item in truth and math.isfinite(score)) for item, score in zip(ranking, values)]
            for k in ks:
                totals[f'recall@{k}'] += sum(hits[:k]) / len(truth)
                dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits[:k]))
                ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(truth))))
                totals[f'ndcg@{k}'] += dcg / ideal
    return {key: value / len(users) for key, value in totals.items()}


def build_model(data, config, device):
    cls = LGMRec if config['model'] == 'LGMRec' else LGMRecOpt
    return cls(data['n_users'], data['visual'], data['text'], data['train'].T, config).to(device)


def diagnostics(model):
    model.eval()
    with torch.no_grad():
        state = model.encode()
        result = []
        for hi in state.incidence:
            mass = hi.sum(0)
            distribution = mass / mass.sum()
            result.append({'edge_mass': mass.cpu().tolist(),
                           'effective_edges': float(torch.exp(-(distribution * distribution.clamp_min(1e-30).log()).sum())),
                           'mean_assignment_entropy': float(-(hi * hi.clamp_min(1e-30).log()).sum(1).mean())})
        return result


def train_one(data, config, output, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(config['seed'])
    rng = np.random.default_rng(config['seed'])
    manifest = {'protocol': 'train_only', 'config': config, 'data_directory': data['directory'],
                'dataset': data['dataset'], 'fingerprint': data['fingerprint'],
                'code_fingerprint': code_fingerprint(), 'torch': torch.__version__,
                'device': str(device), 'deterministic': 'warn_only; sparse CUDA may vary',
                'test_evaluated': False}
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'status.json', {'state': 'running', 'test_evaluated': False})
    try:
        model = build_model(data, config, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                     weight_decay=config['weight_decay'])
        best, best_epoch, stale, best_metrics = -math.inf, 0, 0, None
        started = time.perf_counter()
        for epoch in range(1, config['epochs'] + 1):
            tick = time.perf_counter()
            model.train()
            order = rng.permutation(len(data['train']))
            loss_sum, batches = 0.0, 0
            components = {}
            for begin in range(0, len(order), config['batch_size']):
                pairs = data['train'][order[begin:begin + config['batch_size']]]
                negative = sample_negatives(pairs[:, 0], data['train_history'], model.n_items, rng)
                u, pos, neg = [torch.as_tensor(x, device=device) for x in (pairs[:, 0], pairs[:, 1], negative)]
                optimizer.zero_grad(set_to_none=True)
                loss, details = model.loss(u, pos, neg)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Non-finite gradients')
                optimizer.step()
                loss_sum += float(loss.detach())
                batches += 1
                for key, value in details.items():
                    components[key] = components.get(key, 0.0) + float(value)
            valid = evaluate(model, data['valid'], data['train_history'], config)
            value = valid[config['selection_metric']]
            improved = value > best  # Full precision, no rounding before selection.
            if improved:
                best, best_epoch, stale, best_metrics = value, epoch, 0, valid
                temporary = output / 'best.pt.tmp'
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                            'config_fingerprint': config_fingerprint(config)}, temporary)
                temporary.replace(output / 'best.pt')
            else:
                stale += 1
            record = {'epoch': epoch, 'loss': loss_sum / batches, 'valid': valid,
                      'components': {k: v / batches for k, v in components.items()},
                      'improved': improved, 'seconds': time.perf_counter() - tick}
            with (output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            print(f"{output.name} epoch={epoch} valid={value:.8f} best={best:.8f}", flush=True)
            if stale >= config['patience']:
                break
        checkpoint = torch.load(output / 'best.pt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        write_json(output / 'diagnostics.json', diagnostics(model))
        status = {'state': 'complete', 'best_epoch': best_epoch, 'best_valid': best_metrics,
                  'epochs_completed': epoch, 'seconds': time.perf_counter() - started,
                  'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'checkpoint_sha256': digest(output / 'best.pt'), 'test_evaluated': False}
        write_json(output / 'status.json', status)
        return status
    except Exception as error:
        write_json(output / 'status.json', {'state': 'failed', 'error': str(error), 'test_evaluated': False})
        (output / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def evaluate_frozen(directory, device, data_override=None):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    status = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
    if status['state'] != 'complete' or status['test_evaluated']:
        raise ValueError('Requires a completed, not-yet-tested frozen run')
    if manifest['protocol'] != 'train_only' or manifest['code_fingerprint'] != code_fingerprint():
        raise ValueError('Protocol/source differs from the frozen training run')
    if digest(directory / 'best.pt') != status['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint was changed')
    data = load_data(data_override or manifest['data_directory'], manifest['dataset'], include_test=True)
    if data['fingerprint'] != manifest['fingerprint']:
        raise ValueError('Frozen data fingerprint mismatch')
    checkpoint = torch.load(directory / 'best.pt', map_location=device, weights_only=True)
    if checkpoint['config_fingerprint'] != config_fingerprint(manifest['config']):
        raise ValueError('Frozen configuration was changed')
    # Exclusive marker remains after failure too: never silently re-test a model.
    with (directory / 'test_evaluation.lock').open('x', encoding='utf-8') as stream:
        stream.write('Final TEST evaluation attempted; do not delete to tune against TEST.\n')
    seed_everything(manifest['config']['seed'])
    model = build_model(data, manifest['config'], device)
    model.load_state_dict(checkpoint['state_dict'])
    excluded = {u: history | data['valid'].get(u, set()) for u, history in data['train_history'].items()}
    result = evaluate(model, data['test'], excluded, manifest['config'])
    write_json(directory / 'test.json', {'epoch': checkpoint['epoch'], 'metrics': result,
                                       'checkpoint_sha256': status['checkpoint_sha256']})
    status['test_evaluated'] = True
    write_json(directory / 'status.json', status)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['baby', 'sports', 'clothing'], default='baby')
    parser.add_argument('--data-dir', type=Path, help='Directory containing .inter and both .npy files')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/model/LGMRecOpt.yaml')
    parser.add_argument('--output', type=Path, help='Fresh experiment directory; never overwritten')
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=['baseline', 'slots', 'full'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[999])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--evaluate-frozen', type=Path)
    parser.add_argument('--confirm-final', action='store_true')
    args = parser.parse_args()
    if args.evaluate_frozen:
        if not args.confirm_final:
            parser.error('Final TEST requires --confirm-final after freezing ALL choices')
        evaluate_frozen(args.evaluate_frozen, args.device, args.data_dir)
        return
    if args.confirm_final:
        parser.error('--confirm-final requires --evaluate-frozen')
    if args.output is None:
        parser.error('--output is required')
    if args.output.exists():
        parser.error('Output already exists. Choose a new experiment name.')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        parser.error('Duplicate seeds/variants are not allowed')
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    for key in ('epochs', 'batch_size', 'patience', 'eval_batch_size', 'item_chunk_size'):
        if not isinstance(config[key], int) or config[key] < 1:
            parser.error(f'{key} must be a positive integer')
    if not config['topk'] or any(not isinstance(k, int) or k < 1 for k in config['topk']):
        parser.error('topk must contain positive integers')
    if config['selection_metric'] not in [f'{m}@{k}' for m in ('recall', 'ndcg') for k in config['topk']]:
        parser.error('selection_metric is not in evaluated metrics')
    data = load_data(args.data_dir or ROOT / 'data' / args.dataset, args.dataset)
    # Reserve once, atomically, after data validation. A later process cannot join it.
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for seed in args.seeds:
        for variant in args.variants:
            settings = {**config, **VARIANTS[variant], 'seed': seed, 'variant': variant}
            status = train_one(data, settings, args.output / f'{variant}-seed{seed}', args.device)
            summary.append({'variant': variant, 'seed': seed, **status})
            write_json(args.output / 'summary.json', summary)
    print(f'Validation-only experiment complete: {args.output.resolve()}')


if __name__ == '__main__':
    main()
