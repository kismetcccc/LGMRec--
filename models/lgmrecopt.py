"""Auditable LGMRec reference and its two-point research extension.

The historical models/lgmrec.py is intentionally untouched. This reference
implements LGMRec's CGE/MGE/GHE equations, with deterministic evaluation and
public sparse APIs. Raw features and TRAIN graphs are nonpersistent buffers.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Encoding:
    local: torch.Tensor
    final: torch.Tensor
    incidence: list
    hyper_nodes: list
    prototypes: list
    modal_items: list
    adjacency: torch.Tensor


class LGMRec(nn.Module):
    """Transparent reference; input edges MUST contain TRAIN interactions only."""

    def __init__(self, n_users, visual, text, train_edges, config):
        super().__init__()
        self.config = dict(config)
        self.n_users = int(n_users)
        if visual.ndim != 2 or text.ndim != 2 or len(visual) != len(text):
            raise ValueError("Visual/text features must have the same item rows")
        if not torch.isfinite(visual).all() or not torch.isfinite(text).all():
            raise ValueError("Non-finite features")
        self.n_items = len(visual)
        self.d = int(config['embedding_size'])
        self.k = int(config['hyper_num'])
        self.ui_layers = int(config['n_ui_layers'])
        self.mm_layers = int(config['n_mm_layers'])
        self.hg_layers = int(config['n_hyper_layer'])
        self.alpha = float(config['alpha'])
        self.tau = float(config['assignment_temperature'])
        if min(self.d, self.k, self.hg_layers, self.n_users, self.n_items) < 1:
            raise ValueError("Invalid dimensions/layer count")
        if min(self.ui_layers, self.mm_layers) < 0 or self.tau <= 0:
            raise ValueError("Invalid layers/assignment temperature")
        self.register_buffer('visual', visual.float(), persistent=False)
        self.register_buffer('text', text.float(), persistent=False)
        edges = torch.as_tensor(train_edges, dtype=torch.long)
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] == 0:
            raise ValueError("TRAIN edges must have shape [2, E], E > 0")
        if edges.min() < 0 or edges[0].max() >= n_users or edges[1].max() >= self.n_items:
            raise ValueError("TRAIN edge ID out of range")
        keys = edges[0] * self.n_items + edges[1]
        keys = torch.unique(keys, sorted=True)
        edges = torch.stack((keys // self.n_items, keys % self.n_items))
        self.train_degrees = torch.bincount(edges[0], minlength=self.n_users).cpu().tolist()
        self.register_buffer('train_edges', edges, persistent=False)
        adj, norm = self._graphs(edges)
        self.register_buffer('train_adj', adj, persistent=False)
        self.register_buffer('train_norm', norm, persistent=False)
        self.user_embedding = nn.Embedding(n_users, self.d)
        self.item_embedding = nn.Embedding(self.n_items, self.d)
        self.projections = nn.ParameterList([
            nn.Parameter(torch.empty(visual.shape[1], self.d)),
            nn.Parameter(torch.empty(text.shape[1], self.d)),
        ])
        self.hyper_projections = nn.ParameterList([
            nn.Parameter(torch.empty(visual.shape[1], self.k)),
            nn.Parameter(torch.empty(text.shape[1], self.k)),
        ])
        for parameter in self.parameters():
            nn.init.xavier_uniform_(parameter)
        keep = float(config['keep_rate'])
        if not 0 < keep <= 1:
            raise ValueError("keep_rate must be in (0, 1]")
        self.dropout = nn.Dropout(1 - keep)

    def _graphs(self, edges):
        u, i = edges
        values = torch.ones(len(u), device=edges.device)
        adj = torch.sparse_coo_tensor(edges, values, (self.n_users, self.n_items)).coalesce()
        row = torch.cat((u, i + self.n_users))
        col = torch.cat((i + self.n_users, u))
        degree = torch.bincount(row, minlength=self.n_users + self.n_items).float()
        weight = (degree[row] + 1e-7).rsqrt() * (degree[col] + 1e-7).rsqrt()
        norm = torch.sparse_coo_tensor(torch.stack((row, col)), weight,
                                      (len(degree), len(degree))).coalesce()
        return adj, norm

    def _assignment(self, logits):
        if self.training:
            return F.gumbel_softmax(logits, tau=self.tau, dim=1)
        return F.softmax(logits / self.tau, dim=1)

    def hyper_logits(self, feature, projection):
        return feature @ projection

    def encode(self, edges=None):
        adj, norm = (self.train_adj, self.train_norm) if edges is None else self._graphs(edges)
        # Both CGE and MGE use this SAME graph, including auxiliary support views.
        ego = torch.cat((self.user_embedding.weight, self.item_embedding.weight))
        layers = [ego]
        for _ in range(self.ui_layers):
            ego = torch.sparse.mm(norm, ego)
            layers.append(ego)
        collaborative = torch.stack(layers).mean(0)
        degree = torch.sparse.sum(adj, dim=1).to_dense().unsqueeze(1)
        local = collaborative
        incidences, hyper_nodes, prototypes, modal_items = [], [], [], []
        for feature, projection, hyper_projection in zip(
                (self.visual, self.text), self.projections, self.hyper_projections):
            logits = self.hyper_logits(feature, hyper_projection)
            hi = self._assignment(logits)
            hu = self._assignment(torch.sparse.mm(adj, logits))
            projected = feature @ projection
            user_modal = torch.sparse.mm(adj, projected) / (degree + 1e-7)
            modal = torch.cat((user_modal, projected))
            for _ in range(self.mm_layers):
                modal = torch.sparse.mm(norm, modal)
            local = local + F.normalize(modal, dim=1)
            ih, uh = self.dropout(hi), self.dropout(hu)
            item_global = collaborative[self.n_users:]
            for _ in range(self.hg_layers):
                prototype = ih.T @ item_global
                item_global = ih @ prototype
                user_global = uh @ prototype
            incidences.append(hi)
            hyper_nodes.append(torch.cat((user_global, item_global)))
            prototypes.append(prototype)
            modal_items.append(F.normalize(collaborative[self.n_users:] +
                                           F.normalize(modal[self.n_users:], dim=1), dim=1))
        final = local + self.alpha * F.normalize(sum(hyper_nodes), dim=1)
        return Encoding(local, final, incidences, hyper_nodes, prototypes, modal_items, adj)

    def pair_score(self, state, users, items):
        return (state.final[users] * state.final[self.n_users + items]).sum(-1)

    def score_block(self, state, users, start, end):
        return state.final[users] @ state.final[self.n_users + start:self.n_users + end].T

    def _regularizer(self, state, users, positives, negatives):
        return sum(torch.linalg.vector_norm(x) for x in (
            state.final[users], state.final[self.n_users + positives],
            state.final[self.n_users + negatives])) / len(users)

    def _hcl(self, state, users, positives):
        a, b = (F.normalize(x, dim=1) for x in state.hyper_nodes)
        indices = torch.cat((users, self.n_users + positives))
        # Separate user and item denominators, as in LGMRec.
        loss = a.sum() * 0
        for idx, pool in ((indices[:len(users)], b[:self.n_users]),
                          (indices[len(users):], b[self.n_users:])):
            logits = a[idx] @ pool.T / self.tau
            match = (a[idx] * b[idx]).sum(-1) / self.tau
            loss = loss + (torch.logsumexp(logits, dim=1) - match).sum()
        return loss

    def loss(self, users, positives, negatives):
        state = self.encode()
        bpr = F.softplus(self.pair_score(state, users, negatives) -
                         self.pair_score(state, users, positives)).mean()
        hcl = self._hcl(state, users, positives) if self.config['cl_weight'] else bpr * 0
        total = bpr + self.config['cl_weight'] * hcl
        total = total + self.config['reg_weight'] * self._regularizer(state, users, positives, negatives)
        return total, {'bpr': bpr.detach(), 'hcl': hcl.detach()}


class LGMRecOpt(LGMRec):
    """LGMRec extension: candidate-conditioned slots + held-out role ranking.

    No legacy behavior residual, hyperedge smoothing, new feature encoder,
    hard-negative sampler, or cross-modal slot-index alignment is used.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_temperature = float(self.config['score_temperature'])
        if self.score_temperature <= 0 or self.config['prototype_weight'] < 0:
            raise ValueError("Invalid slot settings")
        if self.config['readout'] not in ('candidate', 'fixed'):
            raise ValueError("readout must be candidate or fixed")
        if self.config['auxiliary'] not in ('none', 'role', 'masked_bpr', 'uniform_role'):
            raise ValueError("Unknown auxiliary objective")
        if self.config.get('incidence_mode', 'learned') not in ('learned', 'random'):
            raise ValueError("Unknown incidence mode")
        generator = torch.Generator().manual_seed(int(self.config.get('seed', 999)) + 131)
        self.register_buffer('structure_permutation', torch.randperm(self.n_items, generator=generator),
                             persistent=False)

    def hyper_logits(self, feature, projection):
        logits = super().hyper_logits(feature, projection)
        if self.config.get('incidence_mode', 'learned') == 'random':
            logits = logits[self.structure_permutation]
        return logits

    def _regularizer(self, state, users, positives, negatives):
        # Regularize the local representations actually used for scoring, not
        # the discarded, fixed-fusion GHE head. Slot directions are unit norm.
        return sum(torch.linalg.vector_norm(x) for x in (
            state.local[users], state.local[self.n_users + positives],
            state.local[self.n_users + negatives])) / len(users)

    def encode(self, edges=None):
        state = super().encode(edges)
        state.slots, state.priors = [], []
        for hi, items, prototype in zip(state.incidence, state.modal_items, state.prototypes):
            mass = torch.sparse.mm(state.adjacency, hi)
            history = torch.stack([
                torch.sparse.mm(state.adjacency, hi[:, k:k + 1] * items)
                for k in range(self.k)], dim=1)
            personal = history / mass.clamp_min(1e-12).unsqueeze(-1)
            global_part = self.config['prototype_weight'] * F.normalize(prototype, dim=1)
            state.slots.append(F.normalize(personal + global_part.unsqueeze(0), dim=-1))
            # Empty histories contribute no hypergraph score, not a random prior.
            state.priors.append(mass / mass.sum(1, keepdim=True).clamp_min(1e-12))
        return state

    def _aggregate(self, scores, priors):
        if self.config['readout'] == 'fixed':
            return (scores * priors).sum(-1)
        value = self.score_temperature * torch.logsumexp(
            scores / self.score_temperature + priors.clamp_min(1e-30).log(), dim=-1)
        return torch.where(priors.sum(-1) > 0, value, torch.zeros_like(value))

    def pair_score(self, state, users, items):
        result = (state.local[users] * state.local[self.n_users + items]).sum(-1)
        for slots, prior, item in zip(state.slots, state.priors, state.modal_items):
            per_slot = (slots[users] * item[items].unsqueeze(1)).sum(-1)
            result = result + self.alpha * self._aggregate(per_slot, prior[users])
        return result

    def score_block(self, state, users, start, end):
        result = state.local[users] @ state.local[self.n_users + start:self.n_users + end].T
        # Online reduction: never materialize [batch, all_items, K, d].
        for slots, prior, item in zip(state.slots, state.priors, state.modal_items):
            accumulated = None
            for k in range(self.k):
                score = slots[users, k] @ item[start:end].T
                weight = prior[users, k:k + 1]
                if self.config['readout'] == 'fixed':
                    term = score * weight
                    accumulated = term if accumulated is None else accumulated + term
                else:
                    term = score / self.score_temperature + weight.clamp_min(1e-30).log()
                    accumulated = term if accumulated is None else torch.logaddexp(accumulated, term)
            if self.config['readout'] == 'candidate':
                accumulated = accumulated * self.score_temperature
            accumulated = torch.where(prior[users].sum(-1, keepdim=True) > 0,
                                      accumulated, torch.zeros_like(accumulated))
            result = result + self.alpha * accumulated
        return result

    def support_view(self, users, positives):
        """One query per eligible user; remove ALL its graph occurrences.

        The selection depends only on the training batch and training degrees.
        Degree-one users are skipped so every selected user keeps support.
        """
        selected = []
        seen = set()
        for index, user in enumerate(users.detach().cpu().tolist()):
            if user not in seen and self.train_degrees[user] >= 2:
                selected.append(index)
                seen.add(user)
        indices = torch.tensor(selected, dtype=torch.long, device=users.device)
        if not len(indices):
            return self.train_edges, indices
        query_keys = users[indices] * self.n_items + positives[indices]
        train_keys = self.train_edges[0] * self.n_items + self.train_edges[1]
        if not torch.isin(query_keys, train_keys).all():
            raise ValueError("Auxiliary queries must be TRAIN positives")
        return self.train_edges[:, ~torch.isin(train_keys, query_keys)], indices

    def loss(self, users, positives, negatives):
        total, metrics = super().loss(users, positives, negatives)
        kind = self.config['auxiliary']
        if kind == 'none' or self.config['role_weight'] == 0:
            return total, metrics
        support_edges, selected = self.support_view(users, positives)
        if not len(selected):
            metrics['auxiliary'] = total.detach() * 0
            return total, metrics
        u, pos, neg = users[selected], positives[selected], negatives[selected]
        # Recompute CGE, MGE, memberships and prototypes on the SUPPORT graph.
        state = self.encode(support_edges)
        if kind == 'masked_bpr':
            auxiliary = F.softplus(self.pair_score(state, u, neg) -
                                   self.pair_score(state, u, pos)).mean()
        else:
            terms = []
            for slots, prior, items, hi in zip(state.slots, state.priors,
                                               state.modal_items, state.incidence):
                responsibility = (prior[u] * hi[pos]).detach()
                if kind == 'uniform_role':
                    responsibility = torch.ones_like(responsibility)
                responsibility = responsibility / responsibility.sum(-1, keepdim=True).clamp_min(1e-12)
                positive = (slots[u] * items[pos].unsqueeze(1)).sum(-1)
                negative = (slots[u] * items[neg].unsqueeze(1)).sum(-1)
                terms.append((responsibility * F.softplus(negative - positive)).sum(-1).mean())
            auxiliary = torch.stack(terms).mean()
        total = total + self.config['role_weight'] * auxiliary
        metrics['auxiliary'] = auxiliary.detach()
        metrics['query_users'] = torch.tensor(len(selected), device=users.device)
        return total, metrics
