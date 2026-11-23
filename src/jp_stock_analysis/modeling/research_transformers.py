"""Research-only PyTorch models (TFT / GRU / cross-sectional transformer).

Status: **research / validation only**, behind an explicit ON/OFF switch. The
adopted model is the OOS tree ensemble (``ml_models.train_primary_model``);
nothing here drives ranking, portfolio, constraints or monitoring. The purpose
is to measure deep sequence models against the adopted model on identical,
leak-free terms.

Switch: ``TransformerResearchConfig(enabled=False)`` by default. OFF => nothing
runs and ``torch`` is not imported. ON without torch => every model reports
``optional_dependency_missing`` (install the ``research-transformers`` extra).

Models
------
``tft`` — Temporal Fusion Transformer (Lim, Arık, Loeff & Pfister, 2021), with
the paper's components:

- *static covariate encoder*: the decision-date cross-sectional feature vector
  (every dataset feature as a within-date rank) + a learned sector embedding,
  passed through a static variable-selection network (VSN) into four context
  vectors (selection, enrichment, LSTM h0, LSTM c0);
- *past observed inputs*: the last ``seq_len`` **daily** bars of the ticker
  (log return, overnight gap, intraday return, log range, candle body ratio,
  log traded-value ratio, equal-weighted market return, presence flag) through
  a context-conditioned VSN and an LSTM encoder;
- *known future inputs*: one decoder step per target horizon carrying
  ``h / h_max``, "a scheduled earnings date falls inside the horizon",
  calendar month (sin/cos) — through a VSN and an LSTM decoder;
- gated skip connections + LayerNorm, static enrichment (GRN with context),
  *interpretable multi-head attention* (per-head Q/K, shared V, head-averaged),
  position-wise GRN and a **quantile** head (0.1 / 0.5 / 0.9) per horizon.

``gru`` — the neural baseline (GRU over the same daily sequence + static
encoder). A transformer has to beat this to justify its complexity.

``cross_sectional_transformer`` — self-attention across all stocks of one
decision date over the static feature vectors (cross-asset interaction).

Training / validation
---------------------
- purged walk-forward OOS (``out_of_sample.walk_forward_oos``) with
  ``purge_horizon = max(target_horizons)``;
- targets: within-date centred ranks of the forward return per horizon
  (missing labels masked);
- loss: pinball loss (quantiles) + ``(1 - per-date Pearson corr)`` of the
  median — a differentiable Rank-IC surrogate;
- inner early stopping on the last training date(s), purged of rows whose
  label window reaches the validation period; best epoch is restored;
- ``n_seeds`` independent seeds averaged (variance reduction on noisy data);
- daily-channel standardisation fitted on the training rows only;
- CPU: bit-deterministic. CUDA (``device="cuda"``): bf16 autocast optional,
  deterministic algorithms requested in warn-only mode (cuDNN RNNs may still
  differ in the last bits — keep golden baselines on CPU).

The score of a row is the predicted median at the evaluation horizon.
"""

from __future__ import annotations

import bisect
import importlib.util
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from jp_stock_analysis.modeling.dataset import ModelingDataset, ModelingObservation
from jp_stock_analysis.modeling.ensemble import rank_average_ensemble
from jp_stock_analysis.modeling.out_of_sample import (
    VALIDATION_SCHEME,
    _centered_ranks,
    label_end_index_for,
    ranked_feature_rows,
    walk_forward_oos,
)
from jp_stock_analysis.modeling.ranking_metrics import (
    ScoredObservation,
    evaluate_horizon,
    spearman,
)
from jp_stock_analysis.modeling.trading_features import (
    adjusted_ohlc,
    equal_weight_market_returns,
)
from jp_stock_analysis.schemas import PriceBar

MODEL_TFT = "tft"
MODEL_GRU = "gru"
MODEL_CS_TRANSFORMER = "cross_sectional_transformer"
RESEARCH_MODELS = (MODEL_TFT, MODEL_GRU, MODEL_CS_TRANSFORMER)

STATUS_DISABLED = "disabled"
STATUS_TRAINED = "trained"
STATUS_MISSING_DEPENDENCY = "optional_dependency_missing"
STATUS_INSUFFICIENT_DATA = "insufficient_data"

QUANTILES = (0.1, 0.5, 0.9)
DAILY_CHANNELS = (
    "log_return", "overnight_gap", "intraday_return", "log_range", "body_ratio",
    "log_traded_value_ratio", "market_return", "row_present",
)
KNOWN_FUTURE_INPUTS = (
    "horizon_fraction", "earnings_in_horizon", "earnings_date_known", "month_sin", "month_cos",
)

RESEARCH_DISCLAIMER = (
    "This output is for analytical and self-directed research purposes. It is not "
    "personalized financial advice. RESEARCH-ONLY deep models: they do not drive "
    "any ranking, portfolio or signal, and make no predictive claim."
)


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


@dataclass(frozen=True)
class TransformerResearchConfig:
    """ON/OFF switch and hyper-parameters. See :meth:`preset` for named setups."""

    enabled: bool = False
    models: tuple[str, ...] = RESEARCH_MODELS
    target_horizons: tuple[int, ...] | None = None  # None -> dataset horizons
    seq_len: int = 40
    d_model: int = 32
    n_heads: int = 4
    dropout: float = 0.1
    epochs: int = 40
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    n_seeds: int = 1
    seed: int = 0
    val_periods: int = 1
    min_train_rows: int = 6
    ic_loss_weight: float = 1.0
    refit_every: int = 1
    device: str = "cpu"  # "cpu" | "cuda" | "auto"
    amp_bf16: bool = False

    def __post_init__(self) -> None:
        unknown = [m for m in self.models if m not in RESEARCH_MODELS]
        if unknown:
            raise ValueError(f"unknown research model(s) {unknown}; expected {RESEARCH_MODELS}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.seq_len < 2 or self.epochs < 1 or self.n_seeds < 1:
            raise ValueError("seq_len >= 2, epochs >= 1 and n_seeds >= 1 are required")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda or auto")

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> TransformerResearchConfig:
        """Named setups: ``cpu_test`` (tests/CI), ``cpu_small``, ``gpu_4060`` (8 GB)."""
        presets: dict[str, dict[str, Any]] = {
            "cpu_test": dict(seq_len=20, d_model=16, n_heads=2, epochs=6, patience=3,
                             batch_size=128),
            "cpu_small": {},
            "gpu_4060": dict(seq_len=60, d_model=64, n_heads=4, dropout=0.2, epochs=100,
                             patience=10, batch_size=256, n_seeds=5, device="cuda",
                             amp_bf16=True),
        }
        if name not in presets:
            raise ValueError(f"unknown preset {name!r}; expected {sorted(presets)}")
        return cls(**{**presets[name], **overrides})

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class ResearchModelResult:
    model_type: str
    status: str
    horizon: int
    scored: list[ScoredObservation] = field(default_factory=list)
    message: str = ""
    oos_folds: list[dict[str, Any]] = field(default_factory=list)
    comparison: dict[str, Any] = field(default_factory=dict)
    interpretation: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    validation: str = VALIDATION_SCHEME
    disclaimer: str = RESEARCH_DISCLAIMER

    @property
    def is_trained(self) -> bool:
        return self.status == STATUS_TRAINED

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "status": self.status,
            "horizon": self.horizon,
            "research_only": True,
            "validation": self.validation,
            "message": self.message,
            "scored_count": len(self.scored),
            "oos_folds": self.oos_folds,
            "training": self.training,
            "comparison_vs_reference": self.comparison,
            "interpretation": self.interpretation,
            "disclaimer": self.disclaimer,
        }


# --------------------------------------------------------------------------- #
# Input construction (all as-of the decision date; no look-ahead)
# --------------------------------------------------------------------------- #
class _InputBuilder:
    """Static rank vectors, daily past sequences and known-future inputs."""

    def __init__(
        self,
        dataset: ModelingDataset,
        prices: Mapping[str, Sequence[PriceBar]],
        seq_len: int,
        target_horizons: Sequence[int],
    ) -> None:
        self.dataset = dataset
        self.seq_len = seq_len
        self.target_horizons = tuple(target_horizons)
        self.feature_names = tuple(dataset.feature_names)
        self.static: dict[tuple[str, date], list[float]] = {
            key: [0.0 if v is None else float(v) for v in row]
            for key, row in ranked_feature_rows(dataset, self.feature_names).items()
        }
        sectors = sorted({o.sector or "" for o in dataset.observations})
        self.sector_index = {s: i for i, s in enumerate(sectors)}
        market = equal_weight_market_returns(prices)
        self.daily: dict[str, tuple[list[date], list[list[float]]]] = {}
        for ticker, bars in prices.items():
            ohlc = adjusted_ohlc(bars)
            dates: list[date] = []
            channels: list[list[float]] = []
            values: list[float] = []
            for i, cur in enumerate(ohlc):
                values.append(cur.raw_close * (cur.volume or 0.0))
                if i == 0 or not ohlc[i - 1].close or not cur.close:
                    continue
                prev = ohlc[i - 1]
                has_ohlc = cur.open and cur.high and cur.low and cur.high > cur.low
                window = values[max(0, len(values) - 20):]
                mean_value = sum(window) / len(window)
                channels.append([
                    math.log(cur.close / prev.close),
                    math.log(cur.open / prev.close) if has_ohlc else 0.0,  # type: ignore[operator]
                    math.log(cur.close / cur.open) if has_ohlc else 0.0,  # type: ignore[operator]
                    math.log(cur.high / cur.low) if has_ohlc else 0.0,  # type: ignore[operator]
                    abs(cur.close - cur.open) / (cur.high - cur.low) if has_ohlc else 0.0,  # type: ignore[operator]
                    math.log(values[-1] / mean_value) if (values[-1] > 0 and mean_value > 0)
                    else 0.0,
                    market.get(cur.date, 0.0),
                    1.0,
                ])
                dates.append(cur.date)
            self.daily[ticker] = (dates, channels)

    @property
    def n_static(self) -> int:
        return len(self.feature_names)

    def static_vector(self, obs: ModelingObservation) -> list[float]:
        return self.static.get((obs.ticker, obs.decision_date), [0.0] * self.n_static)

    def sector_id(self, obs: ModelingObservation) -> int:
        return self.sector_index.get(obs.sector or "", 0)

    def past_sequence(self, obs: ModelingObservation) -> list[list[float]]:
        """Last ``seq_len`` daily channel rows with date <= decision date (zero-padded)."""
        dates, channels = self.daily.get(obs.ticker, ([], []))
        end = bisect.bisect_right(dates, obs.decision_date)
        window = channels[max(0, end - self.seq_len):end]
        pad = [[0.0] * len(DAILY_CHANNELS)] * (self.seq_len - len(window))
        return pad + window

    def known_future(self, obs: ModelingObservation) -> list[list[float]]:
        h_max = max(self.target_horizons)
        days_to = obs.features.get("days_to_next_earnings")
        month = obs.decision_date.month
        out: list[list[float]] = []
        for h in self.target_horizons:
            in_window = 1.0 if (days_to is not None and days_to * 5 / 7 <= h) else 0.0
            out.append([
                h / h_max,
                in_window,
                0.0 if days_to is None else 1.0,
                math.sin(2 * math.pi * month / 12),
                math.cos(2 * math.pi * month / 12),
            ])
        return out


def _rank_targets(
    rows: Sequence[ModelingObservation], horizons: Sequence[int]
) -> list[list[float | None]]:
    """Per-date centred rank of each horizon's forward return (None = missing)."""
    out: list[list[float | None]] = [[None] * len(horizons) for _ in rows]
    by_date: dict[date, list[int]] = {}
    for i, obs in enumerate(rows):
        by_date.setdefault(obs.decision_date, []).append(i)
    for indices in by_date.values():
        for j, h in enumerate(horizons):
            ranked = _centered_ranks([rows[i].labels.get(f"forward_return_h{h}") for i in indices])
            for pos, i in enumerate(indices):
                out[i][j] = ranked[pos]
    return out


# --------------------------------------------------------------------------- #
# Torch modules (lazy)
# --------------------------------------------------------------------------- #
def _torch_modules(torch):  # pragma: no cover - exercised only with torch installed
    nn = torch.nn
    F = nn.functional

    class GRN(nn.Module):
        """Gated residual network with optional context (TFT eq. 2-4)."""

        def __init__(self, d_in, d_hidden, d_out, dropout, d_context=None):
            super().__init__()
            self.fc1 = nn.Linear(d_in, d_hidden)
            self.ctx = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
            self.fc2 = nn.Linear(d_hidden, d_out)
            self.gate = nn.Linear(d_out, 2 * d_out)
            self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
            self.drop = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(d_out)

        def forward(self, x, context=None):
            h = self.fc1(x)
            if self.ctx is not None and context is not None:
                h = h + self.ctx(context)
            h = self.drop(self.fc2(F.elu(h)))
            return self.norm(self.skip(x) + F.glu(self.gate(h), dim=-1))

    class VSN(nn.Module):
        """Variable selection network over scalar inputs [..., V] -> [..., d]."""

        def __init__(self, n_vars, d, dropout, d_context=None):
            super().__init__()
            self.embed = nn.ModuleList([nn.Linear(1, d) for _ in range(n_vars)])
            self.var_grn = nn.ModuleList([GRN(d, d, d, dropout) for _ in range(n_vars)])
            self.select = GRN(n_vars * d, d, n_vars, dropout, d_context)

        def forward(self, x, context=None):
            emb = [self.embed[j](x[..., j : j + 1]) for j in range(len(self.embed))]
            flat = torch.cat(emb, dim=-1)
            if context is not None:
                while context.dim() < flat.dim():
                    context = context.unsqueeze(1)
                context = context.expand(*flat.shape[:-1], context.shape[-1])
            weights = torch.softmax(self.select(flat, context), dim=-1)
            processed = torch.stack([g(e) for g, e in zip(self.var_grn, emb, strict=True)], -2)
            return (weights.unsqueeze(-1) * processed).sum(-2), weights

    class InterpretableMHA(nn.Module):
        """Per-head Q/K, shared V, head-averaged output (TFT sec. 4.4)."""

        def __init__(self, d, n_heads, dropout):
            super().__init__()
            self.n_heads, self.d_head = n_heads, d // n_heads
            self.q = nn.ModuleList([nn.Linear(d, self.d_head) for _ in range(n_heads)])
            self.k = nn.ModuleList([nn.Linear(d, self.d_head) for _ in range(n_heads)])
            self.v = nn.Linear(d, self.d_head)
            self.out = nn.Linear(self.d_head, d)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, mask):
            v = self.v(x)
            heads, attn = [], []
            for q_l, k_l in zip(self.q, self.k, strict=True):
                scores = q_l(x) @ k_l(x).transpose(-1, -2) / math.sqrt(self.d_head)
                weights = self.drop(torch.softmax(scores.masked_fill(mask, -1e9), dim=-1))
                heads.append(weights @ v)
                attn.append(weights)
            return self.out(torch.stack(heads).mean(0)), torch.stack(attn).mean(0)

    class TFT(nn.Module):
        def __init__(self, n_static, n_sectors, n_past, n_known, n_horizons, d, n_heads,
                     dropout):
            super().__init__()
            self.sector_emb = nn.Embedding(max(n_sectors, 1), 1)
            self.static_vsn = VSN(n_static + 1, d, dropout)
            self.ctx = nn.ModuleList([GRN(d, d, d, dropout) for _ in range(4)])
            self.past_vsn = VSN(n_past, d, dropout, d_context=d)
            self.known_vsn = VSN(n_known, d, dropout, d_context=d)
            self.encoder = nn.LSTM(d, d, batch_first=True)
            self.decoder = nn.LSTM(d, d, batch_first=True)
            self.lstm_gate = nn.Linear(d, 2 * d)
            self.lstm_norm = nn.LayerNorm(d)
            self.enrich = GRN(d, d, d, dropout, d_context=d)
            self.attn = InterpretableMHA(d, n_heads, dropout)
            self.attn_gate = nn.Linear(d, 2 * d)
            self.attn_norm = nn.LayerNorm(d)
            self.ff = GRN(d, d, d, dropout)
            self.out_gate = nn.Linear(d, 2 * d)
            self.out_norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, len(QUANTILES))
            self.n_horizons = n_horizons
            self.last_weights: dict[str, Any] = {}

        def forward(self, static, sector, past, known):
            s_in = torch.cat([static, self.sector_emb(sector)], dim=-1)
            s_emb, s_w = self.static_vsn(s_in)
            c_sel, c_enr, c_h, c_c = (g(s_emb) for g in self.ctx)
            past_e, p_w = self.past_vsn(past, c_sel)
            known_e, k_w = self.known_vsn(known, c_sel)
            enc, state = self.encoder(past_e, (c_h.unsqueeze(0), c_c.unsqueeze(0)))
            dec, _ = self.decoder(known_e, state)
            lstm_out = torch.cat([enc, dec], dim=1)
            skip_in = torch.cat([past_e, known_e], dim=1)
            temporal = self.lstm_norm(skip_in + F.glu(self.lstm_gate(lstm_out), dim=-1))
            enriched = self.enrich(temporal, c_enr.unsqueeze(1).expand_as(temporal))
            total = enriched.shape[1]
            causal = torch.triu(
                torch.ones(total, total, dtype=torch.bool, device=enriched.device), diagonal=1
            )
            att, att_w = self.attn(enriched, causal)
            att = self.attn_norm(enriched + F.glu(self.attn_gate(att), dim=-1))
            ff = self.ff(att)
            out = self.out_norm(temporal + F.glu(self.out_gate(ff), dim=-1))
            self.last_weights = {"static": s_w.detach(), "past": p_w.detach()}
            return self.head(out[:, -self.n_horizons :, :])  # [B, H, Q]

    class GRUBaseline(nn.Module):
        def __init__(self, n_static, n_sectors, n_past, n_horizons, d, dropout):
            super().__init__()
            self.sector_emb = nn.Embedding(max(n_sectors, 1), 4)
            self.static = nn.Sequential(nn.Linear(n_static + 4, d), nn.ELU(), nn.Dropout(dropout))
            self.gru = nn.GRU(n_past, d, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(2 * d, d), nn.ELU(), nn.Dropout(dropout),
                nn.Linear(d, n_horizons * len(QUANTILES)),
            )
            self.n_horizons = n_horizons

        def forward(self, static, sector, past, known):
            s = self.static(torch.cat([static, self.sector_emb(sector)], dim=-1))
            _, h = self.gru(past)
            out = self.head(torch.cat([h[-1], s], dim=-1))
            return out.view(-1, self.n_horizons, len(QUANTILES))

    class CrossSectionalTransformer(nn.Module):
        def __init__(self, n_static, n_sectors, n_horizons, d, n_heads, dropout):
            super().__init__()
            self.sector_emb = nn.Embedding(max(n_sectors, 1), 4)
            self.embed = nn.Linear(n_static + 4, d)
            layer = nn.TransformerEncoderLayer(
                d, n_heads, dim_feedforward=2 * d, dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            self.head = nn.Linear(d, n_horizons * len(QUANTILES))
            self.n_horizons = n_horizons

        def forward(self, static, sector, past, known):
            x = self.embed(torch.cat([static, self.sector_emb(sector)], dim=-1)).unsqueeze(0)
            return self.head(self.encoder(x).squeeze(0)).view(-1, self.n_horizons, len(QUANTILES))

    return TFT, GRUBaseline, CrossSectionalTransformer


def _loss(torch, pred, target, mask, ic_weight):  # pragma: no cover - torch only
    """Pinball loss over quantiles + (1 - corr) of the median, masked per horizon."""
    q = torch.tensor(QUANTILES, device=pred.device, dtype=pred.dtype)
    err = target.unsqueeze(-1) - pred  # [B, H, Q]
    pinball = torch.maximum(q * err, (q - 1) * err)
    m = mask.unsqueeze(-1)
    total = (pinball * m).sum() / (m.sum() * len(QUANTILES) + 1e-8)
    median = pred[..., QUANTILES.index(0.5)]
    for j in range(pred.shape[1]):
        sel = mask[:, j] > 0
        if int(sel.sum()) >= 3:
            p, t = median[sel, j], target[sel, j]
            pc, tc = p - p.mean(), t - t.mean()
            corr = (pc * tc).sum() / (torch.sqrt((pc**2).sum() * (tc**2).sum()) + 1e-8)
            total = total + ic_weight * (1.0 - corr) / pred.shape[1]
    return total


# --------------------------------------------------------------------------- #
# Fit / predict for one walk-forward fold
# --------------------------------------------------------------------------- #
def _make_fit_predict(model_type, builder, config, prices, eval_index, log):  # pragma: no cover
    import torch

    tft_cls, gru_cls, cs_cls = _torch_modules(torch)
    horizons = builder.target_horizons
    ends = label_end_index_for(builder.dataset, prices)
    device = torch.device(
        "cuda"
        if (config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )
    use_amp = config.amp_bf16 and device.type == "cuda"

    def tensors(rows, mean=None, std=None):
        static = torch.tensor([builder.static_vector(o) for o in rows], dtype=torch.float32)
        sector = torch.tensor([builder.sector_id(o) for o in rows], dtype=torch.long)
        past = torch.tensor([builder.past_sequence(o) for o in rows], dtype=torch.float32)
        if mean is not None:
            past = (past - mean) / std
        known = torch.tensor([builder.known_future(o) for o in rows], dtype=torch.float32)
        return [t.to(device) for t in (static, sector, past, known)]

    def build_model():
        if model_type == MODEL_TFT:
            return tft_cls(builder.n_static, len(builder.sector_index), len(DAILY_CHANNELS),
                           len(KNOWN_FUTURE_INPUTS), len(horizons), config.d_model,
                           config.n_heads, config.dropout)
        if model_type == MODEL_GRU:
            return gru_cls(builder.n_static, len(builder.sector_index), len(DAILY_CHANNELS),
                           len(horizons), config.d_model, config.dropout)
        return cs_cls(builder.n_static, len(builder.sector_index), len(horizons),
                      config.d_model, config.n_heads, config.dropout)

    def batches(rows):
        groups: dict[date, list[int]] = {}
        for i, o in enumerate(rows):
            groups.setdefault(o.decision_date, []).append(i)
        for d in sorted(groups):
            idx = groups[d]
            step = len(idx) if model_type == MODEL_CS_TRANSFORMER else config.batch_size
            for start in range(0, len(idx), step):
                yield idx[start : start + step]

    def predict(model, rows, mean, std):
        model.eval()
        out = [0.0] * len(rows)
        weights_static, weights_past = [], []
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            for idx in batches(rows):
                sub = [rows[i] for i in idx]
                pred = model(*tensors(sub, mean, std)).float()
                med = pred[:, eval_index, QUANTILES.index(0.5)].cpu().tolist()
                for pos, i in enumerate(idx):
                    out[i] = float(med[pos])
                if model_type == MODEL_TFT and model.last_weights:
                    weights_static.append(model.last_weights["static"].float().mean(0).cpu())
                    weights_past.append(model.last_weights["past"].float().mean((0, 1)).cpu())
        return out, weights_static, weights_past

    def mean_val_ic(model, rows, target, mean, std):
        preds, _, _ = predict(model, rows, mean, std)
        ics = []
        by_date: dict[date, list[int]] = {}
        for i, o in enumerate(rows):
            if target[i][eval_index] is not None:
                by_date.setdefault(o.decision_date, []).append(i)
        for idx in by_date.values():
            if len(idx) >= 3:
                value = spearman([preds[i] for i in idx], [target[i][eval_index] for i in idx])
                if value is not None:
                    ics.append(value)
        return sum(ics) / len(ics) if ics else None

    def fit_predict(train, test, _label_key):
        train = sorted(train, key=lambda o: (o.decision_date, o.ticker))
        dates = sorted({o.decision_date for o in train})
        val_rows: list[ModelingObservation] = []
        fit_rows = list(train)
        if len(dates) >= 3 and config.val_periods >= 1:
            val_dates = set(dates[-config.val_periods :])
            val_start = min(val_dates)
            val_rows = [o for o in train if o.decision_date in val_dates]
            fit_rows = [
                o for o in train
                if o.decision_date not in val_dates
                and ends.end(o, max(horizons)) < val_start
            ]
            if len(fit_rows) < config.min_train_rows:
                fit_rows, val_rows = list(train), []
        past_fit = torch.tensor([builder.past_sequence(o) for o in fit_rows], dtype=torch.float32)
        present = past_fit[..., -1:] > 0
        flat = past_fit.reshape(-1, past_fit.shape[-1])[present.reshape(-1)]
        mean = flat.mean(0) if flat.numel() else torch.zeros(len(DAILY_CHANNELS))
        std = flat.std(0).clamp_min(1e-6) if flat.shape[0] > 1 else torch.ones(len(DAILY_CHANNELS))
        mean[-1], std[-1] = 0.0, 1.0  # keep the presence flag as 0/1
        mean, std = mean.to(device), std.to(device)

        fit_target = _rank_targets(fit_rows, horizons)
        val_target = _rank_targets(val_rows, horizons) if val_rows else []
        target_t = torch.tensor(
            [[0.0 if v is None else v for v in row] for row in fit_target], dtype=torch.float32
        ).to(device)
        mask_t = torch.tensor(
            [[0.0 if v is None else 1.0 for v in row] for row in fit_target], dtype=torch.float32
        ).to(device)
        fit_inputs = tensors(fit_rows, mean, std)

        seed_scores: list[list[float]] = []
        epochs_used: list[int] = []
        for seed_offset in range(config.n_seeds):
            torch.manual_seed(config.seed + seed_offset)
            model = build_model().to(device)
            optim = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )
            best_ic, best_state, best_epoch, stale = None, None, config.epochs, 0
            for epoch in range(config.epochs):
                model.train()
                for idx in batches(fit_rows):
                    index = torch.tensor(idx, device=device)
                    optim.zero_grad()
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                        pred = model(*(t[index] for t in fit_inputs)).float()
                    loss = _loss(torch, pred, target_t[index], mask_t[index],
                                 config.ic_loss_weight)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                if val_rows:
                    ic = mean_val_ic(model, val_rows, val_target, mean, std)
                    if ic is not None and (best_ic is None or ic > best_ic):
                        best_ic, best_epoch, stale = ic, epoch + 1, 0
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    else:
                        stale += 1
                        if stale >= config.patience:
                            break
            if best_state is not None:
                model.load_state_dict(best_state)
            epochs_used.append(best_epoch)
            scores, w_static, w_past = predict(model, list(test), mean, std)
            seed_scores.append(scores)
            if w_static:
                log["static_weights"].append(torch.stack(w_static).mean(0).tolist())
                log["past_weights"].append(torch.stack(w_past).mean(0).tolist())
        log["epochs_used"].append(epochs_used)
        log["early_stopping"].append(bool(val_rows))
        return [sum(col) / len(col) for col in zip(*seed_scores, strict=True)]

    return fit_predict, str(device)


# --------------------------------------------------------------------------- #
# Comparison against the reference (primary) model
# --------------------------------------------------------------------------- #
def compare_to_reference(
    research: Sequence[ScoredObservation],
    reference: Sequence[ScoredObservation],
    *,
    horizon: int,
    reference_label: str,
) -> dict[str, Any]:
    """IC of each on common dates, score correlation and rank-average uplift."""
    if not research or not reference:
        return {"status": "unavailable", "reference_model": reference_label}
    common_dates = {s.decision_date for s in research} & {s.decision_date for s in reference}
    ref_common = [s for s in reference if s.decision_date in common_dates]
    res_common = [s for s in research if s.decision_date in common_dates]
    ref_ic = evaluate_horizon(ref_common, horizon)
    res_ic = evaluate_horizon(res_common, horizon)
    res_by_key = {(s.ticker, s.decision_date): s.score for s in res_common}
    per_date_corr: list[float] = []
    for d in sorted(common_dates):
        pairs = [
            (s.score, res_by_key[(s.ticker, d)])
            for s in ref_common
            if s.decision_date == d and (s.ticker, d) in res_by_key
        ]
        if len(pairs) >= 3:
            value = spearman([a for a, _ in pairs], [b for _, b in pairs])
            if value is not None:
                per_date_corr.append(value)
    blend = rank_average_ensemble({"reference": ref_common, "research": res_common})
    blend_ic = evaluate_horizon(blend.scored, horizon).ic_mean if blend.scored else None
    uplift = (
        blend_ic - ref_ic.ic_mean
        if (blend_ic is not None and ref_ic.ic_mean is not None)
        else None
    )
    return {
        "status": "ok",
        "reference_model": reference_label,
        "common_date_count": len(common_dates),
        "reference_ic_mean": ref_ic.ic_mean,
        "research_ic_mean": res_ic.ic_mean,
        "research_ic_tstat_newey_west": res_ic.ic_tstat_newey_west,
        "mean_score_rank_correlation_with_reference": (
            sum(per_date_corr) / len(per_date_corr) if per_date_corr else None
        ),
        "rank_average_blend_ic_mean": blend_ic,
        "blend_ic_uplift_vs_reference": uplift,
        "note": (
            "Promotion requires a positive, statistically supported uplift over the "
            "adopted model on real data; synthetic fixtures are not evidence."
        ),
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_transformer_research(
    dataset: ModelingDataset,
    *,
    horizon: int,
    config: TransformerResearchConfig | None = None,
    prices: Mapping[str, Sequence[PriceBar]] | None = None,
    reference_scored: Sequence[ScoredObservation] = (),
    reference_label: str = "primary",
) -> list[ResearchModelResult]:
    """Run the enabled research models (purged walk-forward OOS) and compare."""
    config = config or TransformerResearchConfig()
    if not config.enabled:
        return [
            ResearchModelResult(m, STATUS_DISABLED, horizon,
                                message="research models are OFF (enable explicitly)")
            for m in config.models
        ]
    if not torch_available():
        return [
            ResearchModelResult(
                m, STATUS_MISSING_DEPENDENCY, horizon,
                message="torch is not installed; install the 'research-transformers' extra",
            )
            for m in config.models
        ]
    if not prices:
        return [
            ResearchModelResult(m, STATUS_INSUFFICIENT_DATA, horizon,
                                message="daily prices are required for sequence inputs")
            for m in config.models
        ]
    target_horizons = tuple(sorted(set(config.target_horizons or dataset.horizons) | {horizon}))
    config = replace(config, target_horizons=target_horizons)

    import torch  # pragma: no cover - torch only

    if config.device in ("cuda", "auto"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    builder = _InputBuilder(dataset, prices, config.seq_len, target_horizons)
    eval_index = target_horizons.index(horizon)
    previous_threads = torch.get_num_threads()
    previous_det = torch.are_deterministic_algorithms_enabled()
    on_cuda = config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available())
    if not on_cuda:
        torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=on_cuda)
    results: list[ResearchModelResult] = []
    try:
        for model_type in config.models:
            log: dict[str, list[Any]] = {
                "static_weights": [], "past_weights": [], "epochs_used": [], "early_stopping": [],
            }
            fit_predict, device_name = _make_fit_predict(
                model_type, builder, config, prices, eval_index, log
            )
            oos = walk_forward_oos(
                dataset, horizon, fit_predict, prices=prices,
                min_train_rows=config.min_train_rows, purge_horizon=max(target_horizons),
                refit_every=config.refit_every,
            )
            folds = [f.to_dict() for f in oos.folds]
            training = {
                "device": device_name,
                "target_horizons": list(target_horizons),
                "purge_horizon": max(target_horizons),
                "epochs_used_per_fold_and_seed": log["epochs_used"],
                "early_stopping_per_fold": log["early_stopping"],
                "config": config.to_dict(),
            }
            if not oos.ok:
                results.append(ResearchModelResult(
                    model_type, STATUS_INSUFFICIENT_DATA, horizon,
                    message=f"no out-of-sample predictions: {oos.status} — {oos.message}",
                    oos_folds=folds, training=training,
                ))
                continue
            interpretation: dict[str, Any] = {}
            if log["static_weights"]:
                names = [*builder.feature_names, "sector"]
                mean_s = [sum(c) / len(c) for c in zip(*log["static_weights"], strict=True)]
                mean_p = [sum(c) / len(c) for c in zip(*log["past_weights"], strict=True)]
                interpretation["tft_static_variable_importance"] = dict(
                    sorted(((n, round(w, 6)) for n, w in zip(names, mean_s, strict=True)),
                           key=lambda kv: -kv[1])
                )
                interpretation["tft_past_channel_importance"] = dict(
                    sorted(((n, round(w, 6)) for n, w in zip(DAILY_CHANNELS, mean_p, strict=True)),
                           key=lambda kv: -kv[1])
                )
            results.append(ResearchModelResult(
                model_type, STATUS_TRAINED, horizon, scored=oos.scored, oos_folds=folds,
                comparison=compare_to_reference(
                    oos.scored, reference_scored, horizon=horizon,
                    reference_label=reference_label,
                ),
                interpretation=interpretation, training=training,
            ))
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_det)
    return results


__all__ = [
    "DAILY_CHANNELS",
    "KNOWN_FUTURE_INPUTS",
    "MODEL_CS_TRANSFORMER",
    "MODEL_GRU",
    "MODEL_TFT",
    "QUANTILES",
    "RESEARCH_MODELS",
    "STATUS_DISABLED",
    "STATUS_INSUFFICIENT_DATA",
    "STATUS_MISSING_DEPENDENCY",
    "STATUS_TRAINED",
    "ResearchModelResult",
    "TransformerResearchConfig",
    "compare_to_reference",
    "run_transformer_research",
    "torch_available",
]
