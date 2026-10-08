"""Последовательные детекторы и отдельный синтетический benchmark.

BOCPD использует reset-before-observation: новый сегмент получает prior
predictive, затем обновляет NIG-параметры первым наблюдением. Это позволяет
оценивать вероятность нового короткого сегмента, а не постоянную hazard.
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import t as student_t


class Detector:
    def __init__(self, method, threshold, cooldown=3, hazard=1/24):
        self.method, self.threshold, self.cooldown, self.hazard = method, threshold, cooldown, hazard
        self.last_alarm = -10000
        self.plus = self.minus = self.sum_plus = self.sum_minus = self.min_plus = self.min_minus = 0.
        self.mean = 0.; self.count = 0
        self.log_prob = np.array([0.])
        self.mu = np.array([0.]); self.kappa = np.array([1.]); self.alpha = np.array([2.]); self.beta = np.array([2.])

    def update(self, value, index):
        if self.method == "CUSUM":
            self.plus = max(0., self.plus + value - .5)
            self.minus = max(0., self.minus - value - .5)
            score = max(self.plus, self.minus)
        elif self.method == "PageHinkley":
            self.count += 1
            self.mean += (value - self.mean) / self.count
            self.sum_plus += value - self.mean - .1
            self.sum_minus += -value + self.mean - .1
            self.min_plus = min(self.min_plus, self.sum_plus)
            self.min_minus = min(self.min_minus, self.sum_minus)
            score = max(self.sum_plus - self.min_plus, self.sum_minus - self.min_minus)
        elif self.method == "BOCPD":
            scale = np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
            likelihood = student_t.logpdf(value, 2 * self.alpha, loc=self.mu, scale=scale)
            prior_likelihood = student_t.logpdf(value, 4., loc=0., scale=np.sqrt(2.))
            growth = self.log_prob + np.log1p(-self.hazard) + likelihood
            reset = np.log(self.hazard) + prior_likelihood
            new = np.r_[reset, growth]
            self.log_prob = new - logsumexp(new)
            old_mu, old_k = np.r_[0., self.mu], np.r_[1., self.kappa]
            old_a, old_b = np.r_[2., self.alpha], np.r_[2., self.beta]
            self.mu = (old_k * old_mu + value) / (old_k + 1)
            self.kappa = old_k + 1
            self.alpha = old_a + .5
            self.beta = old_b + old_k * (value - old_mu) ** 2 / (2 * (old_k + 1))
            score = float(np.exp(self.log_prob[:3]).sum())
        else:
            raise ValueError(self.method)
        alarm = bool(score >= self.threshold and index - self.last_alarm >= self.cooldown)
        if alarm:
            self.last_alarm = index
            if self.method != "BOCPD":
                self.plus = self.minus = self.sum_plus = self.sum_minus = self.min_plus = self.min_minus = 0.
                self.mean = 0.; self.count = 0
        return float(score), alarm


def residual_stream(panel, config):
    records = []
    for series in panel.columns:
        values = panel[series].to_numpy()
        residuals = []
        for index in range(24, len(values)):
            # Прогноз baseline выпущен до наблюдения значения index.
            past = values[:index]
            predicted = values[index - 12] * past[-12:].mean() / past[-24:-12].mean()
            residual = float(np.log(values[index] / predicted))
            history = residuals[-config["residual_scale_window"]:]
            if len(history) >= config["warmup_observations"]:
                center = np.median(history)
                scale = max(config["minimum_log_scale"], 1.4826 * np.median(np.abs(history - center)))
                records.append({"series_id": series, "index": index, "period": panel.index[index],
                                "log_residual": residual, "z": (residual - center) / scale,
                                "maximum_scaling_history_index": index - 1})
            residuals.append(residual)
    return pd.DataFrame(records)


def detect_stream(residuals, method, threshold, config):
    output = []
    for series, group in residuals.groupby("series_id"):
        detector = Detector(method, threshold, config["cooldown_observations"], config["bocpd_hazard"])
        for k, row in enumerate(group.sort_values("index").itertuples()):
            score, alarm = detector.update(row.z, k)
            output.append({"series_id": series, "index": row.index, "period": row.period,
                           "method": method, "score": score, "alarm": alarm and k >= 3, "z": row.z})
    return pd.DataFrame(output)


def synthetic_scenarios(n, config, seed):
    rng = np.random.default_rng(seed)
    kinds = ["level_up", "level_down", "trend", "variance", "spike", "none"]
    result = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        values = rng.normal(size=config["synthetic_length"])
        cp = config["synthetic_transition_index"]
        if kind == "level_up": values[cp:] += 1.5
        if kind == "level_down": values[cp:] -= 1.5
        if kind == "trend": values[cp:] += np.arange(len(values) - cp) * .08
        if kind == "variance": values[cp:] *= 3
        if kind == "spike": values[cp] += 6
        result.append((kind, values, cp if kind not in ["spike", "none"] else None))
    return result


def benchmark_detector(method, threshold, scenarios, config):
    n_events = n_alarms = tp = fp = 0
    delays = []; rows = []
    for i, (kind, values, cp) in enumerate(scenarios):
        detector = Detector(method, threshold, config["cooldown_observations"], config["bocpd_hazard"])
        alarms = []
        for index, value in enumerate(values):
            _, alarm = detector.update(value, index)
            if alarm and index >= 30: alarms.append(index)
        matched = [a for a in alarms if cp is not None and cp <= a <= cp + config["matching_max_delay"]]
        hit = bool(matched)
        n_events += cp is not None; n_alarms += len(alarms); tp += hit; fp += len(alarms) - int(hit)
        if hit: delays.append(min(matched) - cp)
        rows.append({"scenario": i, "kind": kind, "n_alarms": len(alarms), "detected": hit,
                     "delay_steps": min(matched) - cp if hit else None})
    precision = tp / n_alarms if n_alarms else 0.
    recall = tp / n_events if n_events else 0.
    return {"method": method, "threshold": threshold, "event_precision": precision, "event_recall": recall,
            "event_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "median_delay_steps": float(np.median(delays)) if delays else None,
            "false_alarms_per_100_steps": 100 * fp / (len(scenarios) * (config["synthetic_length"] - 30)),
            "n_events": n_events, "n_alarms": n_alarms}, pd.DataFrame(rows)


def offline_mean_changes(values, penalty, min_segment=6):
    """Точная DP-сегментация L2, не PELT. Используется лишь как proxy-reference."""
    values = np.asarray(values, dtype=float); n = len(values)
    sums, squares = np.r_[0, np.cumsum(values)], np.r_[0, np.cumsum(values ** 2)]
    cost = np.full(n + 1, np.inf); cost[0] = -penalty
    previous = np.full(n + 1, -1)
    for end in range(min_segment, n + 1):
        for start in range(0, end - min_segment + 1):
            if not np.isfinite(cost[start]): continue
            sse = squares[end] - squares[start] - (sums[end] - sums[start]) ** 2 / (end - start)
            value = cost[start] + sse + penalty
            if value < cost[end]: cost[end], previous[end] = value, start
    changes = []; end = n
    while previous[end] > 0:
        end = previous[end]; changes.append(int(end))
    return sorted(changes)


def match_events(event_indices, alarm_indices, delay=3):
    matched = set(); pairs = []
    for event in sorted(event_indices):
        available = [a for a in sorted(alarm_indices) if a not in matched and event <= a <= event + delay]
        if available:
            alarm = available[0]; matched.add(alarm); pairs.append((event, alarm))
    return pairs
