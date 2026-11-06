from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator


class HMMPostprocessor(BaseEstimator):
    """Постпроцессор скрытой марковской модели (HMM) для хронологической цепочки диалогов.

    Моделирует марковский процесс смены состояний матчинга участников в хронологии турнира:
    - Состояние 0: (Человек, Человек)
    - Состояние 1: (Бот, Человек)
    - Состояние 2: (Человек, Бот)

    Выполняет двунаправленное сглаживание Forward-Backward и блендинг апостериорных вероятностей
    с базовыми предсказаниями модели.
    """

    def __init__(
        self,
        alpha_laplace: float = 1.0,
        blend_weight: float = 0.7,
        eps: float = 0.005,
    ) -> None:
        self.alpha_laplace = alpha_laplace
        self.blend_weight = blend_weight
        self.eps = eps

        self.transition_matrix_: np.ndarray | None = None
        self.initial_probs_: np.ndarray | None = None

    def fit(
        self,
        dialog_order: Sequence[Any],
        y_true: np.ndarray,
        participant_indices: np.ndarray,
        dialog_ids: np.ndarray,
    ) -> "HMMPostprocessor":
        """Оценка матрицы переходов и начальных вероятностей по хронологической обучающей выборке."""
        d_ids_str = np.asarray([str(d) for d in dialog_ids])
        p_idx_arr = np.asarray(participant_indices)
        y_arr = np.asarray(y_true, dtype=int)

        # Картирование меток участников для каждого диалога
        p0_dict: dict[str, int] = {}
        p1_dict: dict[str, int] = {}

        for did, p_idx, y_val in zip(d_ids_str, p_idx_arr, y_arr, strict=False):
            if p_idx == 0:
                p0_dict[did] = int(y_val)
            elif p_idx == 1:
                p1_dict[did] = int(y_val)

        # Формирование дискретной последовательности состояний турнира
        states: list[int] = []
        for did in dialog_order:
            d_str = str(did)
            b0 = p0_dict.get(d_str, 0)
            b1 = p1_dict.get(d_str, 0)

            if b0 == 0 and b1 == 0:
                states.append(0)
            elif b0 == 1 and b1 == 0:
                states.append(1)
            elif b0 == 0 and b1 == 1:
                states.append(2)
            else:
                # В случае аномалии двух ботов мапим в нулевое состояние
                states.append(0)

        n_sessions = len(states)
        if n_sessions == 0:
            self.initial_probs_ = np.array([1 / 3, 1 / 3, 1 / 3])
            self.transition_matrix_ = np.full((3, 3), 1 / 3)
            return self

        # Начальные вероятности (априорные доли состояний)
        c0 = states.count(0)
        c1 = states.count(1)
        c2 = states.count(2)
        self.initial_probs_ = np.array([c0, c1, c2], dtype=float) / n_sessions

        # Матрица счетчиков переходов
        trans_counts = np.zeros((3, 3), dtype=float)
        for t in range(1, n_sessions):
            prev_s = states[t - 1]
            curr_s = states[t]
            trans_counts[prev_s, curr_s] += 1.0

        # Сглаживание Лапласа
        smooth_counts = trans_counts + self.alpha_laplace
        self.transition_matrix_ = smooth_counts / smooth_counts.sum(axis=1, keepdims=True)

        return self

    def transform_sequences(
        self,
        p0_seq: np.ndarray,
        p1_seq: np.ndarray,
        blend_weight: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Выполнение Forward-Backward сглаживания над упорядоченными массивами пар вероятностей."""
        if self.transition_matrix_ is None or self.initial_probs_ is None:
            raise ValueError("HMMPostprocessor должен быть обучен методом fit перед вызовом transform.")

        w = self.blend_weight if blend_weight is None else blend_weight
        p0_arr = np.asarray(p0_seq, dtype=float)
        p1_arr = np.asarray(p1_seq, dtype=float)
        n = len(p0_arr)

        if n == 0 or w <= 0.0:
            return p0_arr.copy(), p1_arr.copy()

        # Формирование матрицы правдоподобий эмиссий B размера (N, 3)
        p0_clip = np.clip(p0_arr, self.eps, 1.0 - self.eps)
        p1_clip = np.clip(p1_arr, self.eps, 1.0 - self.eps)

        b_mat = np.zeros((n, 3), dtype=float)
        b_mat[:, 0] = (1.0 - p0_clip) * (1.0 - p1_clip)
        b_mat[:, 1] = p0_clip * (1.0 - p1_clip)
        b_mat[:, 2] = (1.0 - p0_clip) * p1_clip

        row_sums = b_mat.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        b_mat /= row_sums

        pi = self.initial_probs_
        trans = self.transition_matrix_

        # 1. Forward Pass (прямой проход)
        alpha = np.zeros((n, 3), dtype=float)
        alpha[0] = pi * b_mat[0]
        s0 = alpha[0].sum()
        alpha[0] = alpha[0] / (s0 if s0 > 0 else 1.0)

        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ trans) * b_mat[t]
            st = alpha[t].sum()
            alpha[t] = alpha[t] / (st if st > 0 else 1.0)

        # 2. Backward Pass (обратный проход)
        beta = np.zeros((n, 3), dtype=float)
        beta[-1] = 1.0

        for t in range(n - 2, -1, -1):
            beta[t] = trans @ (beta[t + 1] * b_mat[t + 1])
            st = beta[t].sum()
            beta[t] = beta[t] / (st if st > 0 else 1.0)

        # 3. Posterior State Probabilities (сглаженное распределение состояний)
        gamma = alpha * beta
        gamma_sums = gamma.sum(axis=1, keepdims=True)
        gamma_sums = np.where(gamma_sums > 0, gamma_sums, 1.0)
        gamma /= gamma_sums

        # Выделение маргинальных вероятностей принадлежности участников к классу бота
        p0_hmm = gamma[:, 1]
        p1_hmm = gamma[:, 2]

        # 4. Мягкий блендинг с базовыми предсказаниями
        p0_out = (1.0 - w) * p0_arr + w * p0_hmm
        p1_out = (1.0 - w) * p1_arr + w * p1_hmm

        return p0_out, p1_out

    def fit_transform(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        *,
        probabilities: np.ndarray | None = None,
        y_true: np.ndarray | None = None,
        dialog_ids: np.ndarray | None = None,
        participant_indices: np.ndarray | None = None,
        dialog_order: Sequence[Any] | None = None,
        blend_weight: float | None = None,
    ) -> np.ndarray:
        """Совмещенное обучение и сглаживание марковской модели."""
        raw_probs = probabilities if probabilities is not None else X
        target_y = y_true if y_true is not None else y
        if raw_probs is None:
            raise ValueError("Требуется передать массив вероятностей в probabilities или X.")
        if target_y is None or dialog_ids is None or participant_indices is None or dialog_order is None:
            raise ValueError("Для fit_transform требуются y/y_true, dialog_ids, participant_indices и dialog_order.")

        return self.fit(
            dialog_order=dialog_order,
            y_true=target_y,
            participant_indices=participant_indices,
            dialog_ids=dialog_ids,
        ).transform(
            probabilities=raw_probs,
            dialog_ids=dialog_ids,
            participant_indices=participant_indices,
            dialog_order=dialog_order,
            blend_weight=blend_weight,
        )

    def transform(
        self,
        probabilities: np.ndarray | None = None,
        dialog_ids: np.ndarray | None = None,
        participant_indices: np.ndarray | None = None,
        dialog_order: Sequence[Any] | None = None,
        blend_weight: float | None = None,
        X: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Сглаживание плоского массива вероятностей с сохранением соответствия сэмплам."""
        del kwargs
        raw_probs = probabilities if probabilities is not None else X
        if raw_probs is None:
            raise ValueError("Требуется передать массив вероятностей в probabilities или X.")

        if dialog_ids is None or participant_indices is None or dialog_order is None:
            raise ValueError("Требуются параметры dialog_ids, participant_indices и dialog_order.")

        d_ids_str = np.asarray([str(d) for d in dialog_ids])
        p_idx_arr = np.asarray(participant_indices)
        d_order = dialog_order
        p_arr = np.asarray(raw_probs, dtype=float)

        p0_map: dict[str, float] = {}
        p1_map: dict[str, float] = {}

        for did, p_idx, p_val in zip(d_ids_str, p_idx_arr, p_arr, strict=False):
            if p_idx == 0:
                p0_map[did] = float(p_val)
            elif p_idx == 1:
                p1_map[did] = float(p_val)

        # Извлечение упорядоченных последовательностей сессий
        p0_seq = np.array([p0_map.get(str(d), 0.5) for d in d_order], dtype=np.float64)
        p1_seq = np.array([p1_map.get(str(d), 0.5) for d in d_order], dtype=np.float64)

        # Сглаживание через Forward-Backward HMM
        p0_sm, p1_sm = self.transform_sequences(p0_seq, p1_seq, blend_weight=blend_weight)

        # Обратное отображение сглаженных вероятностей в словарь
        d_order_str = [str(d) for d in d_order]
        smoothed_p0_dict = dict(zip(d_order_str, p0_sm, strict=False))
        smoothed_p1_dict = dict(zip(d_order_str, p1_sm, strict=False))

        # Заполнение результирующего массива в исходном порядке
        out_probs = np.zeros_like(p_arr)
        for i in range(len(p_arr)):
            did = d_ids_str[i]
            p_idx = p_idx_arr[i]
            if p_idx == 0:
                out_probs[i] = smoothed_p0_dict.get(did, p_arr[i])
            else:
                out_probs[i] = smoothed_p1_dict.get(did, p_arr[i])

        return out_probs
