import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Triton Python backend processor для преобразования предсказаний 3 GBDT в логиты мета-стекера."""

    def initialize(self, args):
        """Инициализация процессора признаков."""

    def execute(self, requests):
        """Расчет логитов по формуле prob_to_logit и конкатенация 3 мета-признаков."""
        responses = []
        eps = 1e-5

        for request in requests:
            logits_list = []
            for name in ("cb_probabilities", "lgb_probabilities", "xgb_probabilities"):
                tensor = pb_utils.get_input_tensor_by_name(request, name)
                probs = tensor.as_numpy()
                p = probs[:, 1] if probs.ndim == 2 and probs.shape[1] >= 2 else probs.squeeze(-1)
                p_clip = np.clip(p, eps, 1.0 - eps)
                logits_list.append(np.log(p_clip / (1.0 - p_clip)))

            meta_data = np.column_stack(logits_list).astype(np.float32)
            out_tensor = pb_utils.Tensor("meta_input", meta_data)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        """Очистка ресурсов процессора."""
