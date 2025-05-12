# Inspired by https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/metrics/perplexity.py
# But we compute the perplexity correctly: exp(average(nll)), not average(exp(nll))
# Also adapted from https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/text/perplexity.py
# But we pass in the loss to avoid recomputation

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from torchmetrics import Metric
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = torch.nn.CrossEntropyLoss

try:
    from apex.transformer import parallel_state
except ImportError:
    parallel_state = None


class Perplexity(Metric):
    r"""
    Perplexity measures how well a language model predicts a text sample. It's calculated as the average number of bits
    per word a model needs to represent the sample.
    Args:
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    Examples:
        >>> import torch
        >>> preds = torch.rand(2, 8, 5, generator=torch.manual_seed(22))
        >>> target = torch.randint(5, (2, 8), generator=torch.manual_seed(22))
        >>> target[0, 6:] = -100
        >>> metric = Perplexity(ignore_index=-100)
        >>> metric(preds, target)
        tensor(5.2545)
    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    total_log_probs: Tensor
    count: Tensor

    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.add_state("total_log_probs", default=torch.tensor(0.0, dtype=torch.float64),
                       dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

        self.loss_fn = CrossEntropyLoss()

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None) -> None:  # type: ignore
        """Compute and store intermediate statistics for Perplexity.
        Args:
            preds:
                Probabilities assigned to each token in a sequence with shape [batch_size, seq_len, vocab_size].
            target:
                Ground truth values with a shape [batch_size, seq_len].
        """
        count = target.numel()
        if loss is None:
            loss = self.loss_fn(preds, target)
        self.total_log_probs += loss.double() * count
        self.count += count

    def compute(self) -> Tensor:
        """Compute the Perplexity.
        Returns:
           Perplexity
        """
        return torch.exp(self.total_log_probs / self.count)

class NumTokens(Metric):
    """Keep track of how many tokens we've seen.
    """
    # TODO: how do we prevent the reset between the epochs? The reset happens on the 1st batch
    # of the next epoch.
    # Right now the hack is that we override reset(), which would mess up the forward method.
    # We then override forward to do the right thing.

    is_differentiable = False
    higher_is_better = False
    full_state_update = False
    count: Tensor

    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum",
                       persistent=True)  # We want the count to be saved to state-dict
        if parallel_state is not None and not parallel_state.is_unitialized():
            self.tensor_parallel_world_size = parallel_state.get_tensor_model_parallel_world_size()
        else:
            self.tensor_parallel_world_size = 1

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None) -> None:  # type: ignore
        self.count += target.numel() // self.tensor_parallel_world_size

    def compute(self) -> Tensor:
        return self.count

    def reset(self):
        count = self.count
        super().reset()
        self.count = count

    # Adapted from https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/metric.py
    def _forward_reduce_state_update(self, *args: Any, **kwargs: Any) -> Any:
        """forward computation using single call to `update` to calculate the metric value on the current batch and
        accumulate global state.
        This can be done when the global metric state is a sinple reduction of batch states.
        """
        self.update(*args, **kwargs)
        return self.compute()

class AccumulatedMetricBase(Metric):
    """Base class for accumulated metrics that collect predictions before computing the final score."""
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use tensors for state to make metric hashable
        self.add_state("preds", default=torch.tensor([], dtype=torch.long), dist_reduce_fx="cat")
        self.add_state("target", default=torch.tensor([], dtype=torch.long), dist_reduce_fx="cat")

    def update(self, preds, target, loss=None):
        """Update state with predictions and targets.

        Args:
            preds: Tensor with shape (batch_size * seq_len, num_classes)
            target: Tensor with shape (batch_size * seq_len)
            loss: Optional loss value (not used)
        """
        preds = preds.view(-1, preds.shape[-1])
        target = target.view(-1)
        pred_classes = torch.argmax(preds, dim=-1)

        # Append to state tensors
        self.preds = torch.cat([self.preds.to(pred_classes.device), pred_classes])
        self.target = torch.cat([self.target.to(target.device), target])

    def compute(self):
        """Compute metric score over all accumulated predictions and targets."""
        if len(self.preds) == 0 or len(self.target) == 0:
            return torch.tensor(0.0)

        # Move to CPU for sklearn compatibility
        preds_cpu = self.preds.cpu().numpy()
        target_cpu = self.target.cpu().numpy()

        return self._compute_metric(preds_cpu, target_cpu)

    def _compute_metric(self, preds, target):
        """To be implemented by subclasses."""
        raise NotImplementedError

class AccumulatedMCC(AccumulatedMetricBase):
    """Compute Matthews Correlation Coefficient across the entire dataset rather than per-batch."""

    def _compute_metric(self, preds, target):
        # Handle edge cases where all predictions or targets belong to single class
        unique_preds = np.unique(preds)
        unique_targets = np.unique(target)

        if len(unique_preds) <= 1 and len(unique_targets) <= 1:
            # If both predictions and targets are the same class, perfect agreement
            if unique_preds[0] == unique_targets[0]:
                return torch.tensor(1.0)
            # If all preds are one class and all targets are another, perfect disagreement
            else:
                return torch.tensor(-1.0)
        # If all predictions are the same class but targets are mixed, or vice versa
        elif len(unique_preds) <= 1 or len(unique_targets) <= 1:
            return torch.tensor(0.0)

        # Normal case: compute MCC
        mcc_score = matthews_corrcoef(target, preds)
        return torch.tensor(mcc_score, dtype=torch.float32)

class AccumulatedAccuracy(AccumulatedMetricBase):
    """Compute accuracy across the entire dataset rather than per-batch."""

    def _compute_metric(self, preds, target):
        return torch.tensor(accuracy_score(target, preds), dtype=torch.float32)


class AccumulatedF1Binary(AccumulatedMetricBase):
    """Compute F1 score (binary) across the entire dataset rather than per-batch."""

    def _compute_metric(self, preds, target):
        # Check if we have enough classes for proper calculation
        unique_labels = np.unique(np.concatenate([preds, target]))

        # For binary case with only one class present
        if len(unique_labels) == 1:
            # If the only class is positive (usually 1) and all predictions match targets
            if unique_labels[0] > 0 and np.array_equal(preds, target):
                return torch.tensor(1.0)
            # If the only class is negative (usually 0) and all predictions match targets
            elif unique_labels[0] == 0 and np.array_equal(preds, target):
                return torch.tensor(1.0)
            else:
                return torch.tensor(0.0)

        # Handle case where we don't have both positive and negative examples
        if len(unique_labels) < 2:
            return torch.tensor(0.0)

        return torch.tensor(f1_score(target, preds, average='binary'), dtype=torch.float32)

class AccumulatedF1Macro(AccumulatedMetricBase):
    """Compute F1 score (macro) across the entire dataset rather than per-batch."""

    def _compute_metric(self, preds, target):
        # Check if we have enough different classes for macro averaging
        unique_labels = np.unique(np.concatenate([preds, target]))

        # Need at least 2 classes for macro averaging to make sense
        if len(unique_labels) < 2:
            return torch.tensor(0.0)

        return torch.tensor(f1_score(target, preds, average='macro'), dtype=torch.float32)

class AccumulatedPrecisionBinary(AccumulatedMetricBase):
    """Compute precision (binary) across the entire dataset rather than per-batch."""

    def _compute_metric(self, preds, target):
        # Check for the case where there are no positive predictions
        if len(np.unique(preds)) <= 1 and 1 not in preds:
            # If there are no positive predictions, precision is undefined
            # but we return 0 as is common practice
            return torch.tensor(0.0)

        try:
            return torch.tensor(precision_score(target, preds, average='binary', zero_division=0),
                               dtype=torch.float32)
        except:
            # Handle edge cases
            return torch.tensor(0.0)

class AccumulatedPrecisionMacro(AccumulatedMetricBase):
    """Compute precision (macro) across the entire dataset rather than per-batch."""

    def _compute_metric(self, preds, target):
        # Need at least 2 different classes for macro averaging
        unique_labels = np.unique(np.concatenate([preds, target]))
        if len(unique_labels) < 2:
            return torch.tensor(0.0)

        try:
            return torch.tensor(precision_score(target, preds, average='macro', zero_division=0),
                               dtype=torch.float32)
        except:
            # Handle edge cases
            return torch.tensor(0.0)


class AccumulatedRecallBinary(AccumulatedMetricBase):
    """Compute recall/sensitivity (binary) across the entire dataset rather than per-batch."""

    def _compute_metric(self, preds, target):
        # Check for the case where there are no positive targets
        if len(np.unique(target)) <= 1 and 1 not in target:
            # If there are no positive examples in the target, recall is undefined
            # but we return 0 as is common practice
            return torch.tensor(0.0)

        try:
            return torch.tensor(recall_score(target, preds, average='binary', zero_division=0),
                               dtype=torch.float32)
        except:
            # Handle edge cases
            return torch.tensor(0.0)

class AccumulatedRecallMacro(AccumulatedMetricBase):
    """Compute recall/sensitivity (macro) across the entire dataset rather than per-batch."""

    def _compute_metric(self, preds, target):
        # Need at least 2 different classes for macro averaging
        unique_labels = np.unique(np.concatenate([preds, target]))
        if len(unique_labels) < 2:
            return torch.tensor(0.0)

        try:
            return torch.tensor(recall_score(target, preds, average='macro', zero_division=0),
                               dtype=torch.float32)
        except:
            # Handle edge cases
            return torch.tensor(0.0)


class AccumulatedSpecificity(AccumulatedMetricBase):
    """Compute specificity (true negative rate) across the entire dataset."""

    def _compute_metric(self, preds, target):
        # Specificity = TN / (TN + FP)
        # For binary classification:
        # TN = sum((target == 0) & (preds == 0))
        # FP = sum((target == 0) & (preds == 1))

        # Check if we have negative examples
        if 0 not in target:
            return torch.tensor(0.0)

        tn = np.sum((target == 0) & (preds == 0))
        fp = np.sum((target == 0) & (preds == 1))

        if tn + fp == 0:
            return torch.tensor(0.0)

        return torch.tensor(tn / (tn + fp), dtype=torch.float32)

torchmetric_fns = {
    "perplexity": Perplexity,
    "num_tokens": NumTokens,
    "accumulated_mcc": AccumulatedMCC,
    "accumulated_accuracy": AccumulatedAccuracy,
    "accumulated_f1_binary": AccumulatedF1Binary,
    "accumulated_f1_macro": AccumulatedF1Macro,
    "accumulated_precision_binary": AccumulatedPrecisionBinary,
    "accumulated_precision_macro": AccumulatedPrecisionMacro,
    "accumulated_recall_binary": AccumulatedRecallBinary,
    "accumulated_recall_macro": AccumulatedRecallMacro,
    "accumulated_specificity": AccumulatedSpecificity,
}

