import json
from typing import List


class ConfusableSetManager:
    """Manages confusable class pairs for margin loss computation.

    Loads a pre-built mapping from class index to its top-K confusable
    (visually similar) class indices.

    Args:
        confusable_index_path (str): Path to confusable_index.json.
            Format: {"42": [78, 103, 215], "78": [42, 89, 156], ...}
        num_negatives (int): Number of confusable negatives to return.
            Defaults to 3.
    """

    def __init__(self, confusable_index_path: str, num_negatives: int = 3):
        with open(confusable_index_path, 'r') as f:
            raw = json.load(f)
        # Keys are strings in JSON; convert to int
        self._map = {int(k): v[:num_negatives] for k, v in raw.items()}
        self.num_negatives = num_negatives

    def get_negatives(self, gt_label: int) -> List[int]:
        """Return confusable negative class indices for a given GT label.

        Args:
            gt_label (int): Continuous class index of the GT category.

        Returns:
            List[int]: Up to num_negatives confusable class indices.
                Returns empty list if no entry exists for gt_label.
        """
        return self._map.get(gt_label, [])
