from typing import List


class Labels:
    def __init__(self, labels: List, none_value=None):
        self.none_value = none_value
        if none_value is not None:
            self.labels = [none_value] + labels
        else:
            self.labels = labels

    def get_index(self, label: str):
        """Get index of label.

        Return index if label was found.
        Otherwise raise ValueError.
        """
        return self.labels.index(label)

    def __getitem__(self, item):
        return self.labels[item]

    def __len__(self):
        return len(self.labels)
