import torch
import torch.nn as nn


class DeepProtMultiKernelCNN(nn.Module):

    def __init__(
        self,
        vocab_size=21,
        embedding_dim=32,
        num_filters=32,
        num_classes=226,
        dropout=0.4
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.conv5 = nn.Conv1d(
            embedding_dim, num_filters,
            kernel_size=5, padding=2
        )

        self.conv9 = nn.Conv1d(
            embedding_dim, num_filters,
            kernel_size=9, padding=4
        )

        self.conv15 = nn.Conv1d(
            embedding_dim, num_filters,
            kernel_size=15, padding=7
        )

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(num_filters * 3, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):

        mask = (x != 0)

        x = self.embedding(x)
        x = x.transpose(1, 2)

        out5 = self.relu(self.conv5(x))
        out9 = self.relu(self.conv9(x))
        out15 = self.relu(self.conv15(x))

        mask = mask.unsqueeze(1)

        out5 = out5.masked_fill(~mask, float("-inf"))
        out9 = out9.masked_fill(~mask, float("-inf"))
        out15 = out15.masked_fill(~mask, float("-inf"))

        out5 = torch.amax(out5, dim=2)
        out9 = torch.amax(out9, dim=2)
        out15 = torch.amax(out15, dim=2)

        x = torch.cat([out5, out9, out15], dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        return self.fc2(x)
