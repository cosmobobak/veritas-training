before generating data in anger, the old iridium-oxide program with handcrafted policy features ("play near stones")
was used to generate 154872 positions, at 800 playouts per position. These were then used to train a policy network
with this architecture:

```py
class ConvPolicyModel(tch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu    = tch.nn.ReLU()
        # five layers of 3x3 convs mean that information can travel at most five squares away.
        # this is maybe fine idk
        self.conv1   = tch.nn.Conv2d(2, 8, 3, padding=1) # 2x9x9 -> 8x9x9
        self.conv2   = tch.nn.Conv2d(8, 8, 3, padding=1) # 8x9x9 -> 8x9x9
        self.conv3   = tch.nn.Conv2d(8, 8, 3, padding=1) # 8x9x9 -> 8x9x9
        self.conv4   = tch.nn.Conv2d(8, 8, 3, padding=1) # 8x9x9 -> 8x9x9
        self.conv5   = tch.nn.Conv2d(8, 2, 3, padding=1) # 8x9x9 -> 2x9x9
        self.fc      = tch.nn.Linear(2 * SQUARES, SQUARES)
        self.softmax = tch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 2, BOARD_SIDE_LEN, BOARD_SIDE_LEN)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        # flatten
        x = x.view(-1, 2 * SQUARES)
        x = self.fc(x)
        # x = self.softmax(x)
        return x
```

This network was trained for 20 epochs, with a batch-size of 64, using the AdamW optimizer with a learning rate of 1e-3.
The loss function was cross-entropy loss. The training data was split into 90% training and 10% validation data.

The resulting model was much weaker when playing against uniform policy, due to the vast slowdown in rollouts/sec.
However, at fixed playouts, it won 95 out of 100 games against uniform policy.

For the first proper training run, we generated two minutes worth of positions at 800 playouts, using the new `veritas` program,
with fixed uniform policy.

The resulting model was significantly better, so we immediately generated additional training data (higher-quality, at 8000 playouts)
for 8 minutes, and retrained the model.