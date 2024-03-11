# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch as tch
import math
# use proper seaborn styling
import seaborn as sns
sns.set_theme()

# %%
BOARD_SIDE_LEN = int(input("Enter the board size (side length, likely 9, 13, 15, or 19): "))
DATASET_PATH = input("Enter the path to the dataset: ")
RUN_ID = input(f"Enter the run ID: gomoku{BOARD_SIDE_LEN}x{BOARD_SIDE_LEN}-")
RUN_ID = f"gomoku{BOARD_SIDE_LEN}x{BOARD_SIDE_LEN}-" + RUN_ID

# mkdir plots/{RUN_ID}
os.makedirs(f"plots/{RUN_ID}", exist_ok=True)

positions       = np.loadtxt(f"{DATASET_PATH}/positions.csv", delimiter=",")
rollout_counts  = np.loadtxt(f"{DATASET_PATH}/policy-target.csv", delimiter=",")
results         = np.loadtxt(f"{DATASET_PATH}/value-target.csv", delimiter=",")

print(f"{len(positions)} datapoints loaded!")
assert len(positions) == len(rollout_counts)
assert len(positions) == len(results)

# %%
# normalize the rollout counts
r_sums = np.sum(rollout_counts, axis=1)
rollout_counts /= r_sums[:, np.newaxis]

print(f"y has dims {rollout_counts.shape}")
print(f"{rollout_counts[0]}")

# split the dataset into training and validation sets
split = int(len(positions) * 0.9)
x_train = positions[:split]
y_train = rollout_counts[:split]
z_train = results[:split]
x_val = positions[split:]
y_val = rollout_counts[split:]
z_val = results[split:]

# shuffle the datasets
perm_train = np.random.permutation(len(x_train))
x_train = x_train[perm_train]
y_train = y_train[perm_train]
z_train = z_train[perm_train]
perm_val = np.random.permutation(len(x_val))
x_val = x_val[perm_val]
y_val = y_val[perm_val]
z_val = z_val[perm_val]

# convert to tensors
x_train = tch.tensor(x_train, dtype=tch.float)
y_train = tch.tensor(y_train, dtype=tch.float)
z_train = tch.tensor(z_train, dtype=tch.float)
print(f"x_train has dims {x_train.shape}")
print(f"y_train has dims {y_train.shape}")
print(f"z_train has dims {z_train.shape}")
x_val = tch.tensor(x_val, dtype=tch.float)
y_val = tch.tensor(y_val, dtype=tch.float)
z_val = tch.tensor(z_val, dtype=tch.float)
print(f"x_val has dims {x_val.shape}")
print(f"y_val has dims {y_val.shape}")
print(f"z_val has dims {z_val.shape}")

# create a dataset class
class GomokuDataset(tch.utils.data.Dataset):
    def __init__(self, pos, policy, value):
        self.x = pos
        self.y = policy
        self.z = value

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx]

    def __len__(self):
        return len(self.x)

# create dataloaders
train_dataset = GomokuDataset(x_train, y_train, z_train)
val_dataset = GomokuDataset(x_val, y_val, z_val)
train_loader = tch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = tch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

# %%
x_shaped = x_train.reshape(-1, 2, 9, 9)
y_shaped = y_train.reshape(-1, 9, 9)

# show some sample data
fig, axs = plt.subplots(5, 4, figsize=(12, 10))
for i in range(5):
    # left plot is the board state
    board = x_shaped[i, 0] + x_shaped[i, 1] / 2
    axs[i, 0].imshow(board, cmap="gray")
    # right plot is the policy
    policy = y_shaped[i]
    axs[i, 1].imshow(policy, cmap="inferno")

    # plot the nonzero board values
    nonzero_board = board != 0
    axs[i, 2].imshow(nonzero_board, vmin=0, vmax=1)

    # plot the zero policy values
    zero_policy = policy == 0
    axs[i, 3].imshow(zero_policy, vmin=0, vmax=1)

# remove the gridlines
for ax in axs.flatten():
    ax.grid(False)

# save to plots directory
fig.savefig(f"plots/{RUN_ID}/sample_data.png")

# %%

SQUARES = BOARD_SIDE_LEN * BOARD_SIDE_LEN

# define a convolutional model
# this is ever so slightly more complicated than the previous model
# as we need to reshape the input to be 4-dimensional
class ConvPolicyModel(tch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu    = tch.nn.ReLU()
        self.sigmoid = tch.nn.Sigmoid()
        # five layers of 3x3 convs mean that information can travel at most five squares away.
        # this is maybe fine idk
        self.conv1   = tch.nn.Conv2d(2, 64, 3, padding=1) # 2 x SIDE x SIDE -> 64 x SIDE x SIDE
        self.conv2   = tch.nn.Conv2d(64, 64, 3, padding=1) # 64 x SIDE x SIDE -> 64 x SIDE x SIDE
        self.conv3   = tch.nn.Conv2d(64, 32, 3, padding=1) # 64 x SIDE x SIDE -> 32 x SIDE x SIDE
        self.conv4   = tch.nn.Conv2d(32, 16, 3, padding=1) # 32 x SIDE x SIDE -> 16 x SIDE x SIDE
        self.conv5   = tch.nn.Conv2d(16, 2, 3, padding=1) # 8 x SIDE x SIDE -> 2 x SIDE x SIDE
        self.policy1 = tch.nn.Linear(2 * SQUARES, 2 * SQUARES)
        self.policy2 = tch.nn.Linear(2 * SQUARES, SQUARES)
        self.value1  = tch.nn.Linear(2 * SQUARES, 2 * SQUARES)
        self.value2  = tch.nn.Linear(2 * SQUARES, 1)

    def forward(self, x):
        x = x.view(-1, 2, BOARD_SIDE_LEN, BOARD_SIDE_LEN)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        x = self.relu(x)
        # flatten
        latent = x.view(-1, 2 * SQUARES)
        x = self.policy1(latent)
        x = self.relu(x)
        policy_logits = self.policy2(x)
        x = self.value1(latent)
        x = self.relu(x)
        x = self.value2(x)
        value = self.sigmoid(x)
        return policy_logits, value

class SimpleNet(tch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = tch.nn.Sigmoid()
        self.policy  = tch.nn.Linear(2 * SQUARES, SQUARES)
        self.value   = tch.nn.Linear(2 * SQUARES, 1)

    def forward(self, x):
        policy_logits = self.policy(x)
        value = self.sigmoid(self.value(x))
        return policy_logits, value


# %%
# create the model and optimizer
model = ConvPolicyModel()
optimizer = tch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

# create a loss function to match the probability distribution
def loss_fn(prediction_logits, target_distribution):
    return tch.nn.functional.binary_cross_entropy(tch.nn.functional.softmax(prediction_logits, dim=1), target_distribution)

# create a function for masking off illegal moves
def mask_illegal_moves(model_prediction, board):
    # return model_prediction
    # model_prediction is an 81-element vector of probabilities
    # board is a 81 * 2 = 162-element vector of occupancies
    # we need to set all illegal moves to 0,
    # and then renormalize the probabilities
    # so that they sum to 1 again
    # first, get a mask of all illegal moves
    illegal_moves = board[:, :81] + board[:, 81:]
    # now set all illegal moves to 0
    model_prediction = tch.where(illegal_moves != 0, tch.zeros_like(model_prediction), model_prediction)

    return model_prediction

POLICY_SOFTMAX_TEMP = 1.3
def clean_model_prediction(model_prediction, board):
    model_prediction = mask_illegal_moves(model_prediction, board)
    model_prediction = model_prediction * POLICY_SOFTMAX_TEMP
    return model_prediction

# create a training loop
def train(model, optimizer, train_loader, val_loader, epochs=20, device="cpu"):
    losses = []
    val_losses = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        model.train()
        for batch_idx, (board_state, search_policy, game_outcome) in enumerate(train_loader):
            board_state = board_state.to(device)
            search_policy = search_policy.to(device)
            optimizer.zero_grad()
            raw_policy, value = model(board_state)
            masked_policy = clean_model_prediction(raw_policy, board_state)
            policy_loss = loss_fn(masked_policy, search_policy)
            value_loss = tch.nn.functional.mse_loss(value.view(-1), game_outcome)
            loss = policy_loss * 10 + value_loss
            loss.backward()
            optimizer.step()
            policy_loss_value = policy_loss.item()
            value_loss_value = value_loss.item()
            loss_value = loss.item()
            total_batch_idx = epoch * len(train_loader) + batch_idx
            losses.append((total_batch_idx, loss_value, policy_loss_value, value_loss_value))
            if batch_idx % 256 == 0:
                print(f"Training batch {batch_idx}/{len(train_loader)}: loss {loss_value}, policy loss {policy_loss_value}, value loss {value_loss_value}")
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_loss = 0.0
        model.eval()
        with tch.no_grad():
            for batch_idx, (board_state, search_policy, game_outcome) in enumerate(val_loader):
                board_state = board_state.to(device)
                search_policy = search_policy.to(device)
                raw_policy, value = model(board_state)
                masked_policy = clean_model_prediction(raw_policy, board_state)
                policy_loss = loss_fn(masked_policy, search_policy)
                value_loss = tch.nn.functional.mse_loss(value.view(-1), game_outcome)
                val_policy_loss += policy_loss.item()
                val_value_loss += value_loss.item()
                val_loss += (policy_loss + value_loss).item()
            val_policy_loss /= len(val_loader)
            val_value_loss /= len(val_loader)
            val_loss /= len(val_loader)
            total_batch_idx = (epoch + 1) * len(train_loader)
            val_losses.append((total_batch_idx, val_loss, val_policy_loss, val_value_loss))
            print(f"Validation loss {val_loss}, policy loss {val_policy_loss}, value loss {val_value_loss}")
    return losses, val_losses

# %%
# train the model
# print(f"Training on device {tch.cuda.get_device_name(0)}")
loss_trace, val_loss_trace = train(model, optimizer, train_loader, val_loader, epochs=10)

# %%
# plot the loss trace and validation loss trace
loss_trace = np.array(loss_trace)
val_loss_trace = np.array(val_loss_trace)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(loss_trace[:, 0], loss_trace[:, 1])
ax.plot(val_loss_trace[:, 0], val_loss_trace[:, 1])
# add the training loss trace smoothed
smoothed = np.convolve(loss_trace[:, 1], np.ones(1000)/1000, mode="valid")
ax.plot(loss_trace[500:-499, 0], smoothed)
# add dots to the validation loss trace
sc = ax.scatter(val_loss_trace[:, 0], val_loss_trace[:, 1], c="orange", s=10)
# bring the dots to the front
sc.set_zorder(10)
ax.set_xlabel("Batch")
ax.set_ylabel("Loss")
# put the legend in the middle right
ax.legend(["Training loss", "Validation loss", "Smoothed training loss"], loc="best")
ax.set_title("Loss")
fig.savefig(f"plots/{RUN_ID}/loss.png")

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(loss_trace[:, 0], loss_trace[:, 2])
ax.plot(val_loss_trace[:, 0], val_loss_trace[:, 2])
# add the training loss trace smoothed
smoothed = np.convolve(loss_trace[:, 2], np.ones(1000)/1000, mode="valid")
ax.plot(loss_trace[500:-499, 0], smoothed)
# add dots to the validation loss trace
sc = ax.scatter(val_loss_trace[:, 0], val_loss_trace[:, 2], c="orange", s=10)
# bring the dots to the front
sc.set_zorder(10)
ax.set_xlabel("Batch")
ax.set_ylabel("Policy Loss")
# put the legend in the middle right
ax.legend(["Training policy loss", "Validation policy loss", "Smoothed training policy loss"], loc="best")
ax.set_title("Policy Loss")
fig.savefig(f"plots/{RUN_ID}/policy_loss.png")

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(loss_trace[:, 0], loss_trace[:, 3])
ax.plot(val_loss_trace[:, 0], val_loss_trace[:, 3])
# add the training loss trace smoothed
smoothed = np.convolve(loss_trace[:, 3], np.ones(1000)/1000, mode="valid")
ax.plot(loss_trace[500:-499, 0], smoothed)
# add dots to the validation loss trace
sc = ax.scatter(val_loss_trace[:, 0], val_loss_trace[:, 3], c="orange", s=10)
# bring the dots to the front
sc.set_zorder(10)
ax.set_xlabel("Batch")
ax.set_ylabel("Value Loss")
# put the legend in the middle right
ax.legend(["Training value loss", "Validation value loss", "Smoothed training value loss"], loc="best")
ax.set_title("Value Loss")
fig.savefig(f"plots/{RUN_ID}/value_loss.png")

# %%
# save the model
tch.save(model.state_dict(), "model.pt")

# %%
# visualize the model's predictions
model.eval()

# get N random items from the validation set
N_SAMPLES = 100
rand_idx = np.random.randint(len(val_dataset), size=N_SAMPLES)
x_sample = x_val[rand_idx]
y_sample = y_val[rand_idx]
y_pred_raw, value = model(x_sample)
y_pred = clean_model_prediction(y_pred_raw, x_sample)
# apply softmax to get a probability distribution
y_pred = tch.nn.functional.softmax(y_pred, dim=1)

x_sample_re = x_sample.reshape(-1, 2, BOARD_SIDE_LEN, BOARD_SIDE_LEN)
y_sample_re = y_sample.reshape(-1, BOARD_SIDE_LEN, BOARD_SIDE_LEN)
y_pred_re = y_pred.detach().numpy().reshape(-1, BOARD_SIDE_LEN, BOARD_SIDE_LEN)

# %%
# plot the results
# the first column is the board state
# the second column is the search policy
# the third column is the neural network policy
fig, axs = plt.subplots(N_SAMPLES, 3, figsize=(7, int(math.ceil(13 / 5 * N_SAMPLES))))
# label the columns
axs[0, 0].set_title("Board state")
axs[0, 1].set_title("Search policy")
axs[0, 2].set_title("Neural network policy")
for i in range(N_SAMPLES):
    board_one = x_sample_re[i][0]
    board_two = x_sample_re[i][1]
    search = y_sample_re[i]
    nn = y_pred_re[i]
    # pieces_on_first_board = board_one.flatten().sum()
    # pieces_on_second_board = board_two.flatten().sum()
    # x_to_move = pieces_on_first_board != pieces_on_second_board
    # if not x_to_move:
    #     board_one, board_two = board_two, board_one
    board = np.stack([board_one, np.zeros((BOARD_SIDE_LEN, BOARD_SIDE_LEN)), board_two], axis=2) / 1.5
    search_policy_board = np.stack([search, search, search], axis=2)
    nn_policy_board = np.stack([nn, nn, nn], axis=2)

    # renormalise the policies so that the max prediction is 1.0
    search_policy_board *= 1.0 / search_policy_board.max()
    nn_policy_board *= 1.0 / nn_policy_board.max()

    search_policy_board += board
    nn_policy_board += board
    axs[i, 0].imshow(board, vmin=0, vmax=255, cmap="inferno")
    axs[i, 1].imshow(search_policy_board, vmin=0, vmax=255)
    axs[i, 2].imshow(nn_policy_board, vmin=0, vmax=255)

# set gridlines to 1x1
for ax in axs.flatten():
    ax.set_xticks(np.arange(-.5, 9, 1))
    ax.set_yticks(np.arange(-.5, 9, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="grey", linewidth=0.5)

# %%
# export model to ONNX
onnx_model_path = f"nets/{RUN_ID}-model.onnx"

# create a dummy input
dummy_input = tch.randn(1, 162)

# export the model
batch_axis = {0: "batch_size"}

tch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    verbose=True,
    input_names=["input"],
    output_names=["policy", "value"],
    dynamic_axes={"input": batch_axis, "policy": batch_axis, "value": batch_axis},
    opset_version=10,
)

# %%
# verify the model with onnxruntime
common_input_data = x_sample
pytorch_net_output = model(common_input_data)
pytorch_policy, pytorch_value = pytorch_net_output
print(f"PyTorch policy has shape {pytorch_policy.shape}")
print(f"PyTorch value has shape {pytorch_value.shape}")
print(f"x_sample has shape {common_input_data.shape}")
for i in range(len(common_input_data)):
    pytorch_policy[i] = clean_model_prediction(pytorch_policy[i].reshape(1, 81), common_input_data[i].reshape(1, 162))
pytorch_policy = tch.nn.functional.softmax(pytorch_policy, dim=1)
pytorch_policy = pytorch_policy.detach().numpy()
pytorch_value = pytorch_value.detach().numpy()
import onnxruntime as ort
ort_session = ort.InferenceSession(onnx_model_path)
onnx_net_output = []
for i in range(len(common_input_data)):
    input_thing = {"input": common_input_data[i].numpy().reshape(1, 162)}
    ort_session_out = ort_session.run(None, input_thing)
    policy = ort_session_out[0]
    np_policy = policy.reshape(1, 81)
    tensor_policy = tch.tensor(np_policy)
    policy = clean_model_prediction(tensor_policy, common_input_data[i].reshape(1, 162))
    policy = tch.nn.functional.softmax(policy, dim=1)
    onnx_net_output.append(policy)
onnx_net_output = np.squeeze(np.array(onnx_net_output), axis=1)

print(f"ONNX policy has shape {onnx_net_output.shape}")

onnx_policy = onnx_net_output

# compare the outputs
assert pytorch_policy.shape == onnx_policy.shape
assert np.allclose(pytorch_policy, onnx_policy, rtol=1e-03, atol=1e-05)


