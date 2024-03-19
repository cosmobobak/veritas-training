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
BOARD_SIDE_LEN = 7
DATASET_PATH = input("Enter the path to the dataset: ")
RUN_ID = input(f"Enter the run ID: ataxx{BOARD_SIDE_LEN}x{BOARD_SIDE_LEN}-")
RUN_ID = f"ataxx{BOARD_SIDE_LEN}x{BOARD_SIDE_LEN}-" + RUN_ID
POLICY_DIMENSIONALITY = (BOARD_SIDE_LEN * BOARD_SIDE_LEN) ** 2
INPUT_CHANNELS = 3 # white + black + gaps

# mkdir plots/{RUN_ID}
os.makedirs(f"plots/{RUN_ID}", exist_ok=True)

ROW_LIMIT = 550_000

print(f"Loading dataset from {DATASET_PATH}...")
positions       = np.loadtxt(f"{DATASET_PATH}/positions.csv", delimiter=",", max_rows=ROW_LIMIT)
# load rollout counts as float16
rollout_counts  = np.loadtxt(f"{DATASET_PATH}/policy-target.csv", delimiter=",", max_rows=ROW_LIMIT, dtype=np.float16)
results         = np.loadtxt(f"{DATASET_PATH}/value-target.csv", delimiter=",", max_rows=ROW_LIMIT)

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
split = int(len(positions) * 0.95)
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
# do this in-place to save memory

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
class AtaxxDataset(tch.utils.data.Dataset):
    def __init__(self, pos, policy, value):
        self.x = pos
        self.y = policy
        self.z = value

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx]

    def __len__(self):
        return len(self.x)

# create dataloaders
train_dataset = AtaxxDataset(x_train, y_train, z_train)
val_dataset = AtaxxDataset(x_val, y_val, z_val)
train_loader = tch.utils.data.DataLoader(train_dataset, batch_size=64)
val_loader = tch.utils.data.DataLoader(val_dataset, batch_size=64)

# %%
x_shaped = x_train.reshape(-1, INPUT_CHANNELS, BOARD_SIDE_LEN, BOARD_SIDE_LEN)
y_shaped = y_train.reshape(-1, BOARD_SIDE_LEN, BOARD_SIDE_LEN)

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
FINAL_CHANNELS = 128
LATENT_REPR_DIM = FINAL_CHANNELS * BOARD_SIDE_LEN * BOARD_SIDE_LEN
ATTENTION_POLICY_VECTOR_LENGTH = 32
BODY_WIDTH = 128
SE_CHANNELS = 16
class ResidualBlock(tch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = tch.nn.ReLU()
        self.sigmoid = tch.nn.Sigmoid()
        # Two 3x3 convolutions.
        self.conv1 = tch.nn.Conv2d(BODY_WIDTH, BODY_WIDTH, 3, padding=1)
        self.conv2 = tch.nn.Conv2d(BODY_WIDTH, BODY_WIDTH, 3, padding=1)
        # SE layer, i.e.:
        # - Global average pooling layer (FILTERS×7×7 to FILTERS)
        # - Fully connected layer (FILTERS to SE_CHANNELS)
        # - ReLU
        # - Fully connected layer (SE_CHANNELS to 2×FILTERS).
        # - 2×FILTERS is split into two FILTERS sized vectors W and B
        # - Z = Sigmoid(W)
        # - Output of the SE layer is (Z × input) + B.
        self.global_avg_pool = tch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = tch.nn.Linear(BODY_WIDTH, SE_CHANNELS)
        self.fc2 = tch.nn.Linear(SE_CHANNELS, 2 * BODY_WIDTH)

    def forward(self, inp):
        # perform two convolutions
        x = self.relu(self.conv1(inp))
        x = self.relu(self.conv2(x))
        # SE layer
        se = self.global_avg_pool(x).view(-1, BODY_WIDTH)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        w = se[:, :BODY_WIDTH].view(-1, BODY_WIDTH, 1, 1)
        b = se[:, BODY_WIDTH:].view(-1, BODY_WIDTH, 1, 1)
        z = self.sigmoid(w)
        # excitation:
        x = (z * x) + b
        # add residual
        return x + inp

class ConvPolicyModel(tch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu    = tch.nn.ReLU()
        # replace ReLU with an activation function that doesn't kill gradients
        # self.relu    = tch.nn.Mish()
        self.sigmoid = tch.nn.Sigmoid()
        # five layers of 3x3 convs mean that information can travel at most five squares away.
        # this is maybe fine idk

        #########################################
        ####             TRUNK                ###
        #########################################
        self.input_conv   = tch.nn.Conv2d(INPUT_CHANNELS, BODY_WIDTH, 3, padding=1) # INPUT_CHANNELS x SIDE x SIDE -> 64 x SIDE x SIDE
        self.block1  = ResidualBlock()
        self.block2  = ResidualBlock()
        self.block3  = ResidualBlock()
        self.block4  = ResidualBlock()
        self.block5  = ResidualBlock()
        self.trunk_end   = tch.nn.Conv2d(BODY_WIDTH, FINAL_CHANNELS, 3, padding=1) # 16 x SIDE x SIDE -> {FINAL_CHANNELS} x SIDE x SIDE
        #########################################
        ###           POLICY HEAD             ###
        #########################################
        self.policy1 = tch.nn.Conv2d(FINAL_CHANNELS, FINAL_CHANNELS // 2, 1) # {FINAL_CHANNELS} x SIDE x SIDE -> {FINAL_CHANNELS} x SIDE x SIDE
        self.policy_src = tch.nn.Conv2d(FINAL_CHANNELS // 2, ATTENTION_POLICY_VECTOR_LENGTH, 1)
        self.policy_tgt = tch.nn.Conv2d(FINAL_CHANNELS // 2, ATTENTION_POLICY_VECTOR_LENGTH, 1)
        #########################################
        ###        SOFT POLICY HEAD           ###
        #########################################
        self.soft_policy1 = tch.nn.Conv2d(FINAL_CHANNELS, FINAL_CHANNELS // 2, 1) # {FINAL_CHANNELS} x SIDE x SIDE -> {FINAL_CHANNELS} x SIDE x SIDE
        self.soft_policy_src = tch.nn.Conv2d(FINAL_CHANNELS // 2, ATTENTION_POLICY_VECTOR_LENGTH, 1)
        self.soft_policy_tgt = tch.nn.Conv2d(FINAL_CHANNELS // 2, ATTENTION_POLICY_VECTOR_LENGTH, 1)
        #########################################
        ###            VALUE HEAD             ###
        #########################################
        self.value1  = tch.nn.Conv2d(FINAL_CHANNELS, 1, 1) # {FINAL_CHANNELS} x SIDE x SIDE -> 1 x SIDE x SIDE
        self.value2  = tch.nn.Linear(SQUARES, 1)

        # initialize the weights
        for m in self.modules():
            if isinstance(m, tch.nn.Conv2d):
                tch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    tch.nn.init.zeros_(m.bias)
            elif isinstance(m, tch.nn.Linear):
                tch.nn.init.xavier_uniform_(m.weight)
                tch.nn.init.zeros_(m.bias)

    def forward(self, x):
        #########################################
        ###              TRUNK                ###
        #########################################

        # initial transformation
        x = x.view(-1, INPUT_CHANNELS, BOARD_SIDE_LEN, BOARD_SIDE_LEN)
        x = self.relu(self.input_conv(x))

        # residual trunk
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        # latent representation used by all heads
        latent = self.relu(self.trunk_end(x))

        #########################################
        ###           POLICY HEAD             ###
        #########################################
        x = self.relu(self.policy1(latent))
        src = self.policy_src(x)
        tgt = self.policy_tgt(x)
        src = src.reshape(-1, ATTENTION_POLICY_VECTOR_LENGTH, 7 * 7)
        tgt = tgt.reshape(-1, ATTENTION_POLICY_VECTOR_LENGTH, 7 * 7)
        # to get the policy for a certain source-target pair, we take the dot product of the source vector and the target vector.
        # this gives us a 7x7x7x7 tensor.
        src = src.transpose(1, 2)
        policy_logits = (src @ tgt).view(-1, BOARD_SIDE_LEN * BOARD_SIDE_LEN * BOARD_SIDE_LEN * BOARD_SIDE_LEN)

        #########################################
        ###         SOFT POLICY HEAD          ###
        #########################################
        # only run the soft policy head during training:
        if self.training:
            x = self.relu(self.soft_policy1(latent))
            soft_src = self.soft_policy_src(x)
            soft_tgt = self.soft_policy_tgt(x)
            soft_src = soft_src.reshape(-1, ATTENTION_POLICY_VECTOR_LENGTH, 7 * 7)
            soft_tgt = soft_tgt.reshape(-1, ATTENTION_POLICY_VECTOR_LENGTH, 7 * 7)
            soft_src = soft_src.transpose(1, 2)
            soft_policy_logits = (soft_src @ soft_tgt).view(-1, BOARD_SIDE_LEN * BOARD_SIDE_LEN * BOARD_SIDE_LEN * BOARD_SIDE_LEN)

        #########################################
        ###            VALUE HEAD             ###
        #########################################
        x = self.value1(latent).view(-1, SQUARES)
        value = self.sigmoid(self.value2(x))

        if self.training:
            return policy_logits, value, soft_policy_logits
        else:
            return policy_logits, value


# %%
# create the model and optimizer
model = ConvPolicyModel()
optimizer = tch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.03)

# create a loss function to match the probability distribution
def loss_fn(prediction_logits, target_distribution):
    return tch.nn.functional.binary_cross_entropy(tch.nn.functional.softmax(prediction_logits, dim=1), target_distribution)

# create a function for masking off illegal moves
def mask_illegal_moves(model_prediction, board):
    # we just no-op because we don't have ataxx movegen in python atm
    return model_prediction

POLICY_SOFTMAX_TEMP = 1.2
POLICY_SOFTMAX_TEMP_SOFT_POLICY_HEAD = 4.0
def clean_model_prediction(model_prediction, soft_model_prediction, board):
    model_prediction = mask_illegal_moves(model_prediction, board)
    soft_model_prediction = mask_illegal_moves(soft_model_prediction, board)
    model_prediction = model_prediction * POLICY_SOFTMAX_TEMP
    soft_model_prediction = soft_model_prediction * POLICY_SOFTMAX_TEMP_SOFT_POLICY_HEAD
    return model_prediction, soft_model_prediction

def decompose_model_prediction(model_prediction):
    # takes the 7x7x7x7 element model prediction and gives a from-map and a to-map, each of which are only 7x7
    # the from-map is the probability of a piece moving from that square
    # the to-map is the probability of a piece moving to that square
    # with a special case for the "no move" square
    src_map = np.zeros((BOARD_SIDE_LEN, BOARD_SIDE_LEN))
    tgt_map = np.zeros((BOARD_SIDE_LEN, BOARD_SIDE_LEN))
    for i, e in enumerate(model_prediction):
        src = i // (BOARD_SIDE_LEN * BOARD_SIDE_LEN)
        tgt = i % (BOARD_SIDE_LEN * BOARD_SIDE_LEN)
        src_map[src // BOARD_SIDE_LEN, src % BOARD_SIDE_LEN] += e
        tgt_map[tgt // BOARD_SIDE_LEN, tgt % BOARD_SIDE_LEN] += e
    # normalise the maps
    src_map /= src_map.sum()
    tgt_map /= tgt_map.sum()
    return src_map, tgt_map

def decompose_model_prediction_batch(model_prediction):
    # takes a batch of 7x7x7x7 element model predictions and gives a batch of from-maps and a batch of to-maps, each of which are only 7x7
    # so input is (batch, 7*7*7*7) and output is (batch, 7, 7), (batch, 7, 7)
    src_maps = np.zeros((model_prediction.shape[0], BOARD_SIDE_LEN, BOARD_SIDE_LEN))
    tgt_maps = np.zeros((model_prediction.shape[0], BOARD_SIDE_LEN, BOARD_SIDE_LEN))
    for i, e in enumerate(model_prediction):
        src_map, tgt_map = decompose_model_prediction(e)
        src_maps[i] = src_map
        tgt_maps[i] = tgt_map
    return src_maps, tgt_maps

# create a training loop
def train(model, optimizer, train_loader, val_loader, epochs=20, device="cpu"):
    losses = []
    val_losses = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        model.train()
        for batch_idx, (board_state, search_policy, game_outcome) in enumerate(train_loader):
            # convert search_policy from float16 to float32
            search_policy = search_policy.to(tch.float32)
            board_state = board_state.to(device)
            search_policy = search_policy.to(device)
            optimizer.zero_grad()
            raw_policy, value, soft_raw_policy = model(board_state)
            masked_policy, soft_masked_policy = clean_model_prediction(raw_policy, soft_raw_policy, board_state)
            policy_loss = loss_fn(masked_policy, search_policy)
            soft_policy_loss = loss_fn(soft_masked_policy, search_policy)
            value_loss = tch.nn.functional.mse_loss(value.view(-1), game_outcome)
            loss = policy_loss + value_loss + soft_policy_loss
            loss.backward()
            tch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            policy_loss_value = policy_loss.item()
            value_loss_value = value_loss.item()
            loss_value = loss.item()
            total_batch_idx = epoch * len(train_loader) + batch_idx
            losses.append((total_batch_idx, loss_value, policy_loss_value, value_loss_value))
            if batch_idx % 256 == 0:
                print(f"Training batch {batch_idx}/{len(train_loader)}: loss {loss_value:.7f}, policy loss {policy_loss_value:.7f}, value loss {value_loss_value:.7f}")
                val_policy_loss = 0.0
                val_value_loss = 0.0
                val_loss = 0.0
                model.eval()
                with tch.no_grad():
                    for batch_idx, (board_state, search_policy, game_outcome) in enumerate(val_loader):
                        board_state = board_state.to(device)
                        search_policy = search_policy.to(device)
                        raw_policy, value = model(board_state)
                        masked_policy, _ = clean_model_prediction(raw_policy, raw_policy, board_state)
                        policy_loss = loss_fn(masked_policy, search_policy)
                        value_loss = tch.nn.functional.mse_loss(value.view(-1), game_outcome)
                        val_policy_loss += policy_loss.item()
                        val_value_loss += value_loss.item()
                        val_loss += (policy_loss + value_loss).item()
                    val_policy_loss /= len(val_loader)
                    val_value_loss /= len(val_loader)
                    val_loss /= len(val_loader)
                    val_losses.append((total_batch_idx, val_loss, val_policy_loss, val_value_loss))
                    print(f"               Validation loss {val_loss:.7f}, policy loss {val_policy_loss:.7f}, value loss {val_value_loss:.7f}")
                model.train()
    return losses, val_losses

# %%
# train the model
# print(f"Training on device {tch.cuda.get_device_name(0)}")
loss_trace, val_loss_trace = train(model, optimizer, train_loader, val_loader, epochs=2)

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
ax.set_title(f"Loss for {RUN_ID}")
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
ax.set_title(f"Policy Loss for {RUN_ID}")
fig.savefig(f"plots/{RUN_ID}/policy_loss.png")

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(loss_trace[:, 0], loss_trace[:, 3])
ax.plot(val_loss_trace[:, 0], val_loss_trace[:, 3])
# add the training loss trace smoothed
smoothed = np.convolve(loss_trace[:, 3], np.ones(1000)/1000, mode="valid")
ax.plot(loss_trace[500:-499, 0], smoothed)
# add a horizontal line at 0.25, marked "random guessing"
rg = ax.axhline(0.25, color="red", linestyle="--")
# add dots to the validation loss trace
sc = ax.scatter(val_loss_trace[:, 0], val_loss_trace[:, 3], c="orange", s=10)
# bring the dots to the front
sc.set_zorder(10)
ax.set_xlabel("Batch")
ax.set_ylabel("Value Loss")
# put the legend in the middle right
ax.legend(["Training value loss", "Validation value loss", "Smoothed training value loss", "Random Guessing"], loc="best")
ax.set_title(f"Value Loss for {RUN_ID}")
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
y_pred, _ = clean_model_prediction(y_pred_raw, y_pred_raw, x_sample)
# apply softmax to get a probability distribution
y_pred = tch.nn.functional.softmax(y_pred, dim=1)

x_sample_re = x_sample.reshape(-1, INPUT_CHANNELS, BOARD_SIDE_LEN, BOARD_SIDE_LEN)
y_sample_re_src, y_sample_re_tgt = decompose_model_prediction_batch(y_sample)
y_pred_re_src, y_pred_re_tgt = decompose_model_prediction_batch(y_pred.detach().numpy())

# %%
# plot the results
# the first column is the board state
# the second column is the search policy
# the third column is the neural network policy
fig, axs = plt.subplots(N_SAMPLES, 5, figsize=(int(math.ceil(7 / 3 * 5)), int(math.ceil(13 / 5 * N_SAMPLES))))
# label the columns
axs[0, 0].set_title("Board state")
axs[0, 1].set_title("Search policy source")
axs[0, 2].set_title("Search policy target")
axs[0, 3].set_title("Neural network policy source")
axs[0, 4].set_title("Neural network policy target")
# make the column titles slanted
for ax in axs[0]:
    ax.title.set_fontsize(8)
    ax.title.set_fontstyle("italic")
    ax.title.set_rotation(45)
    ax.title.set_position((0.5, 0.5))
    ax.title.set_verticalalignment("bottom")
    ax.title.set_horizontalalignment("center")

for i in range(N_SAMPLES):
    board_one = x_sample_re[i][0]
    board_two = x_sample_re[i][1]
    # search = y_sample_re[i]
    search_src = y_sample_re_src[i]
    search_tgt = y_sample_re_tgt[i]
    # nn = y_pred_re[i]
    nn_src = y_pred_re_src[i]
    nn_tgt = y_pred_re_tgt[i]
    # pieces_on_first_board = board_one.flatten().sum()
    # pieces_on_second_board = board_two.flatten().sum()
    # x_to_move = pieces_on_first_board != pieces_on_second_board
    # if not x_to_move:
    #     board_one, board_two = board_two, board_one
    board = np.stack([board_one, np.zeros((BOARD_SIDE_LEN, BOARD_SIDE_LEN)), board_two], axis=2) / 1.5
    # search_policy_board = np.stack([search, search, search], axis=2)
    search_policy_board_src = np.stack([search_src, search_src, search_src], axis=2)
    search_policy_board_tgt = np.stack([search_tgt, search_tgt, search_tgt], axis=2)
    # nn_policy_board = np.stack([nn, nn, nn], axis=2)
    nn_policy_board_src = np.stack([nn_src, nn_src, nn_src], axis=2)
    nn_policy_board_tgt = np.stack([nn_tgt, nn_tgt, nn_tgt], axis=2)

    # renormalise the policies so that the max prediction is 1.0
    # search_policy_board *= 1.0 / search_policy_board.max()
    search_policy_board_src *= 1.0 / search_policy_board_src.max()
    search_policy_board_tgt *= 1.0 / search_policy_board_tgt.max()
    # nn_policy_board *= 1.0 / nn_policy_board.max()
    nn_policy_board_src *= 1.0 / nn_policy_board_src.max()
    nn_policy_board_tgt *= 1.0 / nn_policy_board_tgt.max()

    search_policy_board_src += board
    search_policy_board_tgt += board
    nn_policy_board_src += board
    nn_policy_board_tgt += board
    axs[i, 0].imshow(board, vmin=0, vmax=255, cmap="inferno")
    axs[i, 1].imshow(search_policy_board_src, vmin=0, vmax=255)
    axs[i, 2].imshow(search_policy_board_tgt, vmin=0, vmax=255)
    axs[i, 3].imshow(nn_policy_board_src, vmin=0, vmax=255)
    axs[i, 4].imshow(nn_policy_board_tgt, vmin=0, vmax=255)

# set gridlines to 1x1
for ax in axs.flatten():
    ax.set_xticks(np.arange(-.5, BOARD_SIDE_LEN, 1))
    ax.set_yticks(np.arange(-.5, BOARD_SIDE_LEN, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="grey", linewidth=0.5)

fig.savefig(f"plots/{RUN_ID}/sample_predictions.png")

# %%
# export model to ONNX
onnx_model_path = f"nets/{RUN_ID}-model.onnx"

# create a dummy input
dummy_input = tch.randn(1, INPUT_CHANNELS * BOARD_SIDE_LEN * BOARD_SIDE_LEN)

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
    t, _ = clean_model_prediction(pytorch_policy[i].reshape(1, POLICY_DIMENSIONALITY), pytorch_policy[i].reshape(1, POLICY_DIMENSIONALITY), common_input_data[i].reshape(1, INPUT_CHANNELS * BOARD_SIDE_LEN * BOARD_SIDE_LEN))
    pytorch_policy[i] = t
pytorch_policy = tch.nn.functional.softmax(pytorch_policy, dim=1)
pytorch_policy = pytorch_policy.detach().numpy()
pytorch_value = pytorch_value.detach().numpy()
import onnxruntime as ort
ort_session = ort.InferenceSession(onnx_model_path)
onnx_net_output = []
for i in range(len(common_input_data)):
    input_thing = {"input": common_input_data[i].numpy().reshape(1, INPUT_CHANNELS * BOARD_SIDE_LEN * BOARD_SIDE_LEN)}
    ort_session_out = ort_session.run(None, input_thing)
    policy = ort_session_out[0]
    np_policy = policy.reshape(1, POLICY_DIMENSIONALITY)
    tensor_policy = tch.tensor(np_policy)
    policy, _ = clean_model_prediction(tensor_policy, tensor_policy, common_input_data[i].reshape(1, INPUT_CHANNELS * BOARD_SIDE_LEN * BOARD_SIDE_LEN))
    policy = tch.nn.functional.softmax(policy, dim=1)
    onnx_net_output.append(policy)
onnx_net_output = np.squeeze(np.array(onnx_net_output), axis=1)

print(f"ONNX policy has shape {onnx_net_output.shape}")

onnx_policy = onnx_net_output

# compare the outputs
assert pytorch_policy.shape == onnx_policy.shape
assert np.allclose(pytorch_policy, onnx_policy, rtol=1e-03, atol=1e-05)


