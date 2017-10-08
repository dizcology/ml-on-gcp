import numpy as np
import torch
from torch.autograd import Variable
import gym

OBSERVATION_DIM = 80 * 80
HIDDEN_DIM = 20
BATCH_SIZE = 3
NUM_BATCHES = 5000
GAMMA = 0.95 # for discounted reward
LEARNING_RATE = 1e-3
DECAY = 0.95 # for RMSProp

RENDER = True
RESTORE = True


# Taken from:
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return discounted_r


def make_observation(state):
    flattened = prepro(state).reshape((1, -1))
    return Variable(torch.Tensor(flattened))
####

# Karpathy's example
model = torch.nn.Sequential(
    torch.nn.Linear(OBSERVATION_DIM, HIDDEN_DIM, bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN_DIM, 3, bias=False),
    torch.nn.Softmax(),
)

# deeper and narrower
model = torch.nn.Sequential(
    torch.nn.Linear(OBSERVATION_DIM, 20, bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 10, bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 3, bias=False),
    torch.nn.Softmax(),
)

batch_gradients = {}
gradients = {}
for w in model.parameters():
    batch_gradients[w] = torch.zeros(w.size())
    gradients[w] = []

    # NOTE: this seemed to make a huge difference in preventing the weights from exploding/dying at the first update.
    torch.nn.init.xavier_normal(w)

# Open AI gym Atari env: 0: 'NOOP', 2: 'UP' and 3: 'DOWN'
ACTIONS = [0, 2, 3]

def action_label(action):
    index = [i for i, a in enumerate(ACTIONS) if a == action][0]
    dummy = [0] * len(ACTIONS)
    dummy[index] = 1
    return dummy

def weighted_choice(as_, ws):
    total_weight = sum(ws)
    r = np.random.uniform(0, total_weight)
    for i, a in enumerate(as_):
        w = ws[i]
        r -= w
        if r <= 0:
            return a
    raise

env = gym.make("Pong-v0")

for i in range(NUM_BATCHES):
    for g in batch_gradients.values():
        g.zero_()

    batch_reward = 0.0

    for j in range(BATCH_SIZE):
        print('>>>>>>> {} / {} of batch {}'.format(j+1, BATCH_SIZE, i))
        state = env.reset()
        previous_x = None

        rewards = []

        for w in gradients:
            gradients[w] = []

        # The while loop for actions/steps
        while True:
            if RENDER:
                env.render()

            current_x = make_observation(state)
            observation = current_x - previous_x if previous_x is not None else Variable(torch.zeros(1, OBSERVATION_DIM))
            previous_x = current_x

            ps = model.forward(observation)[0]

            p = ps.data.tolist()

            action = weighted_choice(ACTIONS, ws=p)
            y = action_label(action)

            loss = -torch.dot(Variable(torch.Tensor(y)), torch.log(ps))
            
            loss.backward()

            state, reward, done, info = env.step(action)
            batch_reward += reward

            rewards.append(reward)

            for w in model.parameters():
                gradients[w].append(w.grad.data)
                            
            if done:
                rewards = discount_rewards(rewards)

                # Centering the rewards - should have the effect of boosting the rarer rewards, e.g. positive rewards at the beginning of training.
                rewards -= np.mean(rewards)
                rewards /= np.std(rewards)

                rewards = torch.Tensor(rewards)

                for w in model.parameters():
                    # policy gradient
                    stacked_gradient = torch.stack(gradients[w], dim=1)
                    add_gradient = torch.matmul(rewards, stacked_gradient)
                    batch_gradients[w] += add_gradient
                    gradients[w] = []

                break

    batch_reward /= BATCH_SIZE
    print('\t\tbatch_reward: {}'.format(batch_reward))

    print('updating weights!!!')
    for w in model.parameters():
        w.data += LEARNING_RATE * batch_gradients[w]

