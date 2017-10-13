import numpy as np

# helpers taken from:
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def onehot(x, xs):
    assert x in xs

    index = [i for i, y in enumerate(xs) if y == x][0]
    dummy = [0] * len(xs)
    dummy[index] = 1
    return dummy


def weighted_choice(xs, ws):
    assert len(xs) == len(ws)

    total_weight = sum(ws)
    r = np.random.uniform(0, total_weight)
    for i, x in enumerate(xs):
        w = ws[i]
        r -= w
        if r <= 0:
            return x
