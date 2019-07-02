import numpy as np
import tempfile
import matplotlib.pyplot as plt
import argparse

############################################
#
# Create Some patterns that we use for testing
#
############################################

strings = []
# strings.append("""
# ..X..
# .X.X.
# X...X
# .X.X.
# ..X..""")

strings.append("""
.....
.XXX.
.X.X.
.XXX.
.....""")

# strings.append("""
# .....
# .....
# ...X.
# ...X.
# ...XX""")

strings.append("""
.....
.....
X....
X....
XX...""")

strings.append("""
XXXXX
X....
X....
X....
X....""")

strings.append("""
.....
.....
XXXXX
.....
.....""")

strings.append("""
X....
.X...
..X..
...X.
....X""")

strings.append("""
....X
...X.
..X..
.X...
X....""")



def string2matrix(s):
    x = np.zeros(shape=(5, 5), dtype=float)
    for i in range(len(s)):
        row, col = i // 5, i % 5
        x[row][col] = -1 if s[i] == 'X' else 1
    return x
def matrix2string(m):
    s = ""
    for i in range(5):
        for j in range(5):
            s = s + ('X' if m[i][j] < 0 else '')
        s = s + chr(10)
    return s

class HopfieldNN:
    def __init__(self, neuron_num):
        self.N = neuron_num
        # self.W = np.random.rand(neuron_num, neuron_num)
        # self.syn = np.random.rand(neuron_num, 1)
        self.W = np.zeros((neuron_num, neuron_num))
        self.syn = np.zeros((neuron_num, 1))

    def Hebbian_train(self, S):
        self.W = np.matmul(S.transpose(), S)

    # Run one simulation step
    def runStep(self):
        i = np.random.randint(0, self.N)
        activation = np.matmul(self.W[i, :], self.syn)
        if activation < 0:
            self.syn[i] = -1
        else:
            self.syn[i] = 1

            #

    # Starting with a given state, execute the update rule
    # N times and return the resulting state
    #
    def run(self, state, steps):
        self.syn = state
        for i in range(steps):
            self.runStep()
        return self.syn

############################################
#
# Parse arguments
#
#############################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memories",
                    type=int,
                    default=3,
                    help="Number of patterns to learn")
    parser.add_argument("--epochs",
                    type=int,
                    default=10,
                    help="Number of epochs")
    parser.add_argument("--iterations",
                    type=int,
                    default=20,
                    help="Number of iterations per epoch")
    parser.add_argument("--errors",
                    type=int,
                    default=5,#20,
                    help="Number of error that we add to each sample")
    parser.add_argument("--save",
                    type=int,
                    default=0,
                    help="Save output")
    return parser.parse_args()

args = get_args()
# Number of epochs. After each
# epoch, we capture one image
#
epochs = args.epochs

#
# Number of iterations
# per epoch
#
iterations = args.iterations

#
# Number of bits that we flip in each sample
#
errors = args.errors

#
# Number of patterns that we try to memorize
#
memories = args.memories

#
# Init network
#
HpNet = HopfieldNN(5*5)

#
# Prepare sample data and train network
#

M = []
for _ in range(memories):
    temp = strings[_].replace(chr(10), '')
    M.append(string2matrix(strings[_].replace(chr(10), '')).reshape(1, 5*5))
# print(M.shape)
S = np.concatenate(M, axis=0)  # (3,25)
HpNet.Hebbian_train(S)

# Run the NN and display results
fig = plt.figure()

for pic in range(memories):
    state = (S[pic, :].reshape(25, 1)).copy()

    ax = fig.add_subplot(memories, epochs + 1, 1+pic * (epochs + 1))
    ax.set_xticks([], [])  # hidden each x+y axis value
    ax.set_yticks([], [])
    ax.imshow(state.reshape(5, 5), "binary_r")

    # Flip a few bits
    state = state.copy()

    for i in range(errors):
        index = np.random.randint(0, 25)
        state[index][0] = -1*state[index][0]

    # Run the network and display the current state at the beginning of each epoch
    for i in range(epochs):
        ax = fig.add_subplot(memories, 1+epochs, i+2+pic * (epochs + 1))
        ax.set_xticks([], [])  # hidden each x+y axis value
        ax.set_yticks([], [])
        ax.imshow(state.reshape(5, 5), "binary_r")
        state = HpNet.run(state, iterations)


if 1 == args.save:
    outfile = tempfile.mktemp() + "_Hopfield.png"
    print("Using outfile ", outfile)
    plt.savefig(outfile)


plt.show()



