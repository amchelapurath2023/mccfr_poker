import matplotlib.pyplot as plt

def plot_regret(regret_history):
    plt.plot(regret_history)
    plt.xlabel("Iteration (x{})".format(100))  # if updateInterval = 100
    plt.ylabel("Cumulative Regret")
    plt.title("CFR Regret Over Time")
    plt.grid(True)
    plt.show()
