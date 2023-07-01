# Neural-Network-ML
Neural Networking code
def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_ih = np.random.randn(input_size, hidden_size)
        self.weights_ho = np.random.randn(hidden_size, output_size)

    def forward(self, inputs):
        hidden = np.dot(inputs, self.weights_ih)
        hidden = np.tanh(hidden)
        output = np.dot(hidden, self.weights_ho)
        return output

def main():
    network = NeuralNetwork(2, 3, 1)
    inputs = np.array([1, 2])
    output = network.forward(inputs)
    print(output)

if __name__ == "__main__":
    main()
