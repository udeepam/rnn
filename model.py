import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self,
                 model,
                 input_size,
                 embed_size,
                 rnn_hidden_size,
                 rnn_num_layers,
                 hidden_layer_size=None,
                 precomputed_embeds=None):
        super().__init__()
        """
        Parameters:
        -----------
        model : `str`
            Recurrent layer type {rnn, gru, lstm}.
        input_size : `int``
            Number of unique words and symbols in the texts.
        embed_size : `int``
            The size of the embedding space.
        rnn_hidden_size : `int`
            The number of features in the hidden state.
        rnn_num_layers : `int`
            Number of recurrent layers. e.g. setting n_layers=2
            would mean stacking two RNNs together to form a stacked RNN,
            with the second RNN taking in outputs of the first RNN and
            computing the final results.
        hidden_layer_size : `list`
            The number of hidden layers for the downstream MLP.
        precomputed_embeds: `numpy.ndarray`
            Precomputed weight matrix for embedding layer.
        """
        super().__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.embed_size = embed_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.hidden_layer_size = hidden_layer_size

        # define embedding layer
        self.encoder = nn.Embedding(input_size, embed_size)

        # define Recurrent Layers
        if self.model == "rnn":
            self.rnn = nn.RNN(input_size=embed_size,
                              hidden_size=rnn_hidden_size,
                              num_layers=rnn_num_layers,
                              batch_first=True)
        elif self.model == "gru":
            self.rnn = nn.GRU(input_size=embed_size,
                              hidden_size=rnn_hidden_size,
                              num_layers=rnn_num_layers,
                              batch_first=True)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(input_size=embed_size,
                               hidden_size=rnn_hidden_size,
                               num_layers=rnn_num_layers,
                               batch_first=True)

        # define decoder: MLP and/or fully connected layer
        if hidden_layer_size is not None:
            self.mlp = nn.ModuleList()
            for i, hidden_layer in enumerate(hidden_layer_size):
                if i == 0:
                    self.mlp.append(nn.Linear(in_features=rnn_hidden_size,
                                              out_features=hidden_layer))
                else:
                    self.mlp.append(nn.Linear(in_features=hidden_layer_size[i-1],
                                              out_features=hidden_layer))
            # define fully connected ouput Layer
            self.fc = nn.Linear(in_features=hidden_layer_size[-1],
                                out_features=input_size)
        else:
            # define fully connected ouput Layer
            self.fc = nn.Linear(in_features=rnn_hidden_size,
                                out_features=input_size)
        self.relu = nn.ReLU(inplace=True)

        # initialise the weights and biases of the network
        self.init_weights()

        # initialise the embedding layer with the precomputed weights
        if precomputed_embeds is not None:
            self.encoder.weight.data.copy_(precomputed_embeds)

    def forward(self, input, hidden):
        """
        Forward pass through the network.

        Parameters:
        -----------
        input : `torch.Tensor`
            Input to the model, [batch_size, seq_len].
        hidden: "rnn" & "gru" = `torch.Tensor` or "lstm" = `tuple` of `torch.Tensor`
            "rnn" & "gru" = [rnn_num_layers, batch_size, rnn_hidden_size]
                   "lstm" = ([rnn_num_layers, batch_size, rnn_hidden_size], [rnn_num_layers, batch_size, rnn_hidden_size])
                            tuple of parameters (hidden_state, cell_state).

        Returns:
        --------
        x : `torch.Tensor`
            Output of predictions, [batch_size, seq_len, input_size].
        hidden :
            The hidden state of the RNN.
        """
        # encode the input using embedding layer
        x = self.encoder(input)
        # pass through RNN
        x, hidden = self.rnn(x, hidden)

        # pass through MLP
        if self.hidden_layer_size is not None:
            for i in range(len(self.mlp)):
                x = self.relu(self.mlp[i](x))

        # pass through final fully connected output layer
        """
        No activation function is applied here during training
        because the cross entropy loss function applies one for us.
        However when predicting we will need to put a softmax layer on top
        to obatin a probability distribution.
        """
        x = self.fc(x)
        return x, hidden

    def init_weights(self):
        """
        Initialise the weights and biases of the network.
        """
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.uniform_(param.data, -0.1, 0.1)

    def init_hidden(self, batch_size, device):
        """
        Initialise the hidden state of the RNN to zeros.
        """
        weight = next(self.parameters()).data
        if self.model=="lstm":
            return (Variable(weight.new(self.rnn_num_layers, batch_size, self.rnn_hidden_size).zero_().to(device)),
                    Variable(weight.new(self.rnn_num_layers, batch_size, self.rnn_hidden_size).zero_()).to(device))
        else:
            return Variable(weight.new(self.rnn_num_layers, batch_size, self.rnn_hidden_size).zero_().to(device))

    def repackage_hidden(self, hidden):
        """
        Reset the hidden state of the RNN.
        Wraps hidden states in new Variables, to detach them from their history.

        https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426/13
        https://discuss.pytorch.org/t/help-clarifying-repackage-hidden-in-word-language-model/226/6
        """
        if self.model=="lstm":
            return tuple(Variable(v.data) for v in hidden)
        else:
            return Variable(hidden.data)
