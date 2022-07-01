from torch import nn
        
class RNNLayer(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.2):

        super(RNNLayer, self).__init__()

        self.norm = nn.LayerNorm(input_size)
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x, (h_n, c_n) = self.bilstm(self.norm(x))
        return self.dropout(x)

class RNN(nn.Module):

    def __init__(
        self,
        input_features,
        output_features,
        lstm_channels=512,
        **args
        ):

        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.lstm_channels = lstm_channels
        
        self.rnn = nn.Sequential(
            *[
            RNNLayer(
            input_size=lstm_channels * 2 if i > 0 else input_features, 
            hidden_size=lstm_channels,
            )
            for i in range(num_lstms)
            ]
            )
        self.fc = nn.Sequential(
            nn.Linear(in_features=lstm_channels * 2, out_features=lstm_channels),
            nn.ReLU(),
            nn.Linear(in_features=lstm_channels, out_features=output_features),
            )
    def forward(self, x):
        # x: (batch_size, num_frames, num_features)
        return self.fc(self.rnn(x.transpose(0, 1)).transpose(0, 1))

