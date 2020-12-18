import torch.nn as nn


class MyLSTM(nn.Module):

    def __init__(self):

        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=128, hidden_size=8)

    def forward(self, x, h0):

        return self.lstm(x, h0)

if __name__ == '__main__':
    my_lstm = MyLSTM()
    target_lstm = MyLSTM()
    target_lstm.load_state_dict(my_lstm.state_dict())
    print('Finish !')

