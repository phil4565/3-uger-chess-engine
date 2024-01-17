import json
import numpy as np
import torch
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


batch_size = 64 
num_epochs = 20

def read_data(json_file):
    count = 0
    data = {}
    with open(json_file, 'r') as f:
        for line in f:
            obj = json.loads(line)
            fen = obj['fen'].split(' ')[0]
            try:
                stockfish_eval = obj['evals'][0]['pvs'][0]['cp']
            except:
                stockfish_eval = obj['evals'][0]['pvs'][0]['mate']
            data[fen] = stockfish_eval
            count += 1
            if count % 10000 == 0:
                print(count)
        json_object = json.dumps(data, indent=4)
        with open("data.json", "w") as outfile:
            outfile.write(json_object)

def get_rep_of_fen(fen):
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    layers = []
    for piece in pieces:
        layers.append(create_rep_layer(fen, piece))
    board_rep = np.stack(layers)
    return torch.stack([torch.from_numpy(board_rep).to(torch.float32)])

def get_input_and_expected(file):
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    with open(file, 'r') as f:
        data = json.load(f)
    count = 0
    board_representations = []
    expected_values = []
    for k, v in data.items():
        layers = []
        # limit the amount of data, since numpy cant handle 14.000.000 entries in a list
        if count % 10 == 0:
            fen = k
            for piece in pieces:
                layers.append(create_rep_layer(fen, piece))
            board_rep = np.stack(layers)
            board_representations.append(board_rep)
            expected_values.append(v)
        count += 1
    return np.array(board_representations), np.stack(expected_values)

def create_rep_layer(board, type):
    mat_rep = []
    board_and_rights = board.split(' ')
    board_only = board_and_rights[0]
    rights_only = board_and_rights[1:]
    board_lines = board_only.split('/')

    for board_line in board_lines:
        new_line = []
        for char in board_line:
            if not char.isalpha():
                for i in range(int(char)):
                    new_line.append(0)
            elif char == type.upper():
                new_line.append(1)
            elif char == type.lower():
                new_line.append(-1)
            else:
                new_line.append(0)
        mat_rep.append(new_line)
    return np.array(mat_rep)

def get_nn_model():
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    #%% Neural network
    net = torch.nn.Sequential(
        torch.nn.Conv2d(6, 16, kernel_size=3), # b x 16 x 6 x 6
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 64, kernel_size=3), # b x 32 x 4 x 4
        torch.nn.ReLU(),
        torch.nn.Flatten(),             # 32 x 4 x 4 = 1024
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 1)
    ).to(device)
    return net

def split_data(file):
    input, expected = get_input_and_expected(file)

    # Define the percentage of data to use for validation
    validation_split = 0.2

    # Calculate the split index
    split_index = int(len(input) * (1 - validation_split))

    # Split the data into training and validation sets
    train_data, val_data = input[:split_index], input[split_index:]
    train_expected, val_expected = expected[:split_index], expected[split_index:]

    train_input_tensor = torch.from_numpy(train_data)
    train_expected_tensor = torch.from_numpy(train_expected).view(train_expected.size, 1)
    val_input_tensor = torch.from_numpy(val_data)
    val_expected_tensor = torch.from_numpy(val_expected).view(val_expected.size, 1)

    #%% Create dataloaders
    train_data = TensorDataset(train_input_tensor, train_expected_tensor)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = TensorDataset(val_input_tensor, val_expected_tensor)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)
    return train_loader, val_loader

def train_nn(net):
    #%% Loss and optimizer
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=.005)

    #%% Metrics
    mse = torchmetrics.MeanAbsoluteError()
    loss_values_train = []
    loss_values_val = []
    #%% Train
    net.train()
    best_val_error = 1000000
    for epoch in range(num_epochs):
        train_errors = []
        for x, y in train_loader:
            # Put data on GPU 
            x = x.to(device)
            y = y.to(device)

            # Compute loss and take gradient step
            x = x.to(torch.float32)
            out = net(x)
            y = y.to(torch.float32)
            loss = loss_function(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            error = mse(out, y)
            train_errors.append(error)

        # Print accuracy for epoch
        train_avarage_error = sum(train_errors) / len(train_errors)
        print(f'Training Error: {train_avarage_error}')
        loss_values_train.append(train_avarage_error.detach().numpy())
        val_error = validate_nn(net)
        print(f'Validation error: {val_error}')
        loss_values_val.append(val_error.detach().numpy())
        if val_error < best_val_error:
            best_val_error = val_error
            # save the model
            torch.save(net.state_dict(), 'model_weights.pth')
    
    # plot loss
    plt.plot(range(1, num_epochs + 1), loss_values_train, label='Training loss')
    plt.plot(range(1, num_epochs + 1), loss_values_val, label='Validation loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return net

def validate_nn(net):
    #%% Metrics
    mse = torchmetrics.MeanAbsoluteError()
    loss_function = torch.nn.L1Loss()
    
    net.eval()
    val_errors = []
    for x, y in val_loader:
        # Put data on GPU 
        x = x.to(device)
        y = y.to(device)

        # Compute loss
        x = x.to(torch.float32)
        out = net(x)
        y = y.to(torch.float32)
        loss = loss_function(out, y)
        error = mse(out, y)

        val_errors.append(error)
    val_avarage_error = sum(val_errors) / len(val_errors)
    return val_avarage_error

def validate_single_position(fen):
    net = get_nn_model()
    net.load_state_dict(torch.load('model_weights.pth'))
    net.eval()
    out = net(get_rep_of_fen(fen))
    return out.item()

if __name__ == '__main__':
    #chess_game = Chess_game_handler()
    #file_path = 'C:\\Users\\phili\\Downloads\\lichess_db_eval.json\\data.json'

    # Run on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    train_loader, val_loader = split_data('data.json')

    net = get_nn_model()

    # Load saved model
    net.load_state_dict(torch.load('model_weights.pth'))

    # train the model
    #net = train_nn(net)

    print(f'Validation error: {validate_nn(net)}')

    print(validate_single_position('rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR'))