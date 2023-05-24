
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

def generate_target():
    num_nodes = 200

    # Generate adjacency matrix representation of the graph
    adjacency_matrix = np.zeros((num_nodes, num_nodes),dtype=int)
    for i in range(num_nodes):
        num_children = np.random.randint(1, 6)  # Random number of children (1 to 5)
        children = np.random.choice(num_nodes, num_children, replace=False)  # Randomly select children
        adjacency_matrix[i, : num_children] = children  # Assign children to adjacency matrix

    # Save the adjacency matrix as "target.npy"
    np.save("target.npy", adjacency_matrix)


def generate_query_graph(target_graph, num_nodes_query):
    num_nodes_target = target_graph.shape[0]
    query_graph = np.zeros((40, 40))
    
    # Randomly select 20-40 nodes from the target graph
    selected_nodes = np.random.choice(num_nodes_target, num_nodes_query, replace=False)
    
    ind_map={v:i for i,v in enumerate(selected_nodes)}
 
    # Assign node IDs and child relationships in the query graph based on the target graph
    for i, node_id in enumerate(selected_nodes):
        t=0
        for v in target_graph[node_id]:
            if v==0:break
            if v in selected_nodes:
                query_graph[i,t]=ind_map[v]
                t+=1
    
    # Generate the position mapping for the query graph in the target graph
    position_mapping = {i:node_id for i, node_id in enumerate(selected_nodes)}
    
    return query_graph, position_mapping

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.fc(output)
        return output, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target):
        batch_size = source.size(0)
        target_len = target.size(1)
        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(source.device)

        hidden, cell = self.encoder(source)

        decoder_input = target[:, 0] - 1

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = output.squeeze(1)
            decoder_input = output.argmax(2)

        return outputs
        
def main():
    # 定义模型和参数
    hidden_size = 64  # 隐藏层大小
    num_layers = 2  # LSTM层数
    learning_rate = 0.001
    num_epochs = 10

    # 生成训练集和测试集
    generate_target()
    target_graph = np.load("target.npy")
    num_samples = 10000
    query_graphs = []
    target_positions = []

    for _ in range(num_samples):
        num_nodes_query = np.random.randint(20, 40)  # 随机生成查询图节点数量
        query_graph, position_mapping = generate_query_graph(target_graph, num_nodes_query)
        query_graphs.append(query_graph)
        target_positions.append([position_mapping[i] for i in range(num_nodes_query)]+[0] * (40 - num_nodes_query))

    query_graphs = np.array(query_graphs)
    target_positions = np.array(target_positions)

    X_train, X_test, y_train, y_test = train_test_split(query_graphs, target_positions, test_size=0.2)

    # 定义模型
    encoder = Encoder(hidden_size, num_layers)
    decoder = Decoder(hidden_size, num_layers)
    model = Seq2Seq(encoder, decoder)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        for i in range(len(X_train)):
            query_graph_tensor = torch.LongTensor(X_train[i])
            target_positions_tensor = torch.Tensor(y_train[i])

            # 前向传播
            hidden = torch.zeros(num_layers, 1, hidden_size)
            cell = torch.zeros(num_layers, 1, hidden_size)
            output, _, _ = model(query_graph_tensor, hidden, cell)

            # 计算损失
            loss = criterion(output.squeeze(), target_positions_tensor)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 1000 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(X_train)}], Loss: {loss.item():.4f}')

    # 在测试集上评估模型准确率
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(len(X_test)):
            query_graph_tensor = torch.LongTensor(X_test[i])
            target_positions_tensor = torch.Tensor(y_test[i])

            hidden = torch.zeros(num_layers, 1, hidden_size)
            cell = torch.zeros(num_layers, 1, hidden_size)
            output, _, _ = model(query_graph_tensor, hidden, cell)
            predicted_positions = output.squeeze().round().tolist()

            if predicted_positions == y_test[i].tolist():
                correct += 1
            total += 1

    accuracy = correct / total * 100
    print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main()