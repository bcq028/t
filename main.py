
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 # encoder(embed +lstm)->(decoder)lstm->output
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
    
    return query_graph, selected_nodes

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        # 输入 40*40的原始数据

        self.embedding = nn.Embedding(num_embeddings=201,embedding_dim=64)  # 初始化嵌入层

        # embed得到 40*40*64的数据

        self.lstm = nn.LSTM(64,40,1)  # 初始化LSTM层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)  # 嵌入层的前向传播
        output, (hidden, cell) = self.lstm(embedded)  # LSTM层的前向传播
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(40,40,1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        return output, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        encoder_hidden, encoder_cell = self.encoder(source)
        decoder_input = torch.randn(1,40,40)  # 假设开始符号的索引为0
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell

        outputs = []
        for t in range(source.size(1)):  # 逐个时间步生成输出
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            outputs.append(decoder_output.squeeze(1))

            # 使用当前时间步的预测结果作为下一个时间步的输入
            decoder_input = decoder_output

        outputs = torch.stack(outputs, dim=1)
        return outputs

def main():
    # Define model and parameters
    hidden_size = 64  # Hidden size
    num_layers = 2  # Number of LSTM layers
    learning_rate = 0.001
    num_epochs = 1

    # Generate training and test data
    generate_target()
    target_graph = np.load("target.npy")
    num_samples = 10000
    query_graphs = []
    target_positions = []

    for _ in range(num_samples):
        num_nodes_query = np.random.randint(20, 40)  # Randomly generate the number of nodes in the query graph
        query_graph, position_mapping = generate_query_graph(target_graph, num_nodes_query)
        query_graphs.append(query_graph)
        target_positions.append([position_mapping[i] for i in range(num_nodes_query)] + [0] * (40 - num_nodes_query))

    query_graphs = np.array(query_graphs)
    target_positions = np.array(target_positions)

    X_train, X_test, y_train, y_test = train_test_split(query_graphs, target_positions, test_size=0.2)

    # Define the model
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    model = Seq2Seq(encoder, decoder).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        for i in range(len(X_train)):
            query_graph_tensor = torch.LongTensor(X_train[i])
            target_positions_tensor = torch.LongTensor(y_train[i])

            # Forward pass
            output = model(query_graph_tensor).to(device)
            output_indices = torch.argmax(output, dim=2).squeeze(0)

            y_train_pred = output_indices[:, 0]

            # Compute loss
            class_labels = torch.zeros_like(target_positions_tensor)
            for j in range(target_positions_tensor.size(0)):
                indices = torch.where(output_indices[j] == target_positions_tensor[j])[0]
                if indices.numel() > 0:
                    class_labels[j] = indices[0]

        # Compute loss
            loss = criterion(output.view(-1, output.size(2)).transpose(0, 1), class_labels.view(-1))


            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 1000 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(X_train)}], Loss: {loss.item():.4f}')

    # Evaluate the model accuracy on the test set
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(len(X_test)):
            query_graph_tensor = torch.LongTensor(X_test[i])
            target_positions_tensor = torch.Tensor(y_test[i])

            output = model(query_graph_tensor).to(device)
            output_indices = torch.argmax(output, dim=2).squeeze(0)

            y_test_pred = output_indices[:, 0]

            # 计算相似度分数
            similarities = []
            for j in range(len(y_test_pred)):
              target_idx = y_test[i][j].item()  # 获取目标节点索引
              predicted_idx = y_test_pred[j].item()  # 获取预测节点索引
              target_node = torch.Tensor(target_graph[target_idx])  # 转换为张量
              predicted_node = torch.Tensor(target_graph[predicted_idx])  # 转换为张量
              # 计算余弦相似度
              similarity = F.cosine_similarity(target_node, predicted_node, dim=0)
              similarities.append(similarity.item())

              # 根据相似度分数和阈值判断预测是否正确
              threshold = 0.6  # 设定相似度阈值
              if all(similarity >= threshold for similarity in similarities):
                correct += 1

              total += 1
    accuracy = correct / total * 100
    print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main()