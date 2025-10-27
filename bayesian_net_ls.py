
from pgmpy.models import BayesianNetwork, BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination
from graphviz import Digraph
import networkx as nx
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

class BayesNet:
    def __init__(self,profs: list, statements: list):
        self.data2022 = pd.read_excel("Non_Promotors_data.xlsx")      
        self.data2024 = pd.read_excel("Promotors_data.xlsx")
        self.profs = profs 
        self.statements = statements 
    def get_vars(self,statement: str) -> list[str]:
        vars = self.profs.copy()
        vars.append(statement)
        return vars
    def build_nodes(self,statement) -> list:
        nodes = []
        for prof in self.profs:
            nodes.append((statement,prof))
        return nodes
    def build_edges(self,statement: str) -> list:
        nodes = self.profs
        edges = [(node, statement) for node in nodes]
        return edges
        
    def train_model(self,edges: list, train_data: pd.DataFrame) -> None:
        self.model = BayesianNetwork(edges)
        self.model.fit(train_data, estimator=MaximumLikelihoodEstimator)
    def refine_structure(self,train_data: pd.DataFrame) ->None:
        # Using HillClimbSearch to refine the structure
        hc = HillClimbSearch(train_data)
        best_model = hc.estimate(scoring_method=BicScore(train_data))    
        self.train_model(best_model.edges(),train_data)
        self.model.check_model()
        # self.add_cpds()
    def save_model(self,statement: str) ->None:
        in_dc,out_dc = self.get_in_out_doc()
        graph = Digraph(format='png')
        for node in self.model.nodes():  
            if in_dc[node] > 0.25 or out_dc[node] > 0.25:          
                graph.node(node,style='filled',color='yellow')
            elif in_dc[node] > 0.15 or out_dc[node] > 0.15:          
                graph.node(node,style='filled',color='green')
            else:
                 graph.node(node,style='filled',color='skyblue')
        # graph.edges(self.model.edges())
        for edge in self.model.edges():
            source, target = edge
            graph.edge(source, target, label=f"in {in_dc[source]:.2f}, out {out_dc[source]:.2f}")
        # graph.render(filename=statement)
        graph.render(filename=statement, format='png', cleanup=True)
        
    def get_degrees_of_centrality(self):
        graph = nx.DiGraph()
        # Add nodes and edges from the Bayesian Network
        graph.add_nodes_from(self.model.nodes())
        graph.add_edges_from(self.model.edges())

        # Calculate degree centrality
        degree_centrality = nx.degree_centrality(graph) 
        return degree_centrality 
    
    def get_in_out_doc(self):
        """_summary_
        In-Degree Centrality focuses on the number of connections coming into a node,
        reflecting its popularity or influence within the network.
        Out-Degree Centrality focuses on the number of connections going out from
        a node, reflecting its activity or influence in spreading information

        Returns:
            _type_: _description_
            in_degree_centrality
            out_degree_centrality  
        """
        graph = nx.DiGraph()
        # Add nodes and edges from the Bayesian Network
        graph.add_nodes_from(self.model.nodes())
        graph.add_edges_from(self.model.edges())
        # Calculate in-degree centrality
        in_degree_centrality = nx.in_degree_centrality(graph)
        # Calculate out-degree centrality
        out_degree_centrality = nx.out_degree_centrality(graph)
        return in_degree_centrality, out_degree_centrality    

    def add_cpds(self):
        # Explicitly add CPDs to the model
        for node in self.model.nodes():
            cpd = self.model.estimate_cpd(node, estimator=MaximumLikelihoodEstimator)
            self.model.add_cpds(cpd)
            
    def update_evidence(self,d: dict) -> dict:
        _d = {}
        for k,v in d.items():
            if k in self.model.nodes():
                _d[k] = v 
        return _d  
    def save_formatted_model(self,file):
        try:
            # Create a directed graph
            G = nx.DiGraph()
            # Add edges (example)
            # G.add_nodes_from(self.model.nodes())
            G.add_edges_from(self.model.edges())

            # Calculate degree centrality
            degree_centrality = nx.degree_centrality(G)

            # Set node attributes based on degree centrality
            for node, centrality in degree_centrality.items():
                G.nodes[node]['degree_centrality'] = centrality
                # Set color based on centrality
                G.nodes[node]['color'] = plt.cm.viridis(centrality)
                # Set font size based on centrality
                G.nodes[node]['font_size'] = 6 #4 + centrality * 20
                
            # Extract node colors
            node_colors = [G.nodes[node]['color'] for node in G.nodes]
            node_sizes = [G.nodes[node]['degree_centrality'] * 1000 for node in G.nodes]

            # Draw the network
            pos = nx.spring_layout(G)
            nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, with_labels=False)

            # Draw labels with varying font sizes
            labels = {node: node for node in G.nodes}
            font_sizes = {node: G.nodes[node]['font_size'] for node in G.nodes}
            for node, (x, y) in pos.items():
                plt.text(x, y, s=node, fontsize=font_sizes[node], ha='center', va='center')
            plt.savefig(file) 
        except Exception as err:
            print(err)        
        
    def simulate_data(self):
        n_statements = len(self.statements)
        new_data = np.zeros(self.data2024.shape[0]*n_statements).reshape(self.data2024.shape[0],n_statements)
        for j, statement in enumerate(self.statements):
            vars = self.get_vars(statement)        
            edges = self.build_edges(statement)
            # print(edges)
            train_data = pd.concat([self.data2022[vars],                         
                self.data2022.sample(self.data2024.shape[0],
                                     replace=True,random_state=42)[vars]],axis=0)
            # train_data = self.data2024[vars]
            self.train_model(edges,train_data)
            self.refine_structure(train_data)
            self.save_model(f"{j+1} {statement.replace('/','_')}")
            # self.save_formatted_model(f"{statement}.png")
            # Perform inference
            inference = VariableElimination(self.model)   
            for idx,row in self.data2024.iterrows():
                try:
                    d_evidence = row[self.profs].to_dict()
                    d_evidence = self.update_evidence(d_evidence)
                    query_result = inference.query(variables=[statement], evidence=d_evidence)
                    result_dict = query_result.values.tolist()
                    categories = query_result.state_names[statement]
                    prob = dict(zip(categories, result_dict))
                    new_data[idx][j] = random.choices(list(prob.keys()),weights=list(prob.values()),k=1)[0]
                    # mq = inference.map_query(variables=[statement], evidence=d_evidence)[statement]
                    # print(f"map query: {mq}")
                except Exception as err:
                     print(f"{row['Respondent key']}:{err}")
        return new_data
    def get_means(self):
        means2024 = self.data2024[self.statements].values.mean(axis=0)
        means2022 = self.data2022[self.statements].values.mean(axis=0)
        return means2022, means2024
    
    def save_meanscores(self):
        new_data = self.simulate_data()
        with pd.ExcelWriter('BBN simulated meanscores.xlsx', engine='xlsxwriter') as writer:               
            new_data = [[col,val] for col,val in zip(self.statements,new_data.mean(axis=0))]
            df_means = pd.DataFrame(new_data,columns=['statement','simulated mean score (2024)'])
            df_means['mean score (2022)'] = self.get_means()[0]
            df_means['mean score (2024)'] = self.get_means()[1]
            df_means.to_excel(writer, sheet_name=f'new sample', index=False)