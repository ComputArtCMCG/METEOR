import numpy as np
from rdkit import Chem,DataStructs
import random,torch,pickle,json,subprocess,os,math,time
from collections import defaultdict
import bisect
from torch.utils.data import IterableDataset,Dataset,DataLoader
import torch.utils.data
import networkx as nx
from typing import List, Tuple, Union
from rdkit import RDLogger
from subprocess import Popen,PIPE
from sascore.sascore_model import SAScorer
from PLANET.PLANET_model import PLANET
from PLANET.chemutils import ProteinPocket,tensorize_molecules
from rdkit.Chem.QED import qed 
from rdkit.Chem import AllChem 
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
RDLogger.DisableLog('rdApp.*')

def mol_to_nx(mol) -> nx.Graph:
    G = nx.Graph()
    for atom in mol.GetAtoms():
        try:
            ring=[not atom.IsInRing(),atom.IsInRingSize(3),atom.IsInRingSize(4),atom.IsInRingSize(5),atom.IsInRingSize(6),atom.IsInRingSize(7)]
        except:  # a highly symmetric fused ring system
            ring = [False,False,False,False,False,True]                   
        G.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   degree=atom.GetDegree(),
                   #valence = atom.GetImplicitValence(),
                   charge = atom.GetFormalCharge(),
                   ring_info=ring,
                   )
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType(),
                   ring_bond=bond.IsInRing(),
                )
    return G

def nx_to_mol(G:nx.Graph):
    mol = Chem.RWMol()
    symbol = nx.get_node_attributes(G, 'symbol')
    charge = nx.get_node_attributes(G, 'charge')
    node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(symbol[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    nodes_N = [node for node in G.nodes() if symbol[node] == 'N']
    nodes_O = []
    def get_valence(node):
        edges = [(i,j) for (i,j) in G.edges() if i == node or j == node]
        bonds = [int(G.edges[i,j]['bond_type']) for (i,j) in edges]  
        return sum(bonds)   
    nodes_N = [node for node in nodes_N if get_valence(node)>3]
    if len(nodes_N) > 0:
        for node_N in nodes_N:
            neis = list(G.neighbors(node_N))
            for nei in neis:
                if symbol[nei] == 'O' and int(G.edges[node_N,nei]['bond_type']) == 1:
                    nodes_O.append(nei)
        for node in nodes_O:
            mol.GetAtomWithIdx(node_to_idx[node]).SetFormalCharge(charge[node])
        for node in nodes_N:
            mol.GetAtomWithIdx(node_to_idx[node]).SetFormalCharge(charge[node])
    Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    return mol

###global definition
POSSIBLE_ATOMS = ['C','N','O','S','F','I', 'Cl','Br']
POSSIBLE_ATOM_TYPES_NUM = len(POSSIBLE_ATOMS)
POSSIBLE_BONDS = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE]
ATOM_FDIM = 20
BOND_FDIM = 5
MAX_ACTION = 50
MAX_ATOM = MAX_ACTION + POSSIBLE_ATOM_TYPES_NUM 
ISOLATED_ATOMS_MOLS = [Chem.RWMol() for _ in range(POSSIBLE_ATOM_TYPES_NUM)]
for ATOM,MOL in zip(POSSIBLE_ATOMS,ISOLATED_ATOMS_MOLS):
    atom_to_add = Chem.Atom(ATOM)
    MOL.AddAtom(atom_to_add)
    Chem.SanitizeMol(MOL,sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
ISOLATED_ATOMS_GRAPHS = [mol_to_nx(m) for m in ISOLATED_ATOMS_MOLS]

NOT_ALLOWED_PATTENS = [
    Chem.MolFromSmarts("*=[#6]=*"),
    Chem.MolFromSmarts("[#8]-[#8]"),
    Chem.MolFromSmarts("[#16]-[#16]"),
    Chem.MolFromSmarts("[#9,#17,#35,#53]=,#*"),
    Chem.MolFromSmarts("[r]#[r]"), #环内三键
    Chem.MolFromSmarts("[r3,r4]=[r3,r4]"), #三 or 四元环内的双键
    Chem.MolFromSmarts("[#16&R&D3,4]"),
    # following are aromatic bridge ring
    Chem.MolFromSmarts('a1(*2)a@2aaaa1'),
    Chem.MolFromSmarts('a1@2aa@1aaa@2'),
    Chem.MolFromSmarts('a1(*2)aaaa@2a1'),   #C1(C2)=CC=CC2=C1
    Chem.MolFromSmarts('a1(**2)aaaa@2a1'),
    Chem.MolFromSmarts('a1(***2)aaaa@2a1'),
    Chem.MolFromSmarts('*1(*2)=**=*@2*=*1'),
    Chem.MolFromSmarts('a1(**2)aaa@2aa1'), 
    Chem.MolFromSmarts('a1(***2)aaa@2aa1'), 
    #five-membered ring
    Chem.MolFromSmarts('a1(*2)a@2aaa1'), 
    Chem.MolFromSmarts('a1@2aa@1aa2'), 
    Chem.MolFromSmarts('a1(*2)aa@2aa1'), 
    Chem.MolFromSmarts('a1(**2)aa@2aa1'), 
    Chem.MolFromSmarts('a1(***2)aa@2aa1'), 
]
# C12=CC=C(C=C2)C1 can not be handled
HALOGEN = ['Cl','F','Br','I']

def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x==item) for item in allowable_set]

def get_node_feature(node):
    #result dimension 20
    node_feature = np.array(onehot_encoding(node['symbol'],POSSIBLE_ATOMS) 
                            + onehot_encoding(node['degree'], [0,1,2,3,4,5]) 
                            #+ onehot_encoding(node['valence'], [0,1,2,3,4])
                            + node['ring_info']) #6
    return node_feature

def tensorize_single_nxgraph(graph:nx.Graph):
    adj_dic = {Chem.rdchem.BondType.SINGLE:[],Chem.rdchem.BondType.DOUBLE:[],Chem.rdchem.BondType.TRIPLE:[]}
    #each tuple in adj_dic store the start_idx and end_idx 
    fatoms = np.zeros([MAX_ATOM,ATOM_FDIM],dtype=np.float32)
    all_mask = np.zeros([MAX_ATOM,1],dtype=np.float32)
    current_mask = np.zeros([MAX_ATOM,1],dtype=np.float32)
    num_nodes = len(graph.nodes())
    
    for i,node_idx in enumerate(graph.nodes()):
        fatoms[i] = get_node_feature(graph.nodes()[node_idx])

    for i,G in enumerate(ISOLATED_ATOMS_GRAPHS):
        fatoms[i+num_nodes] = get_node_feature(G.nodes()[0])

    for (start,end) in graph.edges():
        x = list(graph).index(start) 
        y = list(graph).index(end) 
        
        #graph is seen as a bi-direction graph for adj construction
        adj_dic[graph.edges()[start,end]['bond_type']].append((x,y))

    all_mask[:num_nodes+POSSIBLE_ATOM_TYPES_NUM,:] = 1.
    current_mask[:num_nodes,:] = 1.

    fatoms = torch.from_numpy(fatoms)  # [MAX_ATOM,f]
    adj_matrix = torch.from_numpy(tensorize_adj(adj_dic))
    all_mask = torch.from_numpy(all_mask)
    current_mask = torch.from_numpy(current_mask)

    return (fatoms,adj_matrix,all_mask,current_mask)

def tensorize_adj(adj_dic:dict) -> np.ndarray:
    adj_matrix = np.zeros([3,MAX_ATOM,MAX_ATOM],dtype=np.float32)
    d05_matrix = np.zeros([3,MAX_ATOM,MAX_ATOM],dtype=np.float32)
    
    for b,adj_index in enumerate(adj_dic.values()):
        adj = adj_matrix[b,:,:]  #slice is still a part of adj_matrix 
        for (i,j) in adj_index:
            adj[i,j] = 1.0
            adj[j,i] = 1.0
 
        adj += np.eye(MAX_ATOM) #E
        
        d = np.sqrt(np.reciprocal(np.sum(adj,axis=1)))
        d_05 = d05_matrix[b,:,:]
        row, col = np.diag_indices_from(d_05)
        d_05[row, col] = d
        #for i in range(d_05.shape[0]):
        #    d_05[i,i] = d[i]
        #d05_matrix[b,:,:] = d_05
    
    result_adj = np.matmul(d05_matrix,adj_matrix)
    result_adj = np.matmul(result_adj,d05_matrix)
    
    return np.transpose(result_adj,(1,2,0))

def tensorize_nxgraphs(graphs:List[nx.Graph]):
    if len(graphs) == 1:
        mol_features = tensorize_single_nxgraph(graphs[0])
        mol_features = tuple(map(lambda x:torch.unsqueeze(x,dim=0),mol_features))
    else:
        mol_features = [[],[],[],[]]
        for graph in graphs:
            (fatoms,adj_matrix,all_mask,current_mask) = tensorize_single_nxgraph(graph)
            mol_features[0].append(fatoms)
            mol_features[1].append(adj_matrix)
            mol_features[2].append(all_mask)
            mol_features[3].append(current_mask)
        mol_features = tuple(map(lambda x:torch.stack(x,dim=0),mol_features))
    return mol_features

class Queue():
    def __init__(self):
        self.items = []
    def enqueue(self,item):
        self.items.append(item) 
    def dequeue(self):
        return self.items.pop(0)
    def size(self):
        return len(self.items)
    def is_empty(self):
        return len(self.items)==0
    def __contains__(self,item):
        return item in self.items
    def __iter__(self):
        for item in self.items:
            yield item
    def __len__(self):
        return len(self.items)

class Stack():
    def __init__(self):
        self.items = []
    def push(self,item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()
    def size(self):
        return len(self.items)
    def is_empty(self):
        return len(self.items)==0
    def __contains__(self,item):
        return item in self.items
    def __len__(self):
        return len(self.items)

def ring_first_next_node(G:nx.Graph,node:int,visited_edges:List[Tuple[int]],visited_nodes):
    neighbours = list(nx.all_neighbors(G,node))
    next_nodes = [i for i in neighbours if (i,node) not in visited_edges]
    if len(next_nodes) == 0:
        return None
    elif len(next_nodes) == 1:
        return next_nodes[0]
    else:
        ring_info = nx.cycle_basis(G)
        def ring_member_length(n):
            completeness = [list(set(ring)&set(visited_nodes+[n])) for ring in ring_info if n in ring]
            if len(completeness) > 0:
                return len(max(completeness,key=len))
            else:
                return 0
        next_nodes.sort(key=ring_member_length,reverse=True)
        return next_nodes[0]

def edge_dfs(G:nx.Graph,init_node:int):
    visited_edges = []
    visited_nodes = [init_node]
    stack = Stack()
    stack.push(init_node)
    while not stack.is_empty():
        parent_node = stack.items[-1]
        next_node = ring_first_next_node(G,parent_node,visited_edges,visited_nodes)
        if next_node is not None:
            yield (parent_node,next_node)
            if next_node not in stack:
                stack.push(next_node)
            if next_node not in visited_nodes:
                visited_nodes.append(next_node)
            visited_edges.append((parent_node,next_node))
            visited_edges.append((next_node,parent_node))
        else:
            stack.pop()

def get_expert(mol):
    Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    g = mol_to_nx(mol)
    init_node = random.choice([i for i in g.nodes() if g.nodes()[i]['symbol'] == 'C'])
    edge_traj = list(edge_dfs(g,init_node))
    sub_graphs,focus,finished,actions = [],[],[],[]
    nodes_in_graph = [init_node]
    nodes_finished = []
    unfinished = Stack()
    unfinished.push(init_node)
    
    for i,(start_idx,end_idx) in enumerate(edge_traj):
        bond_type = POSSIBLE_BONDS.index(g.edges()[start_idx,end_idx]['bond_type'])
        if i == 0:
            sub_graph = g.subgraph(nodes_in_graph)
        else:
            sub_graph = g.edge_subgraph(edge_traj[:i])
        sub_graphs.append(sub_graph)
        focus.append(list(sub_graph).index(start_idx))
        
        if end_idx in nodes_in_graph:
            end = list(sub_graph).index(end_idx)
        else:
            end = len(nodes_in_graph) + POSSIBLE_ATOMS.index(g.nodes()[end_idx]['symbol'])
            nodes_in_graph.append(end_idx)
            unfinished.push(end_idx)
        actions.append([end,bond_type,0])
        finished.append([list(sub_graph).index(i) for i in nodes_finished])
        
        sub_graph = g.edge_subgraph(edge_traj[:i+1])
        
        if i != len(edge_traj)-1 and edge_traj[i+1][0] != end_idx:  #not last
            while unfinished.items[-1] != edge_traj[i+1][0]:
                current_focus = unfinished.pop()
                sub_graphs.append(sub_graph)
                focus.append(list(sub_graph).index(current_focus))
                actions.append([end,bond_type,1])
                finished.append([list(sub_graph).index(i) for i in nodes_finished])
                nodes_finished.append(current_focus)

        if i == len(edge_traj)-1:  #last
            while not unfinished.is_empty():
                sub_graphs.append(sub_graph)
                current_focus = unfinished.pop()
                focus.append(list(sub_graph).index(current_focus))
                actions.append([end,bond_type,1])
                finished.append([list(sub_graph).index(i) for i in nodes_finished])
                nodes_finished.append(current_focus)
    
    sub_graphs = [mol_to_nx(nx_to_mol(g)) for g in sub_graphs]
    prob_mask = finished_to_prob_mask(sub_graphs,focus,finished)

    return sub_graphs,focus,prob_mask,actions

def finished_to_prob_mask(sub_graphs:List[nx.Graph],focus:List[int],finished:List[List[int]]) -> torch.Tensor:
    prob_mask = np.zeros([len(focus),MAX_ATOM,1],dtype=np.float32)
    for (mask,sub_graph,fo,f) in zip(prob_mask,sub_graphs,focus,finished):
        nodes_num = len(list(sub_graph))
        mask[:nodes_num+POSSIBLE_ATOM_TYPES_NUM] = 1.0
        for i in f:
            mask[i] = 0.0
        mask[fo] = 0.0
    prob_mask = torch.from_numpy(prob_mask)
    return prob_mask

def finished_to_prob_mask_single(graph:nx.Graph,focus:int,finished:List[int]) -> torch.Tensor:
    """
    used for a single graph\n
    return a "mask" tensor with shape [graph_len,1]
    """
    prob_mask = np.zeros([MAX_ATOM,1],dtype=np.float32)
    nodes_num = len(list(graph))
    prob_mask[:nodes_num+POSSIBLE_ATOM_TYPES_NUM] = 1.0
    for i in finished:
        prob_mask[i] = 0.0
    prob_mask[focus] = 0.0
    prob_mask = torch.from_numpy(prob_mask)
    return prob_mask

def finished_to_critic_features(sub_graphs:List[nx.Graph],focus:List[int],finished:List[List[int]]) -> torch.Tensor:
    extra_features = np.zeros([len(finished),MAX_ATOM,3],dtype=np.float32)
    for (feature,sub_graph,fo,f) in zip(extra_features,sub_graphs,focus,finished):
        nodes_num = len(list(sub_graph))
        feature[:nodes_num,1] = 1.0  #
        for i in f:
            feature[i,0] = 1.0
            feature[i,1] = 0.0
        feature[fo,2] = 1.0
    extra_features = torch.from_numpy(extra_features)
    #extra_features [batch,nodes,(finish,unfinish,focus)]
    return extra_features

class ExpertDataset(Dataset):
    def __init__(self,smi_file,batch_size):
        with open(smi_file,'r') as f:
            smiles = [line.strip().split()[0] for line in f]
        random.shuffle(smiles)
        self.smi_batches = [smiles[i : i + batch_size] for i in range(0,len(smiles),batch_size)]
    
    def __len__(self):
        return len(self.smi_batches)

    def __getitem__(self,idx):
        return self.tensorize(self.smi_batches[idx])

    def tensorize(self,smi_batch):
        filtered_mol_batch,nx_graphs,focus,prob_mask,expert_actions = [],[],[],[],[]
        mol_batch = [Chem.MolFromSmiles(smi) for smi in smi_batch if Chem.MolFromSmiles(smi) is not None]
        for mol in mol_batch:
            flag = False
            if largest_ring(mol) > 7 or mol.GetNumAtoms() > MAX_ACTION:
                flag = True
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in POSSIBLE_ATOMS:
                    flag =True
            if not flag:
                filtered_mol_batch.append(mol)
            
        for mol in filtered_mol_batch:
            sub_graphs,f,p,a = get_expert(mol)
            nx_graphs.extend(sub_graphs)
            focus.extend(f)
            prob_mask.append(p)
            expert_actions.extend(a)
        
        mol_features = tensorize_nxgraphs(nx_graphs)
        focus = torch.LongTensor(focus).view([-1,1])
        prob_mask = torch.cat(prob_mask,dim=0)
        expert_actions = torch.LongTensor(expert_actions)
        return mol_features,focus,prob_mask,expert_actions

class TrajectoryState():
    def __init__(self,nx_graph,focus:int,finished_nodes:List[int],action:torch.LongTensor,step_reward:float):
        """
        focus:int
        action:[1,3]
        """
        self.nx_graph = nx_graph
        self.focus = focus
        self.finished_nodes = finished_nodes
        self.action = action 
        self.reward = step_reward
        self.probs = None
    
    def set_probs(self,probs:torch.Tensor):
        self.probs = probs.detach()

    def set_qvalue(self,qvalue:torch.Tensor):
        self.qvalue = qvalue.to('cpu').detach()
    
    def set_reward(self,reward:float):
        self.reward = reward

    def set_v_target(self,v_target):
        self.v_target = v_target

    def set_advantage(self,advantage:float):
        self.advantage = advantage

class Trajectory():
    def __init__(self,states:List[TrajectoryState],smi):
        self.states = states
        self.smi = smi

    def assign_credit_and_advantage(self,final_reward:float,gamma): 
        # the final reward is calculated from the estimator
        self.states.reverse()
        total_reward = final_reward
        for i,state in enumerate(self.states):
            ### MC
            if i == 0:
                total_reward += state.reward 
            else:
                total_reward = state.reward + total_reward * gamma
            adv = total_reward - state.qvalue
            state.set_advantage(adv)
            state.set_v_target(total_reward)
    
class SAScoreEstimator():
    def __init__(self):
        self.sa_estimator = SAScorer()
    
    def estimate(self,mol) -> float:
        sascore = self.sa_estimator.calculateScore(mol) 
        return sascore

class PlanetEstimator():
    def __init__(self):
        self.estimator = PLANET(300,8,300,300,3,10,1,torch.device('cpu'))  #trained PLANET
        self.estimator.load_parameters()
        for param in self.estimator.parameters():
            param.requires_grad = False
        self.estimator.eval()
        self.device = torch.device('cpu')

    def set_pocket_from_ligand(self,protein_pdb,ligand_sdf):
        try:
            self.pocket = ProteinPocket(protein_pdb=protein_pdb,ligand_sdf=ligand_sdf)
        except:
            raise RuntimeError
        self.res_features = self.estimator.cal_res_features_helper(self.pocket.res_features,self.pocket.alpha_coordinates)

    def set_pocket_from_coordinate(self,protein_pdb,centeriod_x,centeriod_y,centeriod_z):
        try:
            self.pocket = ProteinPocket(protein_pdb,centeriod_x,centeriod_y,centeriod_z)
        except:
            raise RuntimeError
        self.res_features = self.estimator.cal_res_features_helper(self.pocket.res_features,self.pocket.alpha_coordinates)

    def pre_cal_res_features(self):
        self.res_features = self.estimator.cal_res_features_helper(self.pocket.res_features,self.pocket.alpha_coordinates)

    def estimate(self,mol_features,mol_scope) -> Tuple[List[int],List[float]]:
        """
        return best affinity and best isomer
        """
        (fatoms, fbonds, agraph, bgraph, lig_scope) = mol_features
        fatoms = fatoms.to(self.estimator.device)
        fbonds = fbonds.to(self.estimator.device)
        agraph = agraph.to(self.estimator.device)
        bgraph = bgraph.to(self.estimator.device)
        with torch.no_grad():
            fresidues,res_scope = self.estimator.cal_res_features(self.res_features,len(lig_scope))
            mol_feature_batch = (fatoms, fbonds, agraph, bgraph, lig_scope)
            fresidues = fresidues.to(self.estimator.device)
            affinity = self.estimator.screening(fresidues,res_scope,mol_feature_batch).reshape([-1]).cpu().numpy()
        selected_index,selected_affinity = [],[]
        ### CHECK
        for (start_index,interval) in mol_scope:
            start_index,interval = int(start_index),int(interval)
            if interval == 0:
                selected_index.append(-1)
                selected_affinity.append(0)
            else:
                selected_index.append(int(np.argmax(affinity[start_index:start_index+interval])))
                selected_affinity.append(float(np.max(affinity[start_index:start_index+interval])))
        return selected_index,selected_affinity

    def estimate_test(self,mol) -> float:
        ### only used in the notebook 
        mol_H = Chem.AddHs(mol)
        with torch.no_grad():
            fresidues,res_scope = self.estimator.cal_res_features(self.res_features,1)
            mol_feature_batch = tensorize_molecules([mol_H])
            affinity = self.estimator.screening(fresidues,res_scope,mol_feature_batch)
        return affinity.item()
        
    def to_cuda(self,device):
        self.estimator.to(device)
        self.estimator.device = device
        self.estimator.proteinegnn.device = device
        self.estimator.ligandgat.device = device
        self.estimator.prolig.device = device
        self.estimator.prolig.prolig_attention.device = device

    def to_cpu(self):
        self.estimator.to(torch.device('cpu'))
        self.estimator.device = torch.device('cpu')
        self.estimator.proteinegnn.device = torch.device('cpu')
        self.estimator.ligandgat.device = torch.device('cpu')
        self.estimator.prolig.device = torch.device('cpu')
        self.estimator.prolig.prolig_attention.device = torch.device('cpu')

class ActivityStat():
    def __init__(self,activity_dir,feature_files:List[str],planet:PlanetEstimator):
        self.activity_dir = activity_dir
        self.feature_files = feature_files
        self.planet = planet
    
    def _iter_feature_file(self,feature_file):
        file_end = False
        with open(feature_file,'rb') as f:
            while not file_end:
                try:
                    (mol_features,mol_scope,mol_penalty,mol_smis,smis) = pickle.load(f)
                    yield (mol_features,mol_scope,mol_penalty,mol_smis,smis)
                except EOFError:
                    file_end = True

    def estimate_batch_smis(self,feature_file) -> dict:
        """
        return a dict, keys: raw_traj_smi, value: (final_smi,mol_penalty,affinity)
        """
        result = {}
        for (mol_features,mol_scope,mol_penalty,mol_smis,smis) in self._iter_feature_file(feature_file):
            selected_index,selected_affinities = self.planet.estimate(mol_features,mol_scope)
            final_smis = []
            for (start_index,interval),index in zip(mol_scope,selected_index):
                if index == -1:
                    continue
                else:
                    final_smis.append(mol_smis[start_index:start_index+interval][index])
            for origin_smi,final_smi,mol_p,aff in zip(smis,final_smis,mol_penalty,selected_affinities):
                if origin_smi == 'XXX':
                    continue
                else:
                    result[origin_smi] = (final_smi,mol_p,aff)
        return result
    
    def estimate_all(self):
        for feature_file in self.feature_files:
            rank_num = os.path.basename(feature_file).split('_')[0]
            result = self.estimate_batch_smis(feature_file)
            self.write_out(result,rank_num)

    def write_out(self,result:dict,prefix):
        out_file = os.path.join(self.activity_dir,'{}_activity.json'.format(prefix))
        with open(out_file,'w') as f:
            json.dump(result,f)

class PrepareActivityEstimate():
    def __init__(self,raw_traj_file,feature_file,batch_size):
        self.batch_size = batch_size
        self.feature_file = feature_file
        self.raw_traj_file = raw_traj_file

    def gather_smis(self) -> List[str]:
        all_smis = []
        file_end = False
        f = open(self.raw_traj_file,'rb')
        while not file_end:
            try:
                traj:Trajectory = pickle.load(f)
                all_smis.append(traj.smi)
            except EOFError:
                file_end = True
        f.close()
        return all_smis

    def process_smis(self):
        if os.path.exists(self.feature_file):
            os.remove(self.feature_file)
        all_smis = self.gather_smis()
        all_smis = [all_smis[i:i+self.batch_size] for i in range(0,len(all_smis),self.batch_size)]
        for smis in all_smis:
            mols,mol_scope,mol_penalty = [],[],[]
            i = 0
            for smi in smis:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    mol_scope.append((i,0))
                    mol_penalty.append(0.)
                else:
                    mol_p,mol_H = self.cal_mol_penalty(mol)
                    mol_penalty.append(mol_p)
                    mol_scope.append((i,len(mol_H)))
                    i += len(mol_H)
                    mols.extend(mol_H)
            mol_features = tensorize_molecules(mols)     
            mol_smis = [Chem.MolToSmiles(Chem.RemoveHs(mol)) for mol in mols]   
            with open(self.feature_file,'ab') as f:
                pickle.dump((mol_features,mol_scope,mol_penalty,mol_smis,smis),f)
    
    def cal_mol_penalty(self,mol):
        # calculate the basic mol penalty, including atom count and number of chiral centers
        # return mol_penlty and mol_H (enumerated isomers)
        atom_count = mol.GetNumAtoms()
        if atom_count <= 40:
            mol_penalty = min(atom_count/10,1.0)
        else:
            mol_penalty = max(1 - (atom_count - 40) / 10, 0.)
        mol_H = protonation_and_add_H(mol)
        mol_H = list(EnumerateStereoisomers(mol_H))
        isomer_count = len(mol_H)
        if isomer_count > 4:
            mol_H = [random.choice(mol_H)]
            mol_penalty *= 0.5
        return mol_penalty,mol_H

class ActivityStatWithDataset():
    def __init__(self,activity_dir,raw_traj_files:List[str],batch_size:int,planet:PlanetEstimator):
        self.activity_dir = activity_dir
        self.raw_traj_files = raw_traj_files
        self.planet = planet
        self.batch_size = batch_size
    
    def getloader(self,raw_traj_file):
        dataset = ActivityDataset(raw_traj_file,self.batch_size)
        dataloader = DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=lambda x:x[0],num_workers=4,drop_last=False)
        return dataloader

    def estimate_batch_smis(self,raw_traj_file) -> dict:
        """
        return a dict, keys: raw_traj_smi, value: (final_smi,mol_penalty,affinity)
        """
        result = {}
        for (mol_features,mol_scope,mol_penalty,mol_smis,smis) in self.getloader(raw_traj_file):
            selected_index,selected_affinities = self.planet.estimate(mol_features,mol_scope)
            final_smis = []
            for (start_index,interval),index in zip(mol_scope,selected_index):
                if index == -1:
                    continue
                else:
                    final_smis.append(mol_smis[start_index:start_index+interval][index])
            for origin_smi,final_smi,mol_p,aff in zip(smis,final_smis,mol_penalty,selected_affinities):
                if origin_smi == 'XXX':
                    continue
                else:
                    result[origin_smi] = (final_smi,mol_p,aff)
        return result
    
    def estimate_all(self):
        for raw_traj_file in self.raw_traj_files:
            rank_num = os.path.basename(raw_traj_file).split('_')[0]
            result = self.estimate_batch_smis(raw_traj_file)
            self.write_out(result,rank_num)

    def write_out(self,result:dict,prefix):
        out_file = os.path.join(self.activity_dir,'{}_activity.json'.format(prefix))
        with open(out_file,'w') as f:
            json.dump(result,f)

class ActivityDataset(Dataset):
    def __init__(self,raw_traj_file,batch_size):
        self.raw_traj_file = raw_traj_file
        all_smis = self.gather_smis()
        self.all_smis = [all_smis[i:i+batch_size] for i in range(0,len(all_smis),batch_size)]

    def __len__(self):
        return len(self.all_smis)
    
    def __getitem__(self, index):
        return self.process_smis(self.all_smis[index])
    
    def process_smis(self,smis):
        mols,mol_scope,mol_penalty = [],[],[]
        i = 0
        for smi in smis:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                mol_scope.append((i,0))
                mol_penalty.append(0.)
            else:
                mol_p,mol_H = self.cal_mol_penalty(mol)
                mol_penalty.append(mol_p)
                mol_scope.append((i,len(mol_H)))
                i += len(mol_H)
                mols.extend(mol_H)
        mol_features = tensorize_molecules(mols)     
        mol_smis = [Chem.MolToSmiles(Chem.RemoveHs(mol)) for mol in mols]   
        return mol_features,mol_scope,mol_penalty,mol_smis,smis
    
    def cal_mol_penalty(self,mol):
        # calculate the basic mol penalty, including atom count and number of chiral centers
        # return mol_penlty and mol_H (enumerated isomers)
        atom_count = mol.GetNumAtoms()
        if atom_count <= 40:
            mol_penalty = min(atom_count/10,1.0)
        else:
            mol_penalty = max(1 - (atom_count - 40) / 10, 0.)
        mol_H = protonation_and_add_H(mol)
        mol_H = list(EnumerateStereoisomers(mol_H))
        isomer_count = len(mol_H)
        if isomer_count > 4:
            mol_H = [random.choice(mol_H)]
            mol_penalty *= 0.5
        return mol_penalty,mol_H
    
    def gather_smis(self) -> List[str]:
        all_smis = []
        file_end = False
        f = open(self.raw_traj_file,'rb')
        while not file_end:
            try:
                traj:Trajectory = pickle.load(f)
                all_smis.append(traj.smi)
            except EOFError:
                file_end = True
        f.close()
        all_smis = [smi for smi in all_smis if smi is not None]
        return all_smis

class ScaffoldMemory():
    def __init__(self,reward_cutoff,memory_dir):
        self.reward_cutoff = reward_cutoff
        self._good_mols_fp = GoodMolFP(memory_dir)
        self._known_mols = SmilesContainer(memory_dir)

    def add_records(self,smis,raw_rewards,round):
        replaced_smis = [replace_halgon_to_H(smi) for smi in smis]
        good_smis = [smi for (smi,raw_r) in zip(replaced_smis,raw_rewards) if raw_r >= self.reward_cutoff]
        good_pkl_path = '{:0>3d}_good_smi.pkl'.format(round)
        known_pkl_path = '{:0>3d}_known_smi.pkl'.format(round)
        self._good_mols_fp.update(good_smis,good_pkl_path)
        self._known_mols.update(replaced_smis,known_pkl_path)

    def __getitem__(self,smile) -> float:
        '''
        directly return penalty ratio and most similar scaffold according to the given scaffold
        '''
        replaced_smi = replace_halgon_to_H(smile)
        if replaced_smi in self._known_mols:
            return 0.0
        else:
            mol_sim_penalty = 1.0
            mol_similarity = self._good_mols_fp.get_max_similarity(replaced_smi)
            if mol_similarity <= 0.4:
                mol_sim_penalty = 1.0
            elif mol_similarity >= 0.7:
                mol_sim_penalty = 0.0
            else:
                mol_sim_penalty = 1 - ((mol_similarity - 0.4) / 0.3)

            return mol_sim_penalty 
        
class GoodMolFP():
    def __init__(self,dir_path,memory_length=10):
        self.dir_path = dir_path
        self.memory_length = memory_length
        self.molFP_memorys = Queue()
    
    def get_max_similarity(self,smi):
        simiarity = []
        for m in self.molFP_memorys:
            with open(m,'rb') as f:
                memory:_GoodMolFPMemory = pickle.load(f)
            simiarity.append(memory.get_max_similarity(smi))
        if len(simiarity) > 0:
            return max(simiarity)
        else:
            return 0.0
    
    def update(self,smis:List[str],pkl_path:str):
        if len(self.molFP_memorys) > self.memory_length:
            m = self.molFP_memorys.dequeue()
            os.remove(m)
        pkl_path = os.path.join(self.dir_path,pkl_path)
        self.molFP_memorys.enqueue(pkl_path)
        _GoodMolFPMemory(smis,pkl_path)

class _GoodMolFPMemory():
    def __init__(self,smis,pkl):
        self._good_mols_fp = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(s),2) for s in smis]
        with open(pkl,'wb') as f:
            pickle.dump(self,f)

    def get_max_similarity(self,smi):
        # we have make sure this smi is valid for rdkit
        if len(self._good_mols_fp) == 0:
            return 0.0
        else:
            mol_fp = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi),2)
            mol_similarity = float(np.max(np.array([DataStructs.TanimotoSimilarity(mol_fp,fp) for fp in self._good_mols_fp])))
            return mol_similarity

class SmilesContainer():
    def __init__(self,dir_path,memory_length=20):
        self.dir_path = dir_path
        self.containers = Queue()
        self.memory_length = memory_length

    def __contains__(self,smi):
        for c in self.containers:
            with open(c,'rb') as f:
                container = pickle.load(f)
            if smi in container:
                return True
        return False

    def update(self,smis:List[str],pkl_path:str):
        if len(self.containers) >  self.memory_length:
            c = self.containers.dequeue()
            os.remove(c)
        pkl_path = os.path.join(self.dir_path,pkl_path)
        self.containers.enqueue(pkl_path)
        _SmilesContainer(smis,pkl_path)

class _SmilesContainer():
    def __init__(self,smis,pkl):
        self._record = defaultdict(list)
        for smi in smis:
            if smi != 'XXX' and smi is not None:
                self._record[len(smi)].append(smi)
        for k,smi_list in self._record.items():
            smi_list = list(set(smi_list))
            smi_list.sort()
            self._record[k] = smi_list
        with open(pkl,'wb') as f:
            pickle.dump(self,f)

    def __contains__(self,smi):
        if smi == '' or len(self._record[len(smi)]) == 0:
            return False
        else:
            i = bisect.bisect_left(self._record[len(smi)],smi)
            #contain such scaffold
            if i != len(self._record[len(smi)]) and self._record[len(smi)][i] == smi:
                return True
            else:
                return False

class RawTrajectoryProcessor():
    def __init__(self,raw_trajectory_file:str,processed_traj_file:str,activity_json:str,memory_pkl:str,log_file:str,
                sascore_estimator:SAScoreEstimator,scaffold_memory:ScaffoldMemory,\
                qed_cutoff:Tuple[float,float],sascore_cutoff:Tuple[float,float],\
                affinity_cutoff:Tuple[float,float],reward_weights:Tuple[float,float,float],\
                property_meet_cutoff:Tuple[float,float,float],\
                gamma=0.99):
        self.raw_trajectory_file = raw_trajectory_file
        self.processed_traj_file = processed_traj_file
        self.activity_json = activity_json
        self.memory_pkl = memory_pkl
        self.log_file = log_file
        self.sascore_estimator = sascore_estimator
        self.scaffold_memory = scaffold_memory
        self.qed_cutoff,self.sascore_cutoff,self.affinity_cutoff = qed_cutoff,sascore_cutoff,affinity_cutoff
        self.reward_weights = reward_weights
        self.property_meet_cutoff = property_meet_cutoff
        self.gamma = gamma
        self.start_time = time.time()
           
    def iter_raw_trajectory(self) -> List[Trajectory]:
        file_end = False
        f = open(self.raw_trajectory_file,'rb')
        while not file_end:
            try:    
                yield pickle.load(f)
            except EOFError:
                file_end = True
        f.close()
        os.remove(self.raw_trajectory_file)

    def process_raw_trajectories(self):
        '''
        process raw trajs
        '''
        # clean up the processed trajs from previous round of generation
        if os.path.exists(self.processed_traj_file):
            os.remove(self.processed_traj_file)
        all_smis,all_raw_rewards = [],[]
        n_qed,n_sascore,n_affinity,n_trajs = 0,0,0,0
        square_sum,num_sum,n_states = 0.,0.,0
        with open(self.activity_json,'r') as f:
            activity_dict = json.load(f)

        for trajectory in self.iter_raw_trajectory():
            if trajectory.smi == 'XXX':
                qed_score,sascore,aff = 0.0,10.0,0.0
                mol_p,simi_p = 0.0,0.0
                final_smi = 'XXX'
            else:
                try:
                    (final_smi,mol_p,aff) = activity_dict[trajectory.smi]
                except KeyError:
                    continue
                mol = Chem.MolFromSmiles(final_smi)
                qed_score = qed(mol)
                sascore = self.sascore_estimator.estimate(mol)
                simi_p = self.scaffold_memory[trajectory.smi]
            qed_reward,sa_reward,aff_reward,prop_p = self.cal_reward_and_property_penalty(qed_score,sascore,aff)
            if qed_score > self.property_meet_cutoff[0]:
                n_qed += 1
            if sascore < self.property_meet_cutoff[1]:
                n_sascore += 1
            if aff > self.property_meet_cutoff[2]:
                n_affinity += 1
            n_trajs += 1
            
            raw_reward = qed_reward+sa_reward+aff_reward
            final_reward = raw_reward * mol_p * prop_p * simi_p 
            all_smis.append(trajectory.smi)
            all_raw_rewards.append(raw_reward)
            trajectory.assign_credit_and_advantage(final_reward,self.gamma)
            
            square_sum += sum(map(lambda x:x**2,[state.advantage for state in trajectory.states]))
            num_sum += sum([state.advantage for state in trajectory.states])
            n_states += len(trajectory.states)
            self.write_processed_trajectory_pkl(trajectory)  
            self.write_generation_log(final_smi,(qed_score,sascore,aff),(mol_p,prop_p,simi_p),final_reward)
        return all_smis,all_raw_rewards,n_qed,n_sascore,n_affinity,n_trajs,square_sum,num_sum,n_states
    
    def cal_advantage_and_update_memory(self,all_smis,all_raw_rewards,n_qed,n_sascore,n_affinity,n_trajs,square_sum,num_sum,n_states,round_num) -> Tuple[float,float]:
        h,m,s = format_second(time.time() - self.start_time)
        print('Total elapsed time:{:03d} hours {:02d} minutes {:02d} seconds \n'.format(round(h), round(m), round(s)))
        print('{:.1%} of  molecules meet the qed threshold of {}'.format(n_qed/n_trajs,self.property_meet_cutoff[0]))
        print('{:.1%} of  molecules meet the sascore threshold of {}'.format(n_sascore/n_trajs,self.property_meet_cutoff[1]))
        print('{:.1%} of  molecules meet the affinity threshold of {}'.format(n_affinity/n_trajs,self.property_meet_cutoff[2]))
        
        self.scaffold_memory.add_records(all_smis,all_raw_rewards,round_num)
        with open(self.memory_pkl,'wb') as f:
            pickle.dump(self.scaffold_memory,f)
        adv_mean = num_sum / n_states
        adv_std = math.sqrt(square_sum/n_states - adv_mean ** 2)
        return adv_mean,adv_std
        
    def cal_reward_and_property_penalty(self,qed_score,sascore,aff): 
        qed_reward = self.linear_reward(qed_score,self.qed_cutoff) 
        sa_reward = 1.0-self.linear_reward(sascore,self.sascore_cutoff)
        affinity_reward = self.linear_reward(aff,self.affinity_cutoff) 
        prop_p = 1.0
        for r in (qed_reward,sa_reward,affinity_reward):
            prop_p *= min(1.0,r/0.2)
        qed_reward = self.reward_weights[0]*qed_reward
        sa_reward = self.reward_weights[1]*sa_reward
        affinity_reward = self.reward_weights[2]*affinity_reward
        return qed_reward,sa_reward,affinity_reward,prop_p
    
    def linear_reward(self,input_score,bounds) -> float:
        if input_score >= bounds[1]:
            return 1.0
        elif input_score <= bounds[0]:
            return 0.0
        else:
            return (input_score - bounds[0]) / (bounds[1] - bounds[0])

    def write_processed_trajectory_pkl(self,traj:Trajectory):
        with open(self.processed_traj_file,'ab') as f:
            pickle.dump(traj.states,f)

    def write_generation_log(self,final_smi,scores,penalties,final_reward):
        with open(self.log_file,'a') as f:
            to_write = '{: ^130}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.\
                        format(final_smi,scores[0],scores[1],scores[2],penalties[0],penalties[1],penalties[2],final_reward)
            f.write(to_write)

    def write_delimeter(self,round):
        with open(self.log_file,'a') as f:
            f.writelines('----------------------------------------Round: {:03d}-----------------------------------------\n'.format(round))
     
    def synchronize_memory(self):
        with open(self.memory_pkl,'rb') as f:
            self.scaffold_memory = pickle.load(f)

class StatesGenerator():
    def __init__(self,processed_dir,batch_size:int):
        self.processed_dir = processed_dir
        self.batch_size = batch_size
        self.pool_size = batch_size * 10
        self.buffer = []
        
    def __iter__(self):
        for processed_pkl in os.listdir(self.processed_dir):
            pkl_path = os.path.join(self.processed_dir,processed_pkl)
            file_end = False
            with open(pkl_path,'rb') as f:
                while len(self.buffer) < 2 * self.pool_size:
                    try:
                        self.buffer.extend(pickle.load(f))
                    except EOFError:
                        file_end = True
                        break 
                random.shuffle(self.buffer)
                while not file_end:
                    try:
                        while len(self.buffer) > self.pool_size:
                            states,self.buffer = self.buffer[:self.batch_size],self.buffer[self.batch_size:]
                            yield states
                        while len(self.buffer) < 2 * self.pool_size:
                            self.buffer.extend(pickle.load(f))
                        random.shuffle(self.buffer)
                    except EOFError:
                        file_end = True
        while len(self.buffer) > 0:
            states,self.buffer = self.buffer[:self.batch_size],self.buffer[self.batch_size:]
            yield states

class StatesDataset(IterableDataset):
    def __init__(self,states_gen:StatesGenerator,adv_mean,adv_std):
        self.states_gen = states_gen
        self.adv_mean = adv_mean
        self.adv_std = adv_std

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_num = worker_info.num_workers
        worker_id = worker_info.id
        for states in self.states_gen:
            worker_load = int(self.states_gen.batch_size / worker_num)
            outs = [[] for _ in range(10)]
            for state in states[worker_id*worker_load:(worker_id+1)*worker_load]:
                tensors = self.tensorize_state(state)
                for t,o in zip(tensors,outs):
                    o.append(t)
            try:
                outs = tuple(map(lambda x:torch.stack(x,dim=0),outs))
                yield outs
            except:
                continue
    
    def tensorize_state(self,state:TrajectoryState):
        mol_features = tensorize_single_nxgraph(state.nx_graph)
        fatoms,adj_matrix,all_mask,current_mask = mol_features
        focus = torch.LongTensor([state.focus])
        prob_mask = finished_to_prob_mask([state.nx_graph],[state.focus],[state.finished_nodes])
        prob_mask = prob_mask.squeeze(0)
        action = state.action.squeeze()
        probs = state.probs.squeeze()
        v_target = torch.Tensor([state.v_target])
        advantage = torch.Tensor([(state.advantage - self.adv_mean) / (self.adv_std + 1e-7)])
        return (fatoms,adj_matrix,all_mask,current_mask,focus,prob_mask,action,probs,v_target,advantage)

def replace_halgon_to_H(smi) -> Union[str,None]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    else:
        mol = Chem.RWMol(mol)
        for idx in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(idx).GetSymbol() in HALOGEN:
                mol.ReplaceAtom(idx,Chem.Atom('H'))
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if mol is not None:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            return Chem.MolToSmiles(mol)
        else:
            return None
    
def get_scaffold(smi) -> str:
    if smi is None:
        return ''
    else:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ''
        else:
            mol_scaff = MurckoScaffold.GetScaffoldForMol(mol)
            mol_scaff = MurckoScaffold.MakeScaffoldGeneric(mol_scaff)
            try:
                scaff_smi = Chem.MolToSmiles(mol_scaff)
                return scaff_smi
            except:
                return ''

def ring_filter(mol):
    if largest_ring(mol) > 7:
        return False
    else:
        return True

def largest_ring(mol):
    ssr = Chem.GetSymmSSSR(mol)
    if len(ssr) > 0:
        return len(max(ssr,key=len))
    else:
        return 0
    
def format_second(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    #print("%d:%02d:%02d" % (h, m, s))
    return h,m,s

def protonation_and_add_H(mol):
    smi = Chem.MolToSmiles(mol)
    #cmd = "obabel -:'{}' -p 7.0 -o smi".format(smi)
    cmd = "obabel -:'{}' -o smi".format(smi)
    p = Popen(cmd,shell=True,stdin=PIPE,stdout=PIPE,stderr=subprocess.STDOUT)
    outs,err = p.communicate()
    for line in outs.decode().split('\n'):
        if line.strip() == '1 molecule converted' or line.strip() == '':
            pass
        else:
            converted_smi = line.strip()
            protonated_mol = Chem.MolFromSmiles(converted_smi)
            if protonated_mol is not None:
                protonated_mol = Chem.AddHs(protonated_mol)
                return protonated_mol
            else:
                return None
