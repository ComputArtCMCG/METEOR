from typing import List,Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
from states import tensorize_nxgraphs,mol_to_nx,finished_to_prob_mask,finished_to_critic_features

#POSSIBLE_ATOMS = ['C','N','O','S','F','I', 'Cl','Br']
#POSSIBLE_ATOM_TYPES_NUM = len(POSSIBLE_ATOMS)
#MAX_ACTION = 100
#MAX_ATOM = MAX_ACTION + 1  + POSSIBLE_ATOM_TYPES_NUM # 1 for padding
#MAX_NB = 6

class GCN(nn.Module):
    def __init__(self,feature_dims,device):
        super().__init__()
        self.feature_dims = feature_dims
        self.device = device
        self.W_single = nn.Linear(feature_dims,feature_dims)
        self.W_double = nn.Linear(feature_dims,feature_dims)
        self.W_triple = nn.Linear(feature_dims,feature_dims)

    def forward(self,fatoms_input,adj_matrix,all_mask):
        out = []
        for i,w in enumerate([self.W_single,self.W_double,self.W_triple]):
            out.append(F.relu(torch.matmul(adj_matrix[:,:,:,i],w(fatoms_input))).unsqueeze(2))  #[batch,n,1,f]
        fatoms_input = torch.cat(out,dim=2)
        fatoms_input = torch.sum(fatoms_input,dim=2)
        fatoms_input = fatoms_input * all_mask
        return fatoms_input

class ActorCritic(nn.Module):
    def __init__(self,feature_dims=128,update_iters=5,entropy_coeff=0.,clip_epsilon=0.,device=torch.device('cpu')):
        super().__init__()
        self.feature_dims = feature_dims
        self.update_iters = update_iters
        self.entropy_coeff = entropy_coeff
        self.clip_epsilon = clip_epsilon
        self.device = device
        self.fatoms_embedding = nn.Sequential(nn.Linear(20,feature_dims,bias=False),nn.ReLU())
        self.fatoms_critic_embedding = nn.Sequential(nn.Linear(20,feature_dims,bias=False),nn.ReLU())
        self.actor_gcns = nn.ModuleList([GCN(feature_dims,device) for _ in range(update_iters)])
        self.critic_gcns = nn.ModuleList([GCN(feature_dims,device) for _ in range(update_iters)])
        
        ###actor network
        self.end_selection_layer = nn.Sequential(
            nn.Linear(2*feature_dims,feature_dims),
            nn.ReLU(),
            nn.Linear(feature_dims,1,bias=False)
        )

        self.bond_selection_layer = nn.Sequential(
            nn.Linear(2*feature_dims,feature_dims),
            nn.ReLU(),
            nn.Linear(feature_dims,3,bias=False),
            nn.Softmax(dim=-1)
        )
        
        self.stop_selection_layer = nn.Sequential(
            nn.Linear(feature_dims,feature_dims),
            nn.ReLU(),
            nn.Linear(feature_dims,2,bias=False),
            nn.Softmax(dim=-1)
        )
        
        self.qlayer = nn.Sequential(
            nn.Linear(2*feature_dims,feature_dims),
            nn.ReLU(),
            nn.Linear(feature_dims,1),
        )
    
    def actor_feature_update(self,fatoms,adj_matrix,all_mask,current_mask):        
        fatoms_input = self.fatoms_embedding(fatoms) #[batch,atoms,feature_dim]   
        for gcn in self.actor_gcns:
            fatoms_input = gcn(fatoms_input,adj_matrix,all_mask)
        return fatoms_input

    def critic_feature_update(self,fatoms,adj_matrix,all_mask,current_mask):        
        fatoms_input = self.fatoms_critic_embedding(fatoms) #[batch,atoms,feature_dim]   
        for gcn in self.critic_gcns:
            fatoms_input = gcn(fatoms_input,adj_matrix,all_mask)
        q_super_input = self.maxpool_sumpool(fatoms_input,current_mask)
        return q_super_input

    def maxpool_sumpool(self,fatoms:torch.Tensor,current_mask):
        max_mask = torch.where(current_mask==1.,0.,-1e7)
        sum_pool = torch.sum(fatoms*current_mask,dim=1)
        max_pool = torch.max(fatoms+max_mask,dim=1).values
        return torch.cat([sum_pool,max_pool],dim=-1)
    
    def expert_training(self,fatoms_input:torch.Tensor,focus:torch.LongTensor,prob_mask:torch.Tensor,expert_actions:torch.LongTensor):
        
        expert_actions = expert_actions.to(self.device)
        end_selection = expert_actions[:,0]
        bond_selection = expert_actions[:,1]
        stop_selection = expert_actions[:,2]
        #fatoms shape [b,n,f] , focus [b,1]
        #prob_mask [b,n,1] mask =1, a possible selection
        start_atoms = batch_selection(fatoms_input,focus)  # [b,1,f]
        
        end_input = torch.cat(torch.broadcast_tensors(start_atoms,fatoms_input),dim=-1) # [b,n,2f]
        prob_mask = torch.where(prob_mask==1.,0.,-1e7) 
        end_atom_probs = torch.softmax(self.end_selection_layer(end_input)+prob_mask,dim=1).squeeze(2)  #[b,n]  

        end_atom_probs = end_atom_probs.gather(dim=1,index=end_selection.view([-1,1])) #[b,n] -> [b,1] 
        end_atoms = batch_selection(fatoms_input,end_selection)  #[b,f]

        bond_input = torch.cat([start_atoms.squeeze(1),end_atoms],dim=1) #[b,2f]
        bond_probs = self.bond_selection_layer(bond_input).gather(dim=1,index=bond_selection.view([-1,1]))
        stop_probs = self.stop_selection_layer(start_atoms.squeeze(1)).gather(dim=1,index=stop_selection.view([-1,1]))

        masks = torch.where(stop_selection==1.,0.,1.).view([-1,1])  #change [b,1]
        whole_probs = torch.cat([end_atom_probs,bond_probs,stop_probs],dim=-1) #[b,3]

        log_whole_probs = torch.log(torch.masked_select(whole_probs,masks.bool()).view([-1,3]))  #
        masks_stop = torch.where(stop_selection==1.,1.,0.).view([-1,1])
        log_probs_masked = torch.log(torch.masked_select(whole_probs[:,2].unsqueeze(-1),masks_stop.bool()).view([-1,1]))


        final_log_probs = torch.sum(log_whole_probs)  + torch.sum(log_probs_masked)  #[b]
        loss = -1 * final_log_probs / expert_actions.shape[0]
        return loss
    
    def get_prob_and_action(self,fatoms_input:torch.Tensor,focus:torch.LongTensor,prob_mask:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        get action and probs when acting 
        fatoms_input shape [batch,atom,f]
        focus shape [batch,1]
        """
        start_atom = batch_selection(fatoms_input,focus) #[batch,1,f]
        prob_mask = torch.where(prob_mask==1.,0.,-1e7)
        
        end_input = torch.cat(torch.broadcast_tensors(start_atom,fatoms_input),dim=-1) #[batch,atom,2f]
        end_atom_probs = torch.softmax(self.end_selection_layer(end_input)+prob_mask,dim=1).squeeze(-1) #[batch,atom,2f] -> [batch,atom]
        end_atom_dist = Categorical(probs=end_atom_probs) 
        end_atom_selection = end_atom_dist.sample().view([-1]) #[batch]
        end_atom = batch_selection(fatoms_input,end_atom_selection.unsqueeze(1)) #[batch,1,f]
        end_prob = end_atom_probs.gather(dim=1,index=end_atom_selection.unsqueeze(1)) #[batch,1]

        bond_probs = self.bond_selection_layer(torch.cat([start_atom,end_atom],dim=-1)).squeeze(1)  #[batch,1,2f] -> [batch,3]
        bond_dist = Categorical(probs=bond_probs) 
        bond_selection = bond_dist.sample().view([-1]) #[batch]
        bond_prob = bond_probs.gather(dim=1,index=bond_selection.unsqueeze(1)) #[batch,1]

        stop_probs = self.stop_selection_layer(start_atom).squeeze(1)  #[batch,2] 
        stop_dist = Categorical(probs=stop_probs) 
        stop_selection = stop_dist.sample().view([-1]) #[batch]
        stop_prob = stop_probs.gather(dim=1,index=stop_selection.unsqueeze(1)) #[batch,1]
        
        action_batch = torch.stack([end_atom_selection,bond_selection,stop_selection],dim=1) #[batch,3]
        prob_batch = torch.cat([end_prob,bond_prob,stop_prob],dim=-1)  #[batch,3]
        return (action_batch,prob_batch)

    def actorcritic_run(self,mol_features,focus,prob_mask,action_batch):
        """
        calculate prob and entropy during RL process
        focus shape [batch,1]
        prob_mask shape [batch,atom,1]
        action_batch shape [batch,3]
        Return: mask shape [batch] 1 for stop 0 for continue
        """
        prob_mask = torch.where(prob_mask==1.,0.,-1e7)
        fatoms_input = self.actor_feature_update(*mol_features) 
        q_super_input = self.critic_feature_update(*mol_features)
        qvalue:torch.Tensor = self.qlayer(q_super_input)
        
        start_atom = batch_selection(fatoms_input,focus) #[batch,1,f]
        end_input = torch.cat(torch.broadcast_tensors(start_atom,fatoms_input),dim=-1) #[batch,atom,2f]
        end_atom_probs = torch.softmax(self.end_selection_layer(end_input)+prob_mask,dim=1).squeeze(-1) #[batch,n,2f] -> [batch,atom]
        end_atom_entropy = Categorical(probs=end_atom_probs).entropy()
        end_atom = batch_selection(fatoms_input,action_batch[:,0].unsqueeze(1)) #[batch,1,f]
        end_prob_new = end_atom_probs.gather(dim=1,index=action_batch[:,0].unsqueeze(1)) #[batch,1]
        
        bond_probs = self.bond_selection_layer(torch.cat([start_atom,end_atom],dim=-1)).squeeze(1)  #[batch,1,2f] -> [batch,3]
        bond_entropy = Categorical(probs=bond_probs).entropy()
        bond_prob_new = bond_probs.gather(dim=1,index=action_batch[:,1].unsqueeze(1)) #[batch,1]

        stop_probs = self.stop_selection_layer(start_atom).squeeze(1)  #[batch,2] 
        stop_entropy = Categorical(probs=stop_probs).entropy()
        stop_prob_new = stop_probs.gather(dim=1,index=action_batch[:,2].unsqueeze(1)) #[batch,1]
        
        prob_batch_new = torch.cat([end_prob_new,bond_prob_new,stop_prob_new],dim=-1)  #[batch,3]
        entropy = torch.sum(torch.stack([end_atom_entropy,bond_entropy,stop_entropy],dim=1),dim=1) #[batch]
        return prob_batch_new,entropy,qvalue

    def action_qvalue_from_mols(self,mol_features,focus,prob_mask) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        """
        used during actor acting; the actor will recieve one mol from env and make decision
        this method finaly returns an action and prob with shape of [batch,3] 
        input focus with shape [batch,1]
        """
        fatoms,adj_matrix,all_mask,current_mask = map(lambda x:x.to(self.device),mol_features)
        #fatoms = fatoms.to(self.device)
        #adj_matrix = adj_matrix.to(self.device)
        #all_mask = all_mask.to(self.device)
        #current_mask = current_mask.to(self.device)
        focus = focus.to(self.device)
        prob_mask = prob_mask.to(self.device)
        
        mol_features = (fatoms,adj_matrix,all_mask,current_mask)
        fatoms_input = self.actor_feature_update(*mol_features)
        q_super_input = self.critic_feature_update(*mol_features)
        (action_batch,prob_batch) = self.get_prob_and_action(fatoms_input,focus,prob_mask)
        qvalue = self.qlayer(q_super_input)

        action_batch = action_batch.detach().to('cpu')
        prob_batch = prob_batch.detach().to('cpu')
        qvalue = qvalue.detach().to('cpu')

        return (action_batch,prob_batch,qvalue)

    def actor_loss(self,prob_batch_new:torch.Tensor,prob_batch:torch.Tensor,action_batch:torch.Tensor,advantage_batch:torch.Tensor):
        """
        mask with shape [batch]
        """
        masks = torch.where(action_batch[:,2]==1.,0.,1.).view([-1,1])  #change [b,1]
        masks_stop = torch.where(action_batch[:,2]==1.,1.,0.).view([-1,1])

        diff_1 = torch.sum(torch.log(torch.masked_select(prob_batch_new,masks.bool())).view([-1,3]),dim=1) - \
            torch.sum(torch.log(torch.masked_select(prob_batch,masks.bool())).view([-1,3]),dim=1)
        diff_2 = torch.log(torch.masked_select(prob_batch_new[:,2].unsqueeze(-1),masks_stop.bool())) - \
            torch.log(torch.masked_select(prob_batch[:,2].unsqueeze(-1),masks_stop.bool()))
        #mask when stop = 1 ,mask = 1
        advantage_batch = torch.cat([torch.masked_select(advantage_batch,masks.bool()),torch.masked_select(advantage_batch,masks_stop.bool())])
        diff = torch.cat([diff_1,diff_2])  #[b]
        ratio = torch.exp(diff)
        approxy_kl = torch.mean(torch.square(diff))
        option_1 = ratio * advantage_batch.detach()
        option_2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage_batch.detach()
        actor_l = -1 * torch.mean(torch.min(option_1,option_2))   #minimize the loss function
        return actor_l,approxy_kl

    def critic_loss(self,qvalue,v_target_batch:torch.Tensor) -> torch.Tensor:
        #old_qvalue_batch = old_qvalue_batch.to(self.critic.device)
        #clip_q = old_qvalue_batch + torch.clamp(qvalue - old_qvalue_batch, -self.clip_epsilon, self.clip_epsilon)
        #qloss_1 = torch.square(qvalue-reward)
        #qloss_2 = torch.square(clip_q-reward)
        #critic_l = 0.5 * torch.mean(torch.max(qloss_1,qloss_2))
        critic_l = torch.mean(torch.abs(qvalue-v_target_batch.detach()))
        
        return critic_l

    def forward(self,current_mol_features,focus,prob_mask,action_batch,probs_batch,v_target_batch,advantage_batch,train_round):
        prob_batch_new,entropy,qvalue = self.actorcritic_run(current_mol_features,focus,prob_mask,action_batch)
        actor_l,approxy_kl = self.actor_loss(prob_batch_new,probs_batch,action_batch,advantage_batch)
        critic_l = self.critic_loss(qvalue,v_target_batch)
        entropy = torch.mean(entropy)
        if train_round == 1:
            loss = critic_l
        else:
            loss = actor_l + critic_l - entropy * self.entropy_coeff
        return loss,critic_l,entropy,approxy_kl

    def set_device(self,device):
        self.device = device
        self.to(device)

### auxiliary functions
def index_select_ND(source:torch.Tensor,index:torch.Tensor):
    assert len(source.shape) == 2
    index_size = index.size()
    suffix_dim = source.size()[-1:]
    final_size = index_size + suffix_dim
    target = source.index_select(0, index.view(-1))
    return target.view(final_size)

def batch_selection(source:torch.Tensor,index:torch.Tensor):
    '''
    source: a 3D tensor with shape [batch_size,nodes,feature] 
    index: a 2D/3D index tensor with shape [batch_size,nodes] or [batch_size,nodes,nei]
    return: a 3D/4D tensor with shape [batch_size,nodes,feature] or [batch_size,nodes,nei,feature]
    '''
    return torch.stack([index_select_ND(i,j) for i,j in zip(source,index)],dim=0)

def batch_mask(source:torch.Tensor,index:torch.LongTensor):
    result = source.clone()
    for (i,j) in zip(result,index):
        i[j,:] = -1e7
    return result