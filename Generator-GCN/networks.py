from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
from states import tensorize_nxgraphs,mol_to_nx

POSSIBLE_ATOMS = ['C','N','O','S','F','I', 'Cl','Br']
POSSIBLE_ATOM_TYPES_NUM = len(POSSIBLE_ATOMS)
MAX_ACTION = 100
MAX_ATOM = MAX_ACTION + 1  + POSSIBLE_ATOM_TYPES_NUM # 1 for padding
MAX_NB = 6

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
            out.append(F.leaky_relu(torch.matmul(adj_matrix[:,:,:,i],w(fatoms_input)),negative_slope=0.1).unsqueeze(2))  #[batch,n,1,f]
        fatoms_input = torch.cat(out,dim=2)
        fatoms_input = torch.sum(fatoms_input,dim=2)
        fatoms_input = fatoms_input * all_mask
        return fatoms_input

class ActorCritic(nn.Module):
    def __init__(self,feature_dims=128,update_iters=5,device=torch.device('cpu'),entropy_coeff=0.,clip_epsilon=0.):
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
        self.start_selection_layer = nn.Sequential(
            nn.Linear(feature_dims,feature_dims),
            nn.ReLU(),
            nn.Linear(feature_dims,1,bias=False)
        )
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
            nn.Linear(2*feature_dims,feature_dims),
            nn.ReLU(),
            nn.Linear(feature_dims,2,bias=False),
            nn.Softmax(dim=-1)
        )
        
        self.qlayer = nn.Sequential(
            nn.Linear(2*feature_dims,feature_dims),
            nn.ReLU(),
            nn.Linear(feature_dims,1,bias=False),
        )
    
    def actor_feature_update(self,fatoms,adj_matrix,all_mask,current_mask):        
        fatoms_input = self.fatoms_embedding(fatoms) #[batch,atoms,feature_dim]  #equ 
        for gcn in self.actor_gcns:
            fatoms_input = gcn(fatoms_input,adj_matrix,all_mask)
        super_input = self.maxpool_sumpool(fatoms_input,current_mask)
        return  fatoms_input,super_input

    def critic_feature_update(self,fatoms,adj_matrix,all_mask,current_mask):        
        fatoms_input = self.fatoms_critic_embedding(fatoms) #[batch,atoms,feature_dim]  #equ 
        for gcn in self.critic_gcns:
            fatoms_input = gcn(fatoms_input,adj_matrix,all_mask)
        q_super_input = self.maxpool_sumpool(fatoms_input,current_mask)
        return q_super_input

    def maxpool_sumpool(self,fatoms:torch.Tensor,current_mask):
        max_mask = torch.where(current_mask==1.,0.,-1e7)
        sum_pool = torch.sum(fatoms*current_mask,dim=1)
        max_pool = torch.max(fatoms+max_mask,dim=1).values
        return torch.cat([sum_pool,max_pool],dim=-1)
    
    def expert_training(self,fatoms_input:torch.Tensor,super_input:torch.Tensor,all_mask:torch.Tensor,expert_actions):
        whole_probs,expert_stops = [],[]
        for (atom_feature,s,m,expert_action) in zip(fatoms_input,super_input,all_mask,expert_actions):
            
            whole_feature = torch.masked_select(atom_feature,m.bool()).view([-1,self.feature_dims])  #[n,f]
            current_mol_feature = whole_feature[:-POSSIBLE_ATOM_TYPES_NUM,:]
            expert_action = expert_action.to(self.device)
            start_selection = expert_action[:,0]  #shape [n]
            end_selection = expert_action[:,1]
            bond_selection = expert_action[:,2]
            stop_selection = expert_action[:,3]

            start_atom_probs = torch.softmax(self.start_selection_layer(current_mol_feature),dim=0).index_select(dim=0,index=start_selection)#[n,1]  -> [k,1]
            start_atoms = current_mol_feature.index_select(dim=0,index=start_selection)  #[k,f]

            end_input = torch.cat(torch.broadcast_tensors(start_atoms.unsqueeze(1),whole_feature.unsqueeze(0)),dim=-1) #[k,1,f],[1,n,f] ->[k,n,2f]
            end_atom_probs = torch.softmax(self.end_selection_layer(end_input),dim=1).squeeze(-1)  #[k,n,1] -> [k,n]  
            end_atom_probs = end_atom_probs.gather(dim=1,index=end_selection.view([-1,1])) #[k,n] -> [k,1] 
            end_atoms = whole_feature.index_select(dim=0,index=end_selection)  #[k,f]

            bond_probs = self.bond_selection_layer(torch.cat([start_atoms,end_atoms],dim=-1)).gather(dim=1,index=bond_selection.view([-1,1]))  #[k,3]-> [k,1]
            stop_probs = self.stop_selection_layer(s.unsqueeze(0)).view([1,2]).index_select(dim=1,index=stop_selection).view([-1,1]) #[1,2] -> [1,k] -> [k,1]

            whole_probs.append(torch.cat([start_atom_probs,end_atom_probs,bond_probs,stop_probs],dim=-1))
            expert_stops.append(stop_selection)
            
        whole_probs = torch.cat(whole_probs,dim=0)  #[xk,4]
        expert_stops = torch.cat(expert_stops) #[xk]
        log_whole_probs = torch.log(whole_probs)
        expert_masks = torch.where(expert_stops==1.,0.,1.).unsqueeze(1) #[xk,1]

        log_probs_masked = torch.reshape(log_whole_probs[:,3] * torch.where(expert_stops==1.,1.,0.),[-1,1])  #[xk,1]
        final_log_probs = torch.sum(log_whole_probs,dim=1,keepdim=True) * expert_masks + log_probs_masked
        loss = -1 * torch.mean(final_log_probs) 
        return loss
    
    def get_prob_and_action(self,fatoms_input:torch.Tensor,super_input:torch.Tensor,all_mask:torch.Tensor,current_mask:torch.Tensor) -> tuple([torch.Tensor,torch.Tensor]):
        """
        get action and probs when acting 
        expert_action_batch is used for expert training
        """
        prob_mask_all = torch.where(all_mask==1.,0.,-1e7) #[batch,atom,1]
        prob_mask_current = torch.where(current_mask==1.,0.,-1e7)
        mol_features = fatoms_input
        start_atom_probs = torch.softmax(self.start_selection_layer(mol_features)+prob_mask_current,dim=1).squeeze(-1)  #[batch,atom,1] -> [batch,atom]
        start_atom_dist = Categorical(probs=start_atom_probs)  #[batch]
        start_atom_selection = start_atom_dist.sample().view([-1]) #[batch]
        start_atom = batch_selection(mol_features,start_atom_selection.unsqueeze(1)) #[batch,1,f]
        start_prob = start_atom_probs.gather(dim=1,index=start_atom_selection.unsqueeze(1)) #[batch,1]
        
        end_input = torch.cat(torch.broadcast_tensors(start_atom,mol_features),dim=-1) #[batch,n,2f]
        end_atom_probs = torch.softmax(self.end_selection_layer(end_input)+prob_mask_all,dim=1).squeeze(-1) #[batch,n,2f] -> [batch,atom]
        end_atom_dist = Categorical(probs=end_atom_probs) 
        end_atom_selection = end_atom_dist.sample().view([-1]) #[batch]
        end_atom = batch_selection(mol_features,end_atom_selection.unsqueeze(1)) #[batch,1,f]
        end_prob = end_atom_probs.gather(dim=1,index=end_atom_selection.unsqueeze(1)) #[batch,1]

        bond_probs = self.bond_selection_layer(torch.cat([start_atom,end_atom],dim=-1)).squeeze(1)  #[batch,1,2f] -> [batch,3]
        bond_dist = Categorical(probs=bond_probs) 
        bond_selection = bond_dist.sample().view([-1]) #[batch]
        bond_prob = bond_probs.gather(dim=1,index=bond_selection.unsqueeze(1)) #[batch,1]

        stop_probs = self.stop_selection_layer(super_input)  #[batch,2] 
        stop_dist = Categorical(probs=stop_probs) 
        stop_selection = stop_dist.sample().view([-1]) #[batch]
        stop_prob = stop_probs.gather(dim=1,index=stop_selection.unsqueeze(1)) #[batch,1]
        
        action_batch = torch.stack([start_atom_selection,end_atom_selection,bond_selection,stop_selection],dim=1) #[batch,4]
        prob_batch = torch.cat([start_prob,end_prob,bond_prob,stop_prob],dim=-1)  #[batch,4]
        return (action_batch,prob_batch)

    def actorcritic_run(self,mol_features,action_batch):
        """
        calculate prob and entropy during RL process
        """
        #action_batch shape [batch,4]
        _,_,all_mask,current_mask = mol_features
        fatoms_input,super_input = self.actor_feature_update(*mol_features) 
        q_super_input = self.critic_feature_update(*mol_features)
        qvalue:torch.Tensor = self.qlayer(q_super_input)
        prob_mask_all = torch.where(all_mask==1.,0.,-1e7) #[batch,atom,1]
        prob_mask_current = torch.where(current_mask==1.,0.,-1e7)
        mol_features = fatoms_input
        start_atom_probs_input = self.start_selection_layer(mol_features)+prob_mask_current
        start_atom_probs = torch.softmax(start_atom_probs_input,dim=1).squeeze(-1)  #[batch,atom,1] -> [batch,atom]
        start_atom_entropy = Categorical(probs=start_atom_probs).entropy()  #entropy shape [batch]
        start_atom = batch_selection(mol_features,action_batch[:,0].unsqueeze(1))
        start_prob_new = start_atom_probs.gather(dim=1,index=action_batch[:,0].unsqueeze(1))

        end_input = torch.cat(torch.broadcast_tensors(start_atom,mol_features),dim=-1) #[batch,n,2f]
        end_atom_probs = torch.softmax(self.end_selection_layer(end_input)+prob_mask_all,dim=1).squeeze(-1) #[batch,n,2f] -> [batch,atom]
        end_atom_entropy = Categorical(probs=end_atom_probs).entropy()
        end_atom = batch_selection(mol_features,action_batch[:,1].unsqueeze(1)) #[batch,1,f]
        end_prob_new = end_atom_probs.gather(dim=1,index=action_batch[:,1].unsqueeze(1)) #[batch,1]
        
        bond_probs = self.bond_selection_layer(torch.cat([start_atom,end_atom],dim=-1)).squeeze(1)  #[batch,1,2f] -> [batch,3]
        bond_entropy = Categorical(probs=bond_probs).entropy()
        bond_prob_new = bond_probs.gather(dim=1,index=action_batch[:,2].unsqueeze(1)) #[batch,1]

        stop_probs = self.stop_selection_layer(super_input)  #[batch,2] 
        stop_entropy = Categorical(probs=stop_probs).entropy()
        stop_prob_new = stop_probs.gather(dim=1,index=action_batch[:,3].unsqueeze(1)) #[batch,1]
        
        prob_batch_new = torch.cat([start_prob_new,end_prob_new,bond_prob_new,stop_prob_new],dim=-1)  #[batch,4]
        entropy = torch.sum(torch.stack([start_atom_entropy,end_atom_entropy,bond_entropy,stop_entropy],dim=1),dim=1) #[batch]
        return prob_batch_new,entropy,qvalue

    def action_qvalue_from_mols(self,mol_features) -> tuple([torch.Tensor,torch.Tensor]):
        """
        used during actor acting; the actor will recieve one mol from env and make decision
        this method finaly returns an action and prob with shape of [1,4] 
        """
        fatoms,adj_matrix,all_mask,current_mask = map(lambda x:x.to(self.device),mol_features)
        
        mol_features = (fatoms,adj_matrix,all_mask,current_mask)
        fatoms_input,super_input = self.actor_feature_update(*mol_features)
        q_super_input = self.critic_feature_update(*mol_features)
        (action_batch,prob_batch) = self.get_prob_and_action(fatoms_input,super_input,all_mask,current_mask)
        qvalue = self.qlayer(q_super_input)

        action_batch = action_batch.to('cpu')
        prob_batch = prob_batch.to('cpu')
        qvalue = qvalue.to('cpu')

        return (action_batch,prob_batch,qvalue)

    def actor_loss(self,prob_batch_new,probs_batch:torch.Tensor,action_batch,advantage_batch:torch.Tensor):
        masks = torch.where(action_batch[:,3]==1.,0.,1.).view([-1,1])  #change [b,1]
        masks_stop = torch.where(action_batch[:,3]==1.,1.,0.).view([-1,1])

        diff_1 = torch.sum(torch.log(torch.masked_select(prob_batch_new,masks.bool())).view([-1,4]),dim=1) - \
            torch.sum(torch.log(torch.masked_select(probs_batch,masks.bool())).view([-1,4]),dim=1)
        diff_2 = torch.log(torch.masked_select(prob_batch_new[:,3].unsqueeze(-1),masks_stop.bool())) - \
            torch.log(torch.masked_select(probs_batch[:,3].unsqueeze(-1),masks_stop.bool()))
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
        critic_l = 0.5 * torch.mean(torch.square(qvalue-v_target_batch.detach()))
        
        return critic_l

    def forward(self,current_mol_features,action_batch,probs_batch,v_target_batch,advantage_batch,train_round):
        prob_batch_new,entropy,qvalue = self.actorcritic_run(current_mol_features,action_batch)
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
        self.to(self.device)

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
