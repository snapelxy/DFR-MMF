import torch
import cv2
import torch.nn.functional as F
from torch import nn
from vedacore.misc import registry


        


@registry.register_module('common_model')
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.0,feature_h=64,feature_w=64,batch=1,decay=0.1,eps=0.1,unique_embed=False):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self.update_count = 0
        self.decay = decay
        self.eps = eps
        self.cp_embed = True
        self.batch,self.feature_h,self.feature_w,self.unique_embed = batch,feature_h,feature_w,unique_embed
        if self.unique_embed:
            embed = torch.randn( self._embedding_dim,self.feature_h*self.feature_w, self._num_embeddings+1)
        else:
            embed = torch.randn( self._embedding_dim, self._num_embeddings+1)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros( self._num_embeddings))
        self.register_buffer("embed_avg", embed[...,:-1].clone())
        self.register_buffer("dist_min", torch.zeros( 1,requires_grad=False))
        self.register_buffer("dist_midle", torch.zeros( 1,requires_grad=False))
        
        self.register_buffer("embed_update_count", torch.zeros( (self._num_embeddings),requires_grad=False))

    def init_dist_(self):
        self.dist_min.zero_()
        self.dist_midle.zero_()
        self.update_count = 0
 
    def forward(self, inputs,test=False):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()

        if self.unique_embed:
            flatten = inputs.reshape(-1,self.feature_w*self.feature_h, self._embedding_dim)
            flatten_temp = inputs[0:1].reshape(-1,self.feature_w*self.feature_h, self._embedding_dim)
        else:
            flatten = inputs.reshape(-1, self._embedding_dim)
            flatten_temp = inputs[0:1].reshape(-1, self._embedding_dim)

        if not self.training:
            embed_update_count_tmp=self.embed_update_count
            mask_updated = embed_update_count_tmp<1

            self.embed[:,:-1][:,mask_updated] = 9990
        dist = (
            flatten.pow(2).sum(-1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )[...,:-1]


        dist_min, embed_ind = (dist).min(1)
        embed_onehot = F.one_hot(embed_ind, self._num_embeddings).type(flatten.dtype)

        if self.cp_embed and self.training:
            cp_range = min(self.embed.shape[1]-1,flatten_temp.shape[0])
            self.embed[:,:cp_range].data.copy_(flatten_temp[:cp_range,:].transpose(0,1))

            self.cp_embed = False
            print("features copyed ----------------------")

        
        quantize,dist_mask = self.embed_code(embed_ind,(not self.training),dist,inputs)


        if self.training:

            self.embed_update_count += embed_onehot.sum(dim=0)
            uptimes,idx_min = torch.sort(self.embed_update_count)
            if uptimes.min()<3: #小于3 就强制更新
                t1,_ = dist.max(dim=-1)
                t1,t1_idx = torch.sort(t1,descending=False)
                for cp_i in range(5):
                    self.embed[:,idx_min[cp_i]].data.copy_(flatten[t1_idx[cp_i],:])  #拷贝10个
                    self.embed_update_count[idx_min[cp_i]] += 1

            self.update_count += 1
            self.dist_min.data.copy_((dist_min.min() + self.update_count*self.dist_min)/(self.update_count+1))
            self.dist_midle.data.copy_((dist_min.median() + self.update_count*self.dist_midle)/(self.update_count+1))

            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self._num_embeddings * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            mask_one_hot = embed_onehot.sum(dim=0).unsqueeze(0)
            mask_one_hot = mask_one_hot<0.9
            embed_normalized = self.embed.data[:,:-1]* mask_one_hot + embed_normalized * (~mask_one_hot)
            self.embed.data.copy_(torch.cat(
                [embed_normalized,
                 990.0*torch.ones((embed_normalized.shape[0],1),requires_grad=False).
                 to(embed_normalized.device)],axis=-1))
            


        quantize = inputs + (quantize - inputs).detach()
    
        return quantize.permute((0,3,1,2)), dist_min.reshape((1,inputs.shape[1],inputs.shape[2])), embed_ind,dist_mask

    def embed_code(self, embed_id,test,dist,inputs):
        if test:
            dist_gap = abs(self.dist_midle - self.dist_min)
            dis_min, _ = dist.min(dim=1)
            self.embed[:,-1] = 99999.0

            embed_tmp = self.embed.clone()

            embed_id = embed_id.view(*inputs.shape[:-1])
            return F.embedding(embed_id, embed_tmp.transpose(0, 1)),  None #dis_mask

        else:
            embed_id = embed_id.view(*inputs.shape[:-1])
            temp = self.embed.reshape((self.embed.shape[0],-1)).transpose(0, 1)
            return F.embedding(embed_id, temp),None


@registry.register_module('common_model')
class VectorQuantizerList(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.0,feature_h=[64],feature_w=[64],batch=[1],decay=0.1,eps=0.1,unique_embed=[False],**kargs):
        super(VectorQuantizerList, self).__init__()
        self.modules_list=[]
        for i in range(len(num_embeddings)):
            setattr(self,"l"+str(i),VectorQuantizer(num_embeddings[i],embedding_dim[i],
                commitment_cost,feature_h[i],feature_w[i],batch[i],decay,eps))
            self.modules_list.append("l"+str(i))
        self.out_index = kargs["out_index"] if "out_index" in kargs.keys() else 0
    def forward(self,inputs):
        assert len(self.modules_list) == len(inputs)
        out = []
        for m,inp in zip(self.modules_list,inputs):
            obj = getattr(self,m)
            out.append(obj(inp)[self.out_index])
        return out