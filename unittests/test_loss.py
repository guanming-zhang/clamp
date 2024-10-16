import unittest
import torch
import sys
import math
sys.path.append('../model')
from loss_module import EllipsoidPackingLoss
class TestEllipoidsPackingLoss(unittest.TestCase):
    def setUp(self) -> None:
        #-----------------generate the data-----------------#
        self.n_views = 1200
        self.dim = 3
        self.batch_size = 2
        rand1 = torch.normal(mean=0.0,std = 1.0, size =(self.n_views,self.dim))
        rand2 = torch.normal(mean=0.0,std = 1.0, size =(self.n_views,self.dim))
        # eigenvectors and eigenvalues for corr1 and corr2
        # corr1: lambda1 = 1, v1 = [1.0, 0.0, 0.0]
        #        lambda2 = 1, v2 = [0.0, 1.0, 0.0]
        #        lambda3 = 9, v3 = [0.0, 0.0, 1.0]
        # corr2: lambda1 = 1, v1 = [0.0, 0.0, 1.0]
        #        lambda2 = 1, v2 = [-1/sqrt(2), 1/sqrt(2), 0.0]
        #        lambda3 = 9, v3 = [1/sqrt(2),  1/sqrt(2), 0.0]
        corr1 = torch.tensor([[1.0, 0.0, 0.0],[0.0,1.0,0.0],[0.0,0.0,9.0]])
        corr2 = torch.tensor([[5.0, 4.0, 0.0],[4.0,5.0,0.0],[0.0,0.0,1.0]])
        c1 = torch.tensor([0.0,0.0,1.0])
        c2 = torch.tensor([0.0,0.0,0.0]) 
        # shape of data 1&2: n_views*dim
        data1 = torch.matmul(rand1,self.matrix_sqrt(corr1)) + c1
        data2 = torch.matmul(rand2,self.matrix_sqrt(corr2)) + c2
        data = torch.stack((data1,data2)) # shape batch_size*n_views*dim
        data = data.permute(1,0,2)
        data = data.reshape(self.n_views*self.batch_size,self.dim)
        self.data = data.reshape(self.n_views*self.batch_size,self.dim)
        self.loss = EllipsoidPackingLoss(self.n_views,self.batch_size,lw0=0.0,lw1=0.0,lw2=0.0,record=True)
        return super().setUp()
    
    def matrix_sqrt(self,A):
        # calculate the matrix square root
        val,vec = torch.linalg.eigh(A)
        sigma = torch.diag(torch.sqrt(val))
        return vec @ sigma @ vec.T

    def test_loss_size(self):
        #-----------------test the loss function-----------------#
        #loss function 0 = the trace of correlation matrix
        self.loss.lw0 = 1.0
        self.loss.lw1 = 0.0
        self.loss.lw2 = 0.0
        data = self.data.detach().clone()
        l = self.loss(data,[]).item()
        # trace(corr1) = 11.0 trace(corr2) = 11.0
        loss =  math.sqrt(11.0/self.dim) + math.sqrt(11.0/self.dim) 
        err_rel = abs(l-loss)/loss
        err_msg = "estimated loss = {}, true loss = {}".format(l,loss)
        self.assertTrue(err_rel < 0.2,err_msg)
     
    def test_loss_repulsion(self):
        #-----------------test the loss function-----------------#
        #loss function 1 = repuslion between different clusters 
        self.loss.lw0 = 0.0
        self.loss.lw1 = 1.0
        self.loss.lw2 = 0.0
        data = self.data.detach().clone()
        l = self.loss(data,[]).item()
        # center1 = [0,0,1], center2 = [0,0,0]
        # rad1 = sqrt(11/3), rad2 = sqrt(11/3)
        loss =  self.loss.lw1 * 1.092
        err_rel = abs(l-loss)/loss
        err_msg = "estimated loss = {}, true loss = {}".format(l,loss)
        self.assertTrue(err_rel < 0.2,err_msg)
    
    def test_loss_alignment(self):
        #-----------------test the loss function-----------------#
        #loss function 2 = alignment between different clusters 
        self.loss.lw0 = 0.0
        self.loss.lw1 = 0.0
        self.loss.lw2 = 1.0
        data = self.data.detach().clone()
        l = self.loss(data,[]).item()
        # eigenvector1 = [0,0,1] and [1/sqrt(2),1/sqrt(2),0]
        loss =  1.0*2 # 2 -- for the double counting
        err_rel = abs(l-loss)/loss
        err_msg = "estimated loss = {}, true loss = {}".format(l,loss)
        self.assertTrue(err_rel < 0.2,err_msg)
     

if __name__ == '__main__':
    unittest.main()

