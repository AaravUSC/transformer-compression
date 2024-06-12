import torch
import torchaudio
import numpy as np

def compress_DALR(layer, acts, layerWeights, layerBiases, t, lamb):

  Z = np.zeros((acts.size()[0], layerWeights.size()[0], acts.size()[1]), dtype = float)
  layerWeights = layerWeights.detach().numpy() #mxn

  acts = acts.detach().numpy() #nxp
  acts = acts.reshape(acts.shape[0]*acts.shape[1], acts.shape[2])

  Z = np.dot(layerWeights, acts.T)

  U, _, _ = np.linalg.svd(Z, full_matrices = True) #mxm
  print(U.shape)
  XXT= np.dot(acts.T, acts) #nxn
  t = int(t)
  A = U[:,:t]
  inv = np.linalg.inv(XXT + lamb*np.eye(layerWeights.shape[1])) #nxn
  step1 = np.dot(A.T,Z)

  step2 = np.dot(step1, acts)

  B= np.dot(step2, inv)
  #B = B.T

  return [B, A]


def compress_SVD(layer, t):
  layerweights = layer.weight.detach().numpy()
  U, S, Vt = np.linalg.svd(layerweights, full_matrices = True)
  print(U.shape)
  print(S.shape)
  print(Vt.shape)
  U = U[:, :t]
  S = S[:t]
  Vt = Vt[:t, :]

  first_weights = np.dot(S, Vt)
  second_weights = U
  return [first_weights, second_weights]
