import torch, numpy as np

class VAELoss:
	def __init__(self, encoder, decoder):
		self.encoder = encoder
		self.decoer = decoder