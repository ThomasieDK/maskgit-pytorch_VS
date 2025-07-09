import unittest
import torch
from tamg.topology import TopologicalLoss

class TopologyLossTest(unittest.TestCase):
    def test_forward_shapes(self):
        loss_fn = TopologicalLoss()
        logits = torch.randn(2, 16, 4)
        mask = torch.zeros(2, 16, dtype=torch.bool)
        loss, grad = loss_fn(logits, mask)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(grad.shape, logits.shape)

if __name__ == '__main__':
    unittest.main()
