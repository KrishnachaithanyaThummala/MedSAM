import unittest
import numpy as np
from src.cropping import crop_regions
from src.classification import preprocess_crop
class CropTests(unittest.TestCase):
    def test_margin_is_per_side_and_exclusive_bounds(self):
        image=np.zeros((100,100,3),np.uint8);mask=np.zeros((100,100),np.uint8);mask[30:50,20:40]=1
        r=crop_regions(image,mask)
        self.assertEqual([x['box_xyxy_exclusive'] for x in r],[[20,30,40,50],[18,28,42,52],[16,26,44,54]])
    def test_clips_to_boundary(self):
        image=np.zeros((10,10,3),np.uint8);mask=np.ones((10,10),np.uint8)
        self.assertTrue(all(x['image'].shape==(10,10,3) for x in crop_regions(image,mask)))
    def test_empty_mask_has_explicit_failure(self):
        with self.assertRaisesRegex(ValueError,'Empty'): crop_regions(np.zeros((5,5,3),np.uint8),np.zeros((5,5)))
    def test_channel_order_and_normalization(self):
        image=np.full((8,8,3),[255,0,0],np.uint8)
        tensor=preprocess_crop(image)
        np.testing.assert_allclose(tensor[0,:,0,0].numpy(),(np.array([0,0,1])-[.485,.456,.406])/[.229,.224,.225],rtol=1e-6)
if __name__=='__main__':unittest.main()
