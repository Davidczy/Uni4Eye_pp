from torch import nn as nn
from .helpers import to_2tuple, to_3tuple

class PatchEmbed23D(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size2 = to_2tuple(img_size)
        img_size3 = (img_size, img_size, img_size//2)
        patch_size2 = to_2tuple(patch_size)
        patch_size3 = to_3tuple(patch_size)
        self.img_size2 = img_size2
        self.img_size3 = img_size3
        self.patch_size2 = patch_size2
        self.patch_size3 = patch_size3
        self.grid_size2 = (img_size2[0] // patch_size2[0], img_size2[1] // patch_size2[1])
        self.grid_size3 = (img_size3[0] // patch_size3[0], img_size3[1] // patch_size3[1], img_size3[2] // patch_size3[2])
        self.num_patches2 = self.grid_size2[0] * self.grid_size2[1]
        self.num_patches3 = self.grid_size3[0] * self.grid_size3[1] * self.grid_size3[2]
        self.flatten = flatten

        self.proj2= nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size2, stride=patch_size2)
        self.proj3= nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size3, stride=patch_size3)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            # assert H == self.img_size[0] and W == self.img_size[1], \
            #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            x = self.proj2(x)
            t = 2
        else:
            B, C, H, W, D = x.shape
            x = self.proj3(x)
            t = 3
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, t