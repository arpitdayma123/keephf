import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import ResidualBlockNoBN, make_layer
from einops import rearrange

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class CAB(nn.Module):
    def __init__(self, num_feat, is_concat=False, N=3, C=64):
        super(CAB, self).__init__()
        self.N = N
        self.C = C
        self.num_feat = num_feat
        self.is_concat = is_concat
        self.cab = nn.Sequential(
            nn.Linear(num_feat * N, C),
            nn.GELU(),
            nn.Linear(C, C),
            nn.GELU(),
            nn.Linear(C, num_feat * N),
            nn.Sigmoid()
        )
        if is_concat:
            self.cat_conv = nn.Conv2d(num_feat * N, num_feat, 1)

    def forward(self, x):
        # x: (B, N, C, H, W)
        B, N, C, H, W = x.shape
        x_pool = torch.mean(x.reshape(B, N, C, H * W), dim=-1) # (B, N, C)
        x_cab = self.cab(x_pool.reshape(B, N * C)) # (B, N*C)
        x_cab = x_cab.reshape(B, N, C, 1, 1)
        out = x * x_cab # (B, N, C, H, W)
        if self.is_concat:
            out = self.cat_conv(out.reshape(B, N * C, H, W)) # (B, C, H, W)
        return out

class CBAM(nn.Module):
    def __init__(self, c1, c2, K=3, S=1):
        super(CBAM, self).__init__()
        self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c1, c1 // 2, 1),
                nn.ReLU(),
                nn.Conv2d(c1 // 2, c1, 1),
                nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, K, padding=(K - 1) // 2, bias=False), # 7,3
            nn.Sigmoid()
        )

    def forward(self, x):
        x_channel = self.channel_gate(x)
        x = x * x_channel # (B, C, H, W)
        x_spatial = torch.cat([torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)], dim=1) # (B, 2, H, W)
        x_spatial = self.spatial_gate(x_spatial) # (B, 1, H, W)
        x = x * x_spatial # (B, C, H, W)
        return x

class SE(nn.Module):
    def __init__(self, num_feat):
        super(SE, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // 2, 1),
            nn.ReLU(),
            nn.Conv2d(num_feat // 2, num_feat, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert not bias
        assert in_channels % groups == 0, \
            f'in_channels {in_channels} is not divisible by groups {groups}'
        assert out_channels % groups == 0, \
            f'out_channels {out_channels} is not divisible by groups {groups}'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = tuple([0, 0])

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups,
                         *self.kernel_size))
        self.bias = None

        # TPN DCNv2 is not supported for Pytorch 1.10
        from torchvision.ops import DeformConv2d
        self.conv_offset = DeformConv2d(in_channels,
                                     out_channels, # 2 * deformable_groups * kernel_size * kernel_size: in TPN
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     # groups=self.deformable_groups, # in TPN
                                     bias=True)
        self.init_offset()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu') # original

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        # offset = self.conv_offset(x) # TPN style
        # return deform_conv2d(x, offset, self.weight, self.bias, self.stride,
        #                      self.padding, self.dilation, self.groups,
        #                      self.deformable_groups)
        return self.conv_offset(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() # DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class KEEPModule(nn.Module):
    def __init__(self,
                 inc=3,
                 outc=3,
                 midc=64,
                 n_blocks=8,
                 norm_type='bn', # bn, in
                 act_type='relu', # relu, leakyrelu, gelu
                 use_deform=True,
                 use_attn=True,
                 use_cbam=True,
                 use_cab=False,
                 use_se=False,
                 use_res=True,
                 use_skip=True,
                 is_concat=False,
                 num_queries=20,
                 n_frames=1):
        super(KEEPModule, self).__init__()
        self.inc = inc
        self.outc = outc
        self.midc = midc
        self.norm_type = norm_type
        self.act_type = act_type
        self.use_deform = use_deform
        self.use_attn = use_attn
        self.use_cbam = use_cbam
        self.use_cab = use_cab
        self.use_se = use_se
        self.use_res = use_res
        self.use_skip = use_skip
        self.n_frames = n_frames
        self.num_queries = num_queries

        if use_deform:
            self.conv_first = DeformableConv2d(inc * n_frames, midc, 3, 1, 1)
        else:
            self.conv_first = nn.Conv2d(inc * n_frames, midc, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, n_blocks, mid_channels=midc) # vanilla resblock
        if use_attn:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_queries, midc)) # learnable
            self.transformer = TransformerBlock(dim=midc, num_heads=8)
            self.query_embed = nn.Embedding(num_queries, midc) # learnable
            self.frame_token = nn.Embedding(n_frames, midc) # learnable
        if use_cbam:
            self.cbam = CBAM(midc, midc)
        if use_se:
            self.se = SE(midc)
        if use_cab:
            self.cab = CAB(midc, is_concat=is_concat, N=n_frames, C=midc)
        self.conv_last = nn.Conv2d(midc, outc, 3, 1, 1)

        if self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()

        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(midc)
        elif self.norm_type == 'in':
            self.norm = nn.InstanceNorm2d(midc)

    def forward(self, x):
        # x: (B, N, C, H, W)
        B, N, C, H, W = x.shape
        if self.use_cab:
            # cab before reshape
            x_cab = self.cab(x) # (B, N, C, H, W) or (B, C, H, W)
            if self.cab.is_concat:
                x = x_cab
            else:
                x = x_cab.reshape(B, N * C, H, W)
        else:
            x = x.reshape(B, N * C, H, W)
        if self.use_skip:
            x_skip = x
        x_first = self.act(self.norm(self.conv_first(x))) # (B, C, H, W)
        x_body = self.body(x_first) # (B, C, H, W)
        if self.use_attn:
            # x_body: (B, C, H, W)
            # pos_embed: (1, NQ, C)
            # query_embed: (NQ, C)
            # frame_token: (NF, C)
            x_body_attn = x_body.flatten(2).transpose(1, 2) # (B, HW, C)
            # x_body_attn = x_body_attn + self.pos_embed # broadcast
            query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) # (B, NQ, C)
            # frame_token = self.frame_token.weight.unsqueeze(0).repeat(B, 1, 1) # (B, NF, C)
            # x_body_attn = torch.cat([frame_token, query_embed, x_body_attn], dim=1) # (B, NF+NQ+HW, C)
            x_body_attn = torch.cat([query_embed, x_body_attn], dim=1) # (B, NQ+HW, C)
            x_body_attn = self.transformer(x_body_attn) # (B, NQ+HW, C)
            # x_body = x_body_attn[:, -H*W:, :].transpose(1, 2).reshape(B, self.midc, H, W) # (B, C, H, W)
            x_body = x_body_attn[:, self.num_queries:, :].transpose(1, 2).reshape(B, self.midc, H, W) # (B, C, H, W)
        if self.use_cbam:
            x_body = self.cbam(x_body)
        if self.use_se:
            x_body = self.se(x_body)
        if self.use_res:
            x_body = x_body + x_first
        x_last = self.conv_last(x_body) # (B, C, H, W)
        if self.use_skip:
            x_last = x_last + x_skip
        return x_last

@ARCH_REGISTRY.register()
class KEEP_Net(nn.Module):
    def __init__(self,
                 inc=3,
                 outc=3,
                 midc=64,
                 n_blocks=8,
                 num_res_blocks=8, # for sr module
                 norm_type='in', # bn, in
                 act_type='relu', # relu, leakyrelu, gelu
                 use_deform=True,
                 use_attn=True,
                 use_cbam=True,
                 use_cab=True,
                 use_se=False,
                 use_res=True,
                 use_skip=True,
                 is_concat=False,
                 num_queries=20,
                 n_frames=1,
                 scale=4):
        super(KEEP_Net, self).__init__()
        self.scale = scale
        self.n_frames = n_frames
        self.restoration_module = KEEPModule(inc, midc, midc, n_blocks, norm_type, act_type, use_deform, use_attn, use_cbam, use_cab, use_se, use_res, use_skip, is_concat, num_queries, n_frames)
        self.sr_module = nn.Sequential(
            make_layer(ResidualBlockNoBN, num_res_blocks, mid_channels=midc),
            nn.Conv2d(midc, midc * scale ** 2, 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(midc, outc, 3, 1, 1)
        )
        self.norm = LayerNorm(normalized_shape=midc, data_format='channels_first')
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, C, H, W) or (B, N, C, H, W)
        # if x is (B, C, H, W), then reshape to (B, 1, C, H, W)
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        # x: (B, N, C, H, W)
        x_restore = self.restoration_module(x) # (B, C, H, W)
        x_restore = self.act(self.norm(x_restore))
        x_sr = self.sr_module(x_restore)
        base = F.interpolate(x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1]), scale_factor=self.scale, mode='bilinear', align_corners=False)
        x_sr = x_sr + base
        return x_sr

if __name__ == '__main__':
    # Example usage (for testing)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test KEEP_Net
    print("Testing KEEP_Net...")
    inc = 3
    midc = 64 # Must match the midc used for pre-trained weights if loading them
    n_frames_test = 1 # For single image processing
    batch_size = 1
    height, width = 64, 64 # Example dimensions

    # Create a sample input tensor (e.g., for a single frame)
    # Input shape: (B, N, C, H, W) for multi-frame or (B, C, H, W) for single frame which gets unsqueezed
    if n_frames_test == 1:
        sample_input = torch.randn(batch_size, inc, height, width).to(device)
    else:
        sample_input = torch.randn(batch_size, n_frames_test, inc, height, width).to(device)

    print(f"Input shape: {sample_input.shape}")

    # Initialize the network
    # Parameters should match those used for training the loaded checkpoint
    keep_net = KEEP_Net(
        inc=inc,
        outc=3,
        midc=midc,
        n_blocks=8,        # Default in KEEPModule, adjust if different in app.py's config
        num_res_blocks=8,  # Default for sr_module, adjust if different
        norm_type='in',    # Default
        act_type='relu',   # Default
        use_deform=True,   # Default
        use_attn=True,     # Default
        use_cbam=True,     # Default
        use_cab=True,      # Default
        use_se=False,      # Default
        use_res=True,      # Default
        use_skip=True,     # Default
        is_concat=False,   # Default for CAB
        num_queries=20,    # Default for Transformer
        n_frames=n_frames_test, # Number of input frames
        scale=1            # Output scale factor (1 for no SR, >1 for SR)
    ).to(device)
    keep_net.eval()

    with torch.no_grad():
        output = keep_net(sample_input)

    print(f"Output shape: {output.shape}")

    # Example: if scale=1, output H, W should be same as input H, W
    # if scale=4, output H, W should be 4*input H, W
    expected_height = height * keep_net.scale
    expected_width = width * keep_net.scale
    assert output.shape == (batch_size, inc, expected_height, expected_width), \
        f"Output shape mismatch. Expected {(batch_size, inc, expected_height, expected_width)}, got {output.shape}"

    print("KEEP_Net test passed with n_frames_test=1, scale=1.")

    # Test with n_frames > 1 if your use case requires it
    if False: # Set to true to test multi-frame
        n_frames_test_multi = 3
        if n_frames_test_multi == 1:
            sample_input_multi = torch.randn(batch_size, inc, height, width).to(device)
        else:
            sample_input_multi = torch.randn(batch_size, n_frames_test_multi, inc, height, width).to(device)

        keep_net_multi = KEEP_Net(n_frames=n_frames_test_multi, midc=midc, scale=1).to(device) # Adjust other params as needed
        keep_net_multi.eval()
        with torch.no_grad():
            output_multi = keep_net_multi(sample_input_multi)
        print(f"Input shape (multi-frame): {sample_input_multi.shape}")
        print(f"Output shape (multi-frame): {output_multi.shape}")
        assert output_multi.shape == (batch_size, inc, height * keep_net_multi.scale, width * keep_net_multi.scale)
        print(f"KEEP_Net test passed with n_frames_test={n_frames_test_multi}, scale=1.")

    print("All tests passed.")
