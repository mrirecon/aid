import torch as th
from guided_diffusion.unet2 import CausalUNetModel, SwitchDims, SwitchDimsT, DiTBlock, TimestepEmbedSequential, MergeDims, ResBlock, CausalConv1d, CausalGatedResNet, DiTBlockCausalGate
import einops
import debugpy
from torch.profiler import profile, record_function, ProfilerActivity
from torchsummary import summary 
"""
 you might need to modify the source code of torchsummary to support the model around line 99 to define prod function to calculate the size of input
 def prod(data):
    products = [np.prod(item) if item else 1 for item in data]
    return products
"""

debug = False

if debug:
    debugpy.listen(5678)
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    print("debugpy connected")
    debugpy.breakpoint()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = th.device("cuda")

if True:
    model = CausalUNetModel(
        image_size=64,
        seq_len=10,
        in_channels=2,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=[16],
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_causal_gated=True,
        concat_cond=False
    ).to(device)

    # before testing model, we need to disable zero_module in unet2.py
    seq = th.zeros(2, 11, 2, 64, 64).to(device)
    x0 = seq[:, :-1, ...]
    x1 = seq[:, 1:, ...]
    timesteps = th.ones(2).to(device)

    print("start profiling")
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(th.cat([x0, x1], dim=-1), timesteps)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")

    summary(model, [(10, 2, 64, 128), ()], device="cuda", model_kwargs={'multistage': 2})

    print("start testing")
    y = model(th.cat([x0, x1], dim=-1), timesteps)
    print(y[0,:,0,0,0])

    from copy import deepcopy
    x0 = deepcopy(x0)
    x0[:,5, ...] = 1
    y = model(th.cat([x0, x1], dim=-1), timesteps)

    print(y[0,:,0,0,0])
    print("Pass")

    dit = DiTBlock(64, 64, 1, [10, 64, 1, 1], seq_len=10).to(device)

    seq = th.randn(2, 10, 64).to(device)
    timesteps = th.ones(20, 64).to(device)
    y = dit(seq, timesteps)
    print(y[0,:,0])
    seq[:,5,:] = 0
    y = dit(seq, timesteps)
    print(y[0,:,0])
    print("Pass")

    connect_shape = [10, 64, 32, 32]
    layers = [SwitchDims(connect_shape),
    DiTBlockCausalGate(64, 64, 1, connect_shape, seq_len=10),
    SwitchDimsT(connect_shape),
    ResBlock(
            64,
            64,
            0.0,
            out_channels=64,
            dims=2,
            )]
    block = TimestepEmbedSequential(*layers).to(device)

    seq = th.randn(2, 10, 64, 32, 32).to(device)
    merged_s = MergeDims((10, 64, 32, 32))(seq)
    timesteps = th.randn(20, 64).to(device)
    y = block(merged_s, timesteps)
    y = einops.rearrange(y, '(b t) c h w -> b t c h w', t = 10)
    print(y[0,:,0,0,0])
    seq[:,5,:,:,:] = 10
    merged_s = MergeDims((10, 64, 32, 32))(seq)
    y = block(merged_s, timesteps)
    y = einops.rearrange(y, '(b t) c h w -> b t c h w', t = 10)
    print(y[0,:,0,0,0])
    print("Pass")


if True:
    model = CausalUNetModel(
        image_size=64,
        seq_len=10,
        in_channels=2,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=[16],
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        concat_cond=False
    ).to(device)

    seq = th.zeros(2, 11, 2, 64, 64).to(device)
    x0 = seq[:, :-1, ...]
    x1 = seq[:, 1:, ...]
    timesteps = th.ones(2).to(device)

    y = model(th.cat([x0, x1], dim=-1), timesteps)
    print(y[0,:,0,0,0])

    from copy import deepcopy
    x0 = deepcopy(x0)
    x0[:,5, ...] = 1
    y = model(th.cat([x0, x1], dim=-1), timesteps)

    print(y[0,:,0,0,0])
    print("Pass")


    dit = DiTBlock(64, 64, 1, [10, 64, 1, 1], seq_len=10).to(device)

    seq = th.randn(2, 10, 64).to(device)
    timesteps = th.randn(20, 64).to(device)
    y = dit(seq, timesteps)
    print(y[0,:,0])
    seq[:,5,:] = 0
    y = dit(seq, timesteps)
    print(y[0,:,0])
    print("Pass")

    connect_shape = [10, 64, 32, 32]
    layers = [SwitchDims(connect_shape),
    DiTBlock(64, 64, 1, connect_shape, seq_len=10),
    SwitchDimsT(connect_shape),
    ResBlock(
            64,
            64,
            0.0,
            out_channels=64,
            dims=2,
            )]
    block = TimestepEmbedSequential(*layers).to(device)

    seq = th.randn(2, 10, 64, 32, 32).to(device)
    merged_s = MergeDims((10, 64, 32, 32))(seq)
    timesteps = th.randn(20, 64).to(device)
    y = block(merged_s, timesteps)
    y = einops.rearrange(y, '(b t) c h w -> b t c h w', t = 10)
    print(y[0,:,0,0,0])
    seq[:,5,:,:,:] = 0
    merged_s = MergeDims((10, 64, 32, 32))(seq)
    y = block(merged_s, timesteps)
    y = einops.rearrange(y, '(b t) c h w -> b t c h w', t = 10)
    print(y[0,:,0,0,0])
    print("Pass")

if False:

    input_shape = (1, 10, 2, 2, 2) # (Batch size, channels, sequence length, height, width)
    def test_causal_conv(conv_layers, input_tensor):
        output_tensor = conv_layers(input_tensor)
        output_tensor = einops.rearrange(output_tensor, '(b h w) t c -> b t c h w', b=input_shape[0], h=input_shape[3], w=input_shape[4])
        return output_tensor

    print("===========================causal convolution 1d ==================================")

    conv_layers = CausalConv1d(1, 1, 3, [1, 1, 1]).to(device)
    sequence = [ i for i in range(input_shape[1]*input_shape[3]*input_shape[4])]
    input_tensor = th.tensor(sequence, dtype=th.float32).reshape(input_shape).to(device)  # Batch size of 32, 16 channels, sequence length of 50, height of 10, width of 10
    input_tensor_reshaped = einops.rearrange(input_tensor, 'b t c h w -> (b h w) t c')


    print("the first input is ", input_tensor[0,:, 0, 0, 0].squeeze())
    output_tensor = test_causal_conv(conv_layers, input_tensor_reshaped)
    print("the first output is ", output_tensor[0,:, 0, 0, 0].squeeze())

    input_tensor_2 = input_tensor.clone()
    input_tensor_2[:, 5, :, 0, 0] = 0
    input_tensor_reshaped_2 = einops.rearrange(input_tensor_2, 'b t c h w -> (b h w) t c')
    print("the second input is ", input_tensor_2[0,:, 0, 0, 0].squeeze())
    output_tensor_2 = test_causal_conv(conv_layers, input_tensor_reshaped_2)
    print("the second output is ", output_tensor_2[0,:, 0, 0, 0].squeeze())

    print(output_tensor.shape)
    print("the difference between the two input is ", input_tensor_2[..., 0, 0].squeeze() - input_tensor[..., 0, 0].squeeze())
    print("the difference between the two output is ", output_tensor_2[..., 0, 0].squeeze() - output_tensor[..., 0, 0].squeeze())   

    
    print("===========================causal gated resnet==================================")
    causal_gate2 = CausalGatedResNet(1, 1, 3, [1, 1, 1]).to(device)
    def test_causal_gated_resnet(gate, input_tensor):
        output_tensor = gate(input_tensor)
        output_tensor = einops.rearrange(output_tensor, '(b h w) t c -> b t c h w', b=input_shape[0], h=input_shape[3], w=input_shape[4])
        
        return output_tensor
    output_tensor = test_causal_gated_resnet(causal_gate2, input_tensor_reshaped, )
    print("the output is ", output_tensor[..., 0, 0, 0].squeeze())
    print(output_tensor.shape)
    output_tensor_2 = test_causal_gated_resnet(causal_gate2, input_tensor_reshaped_2, )
    print("the output is ", output_tensor_2[..., 0, 0, 0].squeeze())
    print(output_tensor.shape)
    print("the difference between the two output is ", output_tensor_2[...,0, 0, 0].squeeze() - output_tensor[...,0, 0, 0].squeeze())