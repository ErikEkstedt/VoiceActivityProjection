import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from x_transformers.x_transformers import *


def get_device(layer: nn.Module):
    return next(layer.parameters()).device


class AttentionLayersCustom(nn.Module):
    def __init__(
        self,
        dim,
        depth=1,  # Only using 1 layer for now
        heads=8,
        causal=True,  # new default
        cross_attend=True,  # new default
        cross_attend_causal=True,  # new default
        only_cross=False,
        use_scalenorm=False,
        use_rmsnorm=False,
        use_simple_rmsnorm=False,
        alibi_pos_bias=False,
        alibi_num_heads=None,
        rel_pos_bias=False,
        rel_pos_num_buckets=32,
        rel_pos_max_distance=128,
        dynamic_pos_bias=False,
        dynamic_pos_bias_log_distance=False,
        dynamic_pos_bias_mlp_depth=2,
        dynamic_pos_bias_norm=False,
        rotary_pos_emb=False,
        rotary_emb_dim=None,
        rotary_xpos=False,
        rotary_interpolation_factor=1.0,
        rotary_xpos_scale_base=512,
        rotary_base_rescale_factor=1.0,
        custom_layers=None,
        sandwich_coef=None,
        par_ratio=None,
        weight_tie_layers=False,  # Albert - https://arxiv.org/abs/1909.11942
        layers_execute_order=None,  # generalizes weight tying, can do arbitrary layer execution orders
        residual_attn=False,
        cross_residual_attn=False,
        macaron=False,
        pre_norm=True,
        pre_norm_has_final_norm=True,
        gate_residual=False,
        scale_residual=False,
        scale_residual_constant=1.0,
        shift_tokens=0,
        sandwich_norm=False,
        resi_dual=False,
        resi_dual_scale=1.0,
        zero_init_branch_output=False,
        layer_dropout=0.0,
        cross_attn_tokens_dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        rotary_pos_emb = rotary_pos_emb or rotary_xpos

        ff_kwargs, kwargs = groupby_prefix_and_trim("ff_", kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim("attn_", kwargs)

        dim_head = attn_kwargs.get("dim_head", DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.has_pos_emb = rel_pos_bias or rotary_pos_emb

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)

        assert not (
            rotary_xpos and not causal
        ), "rotary xpos is not compatible with bidirectional attention"
        self.rotary_pos_emb = (
            RotaryEmbedding(
                rotary_emb_dim,
                use_xpos=rotary_xpos,
                scale_base=rotary_xpos_scale_base,
                interpolation_factor=rotary_interpolation_factor,
                base_rescale_factor=rotary_base_rescale_factor,
            )
            if rotary_pos_emb
            else None
        )

        assert not (
            alibi_pos_bias and rel_pos_bias
        ), "you can only choose Alibi positional bias or T5 relative positional bias, not both"
        assert (
            rel_pos_num_buckets <= rel_pos_max_distance
        ), "number of relative position buckets must be less than the relative position max distance"

        # relative positional bias

        flash_attn = attn_kwargs.get("flash", False)
        assert (
            int(rel_pos_bias) + int(dynamic_pos_bias) + int(alibi_pos_bias)
        ) <= 1, "you can only choose up to one of t5, alibi, or dynamic positional bias"

        self.rel_pos = None
        if rel_pos_bias:
            assert (
                not flash_attn
            ), "flash attention not compatible with t5 relative positional bias"
            self.rel_pos = RelativePositionBias(
                scale=dim_head**0.5,
                causal=causal,
                heads=heads,
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
            )
        elif dynamic_pos_bias:
            assert (
                not flash_attn
            ), "flash attention not compatible with dynamic positional bias"
            self.rel_pos = DynamicPositionBias(
                dim=dim // 4,
                heads=heads,
                log_distance=dynamic_pos_bias_log_distance,
                depth=dynamic_pos_bias_mlp_depth,
                norm=dynamic_pos_bias_norm,
            )
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert (
                alibi_num_heads <= heads
            ), "number of ALiBi heads must be less than the total number of heads"
            self.rel_pos = AlibiPositionalBias(heads=alibi_num_heads, total_heads=heads)

        assert (
            int(sandwich_norm) + int(resi_dual)
        ) <= 1, "either sandwich norm or resiDual is selected, but not both"
        assert not (
            not pre_norm and sandwich_norm
        ), "sandwich norm cannot be used when not using prenorm"

        if resi_dual:
            pre_norm = False

        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.resi_dual = resi_dual
        assert (
            0 < resi_dual_scale <= 1.0
        ), "resiDual prenorm residual must be scaled by a factor greater than 0 and less than or equal to 1."
        self.resi_dual_scale = resi_dual_scale

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        assert not (
            flash_attn and (residual_attn or cross_residual_attn)
        ), "flash attention is not compatible with residual attention"

        self.cross_attend = cross_attend

        assert (
            int(use_scalenorm) + int(use_rmsnorm) + int(use_simple_rmsnorm)
        ) <= 1, "you can only use either scalenorm, rmsnorm, or simple rmsnorm"

        if use_scalenorm:
            norm_class = ScaleNorm
        elif use_rmsnorm:
            norm_class = RMSNorm
        elif use_simple_rmsnorm:
            norm_class = SimpleRMSNorm
        else:
            norm_class = nn.LayerNorm

        norm_fn = partial(norm_class, dim)

        if cross_attend and not only_cross and not cross_attend_causal:
            default_block = ("a", "c", "f")
        elif cross_attend and not only_cross and cross_attend_causal:
            default_block = ("a", "ac", "f")
        elif cross_attend and only_cross:
            default_block = ("c", "f")
        else:
            default_block = ("a", "f")

        if macaron:
            default_block = ("f",) + default_block

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, "zero_init_output": True}
            ff_kwargs = {**ff_kwargs, "zero_init_output": True}

        # setup weight tying, which is a special case of `layer_execute_order`

        assert not (
            weight_tie_layers
            and any([*map(exists, (custom_layers, par_ratio, sandwich_coef))])
        )

        if weight_tie_layers:
            assert not exists(layers_execute_order)
            layers_execute_order = tuple(range(len(default_block))) * depth
            depth = 1

        # calculate layer block order

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, "par ratio out of range"
            default_block = tuple(filter(not_equals("f"), default_block))
            par_attn = par_depth // par_ratio
            depth_cut = (
                par_depth * 2 // 3
            )  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert (
                len(default_block) <= par_width
            ), "default block is too large for par_ratio"
            par_block = default_block + ("f",) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ("f",) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert (
                sandwich_coef > 0 and sandwich_coef <= depth
            ), "sandwich coefficient should be less than the depth"
            layer_types = (
                ("a",) * sandwich_coef
                + default_block * (depth - sandwich_coef)
                + ("f",) * sandwich_coef
            )
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.layers_execute_order = default(
            layers_execute_order, tuple(range(len(layer_types)))
        )

        assert all([i < len(self.layer_types) for i in self.layers_execute_order])

        self.num_attn_layers = len(list(filter(equals("a"), layer_types)))

        # stochastic depth

        self.layer_dropouts = cast_tuple(layer_dropout, len(layer_types))

        # structured dropout for cross attending

        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        # whether it has post norm

        self.final_norm = norm_fn() if pre_norm or resi_dual else nn.Identity()

        # iterate and construct layers

        for ind, (layer_type, layer_shift_tokens) in enumerate(
            zip(self.layer_types, shift_tokens)
        ):
            is_last_layer = ind == (len(self.layer_types) - 1)

            if layer_type == "a":
                layer = Attention(dim, heads=heads, causal=causal, **attn_kwargs)
            elif layer_type == "ac":
                layer = Attention(dim, heads=heads, causal=causal, **attn_kwargs)
            elif layer_type == "c":
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == "f":
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f"invalid layer type {layer_type}")

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            residual_fn = GRUGating if gate_residual else Residual
            residual = residual_fn(
                dim,
                scale_residual=scale_residual,
                scale_residual_constant=scale_residual_constant,
            )

            pre_branch_norm = norm_fn() if pre_norm else None
            post_branch_norm = norm_fn() if sandwich_norm else None
            post_main_norm = norm_fn() if not pre_norm else None

            norms = nn.ModuleList([pre_branch_norm, post_branch_norm, post_main_norm])

            self.layers.append(nn.ModuleList([norms, layer, residual]))

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        attn_mask=None,
        self_attn_kv_mask=None,
        mems=None,
        seq_start_pos: Optional[Tensor] = None,
        cache: Optional[LayerIntermediates] = None,
        cache_age=1,
        return_hiddens=False,
    ):
        assert not (
            self.cross_attend ^ exists(context)
        ), "context must be passed in if cross_attend is set to True"

        # initialize accums

        hiddens = []
        layer_hiddens = []
        intermediates = []

        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        # handle left padded sequences

        if exists(seq_start_pos):
            seq_arange = torch.arange(x.shape[-2], device=x.device, dtype=torch.long)
            left_pad_mask = seq_arange >= seq_start_pos[..., None]

            if exists(self_attn_kv_mask):
                self_attn_kv_mask = self_attn_kv_mask & left_pad_mask
            else:
                self_attn_kv_mask = left_pad_mask

        # rotary positions

        rotary_pos_emb = None

        if exists(self.rotary_pos_emb):
            max_rotary_emb_length = max(
                list(map(lambda m: (m.shape[1] if exists(m) else 0) + x.shape[1], mems))
            )
            rotary_pos_emb = self.rotary_pos_emb(max_rotary_emb_length)

        # assume cached key / values

        attn_cache = []

        if exists(cache):
            assert (
                not self.training
                and self.causal
                and not any([*map(exists, (mask, attn_mask))])
            )

            if cache_age > 0:
                x = x[:, -cache_age:]  # for spec decoding, may be greater than 1

            attn_cache = cache.attn_intermediates

        iter_attn_cache = iter(attn_cache)

        # outer residual - for resiDual paper

        outer_residual = x * self.resi_dual_scale

        # get layers to be executed

        layer_variables = (self.layer_types, self.layers, self.layer_dropouts)

        layer_variables = tuple(
            tuple(layer_variable[i] for i in self.layers_execute_order)
            for layer_variable in layer_variables
        )

        # go through the attention and feedforward layers

        for ind, (layer_type, (norm, block, residual_fn), layer_dropout) in enumerate(
            zip(*layer_variables)
        ):
            is_last = ind == (len(self.layers) - 1)

            if self.training and layer_dropout > 0.0 and random() < layer_dropout:
                continue

            if layer_type == "a":
                if return_hiddens:
                    hiddens.append(x)
                layer_mem = mems.pop(0) if mems else None

            if layer_type == "ac":
                if self.training and self.cross_attn_tokens_dropout > 0.0:
                    context, context_mask = dropout_seq(
                        context, context_mask, self.cross_attn_tokens_dropout
                    )

            if layer_type == "c":
                if self.training and self.cross_attn_tokens_dropout > 0.0:
                    context, context_mask = dropout_seq(
                        context, context_mask, self.cross_attn_tokens_dropout
                    )

            inner_residual = x

            if return_hiddens:
                layer_hiddens.append(x)

            pre_norm, post_branch_norm, post_main_norm = norm

            if exists(pre_norm):
                x = pre_norm(x)

            if layer_type == "a":
                out, inter = block(
                    x,
                    mask=mask,
                    context_mask=self_attn_kv_mask,
                    attn_mask=attn_mask,
                    rel_pos=self.rel_pos,
                    rotary_pos_emb=rotary_pos_emb,
                    prev_attn=prev_attn,
                    cache=next(iter_attn_cache, None),
                    mem=layer_mem,
                    return_intermediates=True,
                )
            elif layer_type == "ac":
                out, inter = block(
                    x,
                    context=context,
                    mask=mask,
                    context_mask=context_mask,
                    prev_attn=prev_cross_attn,
                    cache=next(iter_attn_cache, None),
                    return_intermediates=True,
                )
            elif layer_type == "c":
                out, inter = block(
                    x,
                    context=context,
                    mask=mask,
                    context_mask=context_mask,
                    prev_attn=prev_cross_attn,
                    cache=next(iter_attn_cache, None),
                    return_intermediates=True,
                )
            elif layer_type == "f":
                out = block(x)

            if self.resi_dual:
                outer_residual = outer_residual + out * self.resi_dual_scale

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            x = residual_fn(out, inner_residual)

            if layer_type in ("a", "c", "ac") and return_hiddens:
                intermediates.append(inter)

            if layer_type == "a" and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == "c" and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn
            elif layer_type == "ac" and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)

        if return_hiddens:
            layer_hiddens.append(x)

        if self.resi_dual:
            x = x + self.final_norm(outer_residual)
        else:
            x = self.final_norm(x)

        if not return_hiddens:
            return x

        intermediates = LayerIntermediates(
            hiddens=hiddens,
            attn_intermediates=intermediates,
            layer_hiddens=layer_hiddens,
        )

        return x, intermediates


class AttentionStereoLayer(AttentionLayersCustom):
    def forward(
        self,
        x1,
        x2,
        mask=None,
        context_mask=None,
        attn_mask=None,
        self_attn_kv_mask=None,
        mems=None,
        seq_start_pos: Optional[Tensor] = None,
        cache: Optional[LayerIntermediates] = None,
        cache_age=1,
        return_hiddens=False,
    ):
        z1 = super().forward(x=x1, context=x2)
        z2 = super().forward(x=x2, context=x1)
        return z1, z2


class VapStereoTower(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        num_self_attn_layers: int = 1,
        num_cross_attn_layers: int = 3,
        rotary_embeddings: bool = True,
        flash: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_self_attn_layers = num_self_attn_layers
        self.num_cross_attn_layers = num_cross_attn_layers
        self.rotary_embeddings = rotary_embeddings

        self.self_attn_layers = (
            AttentionLayers(
                dim=dim,
                depth=num_self_attn_layers,
                heads=num_heads,
                causal=True,
                rotary_xpos=True,
                ff_glu=True,
                attn_flash=flash,
            )
            if num_self_attn_layers > 0
            else nn.Identity()
        )

        if num_cross_attn_layers > 0:
            layer_fn = partial(
                AttentionStereoLayer,
                dim=dim,
                depth=1,
                heads=num_heads,
                rotary_xpos=rotary_embeddings,
                alibi_pos_bias=not rotary_embeddings,
                ff_glu=True,
                attn_flash=flash,
            )
            self.cross_layers = nn.ModuleList(
                [layer_fn() for _ in range(num_cross_attn_layers)]
            )
        else:
            self.cross_layers = nn.Identity()

    def forward(self, x1, x2):
        x1 = self.self_attn_layers(x1)
        x2 = self.self_attn_layers(x2)
        for layer in self.cross_layers:
            x1, x2 = layer(x1, x2)
        return x1, x2


class CausalTest:
    @staticmethod
    def self_attn_causality(layer, t=None, d=None, cuda=False, verbose=False):
        """
        Check causality by gradient information in the input.

        We scale the loss and the gradients with large numbers to always get some activation
        """
        dim = getattr(layer, "dim", d if d else 256)
        t = t if t else 20
        t_focus = t // 2
        large_val = 1e6

        if verbose:
            print("Dim: ", dim)
            print("T: ", t)
            print("T focus: ", t_focus)

        # Input w/ gradients
        x = torch.rand((1, t, dim), requires_grad=True, device=get_device(layer))

        # 2. Forward
        y = layer(x)

        # 3. Gradient calculation
        focus_loss = large_val * y[:, t_focus].pow(2).sum()
        focus_loss.backward()

        # Gradient result
        g = ((large_val * x.grad.abs()) > 0).float()
        pre_grad = g[:, :t_focus].sum(0).sum(0)
        post_grad = g[:, t_focus + 1 :].sum(0).sum(0)
        is_causal = (post_grad.sum() == 0).item()

        if verbose:
            CausalTest.print_is_causal(is_causal, pre_grad, post_grad, focus_loss)

        return is_causal, pre_grad, post_grad

    @staticmethod
    def cross_attn_causality(layer, t=None, d=None, verbose=False):
        """
        Check causality by gradient information in the input.

        We scale the loss and the gradients with large numbers to always get some activation
        """
        dim = getattr(layer, "dim", d if d else 256)
        t = t if t else 20
        t_focus = t // 2

        large_val = 1e6

        if verbose:
            print("Dim: ", dim)
            print("T: ", t)
            print("T focus: ", t_focus)

        # Input w/ gradients
        x = torch.rand((1, t, dim), requires_grad=True, device=get_device(layer))
        x_context = torch.randn((1, t, dim), requires_grad=True)

        # 2. Forward
        y = layer(x, context=x_context)

        # 3. Gradient calculation
        focus_loss = large_val * y[:, t_focus].pow(2).sum()
        focus_loss.backward()

        # Cross Attention Gradient result
        g = ((large_val * x_context.grad.abs()) > 0).float()
        pre_grad = g[:, :t_focus].sum(0).sum(0)
        post_grad = g[:, t_focus + 1 :].sum(0).sum(0)
        is_causal = (post_grad.sum() == 0).item()
        if verbose:
            print("CONTEXT")
            CausalTest.print_is_causal(is_causal, pre_grad, post_grad, focus_loss)

        # Self Attention Gradient result
        g = ((large_val * x.grad.abs()) > 0).float()
        pre_grad = g[:, :t_focus].sum(0).sum(0)
        post_grad = g[:, t_focus + 1 :].sum(0).sum(0)
        is_causal = (post_grad.sum() == 0).item()
        if verbose:
            print("X")
            CausalTest.print_is_causal(is_causal, pre_grad, post_grad, focus_loss)

        return is_causal, pre_grad, post_grad

    @staticmethod
    def stereo_attn_causality(layer, t=None, d=None, grad_channel=1, verbose=False):
        """
        Check causality by gradient information in the input.

        We scale the loss and the gradients with large numbers to always get some activation
        """
        dim = getattr(layer, "dim", d if d else 256)
        t = t if t else 20
        t_focus = t // 2
        large_val = 1e6

        if verbose:
            print("Dim: ", dim)
            print("T: ", t)
            print("T focus: ", t_focus)

        # Input w/ gradients
        x1 = torch.randn((1, t, dim), requires_grad=True, device=get_device(layer))
        x2 = torch.randn((1, t, dim), requires_grad=True, device=get_device(layer))

        # 2. Forward
        y1, y2 = layer(x1, x2)

        # 3. Gradient calculation
        if grad_channel == 1:
            focus_loss = large_val * y1[:, t_focus].pow(2).sum()
        else:
            focus_loss = large_val * y2[:, t_focus].pow(2).sum()
        focus_loss.backward()

        # Channel 1
        g = ((large_val * x1.grad.abs()) > 0).float()
        pre_grad = g[:, :t_focus].sum(0).sum(0)
        post_grad = g[:, t_focus + 1 :].sum(0).sum(0)
        is_causal = (post_grad.sum() == 0).item()
        if verbose:
            print("X1")
            CausalTest.print_is_causal(is_causal, pre_grad, post_grad, focus_loss)

        # Channel 2
        g = ((large_val * x2.grad.abs()) > 0).float()
        pre_grad = g[:, :t_focus].sum(0).sum(0)
        post_grad = g[:, t_focus + 1 :].sum(0).sum(0)
        is_causal = (post_grad.sum() == 0).item()
        if verbose:
            print("X2")
            CausalTest.print_is_causal(is_causal, pre_grad, post_grad, focus_loss)

        return is_causal, pre_grad, post_grad

    @staticmethod
    def print_is_causal(is_causal, pre_grad, post_grad, focus_loss):
        from vap.utils.colors import Colors

        if is_causal:
            c = Colors.GREEN
        else:
            c = Colors.RED

        print(f"{c}Is causal: ", is_causal)
        print("Pre-grad: ", pre_grad.sum())
        print("Post-grad: ", post_grad.sum())
        print(f"Focus loss: {focus_loss.item():.4f}")
        print(Colors.RESET)


if __name__ == "__main__":
    """
    https://github.com/lucidrains/x-transformers
    """

    from x_transformers.x_transformers import AttentionLayers

    def causal_mask(n):
        mask = torch.ones((n, n), dtype=torch.bool)
        mask = torch.tril(mask, diagonal=0)
        return mask

    dim = 256
    T = 100
    B = 1
    x = torch.randn((B, T, dim))

    ###########################################
    # VAP STEREO TOWER
    ###########################################

    vap_tower = VapStereoTower().to("cuda")
    print("rotary_embeddings: ", vap_tower.rotary_embeddings)
    print("num_self_attn_layers: ", vap_tower.num_self_attn_layers)
    print("num_cross_attn_layers: ", vap_tower.num_cross_attn_layers)
    _ = CausalTest.stereo_attn_causality(vap_tower, grad_channel=1, verbose=True)
    _ = CausalTest.stereo_attn_causality(vap_tower, grad_channel=2, verbose=True)

    ###########################################
    # Self Attention
    ###########################################
    rotary = True
    layer = AttentionLayers(
        dim=dim,
        depth=1,
        heads=4,
        causal=True,
        rotary_xpos=rotary,
        alibi_pos_bias=not rotary,
        attn_flash=True,
    ).cuda()
    is_causal, *_ = CausalTest.self_attn_causality(layer, cuda=True, verbose=True)

    ###########################################
    # Cross Attention
    ###########################################
    rotary = True
    cross_layer = AttentionLayersCustom(
        dim=dim,
        heads=4,
        causal=True,
        rotary_xpos=rotary,
        alibi_pos_bias=not rotary,
        cross_attend_causal=True,
        attn_flash=True,
    )
    print(cross_layer.layer_types)
    _ = CausalTest.cross_attn_causality(cross_layer, verbose=True)

    ###########################################
    # Stereo Attention
    ###########################################
    stereo_layer = AttentionStereoLayer(
        dim=dim, depth=1, heads=4, causal=True, rotary_xpos=True, flash=True
    )
    print(stereo_layer.layer_types)
    _ = CausalTest.stereo_attn_causality(stereo_layer, grad_channel=1, verbose=True)
    _ = CausalTest.stereo_attn_causality(stereo_layer, grad_channel=2, verbose=True)

    x = torch.randn((B, T, dim), requires_grad=True)
    x_context = torch.randn((B, T, dim), requires_grad=True)

    n_heads = 8
    rotary_xpos = True
    rotary_xpos_scale_base = 512  # default
    rotary_interpolation_factor = 1  # default

    # norms
    # residual
    rotary_pos_emb = RotaryEmbedding(
        rotary_emb_dim,
        use_xpos=rotary_xpos,
        scale_base=rotary_xpos_scale_base,
        interpolation_factor=rotary_interpolation_factor,
        base_rescale_factor=rotary_base_rescale_factor,
    )
    self_attn = Attention(dim, heads=n_heads, causal=True)
    cross_attn = Attention(dim, heads=n_heads, causal=True)
    ffn = FeedForward(dim, mult=4, glu=True)

    z = self_attn(x)
    z2 = cross_attn(z, context=x_context)
    z3 = ffn(z2)
    z3[:, T // 2].sum().backward()

    g = ((x.grad.abs()) > 0).float()
    pre_grad = g[:, : T // 2].sum(0).sum(0)
    post_grad = g[:, T // 2 + 1 :].sum(0).sum(0)
