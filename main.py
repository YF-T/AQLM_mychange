import os
import time
from argparse import Namespace
from itertools import chain
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
import warnings

import torch
import torch.nn as nn
from tqdm import trange
from tqdm.auto import trange
from transformers import PreTrainedModel

from aq_engine import AQEngine
from src.aq import QuantizedLinear
from src.datautils import get_loaders
from src.finetune import finetune_groupwise
from src.modelutils import (
    FALCON_TYPES,
    find_sublayers,
    get_layers,
    get_lm_logits,
    get_model,
    get_model_head_with_norm,
    get_sequential_groups,
    save_not_quantized_weights,
)
from src.utils import using_tf32

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False


def quantize_model(model: PreTrainedModel, args: Namespace):
    """main entry point to functions for model quantization"""
    tick = time.time()
    print("Loading data ...")
    
    entropy_thresholds = {}
    if args.entropy_percentile_threshold is not None:
        entropy_thresholds['percentile'] = args.entropy_percentile_threshold
    if args.entropy_fixed_threshold is not None:
        entropy_thresholds['fixed'] = args.entropy_fixed_threshold

    data = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model_path=args.model_path,
        seqlen=args.model_seqlen,
        use_fast_tokenizer=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
        use_entropy=args.use_entropy,
        entropy_thresholds=entropy_thresholds,
        weight_from_entropy=args.weight_from_entropy,
        weight_hyperparam_C=args.weight_hyperparam_C,
    )

    if args.use_entropy:
        # Data is a list of tuples (input_ids, mask, weight)
        input_ids_list = [item[0] for item in data]
        masks_list = [item[1] for item in data]
        weights_list = [item[2] for item in data]
    else:
        # Data is a list of tensors (input_ids)
        input_ids_list = data
        masks_list = None
        weights_list = None

    if args.val_size > 0:
        all_indices = torch.randperm(len(input_ids_list))
        train_indices, val_indices = all_indices[args.val_size :], all_indices[: args.val_size]
        
        train_data = [input_ids_list[i] for i in train_indices]
        val_data = [input_ids_list[i] for i in val_indices]
        
        train_masks = [masks_list[i] for i in train_indices] if masks_list else None
        val_masks = [masks_list[i] for i in val_indices] if masks_list else None
        
        train_weights = [weights_list[i] for i in train_indices] if weights_list else None
        val_weights = [weights_list[i] for i in val_indices] if weights_list else None

    else:
        train_data = input_ids_list
        val_data = None
        train_masks, val_masks = masks_list, None
        train_weights, val_weights = weights_list, None

    results = quantize_aq(model, train_data, val_data, args, train_masks, val_masks, train_weights, val_weights)
    print(f"quantization time: {time.time() - tick:.1f}")
    return results


@torch.no_grad()
def get_inps(
    model: PreTrainedModel,
    data: Sequence,
    model_seqlen: int,
    devices: Sequence[torch.device],
    offload_activations: bool,
) -> Tuple[Sequence[torch.Tensor], Dict]:
    """
    mocks model launch to collect inputs to the first model layer
    :returns: a list of torch tensors with activations for each device in args.devices.
    Each tensor has shape [nsample_per_device, seq_len, hid_size]
    """
    print("catching layer inputs from data", flush=True)
    layers = get_layers(model)
    device = devices[0] if not offload_activations else torch.device("cpu")

    if isinstance(data[0], torch.Tensor) and data[0].ndim == 2:
        pass # Data is already a list of tensors
    elif isinstance(data, torch.Tensor) and data.ndim == 2: # given a single long tensor, split it into sequences
        num_sequences, num_tokens_dropped = data.numel() // model_seqlen, data.numel() % model_seqlen
        data = [data[:, i * model_seqlen : (i + 1) * model_seqlen].to(device) for i in range(num_sequences)]
        print(f"Got {len(data)} sequences of {model_seqlen} tokens, dropped last {num_tokens_dropped} tokens")
        del num_sequences, num_tokens_dropped

    assert all(sequence.shape[1] == model_seqlen for sequence in data)

    emb = model.get_input_embeddings()
    emb_device = emb.weight.device
    if emb_device.type != "cuda":
        emb = emb.to(device)
        # opt has other embeddings
        if model.config.model_type == "opt":
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
            if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(device)
    device = emb.weight.device  # now default device is the one where the embeddings are.
    layer_device = next(layers[0].parameters()).device
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    nsamples_per_device = (len(data) - 1) // len(devices) + 1
    inps = [
        torch.zeros(
            (min(nsamples_per_device, len(data) - i * nsamples_per_device), model_seqlen, model.config.hidden_size),
            dtype=dtype,
            device=devices[i] if not offload_activations else "cpu",
            pin_memory=offload_activations,
        )
        for i in range(len(devices))
    ]
    forward_arg_names = ["attention_mask", "position_ids"]
    if model.config.model_type.lower() in FALCON_TYPES:
        forward_arg_names.append("alibi")

    cache = {"i": 0, "alibi": None}

    class CatcherExit(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"] // nsamples_per_device][cache["i"] % nsamples_per_device] = inp
            cache["i"] += 1
            for forward_arg_name in forward_arg_names:
                cache[forward_arg_name] = kwargs.get(forward_arg_name)
            raise CatcherExit()

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch_inps in data:
        try:
            if isinstance(batch_inps, (list, tuple)):
                batch_inps, *_ = batch_inps
            batch_inps = batch_inps.to(device)
            # call model.forward to trigger the Catcher
            model(batch_inps, attention_mask=torch.ones_like(batch_inps))
        except CatcherExit:
            pass  # exit after catcher finished without running the rest of the model layers

    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].to(layer_device)
    model.get_input_embeddings().to(emb_device)
    if model.config.model_type == "opt":
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(emb_device)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(emb_device)
    torch.cuda.empty_cache()

    forward_args = {k: cache[k] for k in forward_arg_names}
    assert cache["i"] == sum(len(inp_tensor) for inp_tensor in inps), "internal error: found empty rows in inps"
    return inps, forward_args


@torch.no_grad()
def quantize_aq(
    model: PreTrainedModel, 
    data: Sequence, 
    val_data: Optional[Sequence], 
    args: Namespace,
    masks: Optional[Sequence[torch.Tensor]] = None,
    val_masks: Optional[Sequence[torch.Tensor]] = None,
    weights: Optional[Sequence[torch.Tensor]] = None,
    val_weights: Optional[Sequence[torch.Tensor]] = None,
):
    assert not torch.backends.cuda.matmul.allow_tf32
    print("\nStarting AQ quantization ...")
    inps, forward_args = get_inps(model, data, args.model_seqlen, args.devices, args.offload_activations)
    outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in inps]

    if val_data:
        run_validation = True
        val_inps, _ = get_inps(model, val_data, args.model_seqlen, args.devices, args.offload_activations)
        val_outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in val_inps]
    else:
        run_validation = False
        val_inps, val_outs = None, None

    use_cache = model.config.use_cache
    num_codebooks = args.num_codebooks
    model.config.use_cache = False

    quantizers = {}
    overall_bits = 0
    number_of_quantized_params = 0
    layers = get_layers(model)

    for layer_index in range(len(layers)):
        print(f"\n---------------- Layer {layer_index} of {len(layers)} ----------------")
        stats_payload = {}
        start_time = time.time()

        layer_device_original = next(layers[layer_index].parameters()).device
        layer_dtype_original = next(layers[layer_index].parameters()).dtype
        print(f"layer_device_original={layer_device_original}")
        layer = layers[layer_index].to(args.devices[0])
        for k, v in forward_args.items():
            forward_args[k] = v.to(args.devices[0]) if isinstance(v, torch.Tensor) else v

        if args.true_sequential:
            sequential = get_sequential_groups(model)
        else:
            sequential = [list(find_sublayers(layer).keys())]

        loaded_layer = False
        if args.resume:
            assert args.save is not None, "using --resume requires a --save path to resume from"
            layer_save_path = os.path.join(args.save, f"{layer_index}.pth")
            if os.path.exists(layer_save_path):
                print(f"Loading layer {layer_index} from {layer_save_path}")
                layer = torch.load(layer_save_path, map_location=args.devices[0])
                loaded_layer = True

        if run_validation and not loaded_layer:
            if len(args.devices) == 1:
                assert len(val_inps) == len(val_outs) == 1
                update_outs(layer, val_inps[0], val_outs[0], compute_mse=not args.skip_out_loss, **forward_args)
            else:
                update_outs_parallel(
                    args.devices, layer, val_inps, val_outs, compute_mse=not args.skip_out_loss, **forward_args
                )

        for names in sequential:
            if loaded_layer:
                print("Skipping quantization: loaded a previously quantized layer")
                break
            
            if len(args.devices) == 1:
                assert len(inps) == len(outs) == 1
                aq_handlers = init_aq_engines(
                    layer,
                    [
                        name
                        for name in names
                        if ((".gate" not in name.lower()) or ("mixtral" not in model.config.model_type.lower()))
                    ],
                    inps[0],
                    outs[0],
                    masks=masks,
                    weights=weights,
                    **forward_args,
                )
            else:
                aq_handlers = init_aq_engines_parallel(
                    args.devices,
                    layer,
                    [
                        name
                        for name in names
                        if ((".gate" not in name.lower()) or ("mixtral" not in model.config.model_type.lower()))
                    ],
                    inps,
                    outs,
                    masks=masks,
                    weights=weights,
                    **forward_args,
                )

            for sublayer_name in aq_handlers.keys():
                print(f"Quantizing module {sublayer_name} of layer {layer_index}")
                if "mixtral" in model.config.model_type.lower() and args.mix_compression:
                    assert "mixtral" in model.config.model_type.lower()
                    if "self_attn" in sublayer_name.lower():
                        args.num_codebooks = 2 * num_codebooks
                    else:
                        args.num_codebooks = num_codebooks
                    print(sublayer_name.lower(), " mixtral num codebooks", args.num_codebooks)
                quantized_weight = aq_handlers[sublayer_name].quantize(args=args, verbose=True)

                with torch.no_grad():
                    assert aq_handlers[sublayer_name].layer.weight in set(
                        layer.parameters()
                    )

                    new_linear = QuantizedLinear(quantized_weight, aq_handlers[sublayer_name].layer.bias)
                    if args.use_checkpointing:
                        new_linear.use_checkpoint = True
                        print("ENABLED CHECKPOINTING FOR", sublayer_name)
                    found_original = False
                    for submodule in layer.modules():
                        for child_name, child_module in submodule.named_children():
                            if child_module is aq_handlers[sublayer_name].layer:
                                setattr(submodule, child_name, new_linear)
                                found_original = True
                    assert found_original, f"could not find {sublayer_name}"

                weight_avg_bits = quantized_weight.estimate_nbits_per_parameter()
                overall_bits += int(weight_avg_bits * torch.numel(aq_handlers[sublayer_name].layer.weight.data))
                number_of_quantized_params += torch.numel(aq_handlers[sublayer_name].layer.weight.data)
                print("curent_avg_bits", overall_bits / number_of_quantized_params)
                quantizers["model.layers.%d.%s" % (layer_index, sublayer_name)] = ()

            del aq_handlers
            assert not loaded_layer

            print("PREPARING TO FINETUNE")
            print(layer)
            layer = layer.to(dtype=torch.float32)
            with using_tf32(enabled=True):
                layer = finetune_groupwise(
                    layer=layer,
                    train_inps=inps,
                    train_outs=outs,
                    args=args,
                    valid_inps=val_inps,
                    valid_outs=val_outs,
                    **forward_args,
                )
            layer = layer.to(dtype=layer_dtype_original)
            print("FINISHED FINETUNING")

        if args.save and not loaded_layer:
            os.makedirs(args.save, exist_ok=True)
            layer_save_path = os.path.join(args.save, f"{layer_index}.pth")
            print(f"Saving layer {layer_index}... to {layer_save_path}")
            torch.save(layer, layer_save_path)
            if args.on_save:
                exec(args.on_save)

        should_compute_mse = not (args.skip_out_loss or loaded_layer)
        if len(args.devices) == 1:
            assert len(inps) == len(outs) == 1
            out_losses = update_outs(layer, inps[0], outs[0], compute_mse=should_compute_mse, **forward_args)
        else:
            out_losses = update_outs_parallel(
                args.devices, layer, inps, outs, compute_mse=should_compute_mse, **forward_args
            )
        stats_payload["out_loss"] = torch.mean(torch.Tensor(out_losses)).item()

        if run_validation:
            if len(args.devices) == 1:
                assert len(val_inps) == len(val_outs) == 1
                out_val_losses = update_outs(
                    layer, val_inps[0], val_outs[0], compute_mse=should_compute_mse, **forward_args
                )
            else:
                out_val_losses = update_outs_parallel(
                    args.devices, layer, val_inps, val_outs, compute_mse=should_compute_mse, **forward_args
                )
            stats_payload["out_val_loss"] = torch.mean(torch.Tensor(out_val_losses)).item()

        layers[layer_index] = layer.to(layer_device_original)
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        if run_validation:
            val_inps, val_outs = val_outs, val_inps

        stats_payload["layer_time"] = time.time() - start_time
        stats_payload["Step"] = layer_index
        if args.wandb:
            wandb.log(stats_payload, step=layer_index)
        if not loaded_layer:
            print(stats_payload)

    print("=====================\nFinal stats:")
    if args.save:
        torch.save(vars(args), os.path.join(args.save, "args.pt"))
        save_not_quantized_weights(model, args.save)
        if args.on_save:
            exec(args.on_save)

    if args.wandb:
        wandb.log({"max_cuda_mem_quantize": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
        if number_of_quantized_params > 0:
            wandb.log({"Avg_bits": overall_bits / number_of_quantized_params})
    model.config.use_cache = use_cache
    print(f"quantize: max_memory_allocated={torch.cuda.max_memory_allocated():,}")
    return quantizers


@torch.no_grad()
def perplexity_eval(model: PreTrainedModel, testenc: torch.LongTensor, args: Namespace) -> float:
    print(f"\nEvaluating perplexity for {args.dataset_name} dataset ...")

    nsamples = testenc.numel() // args.model_seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    inps, forward_args = get_inps(model, testenc, args.model_seqlen, args.devices, args.offload_activations)
    outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in inps]
    device = args.devices[0]
    for k, v in forward_args.items():
        forward_args[k] = v.to(device) if isinstance(v, torch.Tensor) else v

    layers = get_layers(model)
    for i in trange(len(layers), desc="processing eval data by layer"):
        layer = layers[i].to(device)
        if len(args.devices) == 1:
            assert len(inps) == len(outs) == 1
            update_outs(layer, inps[0], outs[0], compute_mse=False, **forward_args)
        else:
            update_outs_parallel(args.devices, layer, inps, outs, compute_mse=False, **forward_args)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    get_model_head_with_norm(model).to(device)
    testenc = testenc.to(device)
    nsamples_per_device = len(inps[0])
    assert len(set(map(len, inps[:-1]))) <= 1 and len(inps[-1]) <= len(inps[0])

    nlls = []
    for i in range(nsamples):
        inp = inps[i // nsamples_per_device][i % nsamples_per_device].to(args.devices[0], non_blocking=True)
        lm_logits = get_lm_logits(inp.to(device), model)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * args.model_seqlen) : ((i + 1) * args.model_seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * args.model_seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * args.model_seqlen)).item()
    print(f"\n{args.dataset_name} perplexity = {ppl:.4f}\n")

    get_model_head_with_norm(model).to(torch.device("cpu"))

    if args.wandb:
        wandb.log({args.dataset_name: ppl})

    model.config.use_cache = use_cache
    return ppl


@torch.no_grad()
def init_aq_engines(
    layer: nn.Module,
    names: Sequence[str],
    inps_tensor: torch.Tensor,
    outs_tensor: torch.Tensor,
    masks: Optional[Sequence[torch.Tensor]] = None,
    weights: Optional[Sequence[torch.Tensor]] = None,
    **forward_args: Dict[str, Any],
) -> Dict[str, AQEngine]:
    """
    Create a dictionary of AQUtil instances for each quantized layer;
    Run forward pass on each sample in inps_tensor; write output activations to outs_tensor (in-plance)
    Accumulate XTX to each one of aq_handlers
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    all_sublayers = find_sublayers(layer)
    subset = {name: all_sublayers[name] for name in names}
    assert len(subset) > 0
    aq_handlers = {}
    for sublayer_name in subset:
        aq_handlers[sublayer_name] = AQEngine(subset[sublayer_name])

    wrapped_layer_to_hander = {aq_handler.layer: aq_handler for aq_handler in aq_handlers.values()}
    for module in list(layer.modules()):
        for child_name, child in list(module.named_children()):
            if child in wrapped_layer_to_hander:
                setattr(module, child_name, _LayerWrapperThatAccumulatesXTX(child, wrapped_layer_to_hander[child]))

    for j in trange(len(inps_tensor), desc="calc outs before quantization", leave=False):
        forward_args_for_sample = forward_args.copy()
        if masks:
            forward_args_for_sample['mask'] = masks[j]
        if weights:
            forward_args_for_sample['weight'] = weights[j]

        outs_tensor[j].copy_(
            layer(inps_tensor[j].to(device).unsqueeze(0), **forward_args_for_sample)[0].view_as(outs_tensor[j]), non_blocking=True
        )

    for module in list(layer.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, _LayerWrapperThatAccumulatesXTX):
                setattr(module, child_name, child.wrapped_layer)
    return aq_handlers


class _LayerWrapperThatAccumulatesXTX(nn.Module):
    def __init__(self, layer: nn.Module, aq_handler: AQEngine):
        super().__init__()
        self.wrapped_layer, self.aq_handler = layer, aq_handler

    def forward(self, input, *args, **kwargs):
        mask = kwargs.pop('mask', None)
        weight = kwargs.pop('weight', None)
        self.aq_handler.add_batch(input, mask=mask, weight=weight)
        return self.wrapped_layer(input, *args, **kwargs)


@torch.no_grad()
def init_aq_engines_parallel(
    devices: Sequence[torch.device],
    layer: nn.Module,
    names: Sequence[str],
    inps: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
    masks: Optional[Sequence[torch.Tensor]] = None,
    weights: Optional[Sequence[torch.Tensor]] = None,
    **forward_args,
):
    if masks is not None or weights is not None:
        # This is a simplification. A full implementation would require splitting
        # the masks and weights lists across devices, similar to how `inps` is handled.
        raise NotImplementedError("Parallel entropy masking/weighting is not implemented.")
    
    """Parallel version of init_aq_engines; works on lists of input/output tensors"""
    layer_replicas = torch.nn.parallel.replicate(layer, devices=devices, detach=True)
    layer_replicas[0] = layer
    funcs_by_device = [init_aq_engines for _ in devices]
    inputs_by_device = []
    kwargs_by_device = []
    for i in range(len(devices)):
        inputs_by_device.append((layer_replicas[i], names, inps[i], outs[i], None, None)) # Pass None for masks/weights
        kwargs_by_device.append(
            {
                k: (v.to(devices[i], non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in forward_args.items()
            }
        )
    aq_handles_by_device: Sequence[Dict[str, AQEngine]] = torch.nn.parallel.parallel_apply(
        funcs_by_device, inputs_by_device, kwargs_by_device, devices=devices
    )
    aq_handlers = aq_handles_by_device[0]
    for key, aq_handler in aq_handlers.items():
        replica_handlers = [device_aq_handlers[key] for device_aq_handlers in aq_handles_by_device]
        replica_nsamples = [replica_handler.nsamples for replica_handler in replica_handlers]
        total_nsamples = sum(replica_nsamples)
        if total_nsamples > 0:
            aq_handler.XTX = sum(
                (replica_handlers[i].XTX * (replica_nsamples[i] / total_nsamples)).to(devices[0], non_blocking=True)
                for i in range(len(devices))
            )
        aq_handler.nsamples = total_nsamples
    return aq_handlers


@torch.no_grad()
def update_outs(
    layer: nn.Module, inps_tensor: torch.Tensor, outs_tensor: torch.Tensor, compute_mse: bool, **forward_args
) -> Sequence[float]:
    """
    Update outs_tensor with new activations and optionally compute sample-wise mse loss with previous activations
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    out_losses = []
    for j in trange(len(inps_tensor), desc="calc outs after quantization", leave=False):
        outs_batch = layer(inps_tensor[j].to(device).unsqueeze(0), **forward_args)[0]
        if compute_mse:
            batch_size = outs_batch.shape[0]
            outs_batch_loss = (
                (outs_batch - outs_tensor[j].to(device)).float().square().view(batch_size, -1).mean(dim=-1)
            )
            outs_batch_loss /= outs_batch.float().square().view(batch_size, -1).mean(dim=-1).clamp(min=1e-6)
            outs_batch_loss = outs_batch_loss.mean()
            out_losses.append(outs_batch_loss.item())
        outs_tensor[j].copy_(outs_batch.reshape_as(outs_tensor[j]), non_blocking=True)
    return out_losses


@torch.no_grad()
def update_outs_parallel(
    devices: Sequence[torch.device],
    layer: nn.Module,
    inps: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
    compute_mse: bool,
    **forward_args,
) -> Sequence[float]:
    """Parallel version of update_outs_and_compute_losses; works on lists of input/output tensors"""
    layer_replicas = torch.nn.parallel.replicate(layer, devices=devices, detach=True)
    funcs_by_device = [update_outs for _ in devices]
    inputs_by_device = []
    kwargs_by_device = []
    for i in range(len(devices)):
        inputs_by_device.append((layer_replicas[i], inps[i], outs[i], compute_mse))
        kwargs_by_device.append(
            {
                k: (v.to(devices[i], non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in forward_args.items()
            }
        )
    out_losses_by_device: Sequence[Sequence[float]] = torch.nn.parallel.parallel_apply(
        funcs_by_device, inputs_by_device, kwargs_by_device, devices=devices
    )
    return list(chain(*out_losses_by_device))


def main():
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "model_path",
        type=str,
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name [c4, pajama, openmathreasoning] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--new_eval",
        action="store_true",
        help="if this is set, evaluate on new (and slightly more realistic!) val dataset versions",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=None,
        help="Number of calibration data samples.If None take all calibration data.",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=4096,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument(
        "--use_checkpointing",
        action="store_true",
        help="Whether to use checkpoining in finetuning",
    )
    parser.add_argument(
        "--mix_compression",
        action="store_true",
        help="Compress .self_attn in 4 bits, .block_sparse_moe.experts to 2.3 for mixtral.",
    )
    parser.add_argument("--load", type=str, default=None, help="Path to load quantized statistics.")
    parser.add_argument("--save", type=str, default=None, help="Path to save quantized statistics.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If true, search for previously saved layers and reuse them to save time. Requires --save path.",
    )
    parser.add_argument(
        "--on_save",
        type=str,
        default=None,
        help="Optional callback (python code string) to call after each saved layer. Example: when "
        "training on preemptible compute, upload partially quantized model and --resume later.",
    )
    parser.add_argument("--devices", metavar="N", type=str, nargs="+", default=None, help="List of devices")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for calibration data and initialization. "
        "Note that the main training is not strictly deterministic.",
    )
    parser.add_argument(
        "--skip_out_loss",
        action="store_true",
        help="Whether to skip computation of out loss.",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
    )
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--num_codebooks",
        type=int,
        default=1,
        help="#Number of codebooks per layer",
    )
    parser.add_argument(
        "--nbits_per_codebook",
        type=int,
        default=16,
        help="each codebook will contain 2 ** nbits_per_codebook vectors",
    )
    parser.add_argument(
        "--out_group_size",
        type=int,
        default=1,
        help="How many output units are quantized together",
    )
    parser.add_argument(
        "--in_group_size",
        type=int,
        default=8,
        help="How many input features are quantized together",
    )
    parser.add_argument(
        "--scale_nbits",
        type=int,
        default=0,
        help="Number of bits dedicated to the learnable group-wise scale. 0 means do not use group-wise scales "
        "(still has row-wise scales), 1-15 means using per-group scales quantized to this many bits, "
        "16+ means use per-group scales but do not quantize them",
    )
    parser.add_argument(
        "--codebook_value_nbits",
        type=int,
        default=16,
        help="If below 16, quantize the values in each codebook with the specified number of bits",
    )
    parser.add_argument(
        "--codebook_value_num_groups",
        type=int,
        default=1,
        help="Split codebook vectors into this many groups for quantizations. Only used when quantized codebooks.",
    )

    parser.add_argument(
        "--init_max_iter",
        type=int,
        default=100,
        help="Number of K-Means iterations used to initialize codebooks and codes",
    )
    parser.add_argument(
        "--use_faiss",
        action="store_true",
        help="Whether to use faiss.Kmeans when initializing codebooks and codes",
    )
    parser.add_argument(
        "--init_max_points_per_centroid",
        type=int,
        default=None,
        help="During K-means initialzation, sample (this_many * 2 ^ nbits_per_codebook) points for training K-means",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Keep top-(this_many) best candidates for each codebook when finding optimal codes",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1000,
        help="Maximum number of beam search rounds before the optimization is forcibly stopped.",
    )
    parser.add_argument(
        "--relative_mse_tolerance",
        type=float,
        default=None,
        help="Stop training when (current_epoch_mse / previous_epoch_mse) > (1 - relative_mse_tolerance)",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=100,
        help="Run (this many) Adam updates before every beam search round",
    )
    parser.add_argument(
        "--finetune_max_epochs",
        type=int,
        default=5,
        help="Run this many passes over training data when doing finetuning; No finetuning if set to 0.",
    )
    parser.add_argument(
        "--finetune_early_stop",
        type=int,
        default=3,
        help="Terminate finetuning if loss doesn't improve after this number of epochs.",
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=1e-5,
        help="Finetuning learning rate",
    )
    parser.add_argument(
        "--finetune_batch_size",
        type=int,
        default=1,
        help="(finetuning only) train on batches of this many sequences, globally across all GPUs",
    )
    parser.add_argument(
        "--finetune_adam_beta1",
        type=float,
        default=0.9,
        help="Finetuning adam_beta1",
    )
    parser.add_argument(
        "--finetune_adam_beta2",
        type=float,
        default=0.95,
        help="Finetuning adam_beta2",
    )
    parser.add_argument("--finetune_keep_best", action="store_true")
    parser.add_argument(
        "--local_batch_size",
        type=int,
        default=None,
        help="(finetuning only) Per-device and per-forward-pass batch size used to accumulate global --batch_size",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=0,
        help="Num validation sequences",
    )
    parser.add_argument(
        "--print_frequency",
        type=int,
        default=10,
        help="Print Adam progress after each print_frequency updates",
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")
    parser.add_argument(
        "--no_quant",
        action="store_true",
        help="Skip model quantization and immediately evaluate the loaded model",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=[None, "eager", "flash_attention_2", "sdpa"],
        help="Attention implementation.",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Whether to use fast tokenizer (some models have only fast tokenizer).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code.",
    )
    # 新增的命令行参数
    parser.add_argument(
        "--use_entropy",
        action="store_true",
        help="Enable entropy-based masking and weighting for XTX calculation."
    )
    parser.add_argument(
        "--entropy_fixed_threshold",
        type=float,
        default=None,
        help="Fixed entropy threshold for masking. Used if --use_entropy is set."
    )
    parser.add_argument(
        "--entropy_percentile_threshold",
        type=float,
        default=None,
        help="Percentile entropy threshold for masking. Used if --use_entropy is set."
    )
    parser.add_argument(
        "--weight_from_entropy",
        action="store_true",
        help="Enable token weighting based on entropy. Used if --use_entropy is set."
    )
    parser.add_argument(
        "--weight_hyperparam_C",
        type=float,
        default=2.0,
        help="Maximum weight for the highest entropy token. Used if --weight_from_entropy is set."
    )


    torch.set_num_threads(min(16, torch.get_num_threads()))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    if args.devices is None:
        if torch.cuda.is_available():
            args.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        else:
            args.devices = [torch.device("cpu")]
    else:
        args.devices = [torch.device(device_str) for device_str in args.devices]
    assert all(isinstance(device, torch.device) for device in args.devices)

    if args.use_entropy and len(args.devices) > 1:
        warnings.warn(
            "Entropy-based quantization with multiple GPUs is experimental and may not be fully optimized."
        )

    if args.nsamples is not None:
        assert args.val_size < args.nsamples, "Number of validation set must be smaller than train + val"

    if args.wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        args.exp_name = (
            os.environ.get("WANDB_NAME", "AQ")
            + f"_num_codebooks_{args.num_codebooks}"
            + f"_out_group_size_{args.out_group_size}"
            + f"_in_group_size_{args.in_group_size}"
            + f"_nbits_per_codebook_{args.nbits_per_codebook}"
            + f"_codebook_value_nbits_{args.codebook_value_nbits}"
            + f"_codebook_value_num_groups_{args.codebook_value_num_groups}"
            + f"_scale_nbits_{args.scale_nbits}"
            + f"_steps_per_epoch_{args.steps_per_epoch}"
            + f"_init_max_iter{args.init_max_iter}"
            + f"_{len(args.devices)}gpus"
        )
        args.group_size = args.in_group_size * args.out_group_size

        wandb.init(
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
        )

    print("\n============ Load model... ============")
    model = get_model(
        args.model_path,
        args.load,
        args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    ).train(False)

    if not args.load and not args.no_quant:
        print("\n============ Quantizing model... ============")
        quantize_model(model, args)

    print("\n============ Evaluating perplexity... ============")
    torch.cuda.reset_peak_memory_stats()
    datasets = ["wikitext2", "c4"]
    if args.new_eval:
        datasets = ["wikitext2", "c4-new"]
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            seed=args.seed,
            model_path=args.model_path,
            seqlen=args.model_seqlen,
            eval_mode=True,
            use_fast_tokenizer=args.use_fast_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
        args.dataset_name = dataset
        perplexity_eval(model, testloader, args)

    print(f"eval: max_memory_allocated={torch.cuda.max_memory_allocated():,}")
    if args.wandb:
        wandb.log({"max_cuda_mem_eval": round(torch.cuda.max_memory_allocated() / 1e9, 2)})


if __name__ == "__main__":
    main()
