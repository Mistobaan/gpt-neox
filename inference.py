# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime
import torch
import json
import threading
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from typing import List
import deepspeed
from deepspeed.launcher.runner import main
from megatron import print_rank_0, mpu
from megatron.training import get_model, get_batch_pipe
from megatron.checkpointing import load_checkpoint
from megatron.utils import get_total_params
from megatron.text_generation_utils import pad_batch, forward_model, filter_logits, switch, stop_tokens_in_completion
from megatron.utils import get_ltor_masks_and_position_ids

from functools import partial
import copy
from torch import functional as F
GENERATE_NUM = 0
lock = threading.Lock()


class MegatronGenerate(Resource):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def send_do_generate():
        choice = torch.cuda.LongTensor([GENERATE_NUM])
        torch.distributed.broadcast(choice, 0)

    def put(self):
        print("request IP: " + str(request.remote_addr))
        print(json.dumps(request.get_json()), flush=True)
        print("current time: ", datetime.datetime.now())

        if not "prompts" in request.get_json():
            return "prompts argument required", 400

        if "max_len" in request.get_json():
            return "max_len is no longer used.  Replace with tokens_to_generate", 400

        if "sentences" in request.get_json():
            return "sentences is no longer used.  Replace with prompts", 400

        prompts = request.get_json()["prompts"]
        if len(prompts) > 128:
            return "Maximum number of prompts is 128", 400

        tokens_to_generate = 64  # Choosing hopefully sane default.  Full sequence is slow
        if "tokens_to_generate" in request.get_json():
            tokens_to_generate = request.get_json()["tokens_to_generate"]
            if not isinstance(tokens_to_generate, int):
                return "tokens_to_generate must be an integer greater than 0"
            if tokens_to_generate < 0:
                return "tokens_to_generate must be an integer greater than or equal to 0"

        logprobs = False
        if "logprobs" in request.get_json():
            logprobs = request.get_json()["logprobs"]
            if not isinstance(logprobs, bool):
                return "logprobs must be a boolean value"

        if tokens_to_generate == 0 and not logprobs:
            return "tokens_to_generate=0 implies logprobs should be True"

        temperature = 1.0
        if "temperature" in request.get_json():
            temperature = request.get_json()["temperature"]
            if not (type(temperature) == int or type(temperature) == float):
                return "temperature must be a positive number less than or equal to 100.0"
            if not (0.0 < temperature <= 100.0):
                return "temperature must be a positive number less than or equal to 100.0"

        top_k = 0.0
        if "top_k" in request.get_json():
            top_k = request.get_json()["top_k"]
            if not (type(top_k) == int):
                return "top_k must be an integer equal to or greater than 0 and less than or equal to 1000"
            if not (0 <= top_k <= 1000):
                return "top_k must be equal to or greater than 0 and less than or equal to 1000"

        top_p = 0.0
        if "top_p" in request.get_json():
            top_p = request.get_json()["top_p"]
            if not (type(top_p) == float):
                return "top_p must be a positive float less than or equal to 1.0"
            if top_p > 0.0 and top_k > 0.0:
                return "cannot set both top-k and top-p samplings."
            if not (0 <= top_p <= 1.0):
                return "top_p must be less than or equal to 1.0"

        add_BOS = False
        if "add_BOS" in request.get_json():
            add_BOS = request.get_json()["add_BOS"]
            if not isinstance(add_BOS, bool):
                return "add_BOS must be a boolean value"

        with lock:  # Need to get lock to keep multiple threads from hitting code
            MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate
            response, response_seg, response_logprobs, _ = \
                generate_and_post_process(
                    self.model,
                    self.tokenizer,
                    prompts=prompts,
                    tokens_to_generate=tokens_to_generate,
                    return_output_log_probs=logprobs,
                    top_k_sampling=top_k,
                    top_p_sampling=top_p,
                    temperature=temperature,
                    add_BOS=add_BOS,
                    use_eod_token_for_early_termination=True)

        return jsonify({"text": response,
                        "segments": response_seg,
                        "logprobs": response_logprobs})


class MegatronServer(object):
    def __init__(self, model, tokenizer):
        self.app = Flask(__name__, static_url_path='')
        api = Api(self.app)
        api.add_resource(MegatronGenerate, '/api',
                         resource_class_args=[model, tokenizer])

    def run(self, host, **options):
        self.app.run(host, threaded=True, debug=False, **options)


def stream_tokens(
    model,
    context_tokens: List[List[int]],
    eos_token_id: int = None,
    maximum_tokens: int = None,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
    stop_tokens=None,
    seq_length: int = 0,
    tokenizer=None,
    is_pipe_parallel: bool = False,
):

    # pad batch in order to allow conversion to tensor
    context_tokens, context_lengths = pad_batch(
        copy.deepcopy(context_tokens),
        pad_id=tokenizer.eod,
        pad_len=seq_length,
    )

    # convert to tensor and broadcast
    context_tokens = torch.cuda.LongTensor(context_tokens)
    if stop_tokens:
        stop_tokens = torch.cuda.LongTensor(stop_tokens)
        if stop_tokens.ndim == 1:
            stop_tokens = stop_tokens.unsqueeze(0)

    # Make sure context tokens + start tokens are the same across all ranks
    token_generation_start_index = torch.cuda.LongTensor(context_lengths)
    torch.distributed.broadcast(
        context_tokens,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group(),
    )
    torch.distributed.broadcast(
        token_generation_start_index,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group(),
    )

    # get attention mask / position ids
    # Move to GPU.
    context_tokens = context_tokens.contiguous().cuda()
    # Get the attention mask and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=context_tokens,
        eod_token=tokenizer.eod,
        eod_mask_loss=False,
    )

    # set variables
    eos_token_id = eos_token_id or tokenizer.eod
    maximum_tokens = maximum_tokens or (
        seq_length - token_generation_start_index.max().item() - 1
    )
    batch_size = context_tokens.size(0)

    # get the context_index at which generation is to start
    # we start generation at the position where the smallest context ends
    token_index_to_generate = token_generation_start_index.min().item()
    first_token_index_to_generate = token_index_to_generate
    last_token_index_to_generate = min(
        seq_length
        - 1,  # never generate more than the model's sequence length
        token_index_to_generate + maximum_tokens - 1,
    )

    with torch.no_grad():
        # initialize generation variables
        state_is_done = torch.zeros([batch_size]).byte().cuda()
        token_generation_end_index = torch.ones(
            [batch_size]).long().cuda() * (-1)

        while token_index_to_generate <= last_token_index_to_generate:
            if recompute:  # recompute all tokens
                model_inputs = (
                    context_tokens,
                    position_ids,
                    attention_mask,
                )
                logits = forward_model(
                    model, model_inputs, is_pipe_parallel)

                if logits is not None:  # if pipe parallel, not all ranks return logits
                    generated_token_logits = logits[
                        :, token_index_to_generate - 1, :
                    ]  # [bs, seq, vocab_size] -> [bs, vocab_size]
            else:  # use kv cache
                if token_index_to_generate == first_token_index_to_generate:
                    tokens_to_use = context_tokens[:, :token_index_to_generate]
                    positions_to_use = position_ids[:,
                                                    :token_index_to_generate]
                else:
                    tokens_to_use = context_tokens[:, token_index_to_generate - 1].view(
                        batch_size, -1
                    )
                    positions_to_use = position_ids[
                        :, token_index_to_generate - 1
                    ].view(batch_size, -1)

                model_inputs = (
                    tokens_to_use,  # input_ids
                    positions_to_use,  # position_ids
                    attention_mask,  # attention_mask
                )

                logits = forward_model(
                    model, model_inputs, is_pipe_parallel)
                if logits is not None:  # if pipe parallel, not all ranks return logits
                    generated_token_logits = (
                        logits[:, -1].view(batch_size, -1).contiguous()
                    )  # [bs, seq, vocab_size] -> [bs, vocab_size]

            if logits is not None:
                # sample token id of the to be generated token
                if temperature == 0.0 and top_k == 0 and top_p == 0.0:
                    generated_tokens = torch.argmax(
                        generated_token_logits, dim=-1
                    ).view(-1)
                else:
                    generated_token_logits = generated_token_logits.float()
                    if temperature > 0.0:
                        generated_token_logits /= temperature
                    generated_token_logits = filter_logits(
                        generated_token_logits, top_k=top_k, top_p=top_p
                    )
                    next_token_log_probs = F.softmax(
                        generated_token_logits, dim=-1)
                    generated_tokens = torch.multinomial(
                        next_token_log_probs, num_samples=1
                    ).view(-1)

            if is_pipe_parallel:
                # broadcast generated tokens to pipe parallel group
                src_rank = model.grid.stage_to_global(model.num_stages - 1)
                generated_tokens = (
                    generated_tokens
                    if logits is not None
                    else torch.zeros(batch_size, dtype=torch.long).cuda()
                )
                torch.distributed.broadcast(
                    tensor=generated_tokens,
                    src=src_rank,
                    group=mpu.get_pipe_parallel_group(),
                )

            # determine if state has started for each batch item
            state_started = (
                token_generation_start_index <= token_index_to_generate
            )  # check which batch items have been started

            # switch out padding tokens for generated tokens
            context_tokens[:, token_index_to_generate] = switch(
                context_tokens[:, token_index_to_generate].view(-1),
                generated_tokens,
                state_started,
            )

            # determine if state has finished for each batch item
            state_done = (
                generated_tokens == eos_token_id
            ).byte() & state_started.byte()  # check which batch items produce an eos_token in the current iteration
            state_just_finished = (state_done & ~state_is_done).bool()
            state_is_done = state_is_done | state_done
            stop_tokens_produced = torch.zeros_like(state_is_done)
            for batch_idx, ctx in enumerate(context_tokens):
                stop_tokens_produced[batch_idx] = stop_tokens_in_completion(
                    stop_tokens, context_tokens, batch_idx, token_index_to_generate
                )
            state_is_done = state_is_done | stop_tokens_produced

            token_generation_end_index[
                (state_started.byte() & ~state_is_done).bool()
            ] = token_index_to_generate

            token_index_to_generate += 1

            if torch.all(state_is_done):
                break
        return context_tokens, token_generation_start_index, token_generation_end_index


def generate_and_post_process(
    model,
    tokenizer,
    prompts,
    tokens_to_generate,
    return_output_log_probs,
    top_k_sampling,
    top_p_sampling,
    temperature,
    add_BOS,
    use_eod_token_for_early_termination,
):

    context_tokens = tokenizer.tokenize(prompts)
    if len(context_tokens) == 0:
        context_tokens = [tokenizer.eod]
    context_length = len(context_tokens)

    max_seq_length = 2048  # FIXME: extract this from the model

    if context_length >= (max_seq_length - 1):
        context_tokens = context_length[:max_seq_length + 1]
        context_tokens[-1] = [tokenizer.eod]
    else:
        context_tokens = tokenizer.tokenize("EMPTY TEXT")
        context_length = len(context_tokens)

    # terminate_runs = broadcast_terminate_signal(terminate_runs)
    # if terminate_runs == 1:
    #     return

    generated_text = ""

    (batch_context_tokens,
        batch_token_generation_start_index,
        batch_token_generation_end_index
     ) = stream_tokens(
        model=model,
        context_tokens=[context_tokens],
        eos_token_id=tokenizer.eod,
        maximum_tokens=max_seq_length,
        recompute=False,
        temperature=temperature,
        top_k=top_k_sampling,
        top_p=top_p_sampling,
        seq_length=tokens_to_generate,
        tokenizer=tokenizer,
    )

    if mpu.get_model_parallel_rank() == 0:
        generated_tokens = (
            batch_context_tokens[0]
            .cpu()
            .numpy()
            .tolist()[
                batch_token_generation_start_index[0]
                .item(): batch_token_generation_end_index[0]
                .item()
            ]
        )

        generated_text = tokenizer.detokenize(generated_tokens)
    return generated_text


def main():
    """
    Generate text/sample model
    """
    from megatron.neox_arguments import NeoXArgs
    from megatron.initialize import initialize_megatron

    _overwrite_values = {
        "checkpoint_activations": False,
        "partition_activations": False,
        "no_load_optim": True,
        # disable zero optimization (won't be used in inference, and loading zero optimizer can cause errors)
        "zero_optimization": None,
    }
    neox_args = NeoXArgs.consume_neox_args(overwrite_values=_overwrite_values)
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()

    if neox_args.load is None:
        raise ValueError("`load` parameter must be supplied to load a model`")

    # initialize megatron
    initialize_megatron(neox_args)
    use_cache = True
    # set up model and load checkpoint.
    model = get_model(neox_args=neox_args, use_cache=use_cache)

    if neox_args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=None,
            args=neox_args,
            lr_scheduler=None,
            dist_init_required=False,
            model_parameters=None,
            config_params=neox_args.deepspeed_config,
            mpu=mpu if not neox_args.is_pipe_parallel else None,
        )

        model.total_params = get_total_params(model.module)
        print_rank_0(f' > total params: {"{:,}".format(model.total_params)}')

        if neox_args.is_pipe_parallel:
            model.set_has_attention_mask(True)
            model.set_batch_fn(partial(get_batch_pipe, neox_args=neox_args))
    else:
        raise ValueError("Must be using deepspeed to run neox")

    # neox_args.iteration = load_checkpoint(
    #     neox_args=neox_args,
    #     model=model,
    #     optimizer=optimizer,
    #     lr_scheduler=lr_scheduler,
    #     iteration=None,
    # )

    print_rank_0(
        f"Loading checkpoint and starting from iteration {neox_args.iteration}"
    )
    print_rank_0("Finished loading model")

    model.module.inference_mode(use_cache=use_cache)
    model.eval()

    model.module.clear_cache()  # clear kv cache between batches
    torch.distributed.barrier(group=mpu.get_model_parallel_group())

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print("run api server")
        server = MegatronServer(model, neox_args.tokenizer)
        server.run("localhost", port=8888)

    # wait for all processes to exit
    torch.distributed.barrier(group=mpu.get_model_parallel_group())
    print("process %d completed", mpu.get_model_parallel_group())

if __name__ == "__main__":
    main()
