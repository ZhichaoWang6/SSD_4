import argparse
import copy
import json
import os
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, get_linear_schedule_with_warmup

from kangaroo_model import KangarooQwenModel
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)

    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--exit_layer", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--save_freq", type=int, default=1)

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--answer_margin_tokens", type=int, default=4)
    parser.add_argument("--speculative_steps", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--post_mismatch_weight", type=float, default=0.3,
                        help="Weight applied to per-token CE loss at and after the first "
                             "draft/verify mismatch. 1.0 = full weight everywhere; "
                             "0.0 = old behavior (only train on prefix).")
    parser.add_argument("--hidden_loss_weight", type=float, default=0.0,
                        help="Auxiliary smooth-L1 loss weight between adapter output hidden "
                             "state and base final hidden state (verify_hidden_normed). "
                             "0.0 = disabled. Recommended: 0.5.")

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_reply_keep_ratio", type=float, default=1.0)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_steps", type=int, default=6)

    return parser.parse_args()


def load_data(path):
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def freeze_base_train_adapter(model):
    model.base_model.model.eval()
    model.head_model.eval()
    model.adapter_model.train()

    for p in model.base_model.model.parameters():
        p.requires_grad = False
    for p in model.head_model.parameters():
        p.requires_grad = False
    for p in model.adapter_model.parameters():
        p.requires_grad = True


def save_adapter(model, outdir, epoch):
    save_dir = os.path.join(outdir, f"epoch_{epoch:03d}")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.adapter_model.state_dict(), os.path.join(save_dir, "adapter_model.bin"))

    adapter_config = {
        "exit_layer": model.early_exit_layer,
        "use_mlp": getattr(model.adapter_model.layers[0], "use_mlp", True),
    }

    with open(os.path.join(save_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2)

    print(f"[save] adapter saved to {save_dir}")


def build_inputs(processor, conversation, device):
    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(conversation)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    return inputs


def estimate_answer_token_len(processor, gt_answer):
    if isinstance(gt_answer, list):
        text = ""
        for item in gt_answer:
            if isinstance(item, dict) and item.get("type") == "text":
                text += item.get("text", "")
    else:
        text = str(gt_answer)

    ids = processor.tokenizer(
        text,
        add_special_tokens=False,
        return_tensors=None,
    )["input_ids"]

    return max(1, len(ids))


def iter_contexts(example):
    messages = example.get("conversation", example.get("messages", []))
    if not messages:
        return

    history = []
    i = 0

    if messages[0]["role"] == "system":
        history.append(messages[0])
        i = 1

    while i < len(messages) - 1:
        user_turn = messages[i]
        asst_turn = messages[i + 1]

        if user_turn["role"] == "user" and asst_turn["role"] == "assistant":
            context = history + [user_turn]
            gt_answer = asst_turn.get("content", "")
            yield copy.deepcopy(context), gt_answer

            history.append(user_turn)
            history.append(asst_turn)
            i += 2
        else:
            history.append(messages[i])
            i += 1


def _cache_len(adapter_past_key_values):
    if adapter_past_key_values and adapter_past_key_values[0] is not None:
        return adapter_past_key_values[0][0].shape[2]
    return 0


def online_train_one_context(
    model,
    processor,
    inputs,
    max_new_tokens=128,
    speculative_steps=6,
    threshold=0.0,
    post_mismatch_weight=0.3,
    hidden_loss_weight=0.0,
    debug=False,
    debug_steps=6,
):
    base_model = model.base_model
    adapter_model = model.adapter_model
    head_model = model.head_model

    device = inputs["input_ids"].device
    input_ids = inputs["input_ids"]
    batch_size, context_length = input_ids.shape
    assert batch_size == 1

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    eos_id = tokenizer.eos_token_id
    eos_set = set(eos_id) if isinstance(eos_id, list) else {eos_id}

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_set.add(im_end_id)

    eos_fill = list(eos_set)[0]
    max_length = context_length + max_new_tokens

    global_tokens = torch.full(
        (1, max_length),
        eos_fill,
        dtype=torch.long,
        device=device,
    )
    global_tokens[:, :context_length] = input_ids

    forward_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "use_cache": True,
        "output_hidden_states": True,
        "return_dict": True,
        "pixel_values": inputs.get("pixel_values"),
        "pixel_values_videos": inputs.get("pixel_values_videos"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "video_grid_thw": inputs.get("video_grid_thw"),
        "second_per_grid_ts": inputs.get("second_per_grid_ts"),
        "drop_method": "none",
        "drop_threshold": 1.0,
        "drop_absolute": True,
    }
    forward_kwargs = {k: v for k, v in forward_kwargs.items() if v is not None}

    with torch.no_grad():
        output = base_model.model(**forward_kwargs)
        base_model.past_key_values = output.past_key_values

        first_token = torch.argmax(output.logits[:, -1, :], dim=-1)
        global_tokens[:, context_length] = first_token.item()

        hidden_state_early = output.hidden_states[model.early_exit_layer]

    with torch.no_grad():
        _, adapter_past_key_values = adapter_model.forward_early_stop(
            inputs_embeds=hidden_state_early.detach(),
            use_cache=True,
        )

    if debug:
        print("\n========== ONLINE DEBUG ==========")
        print(f"context_length = {context_length}")
        print(f"max_new_tokens = {max_new_tokens}")
        print(f"max_length = {max_length}")
        print(f"first_token_id = {first_token.item()}")
        print(f"first_token = {tokenizer.decode(first_token)}")
        print(f"eos_set = {eos_set}")
        print(
            f"[prefill cache] draft_cache={base_model._get_layer_cache_length(0)} "
            f"verify_cache={base_model._get_layer_cache_length(model.early_exit_layer)} "
            f"adapter_cache={_cache_len(adapter_past_key_values)}"
        )

    if first_token.item() in eos_set:
        return None

    start_index = context_length
    total_loss = None
    total_tokens = 0
    correct = 0
    accept_lengths = []

    stop = False
    round_idx = 0

    while start_index < max_length - 1 and not stop:
        round_idx += 1

        start_copy = start_index
        end_index = start_index + 1
        remaining = max_length - 1 - start_index
        round_steps = min(speculative_steps, remaining)

        exited_hidden_states = None
        draft_token_ids = []
        adapter_logits_list = []
        adapter_hidden_list = []

        if debug:
            print(f"\n----- round {round_idx} -----")
            print(f"start_index={start_index}, end_index={end_index}, round_steps={round_steps}")

        for step in range(1 + round_steps):
            in_token = global_tokens[:, end_index - 1:end_index]

            with torch.no_grad():
                hidden_state_early = base_model.forward_draft_or_large_model(
                    in_tokens_small=in_token,
                )

            if step == 0:
                exited_hidden_states = None

            exited_hidden_states = (
                hidden_state_early.detach()
                if exited_hidden_states is None
                else torch.cat([exited_hidden_states, hidden_state_early.detach()], dim=1)
            )

            if step == round_steps:
                if debug:
                    print(f"[draft] step={step} reached bonus/limit; no adapter prediction")
                break

            hidden_state, adapter_past_key_values = adapter_model.forward_early_stop(
                inputs_embeds=hidden_state_early.detach(),
                past_key_values=adapter_past_key_values,
                use_cache=True,
            )

            adapter_logits = head_model(hidden_state[:, -1:, :]).float()
            predicted_token = torch.argmax(adapter_logits[:, -1, :], dim=-1)
            predict_score = adapter_logits.softmax(dim=-1).max().item()

            adapter_logits_list.append(adapter_logits[:, -1, :])
            adapter_hidden_list.append(hidden_state[:, -1:, :])
            draft_token_ids.append(predicted_token.item())

            global_tokens[:, end_index] = predicted_token.item()

            if debug and len(adapter_logits_list) <= debug_steps:
                print(
                    f"[draft] step={step}"
                    f" pos={end_index}"
                    f" in={tokenizer.decode(in_token[0])!r}"
                    f" pred_id={predicted_token.item()}"
                    f" pred={tokenizer.decode(predicted_token)!r}"
                    f" conf={predict_score:.4f}"
                    f" adapter_cache={_cache_len(adapter_past_key_values)}"
                )

            end_index += 1

            if predicted_token.item() in eos_set:
                stop = True
                if debug:
                    print("[draft] EOS predicted; stop=True")
                break

            if predict_score < threshold:
                if debug:
                    print(f"[draft] confidence {predict_score:.4f} < threshold {threshold}; stop draft")
                break

        if not adapter_logits_list:
            if debug:
                print("[warn] no adapter logits; break")
            break

        with torch.no_grad():
            _, verify_hidden_normed = base_model.forward_draft_or_large_model(
                in_features_large=exited_hidden_states,
            )
            verify_logits = head_model(verify_hidden_normed).float()
            verify_ids = torch.argmax(verify_logits, dim=-1)

        verify_len = verify_ids.shape[1]
        draft_len = len(draft_token_ids)
        usable_len = min(draft_len, verify_len)

        if debug:
            print(f"[verify] usable_len={usable_len}, verify_len={verify_len}, draft_len={draft_len}")

        if usable_len <= 0:
            break

        adapter_logits_tensor_full = torch.cat(
            adapter_logits_list[:usable_len],
            dim=0,
        )
        labels_full = verify_ids[0, :usable_len].detach()

        # Find first mismatch and first EOS in the verify window.
        first_mismatch = -1
        first_eos = -1
        for k in range(usable_len):
            verify_id_k = labels_full[k].item()
            draft_id_k = draft_token_ids[k]
            if first_mismatch < 0 and verify_id_k != draft_id_k:
                first_mismatch = k
            if verify_id_k in eos_set:
                first_eos = k
                break

        # Effective training length: cap at first EOS (inclusive); otherwise use full window.
        effective_len = first_eos + 1 if first_eos >= 0 else usable_len
        train_len = effective_len

        adapter_logits_tensor = adapter_logits_tensor_full[:effective_len]
        labels = labels_full[:effective_len]

        # Per-position weights: full weight up to and including first_mismatch
        # (the mismatch position has correct input, only its output was wrong),
        # decayed weight strictly after first_mismatch (those positions had
        # wrong-token inputs and are OOD relative to inference).
        device = adapter_logits_tensor.device
        weights = torch.ones(effective_len, device=device, dtype=adapter_logits_tensor.dtype)
        if 0 <= first_mismatch < effective_len - 1:
            weights[first_mismatch + 1:] = post_mismatch_weight

        ce_per_token = F.cross_entropy(adapter_logits_tensor, labels, reduction='none')
        loss = (ce_per_token * weights).sum() / weights.sum().clamp_min(1.0)

        if hidden_loss_weight > 0 and effective_len > 0:
            adapter_hidden_tensor = torch.cat(
                adapter_hidden_list[:effective_len], dim=1,
            )  # [1, effective_len, H]
            target_hidden = verify_hidden_normed[:, :effective_len, :].detach()
            h_per_token = F.smooth_l1_loss(
                adapter_hidden_tensor.float(),
                target_hidden.float(),
                reduction='none',
            ).mean(-1).squeeze(0)  # [effective_len]
            hidden_loss = (h_per_token * weights).sum() / weights.sum().clamp_min(1.0)
            loss = loss + hidden_loss_weight * hidden_loss
            if debug:
                print(f"[hidden_loss] {hidden_loss.item():.4f}")

        if total_loss is None:
            total_loss = loss * train_len
        else:
            total_loss = total_loss + loss * train_len

        total_tokens += train_len

        with torch.no_grad():
            pred = adapter_logits_tensor.argmax(dim=-1)
            correct += (pred == labels).sum().item()

        if debug:
            print(
                f"[train-window] usable_len={usable_len}, "
                f"train_len={train_len}, "
                f"loss={loss.item():.4f}"
            )
            for j in range(min(train_len, debug_steps)):
                d_id = draft_token_ids[j]
                v_id = labels[j].item()
                print(
                    f"  [cmp-valid] j={j}"
                    f" draft={d_id}({tokenizer.decode([d_id])!r})"
                    f" verify={v_id}({tokenizer.decode([v_id])!r})"
                    f" match={d_id == v_id}"
                )

        mismatch_pos = None

        for i in range(train_len):
            verify_id = labels[i].item()
            draft_id = draft_token_ids[i]

            global_tokens[0, start_index + 1 + i] = verify_id

            if verify_id in eos_set:
                stop = True
                mismatch_pos = i
                start_index = start_index + 1 + i
                break

            if verify_id != draft_id:
                mismatch_pos = i
                start_index = start_index + 1 + i
                break

        if mismatch_pos is None:
            start_index = start_index + train_len

        accept_len = start_index - start_copy
        accept_lengths.append(accept_len)

        base_model.trim_draft_layers_cache(start_index)
        base_model.trim_verify_layers_cache(start_index)

        if adapter_past_key_values and adapter_past_key_values[0][0].shape[2] > start_index:
            adapter_past_key_values = [
                (
                    k[:, :, :start_index, :].detach(),
                    v[:, :, :start_index, :].detach(),
                )
                for k, v in adapter_past_key_values
            ]

        if base_model.past_key_values is not None:
            base_model.past_key_values._seen_tokens = start_index

        if debug:
            print(
                f"[accept] start_copy={start_copy}"
                f" new_start={start_index}"
                f" accept_len={accept_len}"
                f" mismatch_pos={mismatch_pos}"
            )
            print(
                f"[cache after trim] "
                f"draft_cache={base_model._get_layer_cache_length(0)} "
                f"verify_cache={base_model._get_layer_cache_length(model.early_exit_layer)} "
                f"adapter_cache={_cache_len(adapter_past_key_values)}"
            )

    if total_tokens == 0 or total_loss is None:
        if debug:
            print("[warn] total_tokens=0; return None")
            print("========== END DEBUG ==========\n")
        return None

    avg_accept = sum(accept_lengths) / max(len(accept_lengths), 1)
    final_loss = total_loss / total_tokens
    acc = correct / max(total_tokens, 1)

    if debug:
        generated_ids = global_tokens[:, context_length:start_index + 1]
        print(
            f"[result] loss={final_loss.item():.4f}"
            f" acc={acc:.3f}"
            f" avg_accept={avg_accept:.2f}"
            f" tokens={total_tokens}"
        )
        print(f"[generated] {tokenizer.batch_decode(generated_ids, skip_special_tokens=True)}")
        print("========== END DEBUG ==========\n")

    return {
        "loss": final_loss,
        "tokens": total_tokens,
        "acc": acc,
        "avg_accept": avg_accept,
    }


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(args)

    processor = AutoProcessor.from_pretrained(args.model_path)

    model = KangarooQwenModel(
        base_model_path=args.model_path,
        adapter_model_path=args.adapter_path,
        early_exit_layer=args.exit_layer,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(args.device)

    freeze_base_train_adapter(model)

    optimizer = torch.optim.AdamW(
        model.adapter_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    data = load_data(args.data_path)

    if args.max_samples is not None:
        data = data[:args.max_samples]

    total_steps = len(data) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

    global_step = 0

    for epoch in range(args.num_epochs):
        print(f"\n=== Epoch {epoch} ===")
        random.shuffle(data)

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_accept = 0.0
        epoch_tokens = 0

        seen = 0
        seen_no_reply = 0
        seen_long_reply = 0
        skipped_no_reply = 0

        for example in tqdm(data):
            contexts = list(iter_contexts(example))

            for context, gt_answer in contexts:
                gt_text = str(gt_answer).strip()
                is_no_reply = (gt_text == "NO REPLY")

                if is_no_reply and random.random() > args.no_reply_keep_ratio:
                    skipped_no_reply += 1
                    continue

                answer_len = estimate_answer_token_len(processor, gt_answer)
                turn_max_new_tokens = min(
                    args.max_new_tokens,
                    answer_len + args.answer_margin_tokens,
                )

                if args.debug and global_step < args.debug_steps:
                    print("\n==============================")
                    print(f"[GT ANSWER RAW] {gt_answer}")
                    print(f"[GT ANSWER TOKEN LEN] {answer_len}")
                    print(f"[TURN MAX TOKENS] {turn_max_new_tokens}")
                    print(f"[IS NO REPLY] {is_no_reply}")
                    print("==============================")

                model.reset_status()
                freeze_base_train_adapter(model)

                inputs = build_inputs(
                    processor=processor,
                    conversation=context,
                    device=args.device,
                )

                result = online_train_one_context(
                    model=model,
                    processor=processor,
                    inputs=inputs,
                    max_new_tokens=turn_max_new_tokens,
                    speculative_steps=args.speculative_steps,
                    threshold=args.threshold,
                    post_mismatch_weight=args.post_mismatch_weight,
                    hidden_loss_weight=args.hidden_loss_weight,
                    debug=args.debug and global_step < args.debug_steps,
                    debug_steps=args.debug_steps,
                )

                if result is None:
                    continue

                loss = result["loss"]

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.adapter_model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()

                global_step += 1
                seen += 1

                if is_no_reply:
                    seen_no_reply += 1
                else:
                    seen_long_reply += 1

                epoch_loss += loss.item()
                epoch_acc += result["acc"]
                epoch_accept += result["avg_accept"]
                epoch_tokens += result["tokens"]

                if global_step % 10 == 0:
                    print(
                        f"Step {global_step}"
                        f" | LR {optimizer.param_groups[0]['lr']:.2e}"
                        f" | Loss {loss.item():.4f}"
                        f" | Acc {result['acc']:.3f}"
                        f" | AvgAccept {result['avg_accept']:.2f}"
                        f" | Tokens {result['tokens']}"
                        f" | TurnMaxTok {turn_max_new_tokens}"
                        f" | NO_REPLY_seen {seen_no_reply}"
                        f" | LONG_seen {seen_long_reply}"
                        f" | NO_REPLY_skipped {skipped_no_reply}"
                    )

        if seen > 0:
            print(
                f"Epoch {epoch}"
                f" | Loss {epoch_loss / seen:.4f}"
                f" | Acc {epoch_acc / seen:.3f}"
                f" | AvgAccept {epoch_accept / seen:.2f}"
                f" | AvgTokens {epoch_tokens / seen:.2f}"
                f" | Seen {seen}"
                f" | NO_REPLY_seen {seen_no_reply}"
                f" | LONG_seen {seen_long_reply}"
                f" | NO_REPLY_skipped {skipped_no_reply}"
                f" | no_reply_keep_ratio {args.no_reply_keep_ratio}"
            )

        if epoch % args.save_freq == 0:
            save_adapter(model, args.outdir, epoch)

    save_adapter(model, args.outdir, args.num_epochs)


if __name__ == "__main__":
    main()