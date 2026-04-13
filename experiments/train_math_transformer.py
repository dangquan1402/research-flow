#!/usr/bin/env python3
"""
Training script for a small arithmetic transformer.

Implements findings from research:
- Digit-level tokenization (each digit = one token)
- Reversed (LSB-first) output order for carry alignment
- Balanced carry sampling for training data
- Pre-norm (RMSNorm) architecture
- AdamW with cosine LR schedule

Supports MLX (Apple Silicon) with PyTorch MPS fallback.

Usage:
    python experiments/train_math_transformer.py --op add --max_digits 5
    python experiments/train_math_transformer.py --op mul --max_digits 3 --n_layers 4
    python experiments/train_math_transformer.py --op add --max_digits 5 --tokenizer plain  # no reversal
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np

# ── Framework Detection ──────────────────────────────────────────────────────

FRAMEWORK = None

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    FRAMEWORK = "mlx"
except ImportError:
    pass

if FRAMEWORK is None:
    try:
        import torch
        import torch.nn as torch_nn
        import torch.optim as torch_optim

        if not torch.backends.mps.is_available():
            print("WARNING: PyTorch MPS not available, using CPU")
        FRAMEWORK = "pytorch"
    except ImportError:
        raise RuntimeError("Neither MLX nor PyTorch is installed")

print(f"Using framework: {FRAMEWORK}")


# ── Tokenizer ────────────────────────────────────────────────────────────────

# Vocab: 0-9, +, -, *, =, <pad>, <eos>, <bos>
TOKENS = list("0123456789+-*=") + ["<pad>", "<eos>", "<bos>"]
TOK2ID = {t: i for i, t in enumerate(TOKENS)}
ID2TOK = {i: t for t, i in TOK2ID.items()}
VOCAB_SIZE = len(TOKENS)
PAD_ID = TOK2ID["<pad>"]
EOS_ID = TOK2ID["<eos>"]
BOS_ID = TOK2ID["<bos>"]
OP_MAP = {"add": "+", "sub": "-", "mul": "*"}


def encode(s: str) -> list[int]:
    return [TOK2ID[c] for c in s]


def decode(ids: list[int]) -> str:
    return "".join(ID2TOK[i] for i in ids if i not in (PAD_ID, EOS_ID, BOS_ID))


# ── Data Generation ──────────────────────────────────────────────────────────


def compute_carry_length(a: int, b: int) -> int:
    """Count the number of positions with carry in a + b."""
    carry = 0
    length = 0
    while a > 0 or b > 0:
        s = (a % 10) + (b % 10) + carry
        carry = 1 if s >= 10 else 0
        length += carry
        a //= 10
        b //= 10
    return length


def sample_balanced_carry(max_digits: int) -> tuple[int, int]:
    """Sample two numbers with balanced carry chain length."""
    # Pick a target carry length uniformly
    target_carry = random.randint(0, max_digits)
    # Try to find a pair with that carry length (with fallback)
    for _ in range(100):
        n_digits_a = random.randint(1, max_digits)
        n_digits_b = random.randint(1, max_digits)
        a = random.randint(
            10 ** (n_digits_a - 1) if n_digits_a > 1 else 0, 10**n_digits_a - 1
        )
        b = random.randint(
            10 ** (n_digits_b - 1) if n_digits_b > 1 else 0, 10**n_digits_b - 1
        )
        if compute_carry_length(a, b) == target_carry:
            return a, b
    # Fallback: return whatever we got
    return a, b


def sample_balanced_digits(max_digits: int) -> tuple[int, int]:
    """Sample two numbers with balanced digit count distribution."""
    n_digits_a = random.randint(1, max_digits)
    n_digits_b = random.randint(1, max_digits)
    lo_a = 10 ** (n_digits_a - 1) if n_digits_a > 1 else 0
    lo_b = 10 ** (n_digits_b - 1) if n_digits_b > 1 else 0
    a = random.randint(lo_a, 10**n_digits_a - 1)
    b = random.randint(lo_b, 10**n_digits_b - 1)
    return a, b


def generate_example(
    op: str, max_digits: int, reverse_output: bool = True, balanced_carry: bool = True
) -> tuple[str, str, int]:
    """Generate one arithmetic example.

    Returns (input_str, full_sequence, answer_int).
    """
    if balanced_carry and op in ("add", "sub"):
        a, b = sample_balanced_carry(max_digits)
    else:
        a, b = sample_balanced_digits(max_digits)

    if op == "add":
        result = a + b
    elif op == "sub":
        # Ensure a >= b for non-negative results
        if a < b:
            a, b = b, a
        result = a - b
    elif op == "mul":
        # For multiplication, use smaller digit counts
        a = a % (10 ** min(len(str(a)), max_digits))
        b = b % (10 ** min(len(str(b)), max_digits))
        result = a * b
    else:
        raise ValueError(f"Unknown op: {op}")

    op_sym = OP_MAP[op]
    result_str = str(result)
    if reverse_output:
        result_str = result_str[::-1]

    seq = f"{a}{op_sym}{b}={result_str}"
    input_part = f"{a}{op_sym}{b}="
    return input_part, seq, result


def generate_dataset(
    op: str,
    max_digits: int,
    n_samples: int,
    reverse_output: bool = True,
    balanced_carry: bool = True,
) -> list[tuple[str, str, int]]:
    """Generate a dataset of arithmetic examples."""
    seen = set()
    data = []
    attempts = 0
    while len(data) < n_samples and attempts < n_samples * 10:
        attempts += 1
        inp, seq, ans = generate_example(op, max_digits, reverse_output, balanced_carry)
        if seq not in seen:
            seen.add(seq)
            data.append((inp, seq, ans))
    return data


def pad_and_encode(sequences: list[str], max_len: int) -> list[list[int]]:
    """Encode and pad sequences to max_len."""
    result = []
    for seq in sequences:
        ids = [BOS_ID] + encode(seq) + [EOS_ID]
        ids = ids[:max_len]
        ids += [PAD_ID] * (max_len - len(ids))
        result.append(ids)
    return result


# ── MLX Model ────────────────────────────────────────────────────────────────

if FRAMEWORK == "mlx":

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = mx.ones((dim,))
            self.eps = eps

        def __call__(self, x):
            rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
            return x / rms * self.weight

    class MultiHeadAttention(nn.Module):
        def __init__(self, dim: int, n_heads: int):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            self.wq = nn.Linear(dim, dim, bias=False)
            self.wk = nn.Linear(dim, dim, bias=False)
            self.wv = nn.Linear(dim, dim, bias=False)
            self.wo = nn.Linear(dim, dim, bias=False)

        def __call__(self, x, mask=None):
            B, T, C = x.shape
            q = (
                self.wq(x)
                .reshape(B, T, self.n_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            k = (
                self.wk(x)
                .reshape(B, T, self.n_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            v = (
                self.wv(x)
                .reshape(B, T, self.n_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )

            scale = math.sqrt(self.head_dim)
            attn = (q @ k.transpose(0, 1, 3, 2)) / scale

            if mask is not None:
                attn = attn + mask

            attn = mx.softmax(attn, axis=-1)
            out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
            return self.wo(out)

    class FeedForward(nn.Module):
        def __init__(self, dim: int, ff_dim: int):
            super().__init__()
            self.w1 = nn.Linear(dim, ff_dim, bias=False)
            self.w2 = nn.Linear(ff_dim, dim, bias=False)

        def __call__(self, x):
            return self.w2(nn.gelu(self.w1(x)))

    class TransformerBlock(nn.Module):
        def __init__(self, dim: int, n_heads: int, ff_dim: int):
            super().__init__()
            self.norm1 = RMSNorm(dim)
            self.attn = MultiHeadAttention(dim, n_heads)
            self.norm2 = RMSNorm(dim)
            self.ff = FeedForward(dim, ff_dim)

        def __call__(self, x, mask=None):
            x = x + self.attn(self.norm1(x), mask=mask)
            x = x + self.ff(self.norm2(x))
            return x

    class ArithmeticTransformer(nn.Module):
        def __init__(
            self, n_layers: int, dim: int, n_heads: int, ff_dim: int, max_seq_len: int
        ):
            super().__init__()
            self.tok_emb = nn.Embedding(VOCAB_SIZE, dim)
            self.pos_emb = nn.Embedding(max_seq_len, dim)
            self.layers = [
                TransformerBlock(dim, n_heads, ff_dim) for _ in range(n_layers)
            ]
            self.norm = RMSNorm(dim)
            self.output = nn.Linear(dim, VOCAB_SIZE, bias=False)
            self.max_seq_len = max_seq_len

        def __call__(self, x):
            B, T = x.shape
            positions = mx.arange(T)
            h = self.tok_emb(x) + self.pos_emb(positions)

            # Causal mask
            mask = mx.triu(mx.full((T, T), float("-inf")), k=1)

            for layer in self.layers:
                h = layer(h, mask=mask)

            h = self.norm(h)
            return self.output(h)

    def count_parameters(model):
        """Count total parameters in MLX model."""
        total = 0
        for k, v in nn.utils.tree_flatten(model.parameters()):
            total += v.size
        return total

    def train_mlx(args):
        """Training loop for MLX backend."""
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Generate data
        reverse = args.tokenizer == "reversed"
        balanced = args.balanced_carry

        print(f"Generating {args.train_samples} training examples...")
        train_data = generate_dataset(
            args.op, args.max_digits, args.train_samples, reverse, balanced
        )
        test_data = generate_dataset(
            args.op, args.max_digits, args.test_samples, reverse, balanced
        )
        print(f"Generated {len(train_data)} train, {len(test_data)} test examples")

        # Determine max sequence length
        max_len = (
            max(len(seq) for _, seq, _ in train_data + test_data) + 2
        )  # +2 for BOS/EOS
        max_len = min(max_len, 64)  # Cap at 64

        # Encode
        train_seqs = pad_and_encode([seq for _, seq, _ in train_data], max_len)
        test_inputs = [inp for inp, _, _ in test_data]
        test_answers = [ans for _, _, ans in test_data]

        train_x = mx.array(np.array(train_seqs, dtype=np.int32))

        # Model
        model = ArithmeticTransformer(
            n_layers=args.n_layers,
            dim=args.dim,
            n_heads=args.n_heads,
            ff_dim=args.ff_dim,
            max_seq_len=max_len,
        )
        mx.eval(model.parameters())
        n_params = count_parameters(model)
        print(
            f"Model: {args.n_layers}L/{args.n_heads}H/{args.dim}D, {n_params:,} params"
        )

        # Optimizer with cosine schedule
        warmup_steps = args.warmup_steps
        total_steps = args.epochs * (len(train_data) // args.batch_size + 1)

        schedule = optim.cosine_decay(
            args.lr, total_steps - warmup_steps, end=args.lr * 0.1
        )
        if warmup_steps > 0:
            schedule = optim.join_schedules(
                [optim.linear_schedule(0, args.lr, warmup_steps), schedule],
                [warmup_steps],
            )
        optimizer = optim.AdamW(
            learning_rate=schedule, weight_decay=args.weight_decay, betas=[0.9, 0.99]
        )

        # Loss function
        def loss_fn(model, x):
            logits = model(x[:, :-1])  # predict next token
            targets = x[:, 1:]
            # Mask out padding
            mask = (targets != PAD_ID).astype(mx.float32)
            ce = nn.losses.cross_entropy(logits, targets, reduction="none")
            return (ce * mask).sum() / mask.sum()

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        # Training
        results = {
            "framework": "mlx",
            "op": args.op,
            "max_digits": args.max_digits,
            "tokenizer": args.tokenizer,
            "balanced_carry": args.balanced_carry,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "dim": args.dim,
            "ff_dim": args.ff_dim,
            "n_params": n_params,
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "epoch_logs": [],
        }

        print(f"\nTraining for {args.epochs} epochs, batch_size={args.batch_size}")
        print("-" * 70)

        global_step = 0
        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            # Shuffle
            perm = np.random.permutation(len(train_data))
            train_x_shuffled = train_x[mx.array(perm)]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_data), args.batch_size):
                batch = train_x_shuffled[i : i + args.batch_size]
                if batch.shape[0] == 0:
                    continue

                loss, grads = loss_and_grad(model, batch)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            epoch_time = time.time() - epoch_start

            # Evaluate every eval_every epochs
            accuracy = None
            digit_accuracy = {}
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                accuracy, digit_accuracy = evaluate_mlx(
                    model, test_inputs, test_answers, max_len, reverse, args.op
                )

            log = {
                "epoch": epoch,
                "loss": round(avg_loss, 4),
                "time_s": round(epoch_time, 2),
            }
            if accuracy is not None:
                log["accuracy"] = round(accuracy, 4)
                log["digit_accuracy"] = digit_accuracy

            results["epoch_logs"].append(log)

            acc_str = f"  acc={accuracy:.2%}" if accuracy is not None else ""
            digit_str = ""
            if digit_accuracy:
                digit_str = "  " + " ".join(
                    f"{k}d:{v:.0%}" for k, v in sorted(digit_accuracy.items())
                )
            print(
                f"Epoch {epoch:3d} | loss={avg_loss:.4f} | {epoch_time:.1f}s{acc_str}{digit_str}"
            )

        total_time = time.time() - start_time
        results["total_time_s"] = round(total_time, 2)
        results["final_accuracy"] = results["epoch_logs"][-1].get("accuracy", 0)
        results["final_loss"] = results["epoch_logs"][-1]["loss"]

        print("-" * 70)
        print(
            f"Done in {total_time:.1f}s. Final accuracy: {results['final_accuracy']:.2%}"
        )

        return results

    def evaluate_mlx(model, test_inputs, test_answers, max_len, reverse, op):
        """Evaluate model accuracy via greedy autoregressive decoding."""
        correct = 0
        total = 0
        digit_correct = {}
        digit_total = {}

        for inp_str, true_answer in zip(test_inputs, test_answers):
            # Encode input
            ids = [BOS_ID] + encode(inp_str)
            ids_arr = mx.array([ids])

            # Autoregressive generation
            for _ in range(max_len - len(ids)):
                # Pad to generate
                padded = mx.pad(
                    ids_arr,
                    [(0, 0), (0, max_len - ids_arr.shape[1])],
                    constant_values=PAD_ID,
                )
                logits = model(padded)
                next_logit = logits[0, ids_arr.shape[1] - 1]
                next_id = mx.argmax(next_logit).item()

                if next_id == EOS_ID or next_id == PAD_ID:
                    break
                ids_arr = mx.concatenate([ids_arr, mx.array([[next_id]])], axis=1)

            # Decode prediction
            pred_ids = ids_arr[0, len([BOS_ID] + encode(inp_str)) :].tolist()
            pred_str = decode(pred_ids)

            # Reverse back if needed
            if reverse:
                pred_str = pred_str[::-1]

            try:
                pred_val = int(pred_str) if pred_str else -1
            except ValueError:
                pred_val = -1

            # Count by digit count of answer
            n_digits = len(str(true_answer))
            digit_total[n_digits] = digit_total.get(n_digits, 0) + 1

            if pred_val == true_answer:
                correct += 1
                digit_correct[n_digits] = digit_correct.get(n_digits, 0) + 1

            total += 1

        accuracy = correct / max(total, 1)
        digit_accuracy = {
            k: digit_correct.get(k, 0) / digit_total[k]
            for k in sorted(digit_total.keys())
        }
        return accuracy, digit_accuracy


# ── PyTorch Model (fallback) ─────────────────────────────────────────────────

if FRAMEWORK == "pytorch":

    class RMSNormPT(torch_nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = torch_nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
            return x / rms * self.weight

    class ArithmeticTransformerPT(torch_nn.Module):
        def __init__(self, n_layers, dim, n_heads, ff_dim, max_seq_len):
            super().__init__()
            self.tok_emb = torch_nn.Embedding(VOCAB_SIZE, dim)
            self.pos_emb = torch_nn.Embedding(max_seq_len, dim)

            self.layers = torch_nn.ModuleList()
            for _ in range(n_layers):
                self.layers.append(
                    torch_nn.ModuleDict(
                        {
                            "norm1": RMSNormPT(dim),
                            "attn": torch_nn.MultiheadAttention(
                                dim, n_heads, batch_first=True, bias=False
                            ),
                            "norm2": RMSNormPT(dim),
                            "ff": torch_nn.Sequential(
                                torch_nn.Linear(dim, ff_dim, bias=False),
                                torch_nn.GELU(),
                                torch_nn.Linear(ff_dim, dim, bias=False),
                            ),
                        }
                    )
                )
            self.norm = RMSNormPT(dim)
            self.output = torch_nn.Linear(dim, VOCAB_SIZE, bias=False)
            self.max_seq_len = max_seq_len

        def forward(self, x):
            B, T = x.shape
            device = x.device
            positions = torch.arange(T, device=device)
            h = self.tok_emb(x) + self.pos_emb(positions)

            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=device), diagonal=1
            )

            for layer in self.layers:
                h2 = layer["norm1"](h)
                h2, _ = layer["attn"](h2, h2, h2, attn_mask=mask)
                h = h + h2
                h = h + layer["ff"](layer["norm2"](h))

            return self.output(self.norm(h))

    def count_parameters_pt(model):
        return sum(p.numel() for p in model.parameters())

    def train_pytorch(args):
        """Training loop for PyTorch backend."""
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"PyTorch device: {device}")

        reverse = args.tokenizer == "reversed"
        balanced = args.balanced_carry

        print(f"Generating {args.train_samples} training examples...")
        train_data = generate_dataset(
            args.op, args.max_digits, args.train_samples, reverse, balanced
        )
        test_data = generate_dataset(
            args.op, args.max_digits, args.test_samples, reverse, balanced
        )
        print(f"Generated {len(train_data)} train, {len(test_data)} test examples")

        max_len = max(len(seq) for _, seq, _ in train_data + test_data) + 2
        max_len = min(max_len, 64)

        train_seqs = pad_and_encode([seq for _, seq, _ in train_data], max_len)
        test_inputs = [inp for inp, _, _ in test_data]
        test_answers = [ans for _, _, ans in test_data]

        train_x = torch.tensor(train_seqs, dtype=torch.long, device=device)

        model = ArithmeticTransformerPT(
            n_layers=args.n_layers,
            dim=args.dim,
            n_heads=args.n_heads,
            ff_dim=args.ff_dim,
            max_seq_len=max_len,
        ).to(device)

        n_params = count_parameters_pt(model)
        print(
            f"Model: {args.n_layers}L/{args.n_heads}H/{args.dim}D, {n_params:,} params"
        )

        optimizer = torch_optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.99),
        )

        total_steps = args.epochs * (len(train_data) // args.batch_size + 1)
        scheduler = torch_optim.lr_scheduler.CosineAnnealingLR(
            optimizer, total_steps, eta_min=args.lr * 0.1
        )

        results = {
            "framework": "pytorch",
            "device": str(device),
            "op": args.op,
            "max_digits": args.max_digits,
            "tokenizer": args.tokenizer,
            "balanced_carry": args.balanced_carry,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "dim": args.dim,
            "ff_dim": args.ff_dim,
            "n_params": n_params,
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "epoch_logs": [],
        }

        print(f"\nTraining for {args.epochs} epochs, batch_size={args.batch_size}")
        print("-" * 70)

        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            model.train()

            perm = torch.randperm(len(train_data), device=device)
            train_x_shuffled = train_x[perm]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_data), args.batch_size):
                batch = train_x_shuffled[i : i + args.batch_size]
                if batch.shape[0] == 0:
                    continue

                logits = model(batch[:, :-1])
                targets = batch[:, 1:]
                mask = (targets != PAD_ID).float()

                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE),
                    targets.reshape(-1),
                    reduction="none",
                )
                loss = (loss * mask.reshape(-1)).sum() / mask.sum()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            epoch_time = time.time() - epoch_start

            accuracy = None
            digit_accuracy = {}
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                model.eval()
                with torch.no_grad():
                    accuracy, digit_accuracy = evaluate_pytorch(
                        model,
                        test_inputs,
                        test_answers,
                        max_len,
                        reverse,
                        args.op,
                        device,
                    )

            log = {
                "epoch": epoch,
                "loss": round(avg_loss, 4),
                "time_s": round(epoch_time, 2),
            }
            if accuracy is not None:
                log["accuracy"] = round(accuracy, 4)
                log["digit_accuracy"] = digit_accuracy
            results["epoch_logs"].append(log)

            acc_str = f"  acc={accuracy:.2%}" if accuracy is not None else ""
            digit_str = ""
            if digit_accuracy:
                digit_str = "  " + " ".join(
                    f"{k}d:{v:.0%}" for k, v in sorted(digit_accuracy.items())
                )
            print(
                f"Epoch {epoch:3d} | loss={avg_loss:.4f} | {epoch_time:.1f}s{acc_str}{digit_str}"
            )

        total_time = time.time() - start_time
        results["total_time_s"] = round(total_time, 2)
        results["final_accuracy"] = results["epoch_logs"][-1].get("accuracy", 0)
        results["final_loss"] = results["epoch_logs"][-1]["loss"]

        print("-" * 70)
        print(
            f"Done in {total_time:.1f}s. Final accuracy: {results['final_accuracy']:.2%}"
        )

        return results

    def evaluate_pytorch(
        model, test_inputs, test_answers, max_len, reverse, op, device
    ):
        correct = 0
        total = 0
        digit_correct = {}
        digit_total = {}

        for inp_str, true_answer in zip(test_inputs, test_answers):
            ids = [BOS_ID] + encode(inp_str)
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)

            for _ in range(max_len - len(ids)):
                padded = torch.nn.functional.pad(
                    ids_t, (0, max_len - ids_t.shape[1]), value=PAD_ID
                )
                logits = model(padded)
                next_logit = logits[0, ids_t.shape[1] - 1]
                next_id = torch.argmax(next_logit).item()

                if next_id == EOS_ID or next_id == PAD_ID:
                    break
                ids_t = torch.cat(
                    [ids_t, torch.tensor([[next_id]], device=device)], dim=1
                )

            pred_ids = ids_t[0, len([BOS_ID] + encode(inp_str)) :].tolist()
            pred_str = decode(pred_ids)

            if reverse:
                pred_str = pred_str[::-1]

            try:
                pred_val = int(pred_str) if pred_str else -1
            except ValueError:
                pred_val = -1

            n_digits = len(str(true_answer))
            digit_total[n_digits] = digit_total.get(n_digits, 0) + 1

            if pred_val == true_answer:
                correct += 1
                digit_correct[n_digits] = digit_correct.get(n_digits, 0) + 1

            total += 1

        accuracy = correct / max(total, 1)
        digit_accuracy = {
            k: digit_correct.get(k, 0) / digit_total[k]
            for k in sorted(digit_total.keys())
        }
        return accuracy, digit_accuracy


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train arithmetic transformer")
    # Task
    parser.add_argument(
        "--op", choices=["add", "sub", "mul"], default="add", help="Operation"
    )
    parser.add_argument("--max_digits", type=int, default=5, help="Max operand digits")
    parser.add_argument(
        "--tokenizer",
        choices=["reversed", "plain"],
        default="reversed",
        help="Output format: reversed (LSB-first) or plain (MSB-first)",
    )
    parser.add_argument(
        "--balanced_carry", type=bool, default=True, help="Use balanced carry sampling"
    )

    # Architecture
    parser.add_argument(
        "--n_layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument(
        "--n_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--ff_dim", type=int, default=1024, help="FFN hidden dimension")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.5, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps")
    parser.add_argument(
        "--train_samples", type=int, default=50000, help="Training samples"
    )
    parser.add_argument("--test_samples", type=int, default=2000, help="Test samples")
    parser.add_argument(
        "--eval_every", type=int, default=5, help="Evaluate every N epochs"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/results",
        help="Results directory",
    )
    parser.add_argument(
        "--tag", type=str, default="", help="Experiment tag for filename"
    )

    args = parser.parse_args()

    # Run training
    if FRAMEWORK == "mlx":
        results = train_mlx(args)
    else:
        results = train_pytorch(args)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    fname = f"{args.op}_{args.max_digits}d_{args.tokenizer}_{args.n_layers}L{args.n_heads}H{args.dim}D{tag}.json"
    out_path = os.path.join(args.output_dir, fname)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
