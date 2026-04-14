#!/usr/bin/env python3
"""
Training script for a small arithmetic transformer.

Implements findings from research:
- Digit-level tokenization (each digit = one token)
- Reversed (LSB-first) output order for carry alignment
- Balanced carry sampling for training data
- Pre-norm (RMSNorm) architecture
- AdamW with cosine LR schedule
- Position Coupling (Cho 2024) for length generalization

Supports MLX (Apple Silicon) with PyTorch MPS fallback.

Usage:
    python experiments/train_math_transformer.py --op add --max_digits 5
    python experiments/train_math_transformer.py --op mul --max_digits 3 --n_layers 4
    python experiments/train_math_transformer.py --op add --max_digits 5 --tokenizer plain  # no reversal
    python experiments/train_math_transformer.py --op add --max_digits 5 --pos_encoding position_coupling
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

# Vocab: 0-9, +, -, *, =, <pad>, <eos>, <bos>, | (scratchpad step separator)
TOKENS = list("0123456789+-*=") + ["<pad>", "<eos>", "<bos>", "|"]
TOK2ID = {t: i for i, t in enumerate(TOKENS)}
ID2TOK = {i: t for t, i in TOK2ID.items()}
VOCAB_SIZE = len(TOKENS)
PAD_ID = TOK2ID["<pad>"]
EOS_ID = TOK2ID["<eos>"]
BOS_ID = TOK2ID["<bos>"]
SEP_ID = TOK2ID["|"]
OP_MAP = {"add": "+", "sub": "-", "mul": "*"}


def encode(s: str) -> list[int]:
    return [TOK2ID[c] for c in s]


def decode(ids: list[int]) -> str:
    return "".join(ID2TOK[i] for i in ids if i not in (PAD_ID, EOS_ID, BOS_ID))


# ── Position Coupling ────────────────────────────────────────────────────────

# Position IDs for special tokens (low values, distinct from digit positions)
PC_BOS_POS = 0
PC_OP_POS = 1
PC_EQ_POS = 2
PC_EOS_POS = 3
PC_DIGIT_BASE = 4  # Digit significance s → position ID PC_DIGIT_BASE + s
PC_MAX_POS = 256  # Embedding table size for position coupling


def compute_position_coupling_ids(seq_str, training=True):
    """Compute position IDs for position coupling encoding.

    Digits of equal significance across operands and result share the same
    position ID, so the model learns to align by significance regardless of
    operand length.

    For addition with reversed output (e.g., "123+456=975"):
    - Operand digits are MSB-first: index i in k-digit number → significance k-1-i
    - Result digits are LSB-first (reversed): index j → significance j
    - Special tokens (<bos>, +, =, <eos>) get unique position IDs

    During training, a random offset in [1, 100] is added to all position IDs
    so the model cannot memorize absolute positions.
    """
    op_idx = next(i for i, c in enumerate(seq_str) if c in "+-*")
    eq_idx = seq_str.index("=")

    op1 = seq_str[:op_idx]
    op2 = seq_str[op_idx + 1 : eq_idx]
    result = seq_str[eq_idx + 1 :]

    pos_ids = [PC_BOS_POS]

    # Op1 (MSB-first): digit at index i → significance (len-1-i)
    for i in range(len(op1)):
        pos_ids.append(PC_DIGIT_BASE + len(op1) - 1 - i)

    pos_ids.append(PC_OP_POS)

    # Op2 (MSB-first)
    for i in range(len(op2)):
        pos_ids.append(PC_DIGIT_BASE + len(op2) - 1 - i)

    pos_ids.append(PC_EQ_POS)

    # Result (LSB-first / reversed): digit at index j → significance j
    for j in range(len(result)):
        pos_ids.append(PC_DIGIT_BASE + j)

    pos_ids.append(PC_EOS_POS)

    if training:
        offset = random.randint(1, 100)
        pos_ids = [p + offset for p in pos_ids]

    return pos_ids


def compute_input_position_ids(inp_str):
    """Compute position IDs for input-only string (for autoregressive eval).

    inp_str looks like "123+456=" (includes trailing =).
    Returns position IDs including BOS at the start.
    """
    op_idx = next(i for i, c in enumerate(inp_str) if c in "+-*")
    # eq_idx is the last char
    op1 = inp_str[:op_idx]
    op2 = inp_str[op_idx + 1 : -1]  # exclude trailing '='

    pos_ids = [PC_BOS_POS]

    for i in range(len(op1)):
        pos_ids.append(PC_DIGIT_BASE + len(op1) - 1 - i)

    pos_ids.append(PC_OP_POS)

    for i in range(len(op2)):
        pos_ids.append(PC_DIGIT_BASE + len(op2) - 1 - i)

    pos_ids.append(PC_EQ_POS)

    return pos_ids


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


def generate_scratchpad_mul(a: int, b: int) -> str:
    """Generate scratchpad steps for a * b using aligned reversed partial products.

    Format: partial1|partial2|...|final_answer (all reversed/LSB-first)
    Each partial product = (digit_i of b) * a, shifted by i positions.
    Shift is represented as i leading zeros in the reversed representation.

    Example: 12 * 34 = 408
      4*12=48 (shift 0) reversed: "84"
      3*12=36 (shift 1) → 360 reversed: "063"
      final: 408 reversed: "804"
      Output: "84|063|804"
    """
    b_digits = [int(d) for d in str(b)][::-1]  # LSB-first digits of b
    partial_products = []
    for i, d in enumerate(b_digits):
        pp = d * a  # single-digit * multi-digit
        shifted = pp * (10**i)  # shift by position
        pp_str = str(shifted)[::-1]  # reverse
        # Ensure leading zeros for shift (reversed = trailing zeros become leading)
        # The reversed shifted value naturally has the right zeros
        partial_products.append(pp_str)

    result = a * b
    result_str = str(result)[::-1]
    return "|".join(partial_products) + "|" + result_str


def generate_example(
    op: str,
    max_digits: int,
    reverse_output: bool = True,
    balanced_carry: bool = True,
    scratchpad: bool = False,
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

    if scratchpad and op == "mul" and a > 0 and b > 0:
        # Scratchpad format: partial products + final answer (all reversed)
        scratch_str = generate_scratchpad_mul(a, b)
        seq = f"{a}{op_sym}{b}={scratch_str}"
    else:
        result_str = str(result)
        if reverse_output:
            result_str = result_str[::-1]
        seq = f"{a}{op_sym}{b}={result_str}"

    input_part = f"{a}{op_sym}{b}="
    return input_part, seq, result


def generate_example_exact_digits(
    op: str, n_digits: int, reverse_output: bool = True
) -> tuple[str, str, int]:
    """Generate one example where both operands have exactly n_digits digits."""
    lo = 10 ** (n_digits - 1) if n_digits > 1 else 0
    hi = 10**n_digits - 1
    a = random.randint(lo, hi)
    b = random.randint(lo, hi)

    if op == "add":
        result = a + b
    elif op == "sub":
        if a < b:
            a, b = b, a
        result = a - b
    elif op == "mul":
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
    scratchpad: bool = False,
) -> list[tuple[str, str, int]]:
    """Generate a dataset of arithmetic examples."""
    seen = set()
    data = []
    attempts = 0
    while len(data) < n_samples and attempts < n_samples * 10:
        attempts += 1
        inp, seq, ans = generate_example(
            op, max_digits, reverse_output, balanced_carry, scratchpad
        )
        if seq not in seen:
            seen.add(seq)
            data.append((inp, seq, ans))
    return data


def generate_ood_dataset(
    op: str, n_digits: int, n_samples: int, reverse_output: bool = True
) -> list[tuple[str, str, int]]:
    """Generate OOD test examples with exactly n_digits per operand."""
    seen = set()
    data = []
    attempts = 0
    while len(data) < n_samples and attempts < n_samples * 10:
        attempts += 1
        inp, seq, ans = generate_example_exact_digits(op, n_digits, reverse_output)
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


def pad_and_encode_with_positions(
    sequences: list[str], max_len: int, training: bool = True
) -> tuple[list[list[int]], list[list[int]]]:
    """Encode sequences and compute position coupling IDs, padded to max_len."""
    encoded = []
    positions = []
    for seq in sequences:
        ids = [BOS_ID] + encode(seq) + [EOS_ID]
        pos = compute_position_coupling_ids(seq, training=training)

        ids = ids[:max_len]
        pos = pos[:max_len]

        ids += [PAD_ID] * (max_len - len(ids))
        pos += [0] * (max_len - len(pos))

        encoded.append(ids)
        positions.append(pos)

    return encoded, positions


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
            self,
            n_layers: int,
            dim: int,
            n_heads: int,
            ff_dim: int,
            max_seq_len: int,
            pos_encoding: str = "learned",
        ):
            super().__init__()
            self.tok_emb = nn.Embedding(VOCAB_SIZE, dim)
            pos_emb_size = (
                PC_MAX_POS if pos_encoding == "position_coupling" else max_seq_len
            )
            self.pos_emb = nn.Embedding(pos_emb_size, dim)
            self.layers = [
                TransformerBlock(dim, n_heads, ff_dim) for _ in range(n_layers)
            ]
            self.norm = RMSNorm(dim)
            self.output = nn.Linear(dim, VOCAB_SIZE, bias=False)
            self.max_seq_len = max_seq_len
            self.pos_encoding = pos_encoding

        def __call__(self, x, position_ids=None):
            B, T = x.shape
            if position_ids is not None:
                positions = position_ids
            else:
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
        use_scratchpad = args.scratchpad
        use_pc = args.pos_encoding == "position_coupling"

        print(f"Generating {args.train_samples} training examples...")
        if use_scratchpad:
            print("Scratchpad mode enabled for multiplication")
        if use_pc:
            print("Position coupling enabled")
        train_data = generate_dataset(
            args.op,
            args.max_digits,
            args.train_samples,
            reverse,
            balanced,
            use_scratchpad,
        )
        test_data = generate_dataset(
            args.op,
            args.max_digits,
            args.test_samples,
            reverse,
            balanced,
            use_scratchpad,
        )
        print(f"Generated {len(train_data)} train, {len(test_data)} test examples")

        # Determine max sequence length
        max_len = (
            max(len(seq) for _, seq, _ in train_data + test_data) + 2
        )  # +2 for BOS/EOS
        seq_cap = 128 if use_scratchpad else 64
        max_len = min(max_len, seq_cap)

        # For OOD eval, we need a larger max_len
        eval_max_digits = args.eval_max_digits or args.max_digits
        if eval_max_digits > args.max_digits:
            # Max seq: BOS + digits + op + digits + = + (digits+1) + EOS
            ood_max_len = (
                2
                + eval_max_digits
                + 1
                + eval_max_digits
                + 1
                + (eval_max_digits + 1)
                + 2
            )
            max_len = max(max_len, ood_max_len)

        # Encode tokens (position IDs are recomputed each epoch for fresh random offsets)
        if use_pc:
            train_seqs, _ = pad_and_encode_with_positions(
                [seq for _, seq, _ in train_data], max_len, training=True
            )
        else:
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
            pos_encoding=args.pos_encoding,
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
        def loss_fn(model, x, pos=None):
            logits = model(
                x[:, :-1], position_ids=pos[:, :-1] if pos is not None else None
            )
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
            "pos_encoding": args.pos_encoding,
            "tokenizer": args.tokenizer,
            "balanced_carry": args.balanced_carry,
            "scratchpad": use_scratchpad,
            "max_seq_len": max_len,
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

            # For position coupling, re-compute position IDs each epoch
            # (different random offsets per epoch)
            if use_pc:
                _, new_pos = pad_and_encode_with_positions(
                    [seq for _, seq, _ in train_data], max_len, training=True
                )
                train_p_shuffled = mx.array(np.array(new_pos, dtype=np.int32))[
                    mx.array(perm)
                ]
            else:
                train_p_shuffled = None

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_data), args.batch_size):
                batch = train_x_shuffled[i : i + args.batch_size]
                if batch.shape[0] == 0:
                    continue

                batch_pos = (
                    train_p_shuffled[i : i + args.batch_size]
                    if train_p_shuffled is not None
                    else None
                )

                loss, grads = loss_and_grad(model, batch, batch_pos)
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
                    model,
                    test_inputs,
                    test_answers,
                    max_len,
                    reverse,
                    args.op,
                    use_scratchpad,
                    pos_encoding=args.pos_encoding,
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

        # OOD evaluation
        if eval_max_digits > args.max_digits:
            print(f"\n{'=' * 70}")
            print(
                f"OOD Length Generalization Evaluation (train max_digits={args.max_digits})"
            )
            print(f"{'=' * 70}")
            ood_results = {}
            for nd in range(1, eval_max_digits + 1):
                ood_data = generate_ood_dataset(args.op, nd, 500, reverse)
                ood_inputs = [inp for inp, _, _ in ood_data]
                ood_answers = [ans for _, _, ans in ood_data]
                acc, _ = evaluate_mlx(
                    model,
                    ood_inputs,
                    ood_answers,
                    max_len,
                    reverse,
                    args.op,
                    False,
                    pos_encoding=args.pos_encoding,
                )
                ood_results[nd] = round(acc, 4)
                in_dist = "ID" if nd <= args.max_digits else "OOD"
                print(f"  {nd:2d}-digit ({in_dist}): {acc:.2%}")
            results["ood_accuracy"] = ood_results

        return results

    def evaluate_mlx(
        model,
        test_inputs,
        test_answers,
        max_len,
        reverse,
        op,
        scratchpad=False,
        pos_encoding="learned",
    ):
        """Evaluate model accuracy via greedy autoregressive decoding."""
        correct = 0
        total = 0
        digit_correct = {}
        digit_total = {}
        use_pc = pos_encoding == "position_coupling"

        for inp_str, true_answer in zip(test_inputs, test_answers):
            # Encode input
            ids = [BOS_ID] + encode(inp_str)
            ids_arr = mx.array([ids])

            # Position IDs for position coupling
            if use_pc:
                input_pos = compute_input_position_ids(inp_str)
                n_result_tokens = 0

            # Autoregressive generation
            for _ in range(max_len - len(ids)):
                # Build position IDs
                if use_pc:
                    cur_pos = input_pos + [
                        PC_DIGIT_BASE + j for j in range(n_result_tokens)
                    ]
                    cur_pos_padded = cur_pos + [0] * (max_len - len(cur_pos))
                    cur_pos_padded = cur_pos_padded[:max_len]
                    pos_arr = mx.array([cur_pos_padded])
                else:
                    pos_arr = None

                # Pad to generate
                padded = mx.pad(
                    ids_arr,
                    [(0, 0), (0, max_len - ids_arr.shape[1])],
                    constant_values=PAD_ID,
                )
                logits = model(padded, position_ids=pos_arr)
                next_logit = logits[0, ids_arr.shape[1] - 1]
                next_id = mx.argmax(next_logit).item()

                if next_id == EOS_ID or next_id == PAD_ID:
                    break
                ids_arr = mx.concatenate([ids_arr, mx.array([[next_id]])], axis=1)
                if use_pc:
                    n_result_tokens += 1

            # Decode prediction
            pred_ids = ids_arr[0, len([BOS_ID] + encode(inp_str)) :].tolist()
            pred_str = decode(pred_ids)

            # For scratchpad, extract final answer (after last |)
            if scratchpad and "|" in pred_str:
                pred_str = pred_str.split("|")[-1]

            # Reverse back if needed (always reversed in scratchpad mode too)
            if reverse or scratchpad:
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
        def __init__(
            self, n_layers, dim, n_heads, ff_dim, max_seq_len, pos_encoding="learned"
        ):
            super().__init__()
            self.tok_emb = torch_nn.Embedding(VOCAB_SIZE, dim)
            pos_emb_size = (
                PC_MAX_POS if pos_encoding == "position_coupling" else max_seq_len
            )
            self.pos_emb = torch_nn.Embedding(pos_emb_size, dim)

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
            self.pos_encoding = pos_encoding

        def forward(self, x, position_ids=None):
            B, T = x.shape
            device = x.device
            if position_ids is not None:
                positions = position_ids
            else:
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
        use_scratchpad = args.scratchpad
        use_pc = args.pos_encoding == "position_coupling"

        print(f"Generating {args.train_samples} training examples...")
        if use_scratchpad:
            print("Scratchpad mode enabled for multiplication")
        if use_pc:
            print("Position coupling enabled")
        train_data = generate_dataset(
            args.op,
            args.max_digits,
            args.train_samples,
            reverse,
            balanced,
            use_scratchpad,
        )
        test_data = generate_dataset(
            args.op,
            args.max_digits,
            args.test_samples,
            reverse,
            balanced,
            use_scratchpad,
        )
        print(f"Generated {len(train_data)} train, {len(test_data)} test examples")

        max_len = max(len(seq) for _, seq, _ in train_data + test_data) + 2
        seq_cap = 128 if use_scratchpad else 64
        max_len = min(max_len, seq_cap)

        # For OOD eval, we need a larger max_len
        eval_max_digits = args.eval_max_digits or args.max_digits
        if eval_max_digits > args.max_digits:
            ood_max_len = (
                2
                + eval_max_digits
                + 1
                + eval_max_digits
                + 1
                + (eval_max_digits + 1)
                + 2
            )
            max_len = max(max_len, ood_max_len)

        # Encode tokens (position IDs are recomputed each epoch for fresh random offsets)
        if use_pc:
            train_seqs, _ = pad_and_encode_with_positions(
                [seq for _, seq, _ in train_data], max_len, training=True
            )
        else:
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
            pos_encoding=args.pos_encoding,
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
            "pos_encoding": args.pos_encoding,
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

            # Re-compute position IDs each epoch for fresh random offsets
            if use_pc:
                _, new_pos = pad_and_encode_with_positions(
                    [seq for _, seq, _ in train_data], max_len, training=True
                )
                train_p_shuffled = torch.tensor(
                    new_pos, dtype=torch.long, device=device
                )[perm]
            else:
                train_p_shuffled = None

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_data), args.batch_size):
                batch = train_x_shuffled[i : i + args.batch_size]
                if batch.shape[0] == 0:
                    continue

                batch_pos = (
                    train_p_shuffled[i : i + args.batch_size]
                    if train_p_shuffled is not None
                    else None
                )

                logits = model(
                    batch[:, :-1],
                    position_ids=(batch_pos[:, :-1] if batch_pos is not None else None),
                )
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
                        use_scratchpad,
                        pos_encoding=args.pos_encoding,
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

        # OOD evaluation
        if eval_max_digits > args.max_digits:
            print(f"\n{'=' * 70}")
            print(
                f"OOD Length Generalization Evaluation (train max_digits={args.max_digits})"
            )
            print(f"{'=' * 70}")
            ood_results = {}
            model.eval()
            with torch.no_grad():
                for nd in range(1, eval_max_digits + 1):
                    ood_data = generate_ood_dataset(args.op, nd, 500, reverse)
                    ood_inputs = [inp for inp, _, _ in ood_data]
                    ood_answers = [ans for _, _, ans in ood_data]
                    acc, _ = evaluate_pytorch(
                        model,
                        ood_inputs,
                        ood_answers,
                        max_len,
                        reverse,
                        args.op,
                        device,
                        False,
                        pos_encoding=args.pos_encoding,
                    )
                    ood_results[nd] = round(acc, 4)
                    in_dist = "ID" if nd <= args.max_digits else "OOD"
                    print(f"  {nd:2d}-digit ({in_dist}): {acc:.2%}")
            results["ood_accuracy"] = ood_results

        return results

    def evaluate_pytorch(
        model,
        test_inputs,
        test_answers,
        max_len,
        reverse,
        op,
        device,
        scratchpad=False,
        pos_encoding="learned",
    ):
        correct = 0
        total = 0
        digit_correct = {}
        digit_total = {}
        use_pc = pos_encoding == "position_coupling"

        for inp_str, true_answer in zip(test_inputs, test_answers):
            ids = [BOS_ID] + encode(inp_str)
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)

            if use_pc:
                input_pos = compute_input_position_ids(inp_str)
                n_result_tokens = 0

            for _ in range(max_len - len(ids)):
                if use_pc:
                    cur_pos = input_pos + [
                        PC_DIGIT_BASE + j for j in range(n_result_tokens)
                    ]
                    cur_pos_padded = cur_pos + [0] * (max_len - len(cur_pos))
                    cur_pos_padded = cur_pos_padded[:max_len]
                    pos_t = torch.tensor(
                        [cur_pos_padded], dtype=torch.long, device=device
                    )
                else:
                    pos_t = None

                padded = torch.nn.functional.pad(
                    ids_t, (0, max_len - ids_t.shape[1]), value=PAD_ID
                )
                logits = model(padded, position_ids=pos_t)
                next_logit = logits[0, ids_t.shape[1] - 1]
                next_id = torch.argmax(next_logit).item()

                if next_id == EOS_ID or next_id == PAD_ID:
                    break
                ids_t = torch.cat(
                    [ids_t, torch.tensor([[next_id]], device=device)], dim=1
                )
                if use_pc:
                    n_result_tokens += 1

            pred_ids = ids_t[0, len([BOS_ID] + encode(inp_str)) :].tolist()
            pred_str = decode(pred_ids)

            # For scratchpad, extract final answer (after last |)
            if scratchpad and "|" in pred_str:
                pred_str = pred_str.split("|")[-1]

            if reverse or scratchpad:
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
    parser.add_argument(
        "--scratchpad",
        action="store_true",
        default=False,
        help="Enable scratchpad (intermediate steps) for multiplication",
    )
    parser.add_argument(
        "--pos_encoding",
        choices=["learned", "position_coupling"],
        default="learned",
        help="Positional encoding: learned (standard APE) or position_coupling (Cho 2024)",
    )
    parser.add_argument(
        "--eval_max_digits",
        type=int,
        default=None,
        help="Max digits for OOD evaluation (default: same as max_digits, no OOD eval)",
    )

    # Architecture
    parser.add_argument(
        "--n_layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument(
        "--n_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--dim", "--d_model", type=int, default=256, help="Embedding dimension"
    )
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
    pos_tag = f"_{args.pos_encoding}" if args.pos_encoding != "learned" else ""
    fname = f"{args.op}_{args.max_digits}d_{args.tokenizer}_{args.n_layers}L{args.n_heads}H{args.dim}D{pos_tag}{tag}.json"
    out_path = os.path.join(args.output_dir, fname)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
