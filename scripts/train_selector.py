"""Train the LLM selector (termmap/selector.py) on a pipeline run's cached lists, then score its test fold.

    pip install peft accelerate          # once; check `python -c "import torch; print(torch.__version__)"` is unchanged
    hf download Qwen/Qwen2.5-7B-Instruct
    python -m scripts.train_selector --run v2 --fold 0                     # train on folds != 0, score fold 0
    python -m scripts.pipeline --run v2 --folds 0 --stages s5              # stacker with the selector feature
    python -m scripts.train_selector --run v2 --fold 0 --base Qwen/Qwen2.5-0.5B-Instruct --max-steps 20  # smoke

Training data: every training term of the fold's training side with its OUT-OF-FOLD ranker top-k (the lists
the selector will see at test time), both coding systems in one model, options shuffled per term.
Checkpoints: loss every 25 steps; every --eval-every steps the selector's accuracy on a held-out 5% of the
training terms is compared with the ranker's on the same terms. Gate: if the first check is more than
--gate-drop below the ranker, training stops (exit code 2) instead of burning hours.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd

from termmap import config
from termmap.engine import CodeSystem
from termmap.selector import K_SEL, chat, letter_ids, make_examples


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_examples(cache, fold: int, k: int):
    """Training examples (shuffled options), dev examples (5% of training terms) and test examples per system."""
    train_ex, dev_ex, test_ex = [], [], {}
    for system in config.SYSTEMS:
        cs = CodeSystem.load(system)
        trn = pd.read_parquet(cache / f"f{fold}_{system}_train.parquet")
        ti = pd.read_parquet(cache / f"f{fold}_{system}_train_items.parquet").set_index("item")
        texts = {i: (ti.entity[i], ti.chunk_text[i]) for i in ti.index}
        exs = make_examples(trn, texts, cs, system, "oof_rk", k)
        # Ranker's answer on the same list (first option before shuffling): for the dev comparison.
        rk_top = trn.sort_values(["item", "oof_rk", "code"], ascending=[True, False, True]).groupby("item").head(1)
        rk_hit = dict(zip(rk_top.item, rk_top.label))
        for e in exs:
            e["system"], e["ranker_hit"] = system, int(rk_hit.get(e["item"], 0))
            (dev_ex if e["item"] % 20 == 0 else train_ex).append(e)
        test = pd.read_parquet(cache / f"f{fold}_{system}_test.parquet")
        items = pd.read_parquet(cache / f"f{fold}_{system}_test_items.parquet")
        tt = {i: (items.entity[i], items.chunk_text[i]) for i in items.index}
        test_ex[system] = (make_examples(test, tt, cs, system, "score", k, shuffle=False), test, len(items))
    return train_ex, dev_ex, test_ex


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=int, default=K_SEL)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=150, help="optimizer steps between dev checks")
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--gate-drop", type=float, default=0.05)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--test-limit", type=int, default=0, help="smoke tests: score only the first N test terms")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    cache = config.CACHE_DIR / "stages" / args.run
    out = args.out or str(cache / f"selector_f{args.fold}")
    train_ex, dev_ex, test_ex = load_examples(cache, args.fold, args.k)
    rnd = random.Random(config.SEED)
    rnd.shuffle(train_ex)
    rnd.shuffle(dev_ex)
    dev_ex = dev_ex[:args.eval_n]
    log(f"selector: {len(train_ex)} training terms, {len(dev_ex)} dev terms, base={args.base}")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    def encode(ex):
        prompt = tok(chat(tok, ex["prompt"]), add_special_tokens=False)["input_ids"]
        ans = tok(ex["answer"], add_special_tokens=False)["input_ids"]
        return prompt + ans, [-100] * len(prompt) + ans

    def collate(batch):
        enc = [encode(e) for e in batch]
        n = max(len(i) for i, _ in enc)
        ids = torch.full((len(enc), n), tok.pad_token_id)
        lab = torch.full((len(enc), n), -100)
        att = torch.zeros((len(enc), n), dtype=torch.long)
        for r, (i, l) in enumerate(enc):
            ids[r, :len(i)], lab[r, :len(l)], att[r, :len(i)] = torch.tensor(i), torch.tensor(l), 1
        return ids.to(device), att.to(device), lab.to(device)

    def dev_accuracy():
        model.eval()
        tok.padding_side = "left"
        hit = 0
        with torch.inference_mode():
            for s in range(0, len(dev_ex), 8):
                exs = dev_ex[s:s + 8]
                enc = tok([chat(tok, e["prompt"]) for e in exs], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(device)
                logits = model(**enc).logits[:, -1, :].float()
                for e, lg in zip(exs, logits):
                    ids = letter_ids(tok, len(e["codes"]) + 1)
                    hit += int(ids[int(torch.argmax(lg[ids]))] == tok.encode(e["answer"], add_special_tokens=False)[0])
        tok.padding_side = "right"
        model.train()
        return hit / max(len(dev_ex), 1)

    ranker_dev = float(np.mean([e["ranker_hit"] for e in dev_ex])) if dev_ex else float("nan")
    steps_per_epoch = math.ceil(len(train_ex) / (args.batch * args.accum))
    total = args.max_steps or int(args.epochs * steps_per_epoch)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    tok.padding_side = "right"
    model.train()
    step, micro, run_loss, t0, best = 0, 0, 0.0, time.time(), -1.0
    log(f"training {total} optimizer steps (batch {args.batch} x accum {args.accum}); ranker dev acc {ranker_dev:.3f}")
    while step < total:
        for s in range(0, len(train_ex), args.batch):
            ids, att, lab = collate(train_ex[s:s + args.batch])
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / args.accum
            loss.backward()
            run_loss += loss.item()
            micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 25 == 0:
                el = time.time() - t0
                log(f"step {step}/{total} loss {run_loss / 25:.4f}  {el / step:.1f}s/step  ETA {el / step * (total - step) / 60:.0f} min")
                run_loss = 0.0
            if step % args.eval_every == 0 or step == total:
                acc = dev_accuracy()
                log(f"CHECKPOINT step {step}: selector dev acc {acc:.3f} vs ranker {ranker_dev:.3f}")
                if step == args.eval_every and acc < ranker_dev - args.gate_drop and not args.no_gate:
                    log(f"GATE FAIL: selector {acc:.3f} < ranker {ranker_dev:.3f} - {args.gate_drop}; stopping")
                    sys.exit(2)
                if acc >= best:
                    best = acc
                    model.save_pretrained(out)
                    tok.save_pretrained(out)
                    json.dump({"base": args.base, "k": args.k, "fold": args.fold, "run": args.run, "step": step,
                               "dev_acc": acc, "ranker_dev_acc": ranker_dev}, open(os.path.join(out, "selector.json"), "w"))
            if step >= total:
                break
    log(f"training done in {(time.time() - t0) / 60:.1f} min; best dev acc {best:.3f} -> {out}")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None

    from termmap.selector import Selector
    sel = Selector(out)
    for system, (exs, test, n) in test_ex.items():
        if args.test_limit:
            exs, n = exs[:args.test_limit], min(n, args.test_limit)
            test = test[test.item.isin({e["item"] for e in exs})]
        S = sel.score_examples(exs)
        S.to_parquet(cache / f"f{args.fold}_{system}_sel.parquet", index=False)
        F = test.merge(S, on=["item", "code"], how="inner")
        pick = F.sort_values(["item", "sel_p", "code"], ascending=[True, False, True]).groupby("item").head(1)
        rk = test.sort_values(["item", "score", "code"], ascending=[True, False, True]).groupby("item").head(1)
        log(f"fold {args.fold} [{system}] selector top-1 {pick.label.sum() / n:.4f} vs ranker {rk.label.sum() / n:.4f} "
            f"(n={n}) -> {cache / f'f{args.fold}_{system}_sel.parquet'}")


if __name__ == "__main__":
    main()
