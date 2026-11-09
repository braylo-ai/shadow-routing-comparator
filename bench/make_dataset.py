"""Generate the synthetic replay dataset (seeded). Output: data/synthetic_traffic.jsonl"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {
    "order_status": ["Where is my order?", "Has order {o} shipped yet?", "When will {o} arrive?"],
    "refund": ["I want a refund for {o}", "Can I return order {o}?", "Refund please, the item in {o} broke"],
    "faq": ["What are your opening hours?", "Do you ship to Canada?", "How long is the warranty?"],
    "account": ["Change my email address", "Update the email on my account", "I need a new login email"],
    "smalltalk": ["hi", "thanks!", "you are great"],
}
WEIGHTS = {"order_status": 0.30, "refund": 0.20, "faq": 0.25, "account": 0.15, "smalltalk": 0.10}


def main(n=2000, seed=11):
    rng = random.Random(seed)
    cats, w = zip(*WEIGHTS.items())
    lines = []
    for i in range(n):
        c = rng.choices(cats, w)[0]
        o = f"ORD-{rng.randint(10000, 99999)}"
        lines.append(json.dumps({"id": f"req-{i:05d}", "category": c, "prompt": rng.choice(PROMPTS[c]).format(o=o),
                                 "order_id": o, "user_id": f"user-{rng.randint(1, 400)}", "q": c}))
    (ROOT / "data" / "synthetic_traffic.jsonl").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
