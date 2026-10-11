"""Write identity_check.md from validators_now.csv / validatorSummaries cache / post_credit_votes.csv."""
import json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FOUR = ["0x263294039413b96d25e4173a5f7599f8b3801504", "0xda6816df552c3f9e0fb64979fb357800d690d79b",
        "0x58e1b0e63c905d5982324fcd9108582623b8132e", "0xef2364db5db6f5539aa0bc111771a94ee47637fc"]
EXPS = [23, 24, 32, 64, 101, 102]
AFTER_N = 15

n = pd.read_csv(os.path.join(HERE, "validators_now.csv"), dtype=str)
raw = open(os.path.join(HERE, "cache", "validatorSummaries_20261010.json")).read().lower()
v = pd.read_csv(os.path.join(HERE, "votes_raw.csv"))
cb = pd.read_csv(os.path.join(HERE, "credit_blocks.csv")).set_index("experiment_id")

L = ["# 投票者4アドレスの身元確認と着金後の票（2026-10-10）", "",
     "## 1. validators_now.csv との突合",
     f"照合先: validators_now.csv の全{len(n.columns)}列（{', '.join(n.columns)}）×{len(n)}行。"
     "加えて、取得したままの validatorSummaries JSON 全体（description と stats を含む）も文字列で検索した。"
     "大文字と小文字は区別せず、0x を除いた40桁で照合した。", "",
     "| アドレス | 一致した列（validators_now.csv） | 生JSON内での出現 |", "|---|---|---|"]
for a in FOUR:
    cols = sorted({c for c in n.columns for x in n[c].fillna("") if a[2:] in x.lower()})
    L.append(f"| `{a}` | {', '.join(cols) if cols else 'なし'} | {'あり' if a[2:] in raw else 'なし'} |")
L += ["", "→ 4アドレスはどれも、現在（2026-10-10）のバリデータの validator 列・signer 列のどちらにも含まれない。", "",
      "## 2. 着金ブロックに4票目がなかった6件：着金後15ブロックの票"]

st_path = os.path.join(HERE, "logs", "post_credit_status.json")
pc_path = os.path.join(HERE, "post_credit_votes.csv")
if not os.path.exists(st_path):
    L += ["", "未実行（explorer の1日上限が解けるのを待っている）。"]
else:
    st = json.load(open(st_path))
    pc = pd.read_csv(pc_path)
    L += ["", f"取得したブロック数: {st['blocks_fetched']} / {st['blocks_planned']}"
          + (f"（**上限に当たって途中で停止**: {st['stopped_by_quota']}）" if st["stopped_by_quota"] else "（全件取得）"),
          f"件ごとに取得したブロック数: {st['per_exp_fetched']}", ""]
    pc["is_target_deposit"] = pc.is_target_deposit.astype(bool)
    pc["in_known_4"] = pc.in_known_4.astype(bool)
    tgt = pc[pc.is_target_deposit]
    oth = pc[~pc.is_target_deposit]
    L += ["### 当該送金への票（着金ブロック以前の3票と、着金後の票）", "",
          "| exp | 着金前の投票者（3票） | 着金後の票: アドレス, +ブロック, +ms, error |", "|---|---|---|"]
    for e in EXPS:
        pre = ", ".join(f"`{x[:6]}…{x[-4:]}`" for x in v[v.experiment_id == e].validator)
        post = tgt[tgt.experiment_id == e]
        ps = "; ".join(f"`{r.validator[:6]}…{r.validator[-4:]}` +{r.offset_from_credit} blk, +{r.dt_from_credit_ms} ms, {r.tx_error or 'なし'}"
                       for r in post.itertuples()) or ("なし（取得した範囲には票がなかった）" if st["per_exp_fetched"][str(e)] else "未取得")
        L.append(f"| {e} | {pre} | {ps} |")
    newaddr = sorted(set(pc.validator.str.lower()) - set(FOUR))
    L += ["", f"- 当該送金への着金後の票: {len(tgt)} 票。そのうち4アドレス以外からの票: {int((~tgt.in_known_4).sum())} 票",
          f"- 同じ範囲に含まれていた、他の送金への VoteEthDepositAction: {len(oth)} 票"
          f"（投票者: {sorted(set(oth.validator)) if len(oth) else 'なし'}）",
          f"- 取得した範囲で見つかった、4アドレス以外の投票者: {newaddr if newaddr else 'なし'}"]
    missing4 = [e for e in EXPS if st["per_exp_fetched"][str(e)] == AFTER_N and len(set(v[v.experiment_id == e].validator.str.lower()) | set(tgt[tgt.experiment_id == e].validator.str.lower())) < 4]
    L += [f"- 着金後15ブロックをすべて取得した件のうち、4アドレスすべての票がそろわない件: {missing4 if missing4 else 'なし'}", "",
          "詳細: `post_credit_votes.csv`（全行。is_target_deposit, in_known_4, tx_error の列あり）"]
# ---- section 3: userDetails (one request per address, cached)
import collections, datetime as _dt
_f = lambda ms: _dt.datetime.fromtimestamp(ms / 1000, _dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
L += ["", "## 3. 直近の取引（explorer RPC userDetails、2026-10-10 に各アドレス1要求）", "",
      "userDetails は直近300件しか返さない。生の応答は `cache/userDetails/<addr>.json` に保存してある。", "",
      "| アドレス | 件数 | 期間 (UTC) | 取引の種類（件数） | VoteEthDepositAction |", "|---|---|---|---|---|"]
_dep = {}
for a in FOUR:
    p = os.path.join(HERE, "cache", "userDetails", f"{a}.json")
    if not os.path.exists(p):
        L.append(f"| `{a}` | 未取得 | | | |"); continue
    t = json.load(open(p)).get("txs", [])
    c = collections.Counter(x["action"]["type"] for x in t)
    d = [x for x in t if x["action"]["type"] == "VoteEthDepositAction"]
    _dep[a] = {x["action"]["ethId"]["h"] for x in d}
    dv = (f"{len(d)}票（送金{len(_dep[a])}件、error {sum(x['error'] is not None for x in d)}票）、{_f(d[-1]['time'])}〜{_f(d[0]['time'])}"
          if d else "なし（この範囲には含まれない）")
    L.append(f"| `{a}` | {len(t)} | {_f(min(x['time'] for x in t))}〜{_f(max(x['time'] for x in t))} | "
             + ", ".join(f"{k} {v}" for k, v in c.most_common()) + f" | {dv} |")
_k = [a for a in _dep if _dep[a]]
if _k:
    L += ["", f"- 預金投票をしている{len(_k)}アドレスについて、投票先の送金の和集合は{len(set.union(*[_dep[a] for a in _k]))}件、"
          f"共通部分は{len(set.intersection(*[_dep[a] for a in _k]))}件だった。"]
L += ["",
      "**(a) 今も VoteEthDepositAction を出しているか**",
      "- `0x2632…1504`、`0xda68…d79b`、`0xef23…37fc` の3つは、**今も出している**。2026-10-10 の約1.5時間で各59票。3者とも同じ59件の送金に投票していて、error はない。",
      "- `0x58e1…132e` の直近300件は約44秒分しかなく、中身は SetGlobalAction（オラクル価格の設定に見える）15件と evmRawTx 285件だった。この範囲に預金投票はないが、件数の窓が狭すぎるので「今は投票していない」とまでは言えない。",
      "",
      "**(b) 現在のバリデータ集合と無関係に続いているか**",
      "- 上の3アドレスは、現在の validatorSummaries（35件）の validator 列・signer 列のどちらにも含まれない（1節）。それでも、2025-11〜12 と同じ形の預金投票を今も続けている。",
      "- 同じ3アドレスは、出金の投票（VoteEthFinalizedWithdrawalAction、ValidatorSignWithdrawalAction）と、合意のチェックポイント投票（voteAbciDigest、voteAppHash、voteL1Hash）も出している。voteAbciDigest と voteL1Hash は、一覧にあるバリデータの signer（例：ValiDAO の `0x0000…8fde`）も出している種類の投票である。",
      "- したがって、この3つ（と `0x58e1…132e`）は、validatorSummaries に載っている委任型バリデータ集合とは別の固定の署名者集合として、ブリッジの投票（と一部のチェックポイント投票）を継続的に担っているように見える。",
      "- ただし、現在の送金に4者以外が投票しているかどうかは、userDetails だけでは判定できない（該当ブロックの取得が必要）。",
      ]
open(os.path.join(HERE, "identity_check.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
