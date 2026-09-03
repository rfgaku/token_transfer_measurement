#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t4_cctp/scheduler/scheduler.py — T4 CCTP 測定 無人スケジューラ。

役割（指示プロンプト §4 準拠）:
  検証済みの測定スクリプト2本（deposit_cctp_measure.py / withdraw_cctp_measure.py）を
  「都度サブプロセス起動」するだけのオーケストレータ。測定スクリプト本体は一切改変しない。
  本番CSV（result/T4_cctp/*.csv）は read-only（残数計算・成功判定にのみ使用）。

絶対ルール:
  - 測定スクリプトには手を入れない。--exp-id は渡さない（採番はスクリプト任せ）。
  - 実送信（--broadcast --prod）は承認ゲート（start_scheduler.sh / Claude の run_in_background）経由でのみ起動。
  - --plan-only / --summary は計画生成と承認サマリ表示のみで、送信は一切しない。

成功判定（指示 §4.3 + 実体調査の結論）:
  「対象方向の本番CSV行数が実行前比 +1」を主判定とし、**exit_code==0 を安全側ガードに併用**する。
  理由: withdraw は送信失敗時に行を保存してから sys.exit(1) するため、行数+1 のみでは
  送信失敗のゴミ行を成功と誤判定し、連続失敗HALTの安全機構が働かなくなる。exit_code は
  spec が「補助情報」と定める通りの使い方（主判定=行数+1の趣旨は維持）。
"""
import argparse
import csv
import json
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --- リポジトリルートを sys.path に追加（config import 用） ---
SCHED_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCHED_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from t4_cctp.deposit import config as dcfg   # noqa: E402
from t4_cctp.withdraw import config as wcfg   # noqa: E402

# =====================================================================
# 定数
# =====================================================================
JST = timezone(timedelta(hours=9))

DEP_MEASURE = REPO_ROOT / "t4_cctp" / "deposit" / "deposit_cctp_measure.py"
WIT_MEASURE = REPO_ROOT / "t4_cctp" / "withdraw" / "withdraw_cctp_measure.py"
DEP_CSV = dcfg.PROD_CSV
WIT_CSV = wcfg.PROD_CSV

LOG_DIR = SCHED_DIR / "logs"
RUNLOG = SCHED_DIR / "runlog.csv"
STATUS = SCHED_DIR / "STATUS"

SUBPROC_TIMEOUT = 1200          # 20分（通常1〜2分。WS credit検知ブロック最大600s+余裕）
DEFAULT_TARGET = 200
HORIZON_DAYS = 7
MIN_INTRACELL_GAP_S = 20 * 60   # 同一セル内 最低20分間隔（Iris処理サイクルとのエイリアシング回避）
DEP_WIT_GAP_RANGE = (120, 300)  # 往復内 dep→wit の間隔 2〜5分（一様ランダム）
PAST_SLOT_GRACE_S = 30 * 60     # planned_ts を30分超過したスロットは実行せずSKIP
N_SPARE_ROUNDTRIPS = 4          # 計画末尾の予備スロット（target ゲートで overshoot は抑止）
MAX_CONSEC_FAILS = 3            # 連続3件失敗で HALT

# 残高事前チェック閾値（指示 §4.3）
MIN_ARB_USDC = 6.0              # dep 1件 5 + 余裕
MIN_HC_SPOT_USDC = 6.0          # wit 原資 5 + 余裕
PER_DEP_ETH = 0.0000022         # dep 1件あたり概算ガス
ETH_SAFETY = 1.5

# コスト概算（承認サマリ用。1往復 ≈ 0.44 USDC）
COST_DEP_USDC = 0.24            # dep maxFee 概算
COST_WIT_USDC = 0.20           # wit forwarding 固定

# =====================================================================
# 時刻ユーティリティ
# =====================================================================
def now_ms() -> int:
    return int(time.time() * 1000)


def jst_str(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, JST).strftime("%Y-%m-%d %H:%M:%S")


def hm(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, JST).strftime("%H:%M")


# =====================================================================
# CSV 行数（本番CSV: read-only）
# =====================================================================
def csv_data_rows(path) -> int:
    """ヘッダを除くデータ行数。空行は無視。ファイル無しは 0。"""
    p = Path(path)
    if not p.exists():
        return 0
    with open(p, newline="") as f:
        rows = [r for r in csv.reader(f) if any((c or "").strip() for c in r)]
    return max(0, len(rows) - 1)


# =====================================================================
# 残高チェック（measure スクリプトは無改変。config 流用 + 最小ロジック自前実装）
# =====================================================================
_ERC20_BAL = [{
    "name": "balanceOf", "type": "function", "stateMutability": "view",
    "inputs": [{"name": "a", "type": "address"}],
    "outputs": [{"name": "", "type": "uint256"}],
}]


def _read_balances_once() -> dict:
    import requests
    from web3 import Web3
    from web3.middleware import geth_poa_middleware

    addr = Web3.to_checksum_address(os.environ["ARB_SENDER_ADDRESS"])
    w3 = Web3(Web3.HTTPProvider(dcfg.ARB_RPC_URL, request_kwargs={"timeout": 15}))
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    usdc = w3.eth.contract(address=Web3.to_checksum_address(dcfg.ARB_USDC_ADDRESS), abi=_ERC20_BAL)
    arb_usdc = usdc.functions.balanceOf(addr).call() / 1e6
    arb_eth = w3.eth.get_balance(addr) / 1e18

    spot_avail = 0.0
    r = requests.post(wcfg.HL_INFO_URL,
                      json={"type": "spotClearinghouseState", "user": addr.lower()},
                      timeout=15).json()
    for b in (r.get("balances") or []):
        if b.get("coin") == "USDC":
            spot_avail = float(b.get("total") or 0) - float(b.get("hold") or 0)
            break
    return {"arb_usdc": arb_usdc, "arb_eth": arb_eth, "hc_spot_usdc": spot_avail}


def read_balances(retries: int = 3):
    """成功時 dict、全失敗時 None（インフラ起因）。"""
    last = None
    for i in range(retries):
        try:
            return _read_balances_once()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    log(f"[balance] RPC 取得失敗（{retries}回）: {last}")
    return None


def precheck(direction: str, target: int):
    """戻り値: ('OK', bals) / ('FUNDS', detail) / ('INFRA', detail)
    FUNDS は HALT、INFRA は SKIP+fail 扱い（呼び出し側）。"""
    bals = read_balances()
    if bals is None:
        return ("INFRA", "balance RPC failed")
    if direction == "deposit":
        if bals["arb_usdc"] < MIN_ARB_USDC:
            return ("FUNDS", f"arb_usdc={bals['arb_usdc']:.4f} < {MIN_ARB_USDC}")
        dep_rem = max(1, target - csv_data_rows(DEP_CSV))
        need_eth = dep_rem * PER_DEP_ETH * ETH_SAFETY
        if bals["arb_eth"] < need_eth:
            return ("FUNDS", f"arb_eth={bals['arb_eth']:.6f} < need {need_eth:.6f} (dep_rem={dep_rem})")
    else:  # withdraw
        if bals["hc_spot_usdc"] < MIN_HC_SPOT_USDC:
            return ("FUNDS", f"hc_spot_usdc={bals['hc_spot_usdc']:.4f} < {MIN_HC_SPOT_USDC}")
    return ("OK", bals)


# =====================================================================
# 計画生成
# =====================================================================
def _gen_times_in_window(ws_ms: int, we_ms: int, k: int) -> list:
    """窓 [ws,we) に k 個の開始時刻を一様ランダム配置（最低20分間隔を確保）。"""
    if k <= 0:
        return []
    span = max(1, we_ms - ws_ms)
    sub = span / k
    times = []
    prev = ws_ms - MIN_INTRACELL_GAP_S * 1000
    for i in range(k):
        lo = ws_ms + int(i * sub)
        hi = ws_ms + int((i + 1) * sub)
        t = random.randint(lo, max(lo, hi - 1))
        if t < prev + MIN_INTRACELL_GAP_S * 1000:
            t = prev + MIN_INTRACELL_GAP_S * 1000
        if t >= we_ms:
            t = we_ms - 1
        times.append(t)
        prev = t
    return times


def _gen_cells(start_epoch_ms: int, horizon_days: int) -> list:
    """7日 × 4時間帯(00-06/06-12/12-18/18-24 JST) のセル。過去窓は除外、開始セルは start にクランプ。"""
    start_dt = datetime.fromtimestamp(start_epoch_ms / 1000, JST)
    day0 = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    floor_ms = start_epoch_ms + 5 * 60 * 1000  # 開始直後5分は計画に使わない
    cells = []
    for d in range(horizon_days):
        for band in range(4):
            ws = day0 + timedelta(days=d, hours=band * 6)
            we = ws + timedelta(hours=6)
            ws_ms = int(ws.timestamp() * 1000)
            we_ms = int(we.timestamp() * 1000)
            if we_ms <= floor_ms:
                continue  # 完全に過去
            eff_ws = max(ws_ms, floor_ms)
            cells.append({"label": f"D{d}-B{band}", "ws": eff_ws, "we": we_ms})
    return cells


def generate_main_plan(target: int, start_epoch_ms: int, horizon_days: int = HORIZON_DAYS) -> dict:
    random.seed(start_epoch_ms)
    dep_done = csv_data_rows(DEP_CSV)
    wit_done = csv_data_rows(WIT_CSV)
    dep_rem = max(0, target - dep_done)
    wit_rem = max(0, target - wit_done)
    n_round = min(dep_rem, wit_rem)
    dep_singles = max(0, dep_rem - wit_rem)
    wit_singles = max(0, wit_rem - dep_rem)

    types = (["roundtrip"] * n_round + ["dep_single"] * dep_singles + ["wit_single"] * wit_singles)
    random.shuffle(types)
    total = len(types)

    cells = _gen_cells(start_epoch_ms, horizon_days)
    if not cells:
        raise RuntimeError("計画可能なセルがありません（horizon 設定を確認）")

    # 各セルへ near-even 配分（端数を均等分散）
    N = len(cells)
    slots = []  # (epoch_ms, cell_label)
    for i, cell in enumerate(cells):
        k = (total * (i + 1)) // N - (total * i) // N
        for t in _gen_times_in_window(cell["ws"], cell["we"], k):
            slots.append((t, cell["label"]))
    slots.sort(key=lambda x: x[0])

    events = []
    for idx, (t, cell_label) in enumerate(slots):
        ev_type = types[idx] if idx < len(types) else "roundtrip"
        ev = {
            "event_id": idx + 1,
            "cell": cell_label,
            "type": ev_type,
            "planned_epoch_ms": t,
            "planned_ts_jst": jst_str(t),
            "spare": False,
        }
        if ev_type == "roundtrip":
            ev["dep_wit_gap_s"] = random.randint(*DEP_WIT_GAP_RANGE)
        events.append(ev)

    # 予備スロット（末尾に数件・target ゲートで overshoot 抑止）
    last_t = events[-1]["planned_epoch_ms"] if events else start_epoch_ms + 60 * 1000
    for s in range(N_SPARE_ROUNDTRIPS):
        last_t += 25 * 60 * 1000
        events.append({
            "event_id": len(events) + 1,
            "cell": "SPARE",
            "type": "roundtrip",
            "planned_epoch_ms": last_t,
            "planned_ts_jst": jst_str(last_t),
            "dep_wit_gap_s": random.randint(*DEP_WIT_GAP_RANGE),
            "spare": True,
        })

    return {
        "kind": "main",
        "created_epoch_ms": start_epoch_ms,
        "created_ts_jst": jst_str(start_epoch_ms),
        "target": target,
        "horizon_days": horizon_days,
        "dep_done_at_plan": dep_done,
        "wit_done_at_plan": wit_done,
        "n_roundtrip": n_round,
        "n_dep_single": dep_singles,
        "n_wit_single": wit_singles,
        "n_spare": N_SPARE_ROUNDTRIPS,
        "events": events,
    }


def generate_test_plan(target: int, start_in_min: float, roundtrips: int,
                       gap_min: float, gap_max: float) -> dict:
    start = now_ms() + int(start_in_min * 60 * 1000)
    random.seed(start)
    events = []
    t = start
    for i in range(roundtrips):
        events.append({
            "event_id": i + 1,
            "cell": "TEST",
            "type": "roundtrip",
            "planned_epoch_ms": t,
            "planned_ts_jst": jst_str(t),
            "dep_wit_gap_s": random.randint(*DEP_WIT_GAP_RANGE),
            "spare": False,
        })
        gap = random.uniform(gap_min, gap_max) * 60 * 1000
        t = t + int(gap)
    return {
        "kind": "test",
        "created_epoch_ms": now_ms(),
        "created_ts_jst": jst_str(now_ms()),
        "target": target,
        "n_roundtrip": roundtrips,
        "n_dep_single": 0,
        "n_wit_single": 0,
        "n_spare": 0,
        "events": events,
    }


# =====================================================================
# plan / checkpoint / runlog / STATUS の I/O
# =====================================================================
def save_plan(plan: dict, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_plan(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def checkpoint_path_for(plan_path: Path) -> Path:
    stem = plan_path.stem
    if stem == "plan":
        return plan_path.with_name("checkpoint.json")
    return plan_path.with_name(f"checkpoint_{stem}.json")


def load_checkpoint(cp_path: Path, plan_id=None) -> dict:
    """plan_id（= plan.created_epoch_ms）でcheckpointの帰属を検証する。
    plan_id 不一致（＝別の計画で再生成された残骸）の場合は新規扱いにする。
    これにより event_id がリセットされる計画再生成時の「古い完了印の誤流用」を防ぐ。
    同一planファイルでの再投入（真の再開）は plan_id 一致 → 完了印を引き継ぐ。"""
    fresh = {"plan_id": plan_id, "completed_event_ids": [], "last_update_jst": ""}
    if cp_path.exists():
        try:
            with open(cp_path) as f:
                cp = json.load(f)
            if plan_id is not None and cp.get("plan_id") != plan_id:
                log(f"[checkpoint] plan_id 不一致（古い計画の残骸）→ 新規扱い: {cp_path.name} "
                    f"(file={cp.get('plan_id')} vs plan={plan_id})")
                return fresh
            cp.setdefault("plan_id", plan_id)
            cp.setdefault("completed_event_ids", [])
            return cp
        except Exception:  # noqa: BLE001
            pass
    return fresh


def mark_event_completed(cp_path: Path, event_id: int, plan_id=None):
    cp = load_checkpoint(cp_path, plan_id)
    if event_id not in cp["completed_event_ids"]:
        cp["completed_event_ids"].append(event_id)
    cp["plan_id"] = plan_id
    cp["last_update_jst"] = jst_str(now_ms())
    tmp = cp_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cp_path)


RUNLOG_COLS = ["event_id", "cell", "type", "direction", "planned_ts_jst", "start_ts_jst",
               "end_ms_epoch", "duration_s", "exit_code", "csv_rows_before", "csv_rows_after",
               "status", "consecutive_fails", "note"]


def append_runlog(row: dict):
    new = not RUNLOG.exists()
    with open(RUNLOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUNLOG_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RUNLOG_COLS})


def write_status(text: str):
    tmp = STATUS.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(text.rstrip() + "\n")
    os.replace(tmp, STATUS)


def log(msg: str):
    print(f"[{jst_str(now_ms())}] {msg}", flush=True)


# =====================================================================
# 測定スクリプト実行（サブプロセス）
# =====================================================================
def _log_tail(log_path: Path, n_chars: int = 240) -> str:
    try:
        txt = log_path.read_text(errors="replace")
    except Exception:  # noqa: BLE001
        return ""
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    tail = " | ".join(lines[-3:])[-n_chars:]
    return tail.replace("\n", " ")


def run_measure(direction: str) -> dict:
    """measure スクリプトを --broadcast --prod で起動。成功判定 = 行数+1 かつ exit==0。"""
    script = DEP_MEASURE if direction == "deposit" else WIT_MEASURE
    csv_path = DEP_CSV if direction == "deposit" else WIT_CSV
    rows_before = csv_data_rows(csv_path)
    start_ms = now_ms()
    log_path = LOG_DIR / f"{direction}_{start_ms}.log"
    cmd = [sys.executable, "-u", str(script), "--broadcast", "--prod"]

    exit_code = None
    timed_out = False
    t0 = time.time()
    with open(log_path, "w") as lf:
        try:
            p = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=lf,
                               stderr=subprocess.STDOUT, timeout=SUBPROC_TIMEOUT)
            exit_code = p.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
    dur = round(time.time() - t0, 1)
    rows_after = csv_data_rows(csv_path)

    ok = (rows_after == rows_before + 1) and (exit_code == 0)
    note = ("TIMEOUT " if timed_out else "") + _log_tail(log_path)
    return {
        "direction": direction,
        "start_ms": start_ms,
        "end_ms": now_ms(),
        "duration_s": dur,
        "exit_code": ("TIMEOUT" if timed_out else exit_code),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "status": "OK" if ok else "FAIL",
        "log_path": str(log_path),
        "note": note,
    }


# =====================================================================
# イベント実行
# =====================================================================
class HaltSignal(Exception):
    def __init__(self, reason: str):
        self.reason = reason


def _target_for(plan, direction: str) -> int:
    """方向別目標。plan に target_dep/target_wit があれば優先、無ければ単一 target（後方互換）。"""
    if direction == "deposit":
        return int(plan.get("target_dep", plan.get("target", DEFAULT_TARGET)))
    return int(plan.get("target_wit", plan.get("target", DEFAULT_TARGET)))


def _status_line(plan, state, next_ev):
    td = _target_for(plan, "deposit")
    tw = _target_for(plan, "withdraw")
    dep_done = csv_data_rows(DEP_CSV)
    wit_done = csv_data_rows(WIT_CSV)
    nxt = "—"
    if next_ev is not None:
        nxt = f"{next_ev['type']} {hm(next_ev['planned_epoch_ms'])} JST"
    return (f"RUNNING | done {dep_done + wit_done}/{td + tw} "
            f"(dep {dep_done}/{td}, wit {wit_done}/{tw}) | "
            f"last {state['last']} | next {nxt} | fails(consec)={state['consec_fails']}")


def _run_one_direction(direction: str, ev: dict, state: dict, plan: dict):
    target = _target_for(plan, direction)
    csv_path = DEP_CSV if direction == "deposit" else WIT_CSV

    # --- target ゲート（overshoot 防止。spare はここで自然に止まる） ---
    if csv_data_rows(csv_path) >= target:
        log(f"[event {ev['event_id']}] {direction} target {target} 到達済み → SKIP")
        append_runlog({**_base_runrow(ev, direction),
                       "start_ts_jst": jst_str(now_ms()), "end_ms_epoch": now_ms(),
                       "status": "SKIP", "consecutive_fails": state["consec_fails"],
                       "note": "target reached"})
        return

    # --- 事前残高チェック ---
    verdict, detail = precheck(direction, target)
    if verdict == "FUNDS":
        raise HaltSignal(f"残高不足 ({direction}): {detail}")
    if verdict == "INFRA":
        state["consec_fails"] += 1
        log(f"[event {ev['event_id']}] {direction} 残高チェック失敗(INFRA) → SKIP "
            f"(consec={state['consec_fails']})")
        append_runlog({**_base_runrow(ev, direction),
                       "start_ts_jst": jst_str(now_ms()), "end_ms_epoch": now_ms(),
                       "status": "SKIP", "consecutive_fails": state["consec_fails"],
                       "note": f"balance check infra fail: {detail}"})
        if state["consec_fails"] >= MAX_CONSEC_FAILS:
            raise HaltSignal(f"連続{MAX_CONSEC_FAILS}件失敗（直近: balance INFRA）")
        return

    # --- 実送信 ---
    log(f"[event {ev['event_id']}] {direction} 実行 ... (rows_before={csv_data_rows(csv_path)})")
    res = run_measure(direction)
    if res["status"] == "OK":
        state["consec_fails"] = 0
    else:
        state["consec_fails"] += 1
    state["last"] = f"{res['status']} {direction} {hm(res['end_ms'])}"
    log(f"[event {ev['event_id']}] {direction} -> {res['status']} "
        f"(exit={res['exit_code']} rows {res['rows_before']}->{res['rows_after']} "
        f"{res['duration_s']}s consec={state['consec_fails']})")
    append_runlog({
        "event_id": ev["event_id"], "cell": ev["cell"], "type": ev["type"], "direction": direction,
        "planned_ts_jst": ev["planned_ts_jst"], "start_ts_jst": jst_str(res["start_ms"]),
        "end_ms_epoch": res["end_ms"], "duration_s": res["duration_s"], "exit_code": res["exit_code"],
        "csv_rows_before": res["rows_before"], "csv_rows_after": res["rows_after"],
        "status": res["status"], "consecutive_fails": state["consec_fails"], "note": res["note"],
    })
    if state["consec_fails"] >= MAX_CONSEC_FAILS:
        raise HaltSignal(f"連続{MAX_CONSEC_FAILS}件失敗（直近: {direction} FAIL / log={res['log_path']}）")


def _base_runrow(ev: dict, direction: str) -> dict:
    return {"event_id": ev["event_id"], "cell": ev["cell"], "type": ev["type"],
            "direction": direction, "planned_ts_jst": ev["planned_ts_jst"],
            "csv_rows_before": "", "csv_rows_after": "", "exit_code": "", "duration_s": ""}


def execute_event(ev: dict, state: dict, plan: dict):
    if ev["type"] == "roundtrip":
        _run_one_direction("deposit", ev, state, plan)
        gap = ev.get("dep_wit_gap_s", DEP_WIT_GAP_RANGE[0])
        log(f"[event {ev['event_id']}] dep→wit 間隔 {gap}s 待機 ...")
        _interruptible_sleep(gap)
        _run_one_direction("withdraw", ev, state, plan)
    elif ev["type"] == "dep_single":
        _run_one_direction("deposit", ev, state, plan)
    elif ev["type"] == "wit_single":
        _run_one_direction("withdraw", ev, state, plan)
    else:
        log(f"[event {ev['event_id']}] 未知の type={ev['type']} → SKIP")


# =====================================================================
# スリープ（STATUS 更新付き）
# =====================================================================
_STOP = {"flag": False}


def _interruptible_sleep(seconds: float):
    end = time.time() + seconds
    while time.time() < end and not _STOP["flag"]:
        time.sleep(min(5, end - time.time()))


def sleep_until(epoch_ms: int, plan: dict, state: dict, next_ev: dict):
    while not _STOP["flag"]:
        remaining = (epoch_ms - now_ms()) / 1000.0
        if remaining <= 0:
            return
        write_status(_status_line(plan, state, next_ev))
        time.sleep(min(60, remaining))


# =====================================================================
# 計画実行ループ
# =====================================================================
def run_plan(plan: dict, plan_path: Path):
    cp_path = checkpoint_path_for(plan_path)
    plan_id = plan.get("created_epoch_ms")
    cp = load_checkpoint(cp_path, plan_id)
    completed = set(cp["completed_event_ids"])
    events = sorted(plan["events"], key=lambda e: e["planned_epoch_ms"])
    state = {"consec_fails": 0, "last": "—"}

    log(f"=== run_plan: {plan_path.name} kind={plan.get('kind')} target={plan['target']} "
        f"events={len(events)} completed_already={len(completed)} ===")
    write_status(_status_line(plan, state, events[0] if events else None))

    try:
        for i, ev in enumerate(events):
            if _STOP["flag"]:
                raise HaltSignal("シグナル受信による停止")
            if ev["event_id"] in completed:
                continue
            planned = ev["planned_epoch_ms"]
            now = now_ms()

            # 過ぎたスロット（>30分超過）はSKIP
            if now - planned > PAST_SLOT_GRACE_S * 1000:
                log(f"[event {ev['event_id']}] planned {ev['planned_ts_jst']} を超過 → SKIP")
                append_runlog({**_base_runrow(ev, ""), "start_ts_jst": jst_str(now),
                               "end_ms_epoch": now, "status": "SKIP",
                               "consecutive_fails": state["consec_fails"],
                               "note": "past slot (>30min)"})
                mark_event_completed(cp_path, ev["event_id"], plan_id)
                continue

            # 未来なら待機
            if now < planned:
                next_after = events[i + 1] if i + 1 < len(events) else None  # noqa: F841
                sleep_until(planned, plan, state, ev)
            if _STOP["flag"]:
                raise HaltSignal("シグナル受信による停止")

            execute_event(ev, state, plan)
            mark_event_completed(cp_path, ev["event_id"], plan_id)
            nxt = events[i + 1] if i + 1 < len(events) else None
            write_status(_status_line(plan, state, nxt))

        # 全消化
        dep_done = csv_data_rows(DEP_CSV)
        wit_done = csv_data_rows(WIT_CSV)
        write_status(f"DONE | dep {dep_done}/{plan['target']}, wit {wit_done}/{plan['target']} | "
                     f"last {state['last']} | fails(consec)={state['consec_fails']} | "
                     f"finished {jst_str(now_ms())}")
        log("=== 全イベント消化 完了 ===")

    except HaltSignal as h:
        dep_done = csv_data_rows(DEP_CSV)
        wit_done = csv_data_rows(WIT_CSV)
        last_log = sorted(LOG_DIR.glob("*.log"))
        last_log_s = str(last_log[-1]) if last_log else "(none)"
        msg = (f"HALT | 理由: {h.reason} | dep {dep_done}/{plan['target']}, "
               f"wit {wit_done}/{plan['target']} | last {state['last']} | "
               f"last_log={last_log_s} | {jst_str(now_ms())}")
        write_status(msg)
        log("=== " + msg + " ===")
        sys.exit(2)


# =====================================================================
# 承認サマリ
# =====================================================================
def print_summary(plan: dict, plan_path: Path):
    events = plan["events"]
    n_round = sum(1 for e in events if e["type"] == "roundtrip" and not e.get("spare"))
    n_dep = sum(1 for e in events if e["type"] == "dep_single")
    n_wit = sum(1 for e in events if e["type"] == "wit_single")
    n_spare = sum(1 for e in events if e.get("spare"))
    # 実行件数（dep/wit）
    exec_dep = sum(1 for e in events if e["type"] in ("roundtrip", "dep_single") and not e.get("spare"))
    exec_wit = sum(1 for e in events if e["type"] in ("roundtrip", "wit_single") and not e.get("spare"))
    cost = n_round * (COST_DEP_USDC + COST_WIT_USDC) + n_dep * COST_DEP_USDC + n_wit * COST_WIT_USDC
    cost_spare = n_spare * (COST_DEP_USDC + COST_WIT_USDC)

    t0 = min(e["planned_epoch_ms"] for e in events) if events else now_ms()
    t1 = max(e["planned_epoch_ms"] for e in events) if events else now_ms()

    # セル別件数
    from collections import Counter
    cell_counts = Counter(e["cell"] for e in events)

    print("=" * 70)
    print(f" T4 CCTP スケジューラ — 承認サマリ  ({plan.get('kind','?')} / {plan_path.name})")
    print("=" * 70)
    _td = _target_for(plan, "deposit"); _tw = _target_for(plan, "withdraw")
    if _td == _tw:
        print(f" target               : {_td} 件/方向")
    else:
        print(f" target               : dep={_td} / wit={_tw} 件（非対称）")
    print(f" 計画時点のCSV         : dep={plan.get('dep_done_at_plan','?')}  wit={plan.get('wit_done_at_plan','?')}")
    print(f" イベント総数          : {len(events)}  "
          f"(roundtrip={n_round}, dep_single={n_dep}, wit_single={n_wit}, spare={n_spare})")
    print(f" 実送信件数(本体)      : dep={exec_dep} 件 / wit={exec_wit} 件")
    print(f" 推定コスト(本体)      : {cost:.2f} USDC  (+ 予備最大 {cost_spare:.2f} USDC)")
    print(f"   内訳: 1往復≈{COST_DEP_USDC+COST_WIT_USDC:.2f} (dep {COST_DEP_USDC}+wit {COST_WIT_USDC})")
    print(f" 計画期間(JST)         : {jst_str(t0)}  〜  {jst_str(t1)}")
    print("-" * 70)
    print(" セル別件数:")
    for label in sorted(cell_counts):
        print(f"   {label:<10} : {cell_counts[label]} 件")
    print("-" * 70)

    # 残高チェック
    print(" 残高チェック:")
    bals = read_balances()
    if bals is None:
        print("   [WARN] 残高取得に失敗（RPC）。起動前に再確認推奨。")
        ok_all = False
    else:
        dep_rem = max(1, _target_for(plan, "deposit") - csv_data_rows(DEP_CSV))
        need_eth = dep_rem * PER_DEP_ETH * ETH_SAFETY
        c_usdc = bals["arb_usdc"] >= MIN_ARB_USDC
        c_spot = bals["hc_spot_usdc"] >= MIN_HC_SPOT_USDC
        c_eth = bals["arb_eth"] >= need_eth
        print(f"   Arbitrum USDC   : {bals['arb_usdc']:.4f}  (>= {MIN_ARB_USDC}? {'OK' if c_usdc else 'NG'})")
        print(f"   HyperCore spot  : {bals['hc_spot_usdc']:.4f}  (>= {MIN_HC_SPOT_USDC}? {'OK' if c_spot else 'NG'})")
        print(f"   Arbitrum ETH    : {bals['arb_eth']:.6f}  (>= {need_eth:.6f}? {'OK' if c_eth else 'NG'})")
        ok_all = c_usdc and c_spot and c_eth
    print("-" * 70)
    print(f" 充足判定: {'✅ 充足（起動可）' if ok_all else '⚠ 要確認'}")
    print("=" * 70)
    return ok_all


# =====================================================================
# シグナル
# =====================================================================
def _on_signal(signum, frame):  # noqa: ARG001
    _STOP["flag"] = True
    try:
        write_status(f"STOPPED | signal={signum} | {jst_str(now_ms())}")
    except Exception:  # noqa: BLE001
        pass


# =====================================================================
# main
# =====================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="T4 CCTP 無人測定スケジューラ")
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET, help="方向ごとの目標件数（既定200）")
    ap.add_argument("--plan-only", action="store_true", help="本計画を生成し承認サマリのみ表示して終了（送信なし）")
    ap.add_argument("--summary", action="store_true", help="計画の承認サマリのみ表示して終了（送信なし）")
    ap.add_argument("--run", type=str, default=None, help="指定 plan.json を実行")
    ap.add_argument("--out", type=str, default=None, help="生成 plan の保存先（既定 plan.json / plan_test.json）")
    # テストモード
    ap.add_argument("--test", action="store_true", help="即席テスト計画を生成（往復のみ）")
    ap.add_argument("--start-in-min", type=float, default=1.0, help="テスト: 初回開始までの分")
    ap.add_argument("--roundtrips", type=int, default=1, help="テスト: 往復数")
    ap.add_argument("--gap-min", type=float, default=10.0, help="テスト: 往復間 最小間隔(分)")
    ap.add_argument("--gap-max", type=float, default=15.0, help="テスト: 往復間 最大間隔(分)")
    args = ap.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    # --- plan の決定 ---
    if args.run:
        plan_path = Path(args.run)
        plan = load_plan(plan_path)
    elif args.test:
        plan = generate_test_plan(args.target, args.start_in_min, args.roundtrips,
                                  args.gap_min, args.gap_max)
        plan_path = Path(args.out) if args.out else SCHED_DIR / "plan_test.json"
        save_plan(plan, plan_path)
    else:
        start = now_ms()
        plan = generate_main_plan(args.target, start)
        plan_path = Path(args.out) if args.out else SCHED_DIR / "plan.json"
        save_plan(plan, plan_path)

    # --- plan-only / summary は表示して終了 ---
    if args.plan_only or args.summary:
        print_summary(plan, plan_path)
        print(f"\n[plan saved] {plan_path}")
        return

    # --- 実行 ---
    log(f"スケジューラ起動 PID={os.getpid()} plan={plan_path}")
    run_plan(plan, plan_path)


if __name__ == "__main__":
    main()
