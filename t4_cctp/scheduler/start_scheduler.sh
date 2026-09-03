#!/usr/bin/env bash
# =====================================================================
# start_scheduler.sh — T4 CCTP スケジューラ 起動ランチャー（承認ゲート付き）
#
#   1) 与えられた引数で承認サマリ（送信なし）を表示
#   2) 承認: 対話 y/n、または --yes でスキップ（チャット側で承認済みの無人起動用）
#   3) 承認後 nohup でバックグラウンド常駐起動 → PID 表示
#      （ターミナル/シェルを閉じても・PC蓋を閉じても WSL が生きていれば残る:
#       nohup=SIGHUP無視 / </dev/null=stdin切離し / disown=ジョブ表から除去）
#
# 使い方:
#   本番計画(対話):   ./start_scheduler.sh --run plan.json
#   本番計画(無人):   ./start_scheduler.sh --run plan.json --yes
#   テスト(蓋閉め):   ./start_scheduler.sh --test --start-in-min 5 --roundtrips 4 --gap-min 10 --gap-max 15 --yes
#
# --yes はチャット等で承認済みの場合の無人起動用。Phase B もこの経路で起動する。
# =====================================================================
set -euo pipefail

SCHED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCHED_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# --- 引数から --yes/-y を抽出し、残りを scheduler.py へ渡す ---
ASSUME_YES=0
FILTERED=()
for a in "$@"; do
    case "$a" in
        --yes|-y) ASSUME_YES=1 ;;
        *) FILTERED+=("$a") ;;
    esac
done

echo "#######################################################################"
echo "# T4 CCTP スケジューラ 起動ランチャー"
echo "# args: ${FILTERED[*]:-<本計画 --target 既定200>}  (assume_yes=${ASSUME_YES})"
echo "#######################################################################"
echo ""

# 1) 承認サマリ（--summary を足して送信なしで表示）
python3 -u "$SCHED_DIR/scheduler.py" "${FILTERED[@]}" --summary

# 2) 承認
if [ "$ASSUME_YES" -eq 1 ]; then
    echo ""
    echo "[--yes] 承認済みとして起動します（チャット側承認済み・無人起動）。"
else
    echo ""
    read -r -p "上記内容で起動しますか？ 実送信（USDC消費）が発生します [y/N]: " ans
    case "$ans" in
        [yY]|[yY][eE][sS]) ;;
        *) echo "中止しました（送信なし）。"; exit 0 ;;
    esac
fi

# 3) nohup 常駐起動（確実にデタッチ）
OUT="$SCHED_DIR/scheduler.out"
nohup python3 -u "$SCHED_DIR/scheduler.py" "${FILTERED[@]}" </dev/null >> "$OUT" 2>&1 &
PID=$!
disown "$PID" 2>/dev/null || true
echo ""
echo "起動しました。"
echo "  PID         : $PID"
echo "  stdout/err  : $OUT"
echo "  STATUS      : $SCHED_DIR/STATUS"
echo "  runlog      : $SCHED_DIR/runlog.csv"
echo ""
echo "状態確認:    cat $SCHED_DIR/STATUS"
echo "プロセス確認: ps -p $PID"
echo "停止:        kill $PID   （安全に STATUS=STOPPED を書いて終了）"
