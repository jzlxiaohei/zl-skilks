---
name: sglang-pr-submission
description: 向 sgl-project/sglang 提交外部贡献 PR 并推动到 merge 的实战流程——PR 描述怎么写才能过 BBuf review agent 的预审、性能 PR 要准备哪些数据、在 Slack #pull-requests 怎么触发 agent、CI 红了哪些可以忽略、GPU 资源池卡住时怎么办。在准备、提交、跟进 sglang PR，或 PR 卡在 review/CI/merge 时使用。
---

# sglang PR 提交与跟进

来自第一个 PR（#48629，model 文件改用 `apply_qk_norm`，从提交到 merge 约 8 天）的经验。大部分时间耗在"触发方式不对"和"CI 资源池排队"，而不是代码本身。

## 1. 提交前：PR 描述决定能否过预审

- 标题：`[Type] scope: title`，Type 取 Feat/Fix/Refactor/Perf/Doc/Test/CI/Chore（见 `.github/pull_request_template.md`）。
- **描述必须用模板的小节**：`## Motivation`、`## Modifications`、`## Accuracy Tests`、`## Speed Tests and Profiling`。
- **数据放描述里，不放评论里。** Slack 侧 review agent 的预审只读 PR 描述：数据只贴在评论里时，GitHub 侧 agent 回了 "This covers the benchmark request"，Slack 侧 agent 仍然反复回 "before review, please add before/after numbers with hardware and workload"。把同样内容挪进描述后才通过。
- 改描述用 `gh pr edit <N> --body-file <file>`，保留末尾 bot 维护的 `<!-- pr-states:start -->…<!-- pr-states:end -->` 块。
- 改动涉及 `.github/CODEOWNERS` 里有 owner 的路径（如 `srt/managers`、`srt/mem_cache`）也不用逐个找 owner，见第 3 节"有 code owner 的 PR"。

## 2. 性能 PR 要准备的数据

`[Perf]` PR 不给前后对比数据，agent 一定会卡在预审。准备三样，写清硬件、软件版本、workload 命令：

1. **Kernel before/after**：`sglang.kernels.jit.benchmark.marker.do_bench`（CUDA graph + 每次迭代 clone 输入 = 冷 L2；仓库规则禁止固定参数的热 L2 测法）。覆盖 decode（T=1/8）到 prefill（T=2048/8192），并 `assert_close` 证明输出一致。
2. **E2E**：`python -m sglang.benchmark.one_batch_server --batch-size 1 8 32 128 --input-len 1024 --output-len 256`。**旧代码跑两次**，第二次作为噪声参考，一起列进表里。
3. **精度**，二选一：
   - 输出 bit-identical（首选）：加一个 `torch.equal` 的单测证明，不用跑 eval。例如 fused 算子带 `round_sigmoid=True` 这类开关，可以和 torch 两步写法逐位一致；各平台的数值语义保持和改动前完全相同。
   - 输出会变：用 sgl-eval（`.claude/rules/prefer-sgl-eval.md`）跑 gsm8k 全量 1319 题，旧代码同样跑两次。n=1319 时标准误约 1pt，差距在 1pt 内就写"within noise"；需要时对不一致的样本对做符号检验。

**kernel 提速大、E2E 只有几个点是正常的**，可以用每步时间对账来解释：kernel 每层省的 µs × 层数 ≈ 每个 decode step 省的时间。#48629：每层约 8–9µs × 30 层 ≈ 0.25ms/step，而 step 是 4–12ms，所以 E2E 只有 +1–6%。

**范围收窄更容易过**：不能测试的平台（NPU、CPU 新 kernel）保持原行为不动，PR 里写"只改 CUDA/HIP，其他平台输出不变"。

## 3. 触发 review：Slack #pull-requests

BBuf 的 review agent 负责预审、approve、等 CI、merge 的全流程，入口在 Slack #pull-requests。

- **安全红线：默认不要主动发送 Slack 消息！** 除非用户在当前指令中明确要求“直接把 Slack 消息发出去”，否则 Agent **只负责生成文案或将草稿填入输入框并截图展示给用户确认，严禁自动点击 Send / 按回车发送**。
- **只有频道里新发的顶层消息 + 真正的 @BBuf mention 才能触发。** 已验证无效的写法：
  - 编辑旧消息补上 @（Slack 对编辑后的消息不发 mention 事件）；
  - 在线程里回复 @BBuf（等了约一天，agent 一直没处理）。
- 写法：`Hi @BBuf, could your review agent take a look at <PR URL>? <一句话说明改了什么和数据>. Could you add the run-ci label? Thanks!`
- 输入 `@BBuf` 后从弹出的候选里按回车选中，确认变成蓝色 mention 再发。Slack 编辑器会改写撇号和反引号（`they're` 变成 `they''e`，`` `run-ci` `` 的首字母丢失），所以避免使用这两类字符，发送前截图核对一遍。
- agent 的消息顺序：`Picked up sglang#N` → 要么 `before review, please …`（预审没过），要么 `N thing(s) to fix, see the PR comments`，要么 `Approved sglang#N, will merge once CI is green` → `Merged sglang#N, thanks!`。
- **2026-11 起 agent 有准入门槛**：原话 "We're keeping review time for bug fixes and changes with ≥3% end-to-end gains on mainstream models"（#50907），新增 flag/env var 也会被拒（#50924）。只有 microbenchmark、没有 E2E 数据的 `[Perf]` PR 会被挡回：#50932、#50978 等了约一天后，agent 在线程里回复 "Review starts once the PR has end-to-end numbers on current main, e.g. bench_serving on a mainstream model"。要的是**当前 main** 上的数据，PR 落后 main 较多时先 rebase 或在 main 上套 diff 再测。提交前先判断：输出不变、只是更慢的改动就是 `[Perf]`，不要为了过门槛改称 `[Fix]`（违反仓库"不要 host sync"之类的规则也仍是 perf）。perf PR 要么准备主流模型上 ≥3% 的 E2E 数据，要么如实写明收益小、直接找领域 Merge Oncall。
  - 小 diff 的 perf PR 照样被合（11-14~11-18 BBuf 合了多个 +5~+80 行的 `[Perf]`），关键是有**针对受影响场景的 E2E 实测**，不是 diff 大小。最接近的先例 #50197（GDN state reset 去 host sync，+75/-2）挑了 Qwen3.5-0.8B 的 `/v1/score` 负载，测出 +17% req/s 和每请求 sync 次数 53→17；#46781（+5/-2）测了 `run_batch` -13.5%。只有 microbenchmark 的会被跳过或拒（#50907 "about 30 us per layer"）。
- 响应时间：顶层消息从几分钟到半天不等（agent 按批处理）。超过一天没反应，就再发一条顶层消息，不要在线程里催。
- 改完代码后 push 新 commit，也会触发 agent 重新检查。

### 有 code owner 的 PR

规则（`.github/MAINTAINER.md`）：每个受 CODEOWNERS 保护的文件需要一个 owner approve；或者由 **Merge Oncall bypass** 合入。GitHub 会自动给 owner 发 review request，显示 `REVIEW_REQUIRED` / `BLOCKED` 是正常的。

实际情况（2026-11 统计近 400 个 merged PR）：外部贡献者改 `srt/managers`、`srt/mem_cache` 的 PR，多数是 **BBuf agent 一个人 approve 并 merge**，没有任何 code owner approve（如 #50122 改 `mem_cache/evict_policy.py` + `managers/schedule_policy.py`，#50741 改 `tokenizer_manager.py`，都是 1–4 天合入）。所以：

- **不要在 Slack 里逐个 @ code owner**，照常发顶层消息 @BBuf 即可，可以一条消息带多个 PR。
- 领域非常集中的改动也有对应 oncall 自己接：PD/disaggregation 由 ShangmingCai approve+merge，hisparse 由 hzh0425，NPU 由 sglang-npu-bot。
- 只有 BBuf agent 已 approve、CI 绿了但长时间不 merge 时，才去 ping **一个**对应领域的 Merge Oncall（名单见 `MAINTAINER.md`，例如 KV Cache: ispobock / xiezhq-hermann；Scheduler: hnyls2002）。
- 未验证：BBuf agent 是否对所有 owner 路径都会 bypass；统计只看了已 merge 的 PR，没看卡住的。

## 4. CI：哪些红色可以忽略

- `run-ci` 标签由 agent/maintainer 加（`/tag-and-rerun-ci`）。之前 `pr-gate` 全红是因为没有这个标签，属正常现象。
- **必需检查只有 `lint` 和 `Check PR title`**（用 `gh pr checks <N> --required` 确认）。`pr-test-finish` 是 Base CI 的汇总，agent 主要看它和"相关"的 job。
- 可以忽略的红色：
  - `call-gate / pr-gate` 要求 `run-ci-mlu`、`run-ci-amd-extra`、`run-ci-nv-extra`：这些是需要额外标签才会跑的扩展 CI，不加标签就会显示失败，属于预期。
  - 对应的 `*-finish` 汇总 job，跟着上游一起红。
  - 很多 PR 都在挂的测试（例如 MLX 的 `test_attention_patching`）：先看最近 10 次左右同一 workflow 在别的 PR 上的结果，再下结论。
  - runner 环境问题，例如 AMD 的 "VRAM cleanup unsuccessful after 3 attempts"。
- agent 会在 Slack 里标出 "CI failures look related to the change: …"，只有这种才需要修。
- 判断失败是否和自己有关：拿日志里的报错和自己 diff 涉及的文件对照。

## 5. CI 卡在排队：判断资源池是否宕机

job 显示 queued 且没有 `runner_name`，就是还没分到机器。区分"排队慢"和"整个池子挂了"：

```bash
# 最近 N 个 pr-test run 里某个 runner_config 的 job 状态
for r in $(gh run list --repo sgl-project/sglang --workflow pr-test.yml --limit 60 --json databaseId --jq '.[].databaseId'); do
  gh api "repos/sgl-project/sglang/actions/runs/$r/jobs?per_page=100&filter=latest" \
    --jq '.jobs[] | select(.name|test("4-gpu-gb300")) | select(.conclusion!="skipped") | "\(.created_at)\t\(.status)/\(.conclusion)\trunner=\(.runner_name)"'
done | sort | tail -15
```

如果所有 PR 的这个 job 都在排队、连 main 的定时 run 都卡住，说明是基础设施问题。这时：

1. 到 Slack 发顶层消息说明情况：哪些相关检查已通过、哪个 job 排了多久、资源池整体卡住的证据、其余红色为什么无关。#48629 这样做后，agent 几分钟内就回复了 Approved。
2. maintainer 可能会批量取消卡住的 job，之后 `pr-test-finish` 会变红。
3. **等资源池恢复后**（出现新分到 runner 并成功的 job），PR 作者自己评论 `/rerun-failed-ci`。作者在自己的 PR 上有这个权限，见 `scripts/ci/utils/slash_command_handler.py`。池子没恢复前不要 rerun，只会重新回去排队。
4. 不要 push 空 commit 触发 CI，会让全部 job 重新排队。

全绿以后 agent 按批 merge，可能还要再等几个小时。

## 6. 自己租 GPU 跑数据（RunPod）

- 对第一个 PR 来说，A100 80GB 一卡完全够用（约 $1.8/hr；2026-11 已涨到 $2.09–2.19/hr，8B 模型用 L40S $1.12/hr 也够）。SXM 没货时 PCIe 也可以；一次 kernel + E2E + gsm8k 总共花了 1–2 美元。
- 租之前逐项核对文档要求的 VRAM、**主机内存**、磁盘，留出余量。2026-11 曾在 125 GB 内存的机器上跑要求 ≥128 GB 的配置，启动时内存打满、SSH 无响应，白花约 6 美元。
- 部署后立刻打开 pod 的 Logs → System 看拉镜像进度。有进度就等；长时间没日志、没进展才 Stop。
- SSH 公钥没加到账号时，可以在 Jupyter terminal 里把公钥追加到 `~/.ssh/authorized_keys`。

### 源码安装 main（每次都按这个来，以下坑都已重复踩过）

```bash
python3 -m venv venv && . venv/bin/activate && pip install -U pip uv
export SGLANG_BUILD_RUST_EXTS=none   # 必须在第一次 install 之前设；pod 上没有 cargo，不设必失败
uv pip install -e sglang/python
export PYTHONPATH=$PWD/sglang/python  # pyproject 排除了 benchmark*，editable 安装后 import 不到 sglang.benchmark
```

- 服务端 benchmark 入口是 `python -m sglang.benchmark.serving`（旧的 `sglang.bench_serving` 已不存在），需要上面的 `PYTHONPATH`。
- bench_serving 默认 `temperature=0`（greedy），greedy batch 会跳过 top-p/top-k/min-p 路径。测采样相关改动时用 `--backend sglang-oai-chat --dataset-name openai --dataset-path x.jsonl`，在每行里写 `temperature`/`top_p`/`min_p`。
- 远程命令里不要用 `pkill -f sglang.launch_server` / `pgrep -f <脚本名>`：ssh 的命令文本本身包含这个字符串，会把自己杀掉或永远匹配成"运行中"。用 pid 文件、完成标记文件，或 `ps aux | grep "[l]aunch_server"`。
- A100 是 sm80，用不了 fa3。各模型支持的 attention backend 不同（例如 Exaone4 只接受 fa3/triton/trtllm_mha），选 triton 就行。
- **切换 before/after 代码时用 `git checkout <rev> -- <file>`，恢复必须用 `git checkout HEAD -- <file>`。** 用 `git checkout -- <file>` 会从已经被改写的 index 里恢复，结果仍是旧代码。每次跑之前都 `grep` 一下关键调用，确认跑的是哪一版。
- 跑完立刻 Stop 并 Terminate pod，把日志和结果拷回本地留档（放在被 `.git/info/exclude` 忽略的目录里）。

## 7. 杂项

- fork PR 的 Claude Code Review 默认关闭，需要 maintainer 评论 `@claude review` 才会跑，不用指望它。
- commit 作者身份用 `git -c user.name=… -c user.email=…` 单次指定，不要改仓库的 git config。
- 第一个 PR 选"把模型改用已有的、经过验证的工具函数"这类改动最好：diff 小、有现成的测试、性能收益容易量化。
