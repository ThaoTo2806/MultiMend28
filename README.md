# Roadmap cải tiến MultiMend trên Kaggle

Tài liệu này là kế hoạch làm việc chính sau notebook `multimend28.ipynb`. Mục tiêu là tái lập MultiMend đủ tin cậy, sau đó tạo một contribution vừa sức, có ablation rõ ràng và không thay đổi kiến trúc CodeT5, quy trình multi-hunk hoặc validator.

## 1. Hướng nghiên cứu được chọn

**Adaptive Context Budgeting with Dual-View Generation (ACB-DV)**

Thay retrieval cố định của MultiMend bằng cơ chế chọn context theo từng bug:

1. Luôn bảo toàn input bắt buộc: language prefix, buggy hunk và local surrounding context.
2. Chỉ thêm retrieved lines khi độ tin cậy đủ cao và còn token budget.
3. Chọn số dòng linh hoạt `k ∈ [0, 5]`, thay vì luôn lấy tối đa 5 dòng với cùng một threshold.
4. Loại các dòng dư thừa bằng lexical deduplication và Maximal Marginal Relevance (MMR) đơn giản.
5. Với các trường hợp retrieval không chắc chắn, sinh hai view có kiểm soát: `no-RAG` và `adaptive-RAG`, rồi gộp/deduplicate candidate bằng pipeline sẵn có.

Tên làm việc ngắn: **AdaptiveMultiMend**.

### Vì sao đây là hướng phù hợp nhất

MultiMend hiện dùng cố định:

- `all-MiniLM-L6-v2` để embed từng dòng;
- tối đa 5 dòng retrieval;
- cosine distance threshold `0.5`;
- retrieved lines được đặt trước surrounding context;
- toàn bộ input bị truncate ở 512 token.

Paper nói rõ `r=5` và threshold `0.5` chỉ được chọn qua preliminary testing trên một tập bug nhỏ. Ablation của paper cũng cho thấy RAG không luôn tốt: context augmentation tạo 40 correct fixes riêng, nhưng cấu hình không augmentation vẫn tạo 17 correct fixes riêng. Paper nêu trường hợp context nhiễu làm patch sai được xếp trên patch đúng và đề xuất future work cho adaptive context selection.

Vì vậy ACB-DV bám trực tiếp một điểm yếu đã được paper thừa nhận. Nó chỉ tác động tầng tạo input và gộp candidate, nên:

- không cần train lại model cho thí nghiệm chính;
- dùng nguyên checkpoint ensemble;
- giữ nguyên bug-line finder, generator, combiner và validator ở mức tối đa;
- có baseline và ablation dễ bảo vệ;
- phù hợp giới hạn GPU Kaggle.

## 2. Dừng và bảo toàn công việc hiện tại

### 2.1 Sửa mốc checkpoint

Với `2,324,030` training samples, batch size 8 và 2 epochs:

- steps mỗi epoch: `ceil(2,324,030 / 8) = 290,504`;
- tổng steps: `581,008`, **không phải `5,810,008`**;
- `save_steps = 290,504 // 5 = 58,100` theo code hiện tại.

Các mốc tự nhiên là `58100`, `116200`, ..., `581000`. Sai lệch 8 steps cuối là do phép chia nguyên của `save_steps`. Cần đọc `trainer_state.json` thay vì suy luận trạng thái chỉ từ tên thư mục.

Paper dùng `k=5` và mô tả lưu 5 checkpoint tại mỗi 20% của **epoch thứ hai**. Với cách script hiện tại chọn `checkpoints[-5:]`, ensemble dự kiến là:

- `checkpoint-348600`
- `checkpoint-406700`
- `checkpoint-464800`
- `checkpoint-522900`
- `checkpoint-581000`

Các checkpoint `58100`, `116200`, `174300`, `232400`, `290500` hữu ích để resume/reproduce training nhưng không phải ensemble cuối dùng cho generation.

### 2.2 Việc cần làm ngay

- [ ] Giữ nguyên mọi checkpoint hiện có, bao gồm `optimizer.pt`, `scheduler.pt`, `scaler.pt`, RNG state và `trainer_state.json`.
- [ ] Tạo SHA-256 manifest cho từng file trong checkpoint.
- [ ] Nén từng checkpoint riêng, không nén cả thư mục training đang thay đổi.
- [ ] Lưu code commit SHA, versions, dataset cardinality và `trainer_state.json` cùng artifact.
- [ ] Upload artifact bền vững trước khi kết thúc session Kaggle.
- [ ] Không tiếp tục training chỉ để làm AdaptiveMultiMend; trước hết kiểm tra official checkpoints trên Hugging Face collection của tác giả.

Official model nên là baseline ưu tiên. Checkpoint tự train chỉ cần hoàn tất nếu mục tiêu phụ là tái lập training hoặc kiểm chứng model artifact của paper.

## 3. Nguyên tắc thí nghiệm

### 3.1 Không thay nhiều biến cùng lúc

Giữ cố định:

- CodeT5-small và tokenizer;
- 5 checkpoint cuối;
- max input 512, max output 256;
- candidate ranking hiện tại;
- beam/candidate budget;
- bug locations và benchmark versions;
- validation logic.

Chỉ thay context construction trong thí nghiệm đầu tiên.

### 3.2 So sánh công bằng

Nếu baseline sinh `5 checkpoints × 100 candidates`, phương án mới không được âm thầm dùng gấp đôi compute. Có hai chế độ báo cáo:

- **Effectiveness mode:** tối đa 100 candidates cho mỗi view, báo rõ chi phí tăng.
- **Budget-matched mode:** chia cùng tổng budget, ví dụ 50 no-RAG + 50 adaptive-RAG trên mỗi checkpoint.

Kết quả chính phải có budget-matched comparison. Effectiveness mode chỉ là upper bound.

### 3.3 Không tune trên test một cách kín đáo

Dùng Defects4J v1.2 làm development benchmark để chốt rule/threshold. Sau khi freeze cấu hình, đánh giá trên Defects4J v2.0, QuixBugs, Codeflaws, BugAID, BugsInPy và RunBugRun-JS. Nếu sau đó thay rule vì nhìn kết quả test, phải tạo version thí nghiệm mới và ghi rõ.

QuixBugs-Python chỉ dùng smoke test vì 40 chương trình nhỏ, đa số một function và paper cho biết retrieval ít có ích ở đây.

## 4. Thiết kế Adaptive Context Budgeting

### 4.1 Input invariant

Format vẫn tương thích MultiMend:

```text
language_prefix buggy_hunk : selected_retrieved_lines surrounding_context
```

Tuy nhiên builder phải tokenize trước khi ghép để biết budget thật. Không dùng số từ làm đại diện cho số token.

### 4.2 Token budget

Cho mỗi hunk:

1. Tokenize `language_prefix + buggy_hunk + ':'`.
2. Xác định budget còn lại trong 512 token.
3. Dành budget ưu tiên cho surrounding context.
4. Phần dư mới cấp cho retrieved lines.
5. Không thêm một retrieved line nếu nó khiến local context bị truncate thêm.
6. Log số token trước/sau, số token bị truncate và các dòng đã chọn.

Bản đầu nên giữ nguyên toàn bộ surrounding context khi vừa 512 token. Nếu bản thân local context đã quá dài, dùng đúng truncation baseline và đặt retrieval budget bằng 0. Đây là rule đơn giản, deterministic và dễ ablate.

### 4.3 Retrieval confidence

Lấy tối đa 20 raw candidates để lựa chọn, nhưng chỉ đưa tối đa 5 dòng vào model. Với cosine similarity `s = 1 - distance`, dùng các tín hiệu:

- top similarity `s1`;
- margin `s1 - s2`;
- lexical identifier overlap với buggy hunk;
- duplicate ratio giữa candidates;
- available token budget;
- truncation risk của local context.

Version đầu không cần train selector. Dùng rule-based gate, chốt threshold trên Defects4J v1.2:

```text
retrieval_enabled = enough_token_budget
                    and top_similarity >= tau
                    and candidate_has_identifier_overlap
```

Sau gate, chọn candidates theo MMR để cân bằng liên quan và đa dạng:

```text
MMR(line) = lambda * similarity(line, buggy_hunk)
            - (1 - lambda) * max_similarity(line, selected_lines)
```

Grid nhỏ trên development set:

- `tau ∈ {0.45, 0.50, 0.55, 0.60}`;
- `lambda ∈ {0.6, 0.8}`;
- `k_max ∈ {3, 5}`.

Không mở grid lớn; chi phí và nguy cơ overfit không đáng.

### 4.4 Dual-view generation

- Confidence cao: chỉ adaptive-RAG.
- Confidence thấp hoặc sát threshold: no-RAG và adaptive-RAG.
- Không còn token budget: chỉ no-RAG.

Mỗi candidate phải có metadata:

```json
{
  "context_strategy": "adaptive_rag",
  "retrieved_count": 3,
  "retrieved_token_count": 47,
  "input_token_count": 481,
  "was_truncated": false,
  "top_similarity": 0.67,
  "similarity_margin": 0.11
}
```

Không làm learned reranker trong contribution đầu. Gộp candidate bằng rank/score hiện tại và thêm `context_strategy` để phân tích provenance.

## 5. Ablation bắt buộc

| ID | Cấu hình | Mục đích |
|---|---|---|
| B0 | No augmentation | Baseline không RAG của paper |
| B1 | Fixed RAG: top-5, distance 0.5 | Baseline MultiMend chính thức |
| A1 | Token-budget only | Đo lợi ích chống truncation |
| A2 | Token-budget + confidence gate | Đo adaptive `k=0..5` |
| A3 | A2 + MMR | Đo diversity/dedup |
| A4 | A3 + dual-view, budget-matched | Phương án chính |
| A5 | A3 + dual-view, full budget | Upper bound và cost trade-off |

Nếu A2 không thắng B1 trên development set về correct fixes hoặc top-k exact match, dừng MMR/dual-view và phân tích retrieval logs trước. Không tiếp tục thêm độ phức tạp để cứu một giả thuyết đã thất bại.

## 6. Metrics cần báo cáo

### Hiệu quả repair

- `exact_match`: patch sau chuẩn hóa giống developer patch;
- `paper_correct`: nhãn đánh giá thủ công theo tiêu chí của paper (developer
  patch, patch được cộng đồng review, hoặc tương đương ngữ nghĩa);
- plausible bugs: patch vượt qua test suite;
- correct@1, @5, @10, @100, @500;
- số fix riêng của từng strategy và overlap;
- single-hunk so với multi-hunk;
- kết quả theo ngôn ngữ/benchmark.

### Context quality

- tỷ lệ hunk chọn `k=0,1,...,5`;
- input tokens trung bình và p95;
- tỷ lệ input bị truncate;
- số local-context tokens bị mất;
- top similarity và margin;
- retrieved line chứa identifier xuất hiện trong developer patch;
- bug được B1 sửa nhưng adaptive làm mất, và ngược lại.

### Chi phí

- generation seconds/hunk;
- GPU-hours;
- số raw và unique candidates;
- số patch validations trước correct/plausible patch;
- peak VRAM và artifact size.

Kết quả chỉ có ý nghĩa nếu báo cả gain lẫn compute overhead.

## 7. Các giai đoạn thực hiện

### Giai đoạn 0 - Đóng băng reproduction

**Mục tiêu:** có baseline không thay đổi được nữa.

- [ ] Fork/clone source và tạo branch `baseline-reproduction`.
- [ ] Ghi upstream commit SHA.
- [ ] Dùng official checkpoints nếu tải được.
- [ ] Chạy end-to-end QuixBugs-Python: extraction → RAG DB → generation → combine → syntax/test validation.
- [ ] Xác nhận dùng đúng 5 checkpoint cuối.
- [ ] Lưu `generated_input.jsonl`, `sequences_100.jsonl`, `final_candidates_100.jsonl` và summary.
- [ ] So sánh ít nhất exact-match count với artifact chính thức.

**Gate:** pipeline chạy lại cho cùng output với seed 42, hoặc mọi sai khác đã được giải thích và ghi lại.

### Giai đoạn 1 - Instrument baseline

**Mục tiêu:** đo điểm yếu trước khi sửa.

- [ ] Log retrieval distance/similarity và raw candidates.
- [ ] Log token count trước và sau truncation.
- [ ] Log phần local context bị mất do retrieved lines.
- [ ] Thêm `context_strategy=fixed_rag` vào output.
- [ ] Chạy B0 và B1 trên QuixBugs-Python, sau đó một subset Defects4J v1.2.

**Gate:** tìm được các case cụ thể cho ba nhóm: RAG giúp, RAG không đổi, RAG gây hại.

### Giai đoạn 2 - Implement A1/A2

**Mục tiêu:** có adaptive builder tối thiểu.

Các file dự kiến chạm:

- `src/rag_utils.py`: trả score rõ ràng và hỗ trợ lấy raw candidates;
- `src/generate_primary_candidates.py`: context builder token-aware, strategy config và logging;
- một file config/CLI nhỏ nếu cần, không tiếp tục hard-code thí nghiệm trong script.

- [ ] Viết unit tests cho budget 512, empty hunk, empty retrieval, exact-boundary và long context.
- [ ] Đảm bảo fixed mode tạo input giống baseline.
- [ ] Chạy A1/A2 trên development subset.

**Gate:** không input nào vượt 512 sau tokenization; fixed mode không regression; A2 có tín hiệu tốt hơn B1 hoặc giảm truncation rõ ràng mà không mất effectiveness.

### Giai đoạn 3 - MMR và dual-view

**Mục tiêu:** hoàn thiện contribution chính.

- [ ] Thêm MMR và provenance metadata.
- [ ] Thêm budget-matched dual-view.
- [ ] Gộp/deduplicate candidates mà không phá schema validator.
- [ ] Chạy đủ B0, B1, A1-A4 trên Defects4J v1.2.
- [ ] Freeze `tau`, `lambda`, `k_max` và policy chia beam.

**Gate:** chọn đúng một cấu hình final trước khi chạy test benchmarks.

### Giai đoạn 4 - Evaluation đã freeze

Thứ tự để giảm rủi ro và chi phí:

1. QuixBugs Java/Python và BugAID: smoke/cross-language.
2. RunBugRun-JS: benchmark executable vừa phải.
3. Defects4J v2.0: real-project Java.
4. BugsInPy: real-project Python.
5. Codeflaws: chạy cuối vì quy mô lớn.

- [ ] Chạy B0, B1 và A4 trên tất cả benchmark.
- [ ] Chạy full ablation A1-A3 ít nhất trên Defects4J v1.2/v2.0 và một benchmark ngôn ngữ khác.
- [ ] Chỉ dùng A5 khi còn ngân sách.
- [ ] Lưu manifest cho mỗi shard.

**Gate:** mỗi kết quả có config hash, code commit, checkpoint hashes, seed, shard range và completion status.

### Giai đoạn 5 - Phân tích và viết contribution

- [ ] Báo aggregate và per-benchmark.
- [ ] Làm Venn/overlap B0-B1-A4.
- [ ] Phân tích case study retrieval giúp/hại.
- [ ] Báo confidence calibration theo bins.
- [ ] Báo trade-off effectiveness/GPU-hours/validation count.
- [ ] Nêu rõ perfect fault localization và test-suite overfitting là threats to validity.

Contribution có thể viết gọn như sau:

> We introduce a token-aware adaptive context selection strategy for multilingual neural program repair. Unlike MultiMend's fixed top-k retrieval, the strategy conditionally selects zero to five non-redundant source lines according to retrieval confidence and the model's remaining token budget, while preserving local buggy context. A budget-matched dual-view ensemble retains complementary repairs from augmented and non-augmented inputs without changing the repair model.

## 8. Vận hành Kaggle qua session 12 giờ

### 8.1 Tách job nhỏ, idempotent

Mỗi job chỉ làm một việc:

- prepare benchmark/index;
- generate một `(benchmark, strategy, checkpoint, shard)`;
- combine shard;
- validate một bug range;
- aggregate metrics.

Không chạy một notebook tuyến tính 12 giờ chứa mọi bước. Mỗi job phải có thể chạy lại mà bỏ qua output đã hoàn tất.

### 8.2 Run manifest

Mỗi run ghi `run_manifest.json`:

```json
{
  "run_id": "d4j2-a4-ckpt348600-shard00",
  "upstream_commit": "...",
  "experiment_commit": "...",
  "checkpoint": "checkpoint-348600",
  "checkpoint_sha256": "...",
  "benchmark": "Defects4J-v2.0",
  "strategy": "adaptive_dual_budget_matched",
  "seed": 42,
  "start_bug": 0,
  "end_bug": 49,
  "status": "complete"
}
```

Chỉ đánh dấu `complete` sau khi output đóng file, đếm record đúng và checksum được tạo.

### 8.3 Sharding đề xuất

- Generation: shard theo checkpoint và range bug.
- Validation: shard theo bug, vì thời gian test rất lệch.
- Không để hai session ghi cùng một output path.
- Tên artifact phải chứa benchmark, strategy, checkpoint và shard.
- Combine chỉ nhận shard có `status=complete` và checksum hợp lệ.

### 8.4 Lưu trữ

Phân biệt ba loại artifact:

- **Training-resume checkpoint:** cần model + optimizer/scheduler/scaler/RNG/trainer state.
- **Inference checkpoint:** chỉ cần model/tokenizer/config/generation config.
- **Experiment output:** JSONL + manifest + logs + checksums, không kèm cache có thể tái tạo.

Upload artifact định kỳ theo ranh giới job, không chờ gần hết 12 giờ. Git chỉ lưu code/config nhỏ; checkpoint và outputs lớn lưu trong Kaggle Dataset hoặc artifact storage được phép. Việc sử dụng tài nguyên phải tuân thủ điều khoản Kaggle; roadmap này dựa trên resume/sharding chứ không phụ thuộc vào việc né quota tài khoản.

## 9. Cấu trúc branch và output

```text
main
baseline-reproduction
feature/adaptive-context
experiment/defects4j-dev
```

```text
experiments/
  configs/
    b0_no_rag.json
    b1_fixed_rag.json
    a1_token_budget.json
    a2_adaptive_gate.json
    a3_adaptive_mmr.json
    a4_dual_budget_matched.json
  runs/
    <run_id>/
      run_manifest.json
      retrieval_log.jsonl
      generated_input.jsonl
      sequences.jsonl
      final_candidates.jsonl
      metrics.json
      sha256.txt
```

Không commit generated checkpoints, Chroma DB hoặc benchmark working copies vào Git.

## 10. Thứ tự hành động từ bây giờ

1. Backup và checksum các checkpoint đang có.
2. Kiểm tra official Hugging Face collection; ưu tiên dùng official 5 checkpoints cuối cho baseline.
3. Chuyển notebook thành tài liệu tham khảo, không tiếp tục thêm cell production.
4. Tạo branch baseline và chạy QuixBugs-Python end-to-end từ source.
5. Reproduce B0/B1 trước khi viết adaptive code.
6. Instrument token truncation và retrieval scores.
7. Implement A1 rồi validate ngay; sau đó A2, A3 và A4 theo từng gate.
8. Freeze config trên Defects4J v1.2.
9. Shard evaluation trên các benchmark còn lại.
10. Viết báo cáo từ manifests và metrics, không tổng hợp thủ công từ notebook output.

## 11. Tiêu chí hoàn thành

AdaptiveMultiMend được xem là contribution đạt yêu cầu khi:

- fixed mode tái lập baseline;
- có ít nhất B0, B1, A1-A4;
- cấu hình final được freeze trước test evaluation;
- giảm đáng kể local-context truncation;
- cải thiện correct/exact-match hoặc top-k ranking dưới cùng candidate budget;
- gain xuất hiện trên hơn một ngôn ngữ hoặc được phân tích trung thực nếu chỉ tập trung ở một nhóm bug;
- toàn bộ run có thể truy vết bằng commit, config, checkpoint hash và manifest;
- báo cả negative cases và compute overhead.

Nếu adaptive selector không tăng số fix, kết quả vẫn có thể có giá trị nếu chứng minh được khi nào retrieval gây hại và đưa ra policy giảm compute/truncation mà giữ nguyên effectiveness. Tuy nhiên không nên tuyên bố cải thiện repair nếu chỉ giảm token hoặc thời gian mà không có kiểm định tương ứng.

## 12. Tài liệu gốc

- MultiMend source: <https://github.com/h4iku/MultiMend>
- Paper (arXiv v2): <https://arxiv.org/abs/2501.16044>
- Official data/model collection: liên kết Hugging Face trong README upstream.

Paper hiện tại được ghi nhận là arXiv `2501.16044`, không phải `2307.10044`.
