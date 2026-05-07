# Lab 21 — Báo cáo đánh giá LoRA/QLoRA

**Họ tên**: Lê Thành Long — 2A202600105  
**Hình thức nộp**: Option B — GitHub + Hugging Face Hub  
**GitHub repo**: https://github.com/ltlongg/Lab21_2A202600105  
**Hugging Face adapter**: https://huggingface.co/ltlonggg/lab21-qwen2.5-3b-bnb-4bit-r16

## 1. Setup

Trong bài lab này, em fine-tune model `unsloth/Qwen2.5-3B-bnb-4bit` bằng QLoRA 4-bit. Dataset em dùng là `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, lấy 200 samples và chia thành 180 mẫu train, 20 mẫu eval với seed = 42.

Các mẫu dữ liệu được đưa về format Alpaca gồm `instruction`, `input`, `output`. Sau khi phân tích độ dài token, em đặt `max_seq_length = 1024`. GPU sử dụng là NVIDIA L4, khoảng 23 GB VRAM. Cấu hình train chính gồm 3 epochs, learning rate `2e-4`, cosine scheduler, warmup ratio `0.10`, effective batch size = 8. LoRA được gắn vào 2 module `q_proj` và `v_proj`.

Adapter em chọn để upload lên Hugging Face là `r=16`, vì dù `r=64` có perplexity tốt hơn, `r=16` cân bằng hơn giữa chất lượng và chi phí.

## 2. Kết quả thí nghiệm rank

| Rank | Alpha | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 1,843,200 | 2.76 phút | 13.52 GB | 1.5598 | 4.7578 |
| 16 | 32 | 3,686,400 | 2.78 phút | 12.92 GB | 1.5186 | 4.5656 |
| 64 | 128 | 14,745,600 | 2.79 phút | 14.30 GB | 1.4771 | 4.3804 |

Nhìn vào bảng thì rank càng cao, eval loss và perplexity càng giảm. `r=64` là rank có metric tốt nhất. Tuy nhiên, số trainable parameters của `r=64` cao hơn `r=16` rất nhiều, trong khi mức cải thiện perplexity không quá lớn. Vì vậy em xem `r=16` là lựa chọn hợp lý hơn nếu cần chọn một adapter để nộp hoặc deploy.

## 3. Phân tích loss curve

Loss curve được lưu ở `results/loss_curve.png`.

Trong quá trình train, training loss giảm khá ổn định, nên có thể thấy model thật sự học được từ dataset tiếng Việt. Notebook không bật eval giữa training để tránh OOM, nên em chủ yếu dựa vào eval loss cuối cùng, perplexity và output qualitative để đánh giá.

Kết quả cũng khá đúng với kỳ vọng: `r=8` có ít capacity nhất nên perplexity cao nhất, `r=64` có nhiều capacity nhất nên perplexity thấp nhất. Nhưng điểm đáng chú ý là từ `r=16` lên `r=64`, số tham số tăng 4 lần trong khi perplexity chỉ giảm từ 4.5656 xuống 4.3804. Với em, đây là dấu hiệu diminishing returns khá rõ.

Em cũng có dùng W&B để tracking các run. Tuy nhiên link W&B bị 404 khi mở từ ngoài vì workspace nằm dưới entity bị giới hạn quyền truy cập. Vì vậy em lưu ảnh chụp W&B vào repo tại `results/wandb_r8_16_64.png` để làm bằng chứng.

## 4. So sánh qualitative

File đầy đủ nằm ở `results/qualitative_comparison.csv`. Dưới đây là 5 prompts em dùng để so sánh base model với model fine-tuned `r=16`.

| Prompt | Base Model | Fine-tuned r=16 | Nhận xét |
|---|---|---|---|
| Giải thích khái niệm machine learning cho người mới bắt đầu. | Có giải thích được ý chính nhưng hơi dài dòng và lặp. | Câu trả lời dễ hiểu hơn, hợp với người mới bắt đầu hơn. | Cải thiện |
| Viết đoạn code Python tính số Fibonacci thứ n. | Có nói đến đệ quy/vòng lặp nhưng output bị cắt. | Có cấu trúc hơn nhưng code vẫn bị cắt, chưa thật sự tốt. | Chưa ổn định |
| Liệt kê 5 nguyên tắc thiết kế UI/UX. | Một số ý nghe chưa đúng, ví dụ "Kỷ luật" và "Thảm họa". | Các ý như thuận tiện, thống nhất, dễ đọc, đơn giản hợp lý hơn. | Cải thiện |
| Tóm tắt sự khác biệt giữa LoRA và QLoRA. | Giải thích sai tên LoRA và còn khá mơ hồ. | Nhận diện đúng LoRA là Low-Rank Adaptation và QLoRA là bản quantized. | Cải thiện rõ |
| Phân biệt prompt engineering, RAG, và fine-tuning. | Có trả lời nhưng còn chung chung. | Cấu trúc tốt hơn, nhưng vẫn chưa phân biệt thật sắc giữa RAG và fine-tuning. | Cải thiện nhẹ |

Nhìn chung, adapter fine-tuned giúp câu trả lời tiếng Việt tự nhiên hơn và dùng thuật ngữ đúng hơn, đặc biệt ở câu hỏi về LoRA/QLoRA. Tuy vậy, model vẫn chưa hoàn hảo. Với prompt yêu cầu code, output vẫn có thể bị cắt hoặc chưa đủ chính xác. Vì vậy em thấy qualitative evaluation rất cần thiết, vì chỉ nhìn perplexity thì chưa đủ.

## 5. Kết luận về trade-off của rank

Nếu chỉ nhìn vào perplexity, `r=64` là rank tốt nhất vì đạt 4.3804, thấp hơn cả `r=8` và `r=16`. Tuy nhiên, em không chọn `r=64` làm adapter chính vì nó dùng 14,745,600 trainable parameters, tức là gấp 4 lần `r=16`. Trong khi đó, `r=16` đạt perplexity 4.5656, không quá xa so với `r=64`, nhưng nhẹ hơn nhiều. So với `r=8`, `r=16` cải thiện khá rõ từ 4.7578 xuống 4.5656, nên `r=16` là bước nâng cấp đáng giá. Với dataset chỉ có 200 samples, em nghĩ tăng rank lên quá cao có thể không đem lại lợi ích tương xứng, thậm chí nếu train lâu hơn hoặc dataset nhỏ hơn còn có nguy cơ overfit. Vì vậy, kết luận của em là `r=64` tốt nhất về metric, nhưng `r=16` là lựa chọn tốt nhất về trade-off và phù hợp hơn để upload/deploy.

## 6. Em học được gì

- Rank trong LoRA giống như capacity của adapter: tăng rank thì model học được nhiều hơn, nhưng chi phí cũng tăng.
- QLoRA rất hữu ích vì có thể fine-tune model 3B trên một GPU L4 mà không cần full fine-tune.
- Perplexity giúp so sánh định lượng, nhưng vẫn phải đọc output thật vì model có thể metric tốt hơn nhưng câu trả lời vẫn bị lỗi format hoặc bị cắt.

## Bonus Evidence

- **Hugging Face Hub adapter**: https://huggingface.co/ltlonggg/lab21-qwen2.5-3b-bnb-4bit-r16
- **GitHub repository**: https://github.com/ltlongg/Lab21_2A202600105
- **Bằng chứng W&B tracking**: `results/wandb_r8_16_64.png`
