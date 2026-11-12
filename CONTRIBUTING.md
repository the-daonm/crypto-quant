# Hướng dẫn đóng góp

Cảm ơn bạn quan tâm tới **crypto-quant**! Mọi đóng góp đều được chào đón: sửa lỗi, thêm thí nghiệm, cải thiện tài liệu,
báo cáo kết quả (kể cả kết quả "không tốt"), hay chỉ là đặt câu hỏi.

Khi tham gia dự án, bạn đồng ý tuân theo [Quy tắc ứng xử](CODE_OF_CONDUCT.md).
Mới làm quen với crypto, ML hay backtest? Đọc [docs/GIAI-THICH.md](docs/GIAI-THICH.md) trước.

## Báo lỗi hoặc đề xuất

- Tìm trong [Issues](https://github.com/the-daonm/crypto-quant/issues) xem đã có ai báo chưa.
- Chưa có thì mở issue mới bằng một trong các template: **Báo lỗi**, **Đề xuất tính năng**, **Thí nghiệm**.
- Issue có label [`good first issue`](https://github.com/the-daonm/crypto-quant/labels/good%20first%20issue) phù hợp cho người mới.

## Cài đặt môi trường phát triển

```bash
git clone https://github.com/<username-của-bạn>/crypto-quant.git   # sau khi fork
cd crypto-quant
python -m venv .venv && source .venv/bin/activate                 # fish: source .venv/bin/activate.fish
pip install -e ".[dev,notebook]"
pytest                                                            # phải pass trước khi bắt đầu
```

## Quy trình làm việc

1. **Fork** repo (thành viên nhóm có quyền ghi thì có thể tạo branch trực tiếp).
2. Tạo branch từ `main` với tên mô tả: `feat/funding-rate`, `fix/backtest-fee`, `docs/giai-thich-sharpe`, `exp/ablation`.
3. Code, viết test, chạy kiểm tra:
   ```bash
   ruff check . && ruff format --check .
   pytest
   ```
4. Commit (xem quy tắc bên dưới), push, rồi mở **Pull Request** vào `main` theo template.
5. Cần **ít nhất 1 người review** và CI pass trước khi merge. Người review nên chạy thử code, không chỉ đọc.
6. Ưu tiên **squash merge** hoặc rebase để lịch sử `main` gọn.

## Quy tắc commit

- **Một commit = một mục đích.** Không gộp sửa lỗi, format code và tính năng mới vào cùng một commit.
- Tiêu đề viết ở thể mệnh lệnh, ngắn gọn (khoảng ≤ 72 ký tự), tiếng Anh hoặc tiếng Việt đều được,
  ví dụ `Add funding rate features`, `Fix fee double-count on exit`.
- Nếu tiêu đề cần chữ "and" để mô tả, hãy cân nhắc tách thành hai commit.
- Mỗi commit nên tự build và tự pass test được.

## Quy tắc code

- Python ≥ 3.11, format bằng `ruff format`, lint bằng `ruff check` (cấu hình trong `pyproject.toml`).
- Docstring và comment có thể viết tiếng Việt.
- Code mới phải có test trong `tests/`. **Test không được gọi mạng**: dùng dữ liệu giả, xem `_synthetic_ohlcv()` trong `tests/test_features.py`.
- Không commit dữ liệu thô (`data/`), kết quả (`results/`) hay thông tin bí mật (API key, …).

## Quy tắc phương pháp (bắt buộc với thí nghiệm)

Đây là dự án đánh giá chiến lược, nên **phương pháp đúng quan trọng hơn con số đẹp**:

1. **Không leakage.** Feature tại nến t chỉ dùng dữ liệu ≤ t, và phải có test kiểm tra, xem `test_features_do_not_use_future_data`.
2. **Chỉ dùng giai đoạn design** (trước `holdout_start` trong `config.yaml`) để chọn cấu hình hoặc tham số.
   Giai đoạn holdout chỉ được chạy **một lần** cho cấu hình cuối cùng.
3. Luôn so sánh với **Buy & Hold** và báo cáo kết quả **sau phí**.
4. Ghi lại **số cấu hình đã thử**; con số này cần để hiệu chỉnh khi đánh giá thống kê.
5. Thí nghiệm "thất bại" cũng mở PR hoặc ghi vào issue: đó là kết quả có giá trị.

## Phát hành phiên bản

Dự án dùng [Semantic Versioning](https://semver.org/lang/vi/) và ghi thay đổi trong [CHANGELOG.md](CHANGELOG.md)
theo định dạng [Keep a Changelog](https://keepachangelog.com/vi/1.1.0/). Mỗi PR có thay đổi đáng kể nên thêm một dòng vào mục
`[Unreleased]` của CHANGELOG.

## License

Khi đóng góp, bạn đồng ý rằng đóng góp của mình được phát hành theo [MIT License](LICENSE) của dự án.
