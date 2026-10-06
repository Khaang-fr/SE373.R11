# BTVN #3 — Agent đặt vé máy bay

## Mục tiêu

Xây dựng một agent đặt vé máy bay có thể chạy offline trên dữ liệu giả, dùng
tool của LangChain và LangGraph; cài đặt lớp harness để giới hạn hành động,
kiểm tra quyền, xác định hoàn tất bằng code và bàn giao tác vụ không thể xử lý.
Không cần API key hoặc gọi dịch vụ hàng không thật.

## Thành phần

- `tools.py`: bốn tool `search_flights`, `get_flight_details`, `book_flight`,
  `cancel_booking`; dữ liệu chuyến bay HAN/SGN/DAD, booking giả trong bộ nhớ,
  kiểm tra dữ liệu, ghế trống, quyền sở hữu booking và reset fixture. Bốn hàm
  cũng được bọc thành `StructuredTool` trong `LANGCHAIN_TOOLS`.
- `harness.py`: `HarnessPolicy` chứa quyền theo vai trò và ràng buộc dưới dạng
  dữ liệu; `HarnessContext` kiểm tra từng lời gọi tool, ghi log, chặn vượt giới
  hạn; `Harness._check_completion` xác nhận kết quả bằng code; phản hồi handoff
  dùng khi yêu cầu thiếu thông tin, không hỗ trợ hoặc bị policy chặn.
- `agents.py`:
  - **ReAct**: chọn hành động kế tiếp từ quan sát mới nhất; khi chưa có mã
    chuyến, tìm và chọn vé rẻ nhất trước khi đọc chi tiết và đặt.
  - **Plan-then-Execute**: tạo danh sách bước rồi thực hiện tuần tự.
  - **Hybrid (LangGraph)**: dùng `StateGraph` để lập kế hoạch, chạy từng bước
    và kết thúc hoặc dừng khi có lỗi.
- `eval.py`: 10 tình huống được chạy trên cả 3 agent, gồm tìm/đọc/đặt/hủy vé,
  xác nhận còn thiếu, tác vụ không hỗ trợ, khách không có quyền đặt, ngày bị
  chặn và hủy booking của người khác.
- `main.py`: lệnh chạy demo đánh giá; trả mã lỗi khác 0 nếu có kết quả sai.

## Harness và chính sách

Quyền `guest/customer/admin`, giới hạn tối đa 6 hành khách, ngày bay được phép,
giới hạn 8 lượt gọi tool và yêu cầu xác nhận trước khi đặt được khai báo trong
`HarnessPolicy`, không rải thành điều kiện trong từng agent. `HarnessContext`
thực thi policy ngay tại ranh giới tool; email khi đặt/hủy lấy từ danh tính đã
đăng nhập của harness, không tin email do agent truyền vào. Tool hủy còn kiểm
tra chủ booking độc lập.

Tiêu chí hoàn tất được ánh xạ theo intent trong code: tìm chuyến cần có lời gọi
tìm kiếm thành công; xem chi tiết cần lấy được chi tiết; đặt/hủy cần tool tương
ứng trả thành công. Với tình huống yêu cầu handoff, đánh giá xác nhận agent đã
bàn giao và không báo hoàn tất. Mọi lời gọi và policy rejection đều có trong
trace kết quả.

## Cách chạy

Từ thư mục bài tập, cài dependency và chạy:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Các phiên bản được khai báo trong `requirements.txt`; dùng Python 3.10 trở lên.
Có thể chạy riêng benchmark bằng `python eval.py`.

## Đánh giá

Chạy trên 10 tình huống, mỗi tình huống một lần cho mỗi agent (30 lượt tổng).
`Correct outcomes` gồm cả tác vụ hoàn tất đúng và handoff đúng. `Avg tool calls`
bao gồm cả lời gọi bị policy từ chối. `Avg trace events` phản ánh độ chi tiết
trace. Thời gian là trung bình một lần chạy trên môi trường bài tập, chỉ để tham
khảo và không phải benchmark thống kê.

Kết quả một lần chạy:

| Strategy | Correct outcomes | Accuracy | Correct handoffs | Policy blocks | Avg tool calls | Avg trace events | Avg time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReAct | 10/10 | 100% | 5 | 2 | 1.20 | 3.00 | 0.113 |
| Plan-then-Execute | 10/10 | 100% | 5 | 2 | 1.20 | 3.80 | 0.111 |
| Hybrid (LangGraph) | 10/10 | 100% | 5 | 2 | 1.20 | 3.80 | 2.007 |

## Nhận xét và giới hạn

Ba thiết kế đạt cùng độ chính xác và số tool call trên bộ fixture này; ReAct
ghi trace ngắn hơn, còn hai thiết kế có kế hoạch tường minh. LangGraph có thời
gian cao hơn trong lần chạy này do overhead của graph, không phải tốc độ của
một LLM. Vì agent dùng quyết định code-based thay vì gọi LLM, kết quả chứng minh
luồng điều khiển, xử lý quyền và tính nhất quán trên mock data, không đo chất
lượng suy luận ngôn ngữ tự nhiên. Dữ liệu booking chỉ ở bộ nhớ và sẽ mất khi
process kết thúc.
