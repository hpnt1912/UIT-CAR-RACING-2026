======================================================================
              HƯỚNG DẪN CHẠY CHƯƠNG TRÌNH ĐIỀU KHIỂN XE UCR
======================================================================

TÊN FILE CODE CHÍNH: main.py

BUỚC 1: THỰC THI CHƯƠNG TRÌNH ĐIỀU KHIỂN
1. BTC bam chuyen doi thanh AV Mode
2. Tại thư mục /workspace, nhập câu lệnh sau để khởi chạy chương trình:
   python main.py
3. Nhấn Enter để bắt đầu chạy script.

BUỚC 2: TÍN HIỆU SẮN SÀNG VÀ XUẤT PHÁT
1. Chương trình sẽ tự động load mô hình nhận diện (lane_seg.onnx) và bộ điều khiển PID.
2. Khi mô hình khởi tạo thành công, Terminal sẽ xuất hiện thông báo:

   ==============================================
                    SAN SANG                     
   ==============================================

3. Ngay khi dòng chữ "SAN SANG" xuất hiện, BTC chỉ cần bấm nút "START" trên mô phỏng Unity. Xe sẽ tự động nhận lệnh và xuất phát.

GHI CHÚ / THÔNG TIN BỔ SUNG:
- Thời gian chạy tối đa được cấu hình là 7 phút (420 giây), chương trình sẽ tự động ngắt an toàn sau khi hoàn thành lượt thi.
======================================================================