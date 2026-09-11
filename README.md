# SL-YOLO

SL-YOLO mô phỏng quá trình huấn luyện một mô hình phát hiện đối tượng lần lượt qua nhiều thiết bị có phân phối dữ liệu IID hoặc non-IID. Mô hình sử dụng kiến trúc YOLO11n được dựng lại trực tiếp bằng PyTorch thay vì gọi model từ thư viện Ultralytics. Dữ liệu dùng cho huấn luyện và kiểm thử là VisDrone2019-DET với 10 lớp đối tượng.

Project hỗ trợ hai luồng huấn luyện:

- `full`: huấn luyện trên toàn bộ tập train như một dataset thông thường.
- `devices`: đưa cùng một model qua từng device; khi tất cả device được chọn hoàn tất thì kết thúc một round.

Luồng `devices` là mô phỏng huấn luyện tuần tự/continual trên các partition IID hoặc non-IID. Đây chưa phải federated learning hoàn chỉnh vì không tạo model cục bộ độc lập và không có bước tổng hợp trọng số như FedAvg.

## Thành phần chính

```text
SL_YOLO/
├── config.yaml                     # Cấu hình train, predict và test
├── model.py                        # Kiến trúc YOLO11n dựng lại bằng PyTorch
├── loss.py                         # Detection loss, assigner và DFL
├── dataset.py                      # Đọc ảnh và annotation YOLO/VisDrone
├── load_pretrained.py              # Nạp pretrained weight theo từng vùng model
├── train.py                        # Huấn luyện full hoặc tuần tự theo device
├── predict_image.py                # Dự đoán trên một ảnh
├── test.py                         # So sánh trực quan prediction và ground truth
├── scripts/data/
│   ├── generate_non_iid_labels.py  # Tạo nhãn số object cho việc chia dữ liệu
│   ├── create_data_partitions.py    # Chia tập train thành device IID/non-IID
│   └── plot_label_distribution.py  # Vẽ phân phối số object trên mỗi ảnh
├── yolo_state_dict.pt              # Pretrained checkpoint
└── requirement.txt
```

## Cài đặt

Yêu cầu Python 3.10 trở lên. Tạo và kích hoạt môi trường ảo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirement.txt
```

Nếu sử dụng GPU NVIDIA, nên cài bản PyTorch phù hợp với phiên bản CUDA của máy trước khi cài các dependency còn lại.

Môi trường phát triển hiện tại của project là `/home/truong/truongtd`:

```bash
source /home/truong/truongtd/bin/activate
```

## Chuẩn bị VisDrone2019

Tải và giải nén `VisDrone2019-DET-train` và `VisDrone2019-DET-val` vào `data/` theo cấu trúc:

```text
data/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations/
└── VisDrone2019-DET-val/
    ├── images/
    └── annotations/
```

Mỗi annotation VisDrone có dạng:

```text
left,top,width,height,score,category_id,truncation,occlusion
```

Category `1..10` được ánh xạ thành class `0..9`. Category `0` (ignored regions) và `11` (others) không được dùng làm class huấn luyện.

## Chia dữ liệu IID hoặc non-IID

### 1. Tạo label theo số lượng object

Chạy:

```bash
python3 scripts/data/generate_non_iid_labels.py
```

Script đọc annotation VisDrone và tạo một file trong `labels/` cho mỗi ảnh. Mỗi file chỉ chứa một số nguyên là tổng số object hợp lệ trong ảnh:

```text
VisDrone2019-DET-train/
├── images/
├── annotations/    # Bounding box dùng để huấn luyện detection
└── labels/         # Số object dùng riêng để chia non-IID
```

Nếu cần ghi đè label đã thay đổi:

```bash
python3 scripts/data/generate_non_iid_labels.py --overwrite
```

### 2. Quan sát phân phối label

```bash
python3 scripts/data/plot_label_distribution.py
```

Biểu đồ mặc định được lưu tại `runs/label_distribution.png`.

### 3. Tạo các device non-IID

```bash
/home/truong/truongtd/bin/python scripts/data/create_data_partitions.py \
  --strategy non-iid --devices 8
```

Cách chia dữ liệu:

1. Lấy `source_id` từ phần đứng trước dấu gạch dưới đầu tiên trong tên ảnh.
2. Gom các ảnh có cùng `source_id` để chúng không bị tách sang nhiều device.
3. Tính số object trung bình của từng source.
4. Sắp xếp source theo số object trung bình.
5. Dùng dynamic programming để chia các đoạn source liên tiếp, giữ đặc tính non-IID nhưng giảm chênh lệch số ảnh giữa các device.

Kết quả mặc định:

```text
data/VisDrone2019-DET-train-non-iid-8/
├── manifest.csv
├── summary.json
├── device_01/
│   ├── images/
│   ├── annotations/
│   ├── labels/
│   └── sources.txt
├── device_02/
└── ...
```

Các file dữ liệu trong device là symbolic link đến dataset gốc để không nhân đôi ảnh. Vì vậy không nên di chuyển hoặc xóa dataset nguồn sau khi chia. Script không ghi đè thư mục output đã tồn tại; hãy chọn `--output-dir` khác nếu muốn giữ kết quả cũ.

### 4. Tạo 8 device IID

```bash
/home/truong/truongtd/bin/python scripts/data/create_data_partitions.py \
  --strategy iid --devices 8 --seed 42
```

Với chiến lược IID, mỗi ảnh chỉ xuất hiện đúng một lần nhưng các ảnh có cùng
`source_id` được phép nằm trên nhiều device. Script phân tầng theo tổng số
object trong ảnh, sau đó cân bằng thêm số object của 10 class. Vì vậy:

- chênh lệch tổng số ảnh giữa các device không quá 1;
- tại mỗi giá trị `object_count`, số ảnh giữa các device chênh không quá 1;
- class hiếm được ưu tiên đưa tới device đang thiếu class đó.

Kết quả mặc định nằm tại
`data/VisDrone2019-DET-train-iid-8/`. File `summary.json` chứa histogram
`object_count`, tổng object và tỷ lệ của từng class trên mỗi device để kiểm tra
mức độ IID. Thay đường dẫn trong `train.devices.items` của `config.yaml` từ
`VisDrone2019-DET-train-non-iid-8` sang `VisDrone2019-DET-train-iid-8` để train
trên partition mới.

## Cấu hình huấn luyện

Toàn bộ cấu hình nằm trong `config.yaml`. Ví dụ huấn luyện theo device:

```yaml
train:
  data_mode: devices
  rounds: 50
  batch_size: 16
  workers: 4

  devices:
    order: sequential
    local_epochs: 1
    items:
      - id: 1
        name: device_01
        path: data/VisDrone2019-DET-train-non-iid-8/device_01
        enabled: true

      - id: 8
        name: device_08
        path: data/VisDrone2019-DET-train-non-iid-8/device_08
        enabled: true
```

Thứ tự khai báo trong `items` là thứ tự cơ sở. `id` chỉ dùng để định danh; code không tự sắp xếp lại theo ID. Đặt `enabled: false` để bỏ qua một device.

Ba chiến lược thứ tự:

- `sequential`: giữ nguyên thứ tự trong `items` ở mọi round.
- `shuffle`: tạo một hoán vị mới ở mỗi round, có thể tái lập bằng `train.seed`.
- `rotate`: dịch device bắt đầu sau mỗi round nhưng giữ thứ tự tương đối.

Ví dụ với thứ tự cấu hình `[1, 2, 8]`, chế độ `rotate` chạy:

```text
Round 1: 1 → 2 → 8
Round 2: 2 → 8 → 1
Round 3: 8 → 1 → 2
```

`local_epochs` là số lần duyệt dữ liệu của một device trước khi chuyển sang device kế tiếp. Model, optimizer và AMP scaler được giữ liên tục giữa các device. Validation và learning-rate scheduler chỉ chạy sau khi toàn bộ device trong round hoàn tất.

Để dùng toàn bộ tập train, đổi:

```yaml
train:
  data_mode: full
```

Trong chế độ `full`, một round tương đương một lần duyệt toàn bộ tập train và phần cấu hình `devices` không được sử dụng.

## Huấn luyện

Sau khi chỉnh `config.yaml`:

```bash
python3 train.py
```

Hoặc dùng một file config khác:

```bash
python3 train.py --config path/to/config.yaml
```

Khi `output_dir: null`, kết quả được lưu tự động tại:

- `runs/visdrone_full/` với `data_mode: full`.
- `runs/visdrone_devices/` với `data_mode: devices`.

Các checkpoint chính:

- `last.pt`: trạng thái sau round mới nhất.
- `best.pt`: checkpoint có validation loss thấp nhất.

Checkpoint chứa model, optimizer, scheduler, round hiện tại, thứ tự device thực tế và toàn bộ cấu hình huấn luyện.

## Dự đoán trên một ảnh

Chỉnh section `predict` trong `config.yaml`:

```yaml
predict:
  checkpoint: runs/visdrone_devices/best.pt
  image: data/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg
  output: runs/visdrone_devices/prediction.jpg
  image_size: 640
  confidence_threshold: 0.25
  iou_threshold: 0.45
  max_detections: 300
  device: auto
```

Sau đó chạy:

```bash
python3 predict_image.py
```

`device: auto` ưu tiên CUDA, sau đó MPS và cuối cùng là CPU. Script in danh sách detection và lưu ảnh đã vẽ bounding box tại đường dẫn `output`.

## Kiểm tra hiệu quả mô hình

### Theo dõi trong quá trình train

Sau mỗi round, chương trình in:

- Train loss trên mỗi ảnh của toàn round.
- Validation loss trên tập `VisDrone2019-DET-val`.
- Learning rate hiện tại.
- Best validation loss.

Có thể dùng validation loss và `best.pt` để so sánh các cấu hình huấn luyện trong cùng điều kiện.

### So sánh prediction với ground truth

Chỉnh section `test` trong `config.yaml`, sau đó chạy:

```bash
python3 test.py
```

Script sẽ:

1. Chạy inference trên ảnh được cấu hình.
2. Lưu ảnh prediction.
3. Tìm annotation VisDrone tương ứng và lưu ảnh ground truth.
4. Mở hai cửa sổ để so sánh trực quan.

Chức năng hiển thị yêu cầu môi trường desktop và gói `opencv-python` có GUI. Trên server headless, có thể dùng `predict_image.py` và xem các file output sau khi tải về.

`test.py` hiện chỉ hỗ trợ đánh giá định tính trên một ảnh; project chưa triển khai mAP, precision, recall hoặc đánh giá toàn bộ validation set. Để báo cáo theo chuẩn object detection, cần bổ sung evaluator cho mAP@0.5 và mAP@0.5:0.95.

## Kiểm tra pretrained checkpoint

Có thể xác nhận checkpoint pretrained tương thích với model bằng:

```bash
python3 load_pretrained.py
```

Section `pretrained` trong `config.yaml` cho phép chọn phạm vi load:

- `backbone`: block `0..10`.
- `backbone_neck_partial`: block `0..16`.
- `backbone_neck`: block `0..22`.

Detection head không được load từ checkpoint COCO vì VisDrone có 10 class thay vì 80 class.
