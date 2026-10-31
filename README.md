# Document OCR Service

企业文档 OCR 与结构化解析服务。接收图片、PDF 或 Office 文件，经过图像预处理、
版面与表格分析、文字识别，输出带字符级坐标和置信度的结构化 JSON。

## 功能

- 多格式输入：图片、PDF、Office，支持页码范围
- 扫描件 / 拍照件自适应，拍照件可做透视矫正
- 多页投票去水印，保留正文与表格内容
- YOLO 红章检测、章面文字识别与像素级去除
- 整页与表格方向矫正（0 / 90 / 180 / 270 度）
- 版面分析、表格检测、单元格 OCR 与坐标回填
- 页眉页脚、页码、重复水印过滤
- MinIO / S3 与 HTTP 代理双通道文件存取
- Docker 部署、模型懒加载、版面缓存与 GPU 推理锁

## 处理流程

```text
文件下载 -> 格式转换 -> 二维码/水印去除 -> 方向矫正
        -> 红章去除 -> 页眉页脚过滤 -> 版面分析
        -> 表格检测/单元格 OCR -> 文本 OCR -> 结构化 JSON
```

## 技术栈

- Python, Flask, Gunicorn
- PaddleOCR PP-OCRv6, PP-DocLayoutV2
- RT-DETR 表格单元格检测, PP-LCNet 表格分类
- YOLO11 印章检测, ONNX Runtime
- OpenCV, NumPy, PyMuPDF, pdfplumber
- MinIO / S3, Docker, CUDA

## 目录结构

```text
.
├── index.py                  # Flask 入口与流水线编排
├── config.py                 # 模型初始化与懒加载
├── ocr.py                    # OCR 结果整理、字符级输出
├── parse_file_director.py    # 段落/目录/页码过滤
├── s3.py                     # 对象存储客户端
├── cryptor.py                # 模型目录加密工具
├── algo/
│   ├── enhancement/          # 背景归一化、多页投票去水印
│   ├── redmask_remove/       # 红章检测与去除
│   ├── table_detect/         # 表格结构识别
│   ├── header_footer/        # 页眉页脚处理
│   ├── pdf/                  # PDF 转图与图像分析
│   ├── handwriting/          # 手写内容处理
│   ├── remove_qrcode/        # 二维码定位
│   └── images_detect/        # 文档图像相似度
├── Dockerfile
└── requirements.txt
```

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # 按实际环境填写
python index.py
```

服务默认监听 `0.0.0.0:5000`。

## 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `SEND_SERVER` | 结果图片上传服务 | `http://127.0.0.1:5700` |
| `hostname` | 服务对外地址 | `http://127.0.0.1:5000` |
| `MODEL_KEY` | 模型目录 Fernet 加密密钥 | 空 |
| `STORAGE_MODE` | `proxy` 或 `minio` | `proxy` |
| `S3_PROXY_SERVER` | 对象存储代理地址 | `http://127.0.0.1:30230` |
| `MINIO_ENDPOINT` | MinIO 地址 | 空 |
| `MINIO_ACCESS_KEY` | MinIO Access Key | 空 |
| `MINIO_SECRET_KEY` | MinIO Secret Key | 空 |
| `MINIO_BUCKET` | MinIO Bucket | 空 |
| `DEBUG_SAVE_DIR` | 调试图片输出目录 | 空 |

## API

### POST /ocr_process

```json
{
  "fileType": "imagePdf | office | textPdf | image",
  "taskId": "string",
  "fileKey": "S3 key or HTTP URL",
  "forceImage": true,
  "ignoreSeals": false,
  "ignorePageHeaderAndFooter": false,
  "ignoreContentTable": false,
  "pageRange": [[1, 5], [7, 10]],
  "fileExt": "",
  "detectOrientation": true,
  "removeWatermark": true
}
```

返回按页组织的 `image`、`tables`、`lines`、`red_masks` 和 `page` 字段。

## 模型文件

模型权重不包含在本仓库中。请将 OCR、版面、表格、方向、红章等模型放到 `models/`
目录，或通过 `MODEL_KEY` 配合 `cryptor.py` 管理加密模型。

## 说明

本仓库为代码展示快照，已移除客户文档、模型权重、内部服务地址与密钥。
请勿将仓库中的代码或数据用于未经授权的商业用途。
