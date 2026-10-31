import os
import cv2
import json
import statistics
import numpy as np
from enum import Enum
import paddle
import tensorflow as tf
import onnxruntime as ort
from paddleocr import PaddleOCR
from paddlex import create_model, create_predictor
from scipy.special import softmax
from paddleocr import LayoutDetection, TableClassification
from keras.applications.vgg16 import VGG16
from algo.handwriting.remove_handwriting import load_model


# ============================================================
# 常量
# ============================================================
slope_width = 1985
hot_area = 0.12  # 页眉页脚热区占比

rotate_batch = 8
rotate_batch_time = 6
rotate_step = 0.2

cutoff = 150  # 截止频率
order = 2  # 滤波器阶数
window = 4  # 方差滑动窗口大小
throttle = 39  # 线条阀门

ORIENT_CLASSES = ["0°", "90°", "180°", "270°"]

CLASS_NAMES = {0: 'seal'}


# ============================================================
# 执行提供者枚举
# ============================================================
class EP(Enum):
    """执行提供者枚举"""
    CPU_EP = "CPUExecutionProvider"
    CUDA_EP = "CUDAExecutionProvider"
    DIRECTML_EP = "DmlExecutionProvider"
    CANN_EP = "CANNExecutionProvider"


# ============================================================
# ONNX 推理会话封装
# ============================================================
class OrtInferSession:
    """支持多种执行提供者的ONNX模型加载类"""

    def __init__(self, model_path, providers=None):
        if providers is None:
            providers = [EP.CPU_EP.value]

        available_providers = ort.get_available_providers()
        selected_providers = [p for p in providers if p in available_providers]

        if not selected_providers:
            selected_providers = [EP.CPU_EP.value]
            print(f"警告: 没有可用的执行提供者，将使用默认的{EP.CPU_EP.value}")

        self.session = ort.InferenceSession(model_path, providers=selected_providers)
        print(f"模型加载完成，使用的执行提供者: {selected_providers}")

    def run(self, input_name, input_data):
        return self.session.run(None, {input_name: input_data})

    def get_input_name(self):
        return self.session.get_inputs()[0].name


# ============================================================
# 印章识别模型
# ============================================================
class RedMaskRec:
    def __init__(self, model_path, use_cpu=True, use_gpu=False, use_dml=False, use_cann=False):
        self.model_path = model_path

        providers = []
        if use_cann:
            providers.append(EP.CANN_EP.value)
        if use_gpu:
            providers.append(EP.CUDA_EP.value)
        if use_dml:
            providers.append(EP.DIRECTML_EP.value)
        if use_cpu or not providers:
            providers.append(EP.CPU_EP.value)

        encoder_path = os.path.join(model_path, "encoder_model.onnx")
        decoder_path = os.path.join(model_path, "decoder_model.onnx")

        self.encoder = OrtInferSession(encoder_path, providers)
        self.decoder = OrtInferSession(decoder_path, providers)

        self.vocab = self._read_vocab(os.path.join(model_path, "vocab.json"))
        self.vocab_inp = {self.vocab[key]: key for key in self.vocab}
        self.threshold = 0.5
        self.max_len = 50

    @staticmethod
    def _read_vocab(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _do_norm(x):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        x = x / 255.0
        x[0, :, :] -= mean[0]
        x[1, :, :] -= mean[1]
        x[2, :, :] -= mean[2]
        x[0, :, :] /= std[0]
        x[1, :, :] /= std[1]
        x[2, :, :] /= std[2]
        return x

    def _decode_text(self, tokens):
        s_start = self.vocab.get('<s>')
        s_end = self.vocab.get('</s>')
        unk = self.vocab.get('<unk>')
        pad = self.vocab.get('<pad>')
        text = ''
        for tk in tokens:
            if tk == s_end:
                break
            if tk not in [s_end, s_start, pad, unk]:
                text += self.vocab_inp[tk]
        return text

    def run(self, image):
        image = cv2.resize(image, (384, 384))
        pixel_values = cv2.split(np.array(image))
        pixel_values = self._do_norm(np.array(pixel_values))
        pixel_values = np.array([pixel_values])

        encoder_output = self._run_encoder(pixel_values)
        text, scores = self._run_decoder(encoder_output)
        return text, scores

    def _run_encoder(self, image):
        input_name = self.encoder.get_input_name()
        encoder_output = self.encoder.run(input_name, np.asarray(image, dtype='float32'))[0]
        return encoder_output

    def _run_decoder(self, encoder_output):
        ids = [self.vocab["<s>"], ]
        mask = [1, ]
        scores = []

        for i in range(self.max_len):
            attention_mask = np.array([mask]).astype('int64')
            input_ids = np.array([ids]).astype('int64')

            onnx_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "encoder_hidden_states": encoder_output
            }

            input_names = [inp.name for inp in self.decoder.session.get_inputs()]
            input_feed = {name: onnx_inputs[name] for name in input_names}

            decoder_output = self.decoder.session.run(None, input_feed)[0]
            pred = softmax(decoder_output[0], axis=1)
            max_index = pred.argmax(axis=1)

            if max_index[-1] == self.vocab["</s>"]:
                break

            score = float(pred[max_index.shape[0] - 1, max_index[-1]])
            scores.append(score)
            ids.append(max_index[-1])
            mask.append(1)

        print("解码单字评分：{}".format(scores))
        print("解码平均评分：{}".format(statistics.mean(scores)))

        if statistics.mean(scores) >= self.threshold:
            text = self._decode_text(ids)
        else:
            text = ""
        return text, scores


# ============================================================
# YOLO11 印章检测
# ============================================================
class YOLO11:
    def __init__(self, onnx_model_path, input_size=(640, 640), confidence_thres=0.7, iou_thres=0.45,
                 use_cpu=True, use_gpu=False, use_dml=False, use_cann=False):
        self.onnx_model_path = onnx_model_path
        self.input_size = input_size
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        providers = []
        if use_cann:
            providers.append(EP.CANN_EP.value)
        if use_gpu:
            providers.append(EP.CUDA_EP.value)
        if use_dml:
            providers.append(EP.DIRECTML_EP.value)
        if use_cpu or not providers:
            providers.append(EP.CPU_EP.value)

        self.session = OrtInferSession(onnx_model_path, providers)
        self.input_name = self.session.get_input_name()

    def pre_image(self, img):
        img_height, img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, ratio, (dw, dh) = self.letterbox(img, new_shape=self.input_size)
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data, ratio, dw, dh, img_width, img_height

    @staticmethod
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        shape = img.shape[:2]

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = int(np.mod(dw, 2) / 2), int(np.mod(dh, 2) / 2)

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = dh, new_shape[0] - new_unpad[1] - dh
        left, right = dw, new_shape[1] - new_unpad[0] - dw
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img, (r, r), (dw, dh)

    def postprocess(self, outputs, img_width, img_height, ratio, dw, dh):
        outputs = np.transpose(np.squeeze(outputs[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                x -= dw
                y -= dh
                x /= ratio[0]
                y /= ratio[1]
                w /= ratio[0]
                h /= ratio[1]

                x = max(x, 0)
                y = max(y, 0)
                w = max(w, 0)
                h = max(h, 0)

                x = min(x, img_width)
                y = min(y, img_height)
                w = min(w, img_width - x)
                h = min(h, img_height - y)

                left = int(x - w / 2)
                top = int(y - h / 2)
                width = int(w)
                height = int(h)

                left = max(left, 0)
                top = max(top, 0)
                width = max(width, 0)
                height = max(height, 0)
                right = min(left + width, img_width)
                bottom = min(top + height, img_height)

                width = right - left
                height = bottom - top

                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        detections = [boxes[i] for i in indices]
        return detections

    def detect_objects(self, image):
        img = cv2.imread(image) if isinstance(image, str) else image
        img_data, ratio, dw, dh, img_width, img_height = self.pre_image(img)
        outputs = self.session.run(self.input_name, img_data)
        detections = self.postprocess(outputs, img_width, img_height, ratio, dw, dh)
        return detections


# ============================================================
# PaddlePaddle 原生推理封装（替代 ONNX，接口兼容 .run(None, {'x': data})）
# ============================================================
class PaddlePredictor:
    def __init__(self, model_dir, device="gpu"):
        import paddle.inference as paddle_infer
        cfg = paddle_infer.Config(
            os.path.join(model_dir, "inference.json"),
            os.path.join(model_dir, "inference.pdiparams"),
        )
        if device == "gpu":
            cfg.enable_use_gpu(100, 0)
        cfg.disable_glog_info()
        self._predictor = paddle_infer.create_predictor(cfg)
        self._input_names = self._predictor.get_input_names()
        self._output_names = self._predictor.get_output_names()

    def run(self, _, feed_dict):
        for name, data in feed_dict.items():
            handle = self._predictor.get_input_handle(name)
            handle.reshape(data.shape)
            handle.copy_from_cpu(data)
        self._predictor.run()
        return [self._predictor.get_output_handle(n).copy_to_cpu() for n in self._output_names]


# ============================================================
# 模型懒加载（按需初始化，首次访问时实例化并缓存）
# ============================================================
_models = {}


def _load_paddleocr_small():
    return PaddleOCR(
        use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
        text_detection_model_name='PP-OCRv6_small_det', text_recognition_model_name='PP-OCRv6_small_rec',
        return_word_box=True, text_det_box_thresh=0.5, text_det_thresh=0.4,
        text_det_unclip_ratio=1.5, device="gpu", engine="paddle"
    )


def _load_paddleocr_medium():
    return PaddleOCR(
        use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
        text_detection_model_name='PP-OCRv6_medium_det', text_recognition_model_name='PP-OCRv6_medium_rec',
        return_word_box=True, text_det_box_thresh=0.5, text_det_thresh=0.4,
        text_det_unclip_ratio=1.5, device="gpu", engine="paddle"
    )


def _load_vgg16():
    weights_path = "./models/similarity_vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
            )
            print("Using GPU:", gpus[0])
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found, using CPU.")

    model = VGG16(weights=None, include_top=False)
    model.load_weights(weights_path)
    return model


def _safe_load(name, loader):
    """安全加载模型，失败时返回 None 并打印警告。"""
    try:
        return loader()
    except Exception as e:
        print(f"[config] 模型 {name} 加载失败: {e}")
        return None


_MODEL_LOADERS = {
    'handwriting_model':           lambda: load_model('./models/handwriting/handwriting'),
    'structure':                   lambda: LayoutDetection(model_name="PP-DocLayoutV2"),
    'paddleOCR':                   _load_paddleocr_small,
    'OCR':                         _load_paddleocr_medium,
    'table_OCR':                   lambda: _models['OCR'],  # 与 OCR 共用同一实例
    'uv_model':                    lambda: create_model("UVDoc", device="gpu"),
    'wired_predictor':             lambda: create_predictor(model_name="RT-DETR-L_wired_table_cell_det"),
    'cls_model':                   lambda: TableClassification(model_name="PP-LCNet_x1_0_table_cls"),
    'text_det_predictor':          lambda: create_predictor(model_name="PP-OCRv6_small_det", device="gpu"),
    'orientation_model':           lambda: ort.InferenceSession(
        "./models/rapid_orientation.onnx",
        providers=['CPUExecutionProvider']
    ),
    'textline_orientation_model':  lambda: create_predictor(
        model_name="PP-LCNet_x1_0_textline_ori",
        model_dir="./models/textline_orientation/",
        device="gpu"
    ),
    'redmaskRec_model':            lambda: _safe_load("redmaskRec_model", lambda: RedMaskRec("models/redmask_rec")),
    'yolo_model':                  lambda: _safe_load("yolo_model", lambda: YOLO11("models/redmask_inference_model/seal_yolo11n_first.onnx")),
    'model_similarity':            lambda: _safe_load('model_similarity', _load_vgg16),
}


def __getattr__(name):
    if name in _MODEL_LOADERS:
        if name not in _models:
            _models[name] = _MODEL_LOADERS[name]()
        return _models[name]
    raise AttributeError(f"module 'config' has no attribute '{name}'")