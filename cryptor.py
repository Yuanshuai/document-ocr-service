import os
import argparse
from cryptography.fernet import Fernet


class FolderCryptor:
    def __init__(self, key):
        self.key = key[:44]

    @staticmethod
    def is_encrypted_file(file_path):
        with open(file_path, "rb") as f:
            header = f.read(6)
        return header == b"gAAAAA"

    def decrypt_file(self, file_path):
        if not self.is_encrypted_file(file_path):
            return False
        with open(file_path, "rb") as file:
            encrypted_data = file.read()
        try:
            data = Fernet(self.key).decrypt(encrypted_data)
            with open(file_path, "wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            return True
        except Exception as e:
            print(f"解密文件 {file_path} 失败: {e}")
            return False

    def decrypt_folder(self, folder_path):
        result = {"decrypted": 0, "skipped": 0, "errors": 0}
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if self.decrypt_file(file_path):
                    result["decrypted"] += 1
                else:
                    result["skipped"] += 1
        return result

    def encrypt_file(self, file_path):
        if self.is_encrypted_file(file_path):
            return False
        with open(file_path, "rb") as file:
            data = file.read()
        encrypted_data = Fernet(self.key).encrypt(data)
        with open(file_path, "wb") as file:
            file.write(encrypted_data)
        return True

    def encrypt_folder(self, folder_path):
        result = {"encrypted": 0, "skipped": 0, "errors": 0}
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if self.encrypt_file(file_path):
                        result["encrypted"] += 1
                    else:
                        result["skipped"] += 1
                except Exception as e:
                    print(f"加密文件 {file_path} 失败: {e}")
                    result["errors"] += 1
        return result

    def status(self, folder_path):
        result = {"encrypted": 0, "decrypted": 0, "total": 0}
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                result["total"] += 1
                if self.is_encrypted_file(file_path):
                    result["encrypted"] += 1
                else:
                    result["decrypted"] += 1
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="模型文件夹加密/解密/检查工具")
    parser.add_argument("action", choices=["check", "decrypt", "encrypt"],
                        help="check: 仅检查状态; decrypt: 解密; encrypt: 加密")
    parser.add_argument("folder", nargs="?", default="models",
                        help="目标文件夹 (默认: models)")
    parser.add_argument("--key", default=None,
                        help="加密密钥 (也可通过 MODEL_KEY 环境变量提供)")

    args = parser.parse_args()

    scret_key = (args.key or os.environ.get("MODEL_KEY", "")).encode()
    if not scret_key:
        parser.error("请通过 --key 或 MODEL_KEY 环境变量提供加密密钥")
    cryptor = FolderCryptor(scret_key)

    if args.action == "check":
        s = cryptor.status(args.folder)
        pct = (s["encrypted"] / s["total"] * 100) if s["total"] else 0
        print(f"[{args.folder}] 共 {s['total']} 个文件, "
              f"已加密 {s['encrypted']} ({pct:.0f}%), "
              f"已解密 {s['decrypted']}")
    elif args.action == "decrypt":
        s = cryptor.status(args.folder)
        if s["encrypted"] == 0:
            print(f"[{args.folder}] 已全部解密，无需操作")
        else:
            r = cryptor.decrypt_folder(args.folder)
            print(f"[{args.folder}] 解密 {r['decrypted']} 个文件, "
                  f"跳过 {r['skipped']} 个, 错误 {r['errors']} 个")
    elif args.action == "encrypt":
        s = cryptor.status(args.folder)
        if s["encrypted"] == s["total"]:
            print(f"[{args.folder}] 已全部加密，无需操作")
        else:
            r = cryptor.encrypt_folder(args.folder)
            print(f"[{args.folder}] 加密 {r['encrypted']} 个文件, "
                  f"跳过 {r['skipped']} 个, 错误 {r['errors']} 个")
