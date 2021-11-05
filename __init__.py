from fastapi import APIRouter, Body, File

from app.core.module_class import ApiModule
from app.core.package_manager import PackageManager


class HogFaceDetection(ApiModule):
    def _register_api_bp(self, bp: APIRouter):
        @bp.post('/init')
        def init():
            PackageManager.install_package(
                ['opencv-python', 'opencv-contrib-python'], 'cv2')
            # dlibのインストールは以下のコマンドが必要です
            # brew install cmake
            # PackageManager.install_package(['numpy'])
            PackageManager.install_package(['wheel'])
            PackageManager.install_package(['dlib'])

        @bp.post('/detect')
        def detect(file: bytes = File(...)):
            try:
                import cv2
                import dlib
            except ModuleNotFoundError as e:
                return '初期化していない、または初期化失敗'

    def _get_tag(self) -> str:
        return 'Hog顔検出'

    def get_module_name(self) -> str:
        return 'hog_face_detection'


hog_face_detection = HogFaceDetection()
