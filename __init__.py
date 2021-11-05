from fastapi import APIRouter, Body, File, HTTPException

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
                face_detect = dlib.get_frontal_face_detector()
                rects = face_detect(gray, 1)
                rt = []
                for (i, rect) in enumerate(rects):
                    rect: dlib.rectangle
                    # print(rect.bottom())
                    m = {}
                    m['top'] = rect.top()
                    m['left'] = rect.left()
                    m['right'] = rect.right()
                    m['bottom'] = rect.bottom()
                    rt.append(m)
                return rt
            except ModuleNotFoundError as e:
                raise HTTPException(412, '初期化していない、または初期化失敗(条件満たしていない)')

        def _get_tag(self) -> str:
            return 'Hog顔検出'

        def get_module_name(self) -> str:
            return 'hog_face_detection'

    hog_face_detection = HogFaceDetection()
