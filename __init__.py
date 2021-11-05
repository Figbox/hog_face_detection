import random

from fastapi import APIRouter, Body, Depends

from app.core.adaptor.DbAdaptor import DbAdaptor
from app.core.module_class import TableModule, ApiModule
from app.core.package_manager import PackageManager
from app.core.page_engine.PageAdaptor import PageAdaptor
from app.modules.sample.table import SampleTable


# TODO: change class name
class HogFaceDetection(ApiModule):
    def _register_api_bp(self, bp: APIRouter):
        @bp.post('/init')
        def init():
            PackageManager.install_package(
                ['opencv-python', 'opencv-contrib-python'], 'cv2')
            PackageManager.install_package(['dlib'])

        @bp.post('/detect')
        def show_body(body: str = Body(..., embed=True)):
            return f'your body is: {body}'

    def _get_tag(self) -> str:
        return 'Hog顔検出'

    def get_module_name(self) -> str:
        return 'hog_face_detection'


hog_face_detection = HogFaceDetection()
