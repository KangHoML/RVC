import logging
import os

# Hubert 모델 다운로드를 위한 명령어 (현재는 주석 처리됨)
# os.system("wget -P cvec/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt")

# Gradio 웹 인터페이스 라이브러리 가져오기
import gradio as gr

# 환경 변수 로드를 위한 라이브러리 가져오기
from dotenv import load_dotenv

# 프로젝트 내부 모듈 가져오기
from configs import Config  # 설정 관리 클래스
from i18n.i18n import I18nAuto  # 다국어 지원 모듈
from infer.modules.vc import VC  # 음성 변환 모듈

# 불필요한 로그 메시지를 줄이기 위해 여러 라이브러리의 로깅 레벨 설정
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# 국제화(i18n) 객체 초기화 - 다국어 지원을 위한 설정
i18n = I18nAuto()
logger.info(i18n)

# .env 파일에서 환경 변수 로드
load_dotenv()
# 프로젝트 설정 객체 생성
config = Config()
# 음성 변환(Voice Conversion) 객체 초기화
vc = VC(config)

# 환경 변수에서 모델 가중치, UVR5 가중치, 인덱스 파일 경로 가져오기
weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")

# 사용 가능한 모델(.pth 파일) 찾기
names = []
hubert_model = None
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)

# 사용 가능한 인덱스 파일(.index 파일) 찾기
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))


# Gradio 웹 애플리케이션 객체 생성
app = gr.Blocks()
with app:
    # 탭 인터페이스 생성
    with gr.Tabs():
        # "Online Demo" 탭 생성
        with gr.TabItem("Online Demo"):
            # 마크다운으로 제목 표시
            gr.Markdown(
                value="""
                RVC Online Demo
                """
            )
            # 모델 선택을 위한 드롭다운 메뉴
            sid = gr.Dropdown(label=i18n("Inferencing voice"), choices=sorted(names))

            # 화자/가수 ID 선택을 위한 슬라이더 (기본적으로 숨겨져 있음)
            with gr.Column():
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("Select Speaker/Singer ID"),
                    value=0,
                    visible=False,
                    interactive=True,
                )
            # 모델 변경 시 화자/가수 ID 정보 갱신
            sid.change(fn=vc.get_vc, inputs=[sid], outputs=[spk_item])

            # 음조 변환 설명 텍스트
            gr.Markdown(
                value=i18n(
                    "Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12)"
                )
            )

            # 오디오 파일 업로드 입력
            vc_input3 = gr.Audio(label="Upload audio (length less than 90 seconds)")

            # 음조 변환 값 입력
            vc_transform0 = gr.Number(
                label=i18n(
                    "Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12)"
                ),
                value=0,
            )

            # 피치 추출 알고리즘 선택 라디오 버튼
            f0method0 = gr.Radio(
                label=i18n(
                    "Select the pitch extraction algorithm ('pm': faster extraction but lower-quality speech; 'harvest': better bass but extremely slow; 'crepe': better quality but GPU intensive), 'rmvpe': best quality, and little GPU requirement"
                ),
                choices=["pm", "harvest", "crepe", "rmvpe"],
                value="pm",
                interactive=True,
            )

            # 피치 필터링 반경 설정 슬라이더
            filter_radius0 = gr.Slider(
                minimum=0,
                maximum=7,
                label=i18n(
                    "If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness."
                ),
                value=3,
                step=1,
                interactive=True,
            )

            # 특성 인덱스 파일 경로 입력을 위한 텍스트 박스 (숨겨져 있음)
            with gr.Column():
                file_index1 = gr.Textbox(
                    label=i18n(
                        "Path to the feature index file. Leave blank to use the selected result from the dropdown"
                    ),
                    value="",
                    interactive=False,
                    visible=False,
                )

            # 자동 탐지된 인덱스 파일 선택을 위한 드롭다운
            file_index2 = gr.Dropdown(
                label=i18n("Auto-detect index path and select from the dropdown"),
                choices=sorted(index_paths),
                interactive=True,
            )

            # 특성 검색 비율 설정 슬라이더
            index_rate1 = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Feature searching ratio"),
                value=0.88,
                interactive=True,
            )

            # 출력 오디오 리샘플링 설정 슬라이더
            resample_sr0 = gr.Slider(
                minimum=0,
                maximum=48000,
                label=i18n(
                    "Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling"
                ),
                value=0,
                step=1,
                interactive=True,
            )

            # 볼륨 스케일링 비율 설정 슬라이더
            rms_mix_rate0 = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n(
                    "Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume"
                ),
                value=1,
                interactive=True,
            )

            # 무성 자음 및 숨소리 보호 설정 슬라이더
            protect0 = gr.Slider(
                minimum=0,
                maximum=0.5,
                label=i18n(
                    "Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy"
                ),
                value=0.33,
                step=0.01,
                interactive=True,
            )

            # F0 곡선 파일 업로드 입력
            f0_file = gr.File(
                label=i18n(
                    "F0 curve file (optional). One pitch per line. Replaces the default F0 and pitch modulation"
                )
            )

            # 음성 변환 실행 버튼
            but0 = gr.Button(i18n("Convert"), variant="primary")

            # 출력 정보 표시를 위한 텍스트 박스
            vc_output1 = gr.Textbox(label=i18n("Output information"))

            # 변환된 오디오 출력
            vc_output2 = gr.Audio(
                label=i18n(
                    "Export audio (click on the three dots in the lower right corner to download)"
                )
            )

            # 변환 버튼 클릭 시 실행할 함수 및 입출력 연결
            but0.click(
                vc.vc_single,  # 호출할 함수
                [  # 입력 파라미터 목록
                    spk_item,  # 화자/가수 ID
                    vc_input3,  # 입력 오디오
                    vc_transform0,  # 음조 변환 값
                    f0_file,  # F0 곡선 파일
                    f0method0,  # 피치 추출 알고리즘
                    file_index1,  # 특성 인덱스 파일 경로
                    file_index2,  # 자동 탐지된 인덱스 파일
                    # file_big_npy1,  # 주석 처리된 파라미터
                    index_rate1,  # 특성 검색 비율
                    filter_radius0,  # 필터 반경
                    resample_sr0,  # 리샘플링 샘플 레이트
                    rms_mix_rate0,  # 볼륨 스케일링 비율
                    protect0,  # 무성 자음 보호 설정
                ],
                [vc_output1, vc_output2],  # 출력 객체 목록
            )


# Gradio 애플리케이션 실행
app.launch()
